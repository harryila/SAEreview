"""Deterministic candidate and feature-evidence preparation.

The module deliberately keeps answer-bearing SCAR fields in a separate qrels
sidecar.  Nothing in this file calls a language model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence
import warnings

import numpy as np
import torch
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.random_projection import SparseRandomProjection

from complementarity_gate import load_exact_sae, load_raw_openai_embeddings
from sae_smoke_test import (
    build_direction,
    l2_normalize,
    load_jsonl,
    normalize_label,
    score_from_embeddings,
    validate_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.json")
DEFAULT_DATA = ROOT / "data" / "scar_system_analogy_en.jsonl"
DEFAULT_EMBEDDINGS = ROOT / ".cache" / "openai_text_embedding_3_small_scar.npz"
DEFAULT_COMPLEMENTARITY = ROOT / "results" / "complementarity_queries.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("outputs") / "development"

TOP_K = 10
DENSE_POOL_SIZE = 30
SHARED_FEATURES_PER_PAIR = 3
DESCRIBED_FEATURES_PER_REPRESENTATION = 128
DESCRIPTION_EXAMPLES_PER_FEATURE = 6
FEATURE_DESCRIPTION_BATCH_SIZE = 4
FEATURE_DESCRIPTION_FREQUENCY_BINS = 8
FEATURE_DESCRIPTION_SHUFFLE_NAMESPACE = (
    "structural-rescue-feature-description-shuffle-v1"
)
RANDOM_SEED_NAMESPACE = "structural-rescue-random-source-oracle-v1"
RANDOM_SEED_PAIR_COUNT = 64
LEGACY_RANDOM_SEED_PAIR = (2026090201, 2026090202)
VERIFIED_RANDOM_PAIR_INDICES = (0, 1, 2)
SCREEN_DENSE_CONTROL_COUNT = 54
VERIFIER_BATCH_SIZE = 64


def _derive_random_seed(pair_index: int, member_index: int) -> int:
    if pair_index < 0 or member_index not in (0, 1):
        raise ValueError("Random seed coordinates are out of range")
    payload = (
        f"{RANDOM_SEED_NAMESPACE}\0pair={pair_index}\0member={member_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


RANDOM_SEED_PAIRS = (
    LEGACY_RANDOM_SEED_PAIR,
    *(
        tuple(
            _derive_random_seed(pair_index, member_index)
            for member_index in (0, 1)
        )
        for pair_index in range(1, RANDOM_SEED_PAIR_COUNT)
    ),
)
# Kept as a narrow compatibility alias for callers that previously imported the
# single control's two seeds. New preparation always uses RANDOM_SEED_PAIRS.
RANDOM_SEEDS = LEGACY_RANDOM_SEED_PAIR

CHECKPOINTS = {
    "cslg": {
        "path": ROOT / "weights" / "csLG_64_9216.pth",
        "sha256": "29073be46ce5ddceee53f7e9ebf46449e239c1bc29f57dfebced041833698752",
    },
    "astroph": {
        "path": ROOT / "weights" / "astroPH_64_9216.pth",
        "sha256": "112e8a006ff0cc8e3b4439e1ef28df816564c5d9054974a763eaa69804cf02ed",
    },
}

ARM_NAMES = (
    "dense_ranking",
    "dense30_structure",
    "sae_union_padded30_structure",
    "random_union_padded30_structure_1",
    "random_union_padded30_structure_2",
    "random_union_padded30_structure_3",
    "sae_union_padded30_activation_only",
    "sae_union_padded30_aligned_description",
    "sae_union_padded30_shuffled_description",
)


@dataclass(frozen=True)
class Representation:
    """Sparse activations and their normalized retrieval representation."""

    values: np.ndarray
    unit: np.ndarray
    idf: np.ndarray
    active_value_distributions: dict[int, np.ndarray]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


RANDOM_SEED_PAIRS_SHA256 = canonical_json_sha256(RANDOM_SEED_PAIRS)


def deterministic_candidate_order(
    query_id: str, candidate_ids: Iterable[str]
) -> list[str]:
    return sorted(
        set(candidate_ids),
        key=lambda candidate: (
            hashlib.sha256(f"{query_id}\0{candidate}".encode()).hexdigest(),
            candidate,
        ),
    )


def verifier_batch_plan(
    prepared: Sequence[dict[str, Any]], screen: dict[str, Any]
) -> dict[str, Any]:
    by_id = {str(row["query_id"]): row for row in prepared}
    queries: list[dict[str, Any]] = []
    for query_id in map(str, screen["query_ids"]):
        candidate_ids = deterministic_candidate_order(
            query_id, by_id[query_id]["superpool"]
        )
        batches = []
        for offset in range(0, len(candidate_ids), VERIFIER_BATCH_SIZE):
            batch_ids = candidate_ids[offset : offset + VERIFIER_BATCH_SIZE]
            aliases = {
                f"C{index:03d}": candidate_id
                for index, candidate_id in enumerate(batch_ids, start=1)
            }
            batches.append(
                {
                    "batch_id": canonical_json_sha256(
                        {"query_id": query_id, "candidate_ids": batch_ids}
                    ),
                    "candidate_ids": batch_ids,
                    "alias_map_sha256": canonical_json_sha256(aliases),
                }
            )
        queries.append({"query_id": query_id, "batches": batches})
    return {
        "selection": screen["selection"],
        "batch_size": VERIFIER_BATCH_SIZE,
        "query_count": len(queries),
        "batch_count": sum(len(row["batches"]) for row in queries),
        "queries": queries,
    }


def largest_batch_preflight(
    prepared: Sequence[dict[str, Any]], *, query_count: int = 2
) -> dict[str, Any]:
    """Select largest verifier contexts without consulting outcomes or qrels."""

    if query_count <= 0:
        raise ValueError("query_count must be positive")
    if not prepared:
        raise ValueError("No prepared queries available for preflight")
    query_ids = [str(row["query_id"]) for row in prepared]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Prepared queries contain duplicate query IDs")
    ordered = sorted(
        prepared,
        key=lambda row: (
            -len(set(map(str, row["superpool"]))),
            hashlib.sha256(
                f"structural-rescue-largest-batch-v1\0{row['query_id']}".encode()
            ).hexdigest(),
            str(row["query_id"]),
        ),
    )
    chosen = ordered[:query_count]
    if len(chosen) != query_count:
        raise ValueError(
            f"Requested {query_count} preflight queries, found {len(chosen)}"
        )
    return {
        "selection": "qrels_free_largest_superpool_preflight",
        "query_ids": [str(row["query_id"]) for row in chosen],
        "superpool_sizes": [len(set(map(str, row["superpool"]))) for row in chosen],
        "qrels_used": False,
        "population_estimate_allowed": False,
    }


def feature_description_shuffle_map(
    catalog: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Freeze a frequency-matched cyclic derangement before descriptions exist."""

    by_representation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_row in catalog:
        row = dict(raw_row)
        by_representation[str(row["representation"])].append(row)
    if not by_representation:
        raise ValueError("Cannot construct a shuffle map from an empty catalog")

    mappings: list[dict[str, Any]] = []
    bin_counts: dict[str, list[int]] = {}
    for representation in sorted(by_representation):
        rows = sorted(
            by_representation[representation],
            key=lambda row: (
                int(row["corpus_active_count"]),
                str(row["feature_key"]),
            ),
        )
        if len(rows) != DESCRIBED_FEATURES_PER_REPRESENTATION:
            raise ValueError(
                f"Expected {DESCRIBED_FEATURES_PER_REPRESENTATION} catalog features "
                f"for {representation}, found {len(rows)}"
            )
        bins = np.array_split(
            np.asarray(rows, dtype=object), FEATURE_DESCRIPTION_FREQUENCY_BINS
        )
        bin_counts[representation] = [int(len(bin_rows)) for bin_rows in bins]
        if len(set(bin_counts[representation])) != 1:
            raise AssertionError("Frequency bins must have equal feature counts")
        for bin_index, raw_bin_rows in enumerate(bins):
            bin_rows = list(raw_bin_rows)
            if len(bin_rows) < 2:
                raise ValueError("Every frequency bin needs at least two features")
            ordered = sorted(
                bin_rows,
                key=lambda row: (
                    hashlib.sha256(
                        (
                            f"{FEATURE_DESCRIPTION_SHUFFLE_NAMESPACE}\0"
                            f"{representation}\0{bin_index}\0{row['feature_key']}"
                        ).encode("utf-8")
                    ).hexdigest(),
                    str(row["feature_key"]),
                ),
            )
            for source_index, source in enumerate(ordered):
                donor = ordered[(source_index + 1) % len(ordered)]
                if source["feature_key"] == donor["feature_key"]:
                    raise AssertionError("Description shuffle must be a derangement")
                mappings.append(
                    {
                        "source_feature_key": str(source["feature_key"]),
                        "source_corpus_active_count": int(
                            source["corpus_active_count"]
                        ),
                        "donor_feature_key": str(donor["feature_key"]),
                        "donor_corpus_active_count": int(
                            donor["corpus_active_count"]
                        ),
                        "representation": representation,
                        "frequency_bin": bin_index,
                        "bin_feature_count": len(ordered),
                    }
                )
    mappings.sort(key=lambda row: row["source_feature_key"])
    if len({row["source_feature_key"] for row in mappings}) != len(mappings):
        raise AssertionError("Shuffle sources must be unique")
    if len({row["donor_feature_key"] for row in mappings}) != len(mappings):
        raise AssertionError("Shuffle donors must form a permutation")
    return {
        "scheme": "frequency_bin_hash_ordered_one_step_cyclic_derangement",
        "namespace": FEATURE_DESCRIPTION_SHUFFLE_NAMESPACE,
        "frequency_bins_per_representation": FEATURE_DESCRIPTION_FREQUENCY_BINS,
        "bin_feature_counts": bin_counts,
        "mapping_count": len(mappings),
        "mappings": mappings,
    }


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            )


def _strict_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "exploratory_development_only":
        raise ValueError("Structural Rescue protocol must remain development-only")
    if protocol.get("scope", {}).get("latent_choice_test_prompts_may_be_used") is not False:
        raise ValueError("Latent Choice test prompts must be forbidden")
    candidate_generation = protocol["candidate_generation"]
    if candidate_generation.get("random_seed_namespace") != RANDOM_SEED_NAMESPACE:
        raise ValueError("Protocol random seed namespace does not match implementation")
    if candidate_generation.get("random_seed_pair_count") != RANDOM_SEED_PAIR_COUNT:
        raise ValueError(
            "Protocol random seed pair count does not match implementation"
        )
    if (
        candidate_generation.get("random_seed_pairs_sha256")
        != RANDOM_SEED_PAIRS_SHA256
    ):
        raise ValueError("Protocol random seed pairs do not match implementation")
    if (
        tuple(candidate_generation.get("verified_random_pair_indices", ()))
        != VERIFIED_RANDOM_PAIR_INDICES
    ):
        raise ValueError(
            "Protocol verified random controls do not match implementation"
        )
    return protocol


def _topk(scores: np.ndarray, k: int = TOP_K) -> list[int]:
    candidate_indices = np.arange(len(scores), dtype=np.int64)
    return np.lexsort((candidate_indices, -np.asarray(scores)))[:k].astype(int).tolist()


def stable_union(*rankings: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for ranking in rankings:
        for value in ranking:
            candidate = int(value)
            if candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
    return ordered


def pad_pool_to_size(
    source_pool: Sequence[int],
    dense_ranking: Sequence[int],
    *,
    size: int = DENSE_POOL_SIZE,
) -> list[int]:
    """Append the next unused dense candidates without changing source membership."""

    source = [int(value) for value in source_pool]
    if len(source) != len(set(source)):
        raise ValueError("Source pool must not contain duplicates")
    if len(source) > size:
        raise ValueError(
            f"Source pool of size {len(source)} exceeds target size {size}"
        )
    padded = stable_union(source, dense_ranking)
    if len(padded) < size:
        raise ValueError(
            f"Only {len(padded)} unique candidates available for size {size}"
        )
    result = padded[:size]
    if result[: len(source)] != source or not set(source).issubset(result):
        raise AssertionError("Padding changed the unpadded source pool")
    return result


def canonical_pair_group(first: str, second: str) -> str:
    labels = sorted((normalize_label(first), normalize_label(second)))
    return "\u241f".join(labels)


def system_id(side: str, row: dict[str, Any]) -> str:
    if side not in {"a", "b"}:
        raise ValueError(f"Unknown corpus side {side}")
    return f"{side}:{int(row['id'])}"


def parse_system_id(value: str, rows_by_id: dict[int, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    try:
        side, raw_id = value.split(":", 1)
        row = rows_by_id[int(raw_id)]
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Invalid system id: {value}") from exc
    if side not in {"a", "b"}:
        raise ValueError(f"Invalid system side: {side}")
    return side, row


def system_payload(value: str, rows_by_id: dict[int, dict[str, Any]]) -> dict[str, str]:
    """Return the only two SCAR fields allowed into mechanism extraction."""

    side, row = parse_system_id(value, rows_by_id)
    prefix = "system_a" if side == "a" else "system_b"
    return {
        "system_id": value,
        "name": str(row[prefix]),
        "background": str(row[f"{prefix}_background"]),
    }


def _activation_distributions(values: np.ndarray) -> dict[int, np.ndarray]:
    row_ids, feature_ids = np.nonzero(values > 0)
    del row_ids
    order = np.argsort(feature_ids, kind="stable")
    sorted_features = feature_ids[order]
    sorted_values = values[values > 0][order]
    unique, starts = np.unique(sorted_features, return_index=True)
    distributions: dict[int, np.ndarray] = {}
    for offset, feature_id in enumerate(unique):
        start = int(starts[offset])
        end = int(starts[offset + 1]) if offset + 1 < len(starts) else len(order)
        distributions[int(feature_id)] = np.sort(sorted_values[start:end])
    return distributions


def make_representation(values: np.ndarray) -> Representation:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 9216:
        raise ValueError(f"Expected [n, 9216] sparse values, found {values.shape}")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Sparse activations must be finite and nonnegative")
    document_frequency = np.count_nonzero(values > 0, axis=0)
    idf = (
        np.log((len(values) + 1.0) / (document_frequency + 1.0)) + 1.0
    ).astype(np.float32)
    return Representation(
        values=values,
        unit=l2_normalize(values),
        idf=idf,
        active_value_distributions=_activation_distributions(values),
    )


def encode_sae(raw_embeddings: np.ndarray, checkpoint: Path) -> Representation:
    model = load_exact_sae(checkpoint)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(raw_embeddings), 64):
            batch = torch.from_numpy(raw_embeddings[start : start + 64]).float()
            batches.append(model.encode(batch).cpu().numpy().astype(np.float32))
    return make_representation(np.concatenate(batches, axis=0))


def _random_sparse_values(raw_embeddings: np.ndarray, *, seed: int) -> np.ndarray:
    projector = SparseRandomProjection(
        n_components=9216,
        density="auto",
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataDimensionalityWarning)
        projected = np.asarray(
            projector.fit_transform(l2_normalize(raw_embeddings)), dtype=np.float32
        )
    top_indices = np.argpartition(projected, -64, axis=1)[:, -64:]
    row_indices = np.arange(len(projected))[:, None]
    top_values = np.maximum(projected[row_indices, top_indices], 0.0)
    sparse = np.zeros_like(projected, dtype=np.float32)
    sparse[row_indices, top_indices] = top_values
    return sparse


def random_representation(raw_embeddings: np.ndarray, *, seed: int) -> Representation:
    """Dimension/sparsity-matched random projection; not an SAE surrogate."""

    return make_representation(_random_sparse_values(raw_embeddings, seed=seed))


def _percentile(representation: Representation, feature_id: int, value: float) -> float:
    distribution = representation.active_value_distributions.get(int(feature_id))
    if distribution is None or value <= 0:
        return 0.0
    return float(np.searchsorted(distribution, value, side="right") / len(distribution))


def shared_feature_rows(
    representation: Representation,
    query_global: int,
    candidate_global: int,
    *,
    namespace: str,
    limit: int,
    allowed_features: set[int] | None = None,
) -> list[dict[str, Any]]:
    query_values = representation.values[query_global]
    candidate_values = representation.values[candidate_global]
    common = np.flatnonzero((query_values > 0) & (candidate_values > 0))
    evidence: list[dict[str, Any]] = []
    for raw_feature_id in common:
        feature_id = int(raw_feature_id)
        if allowed_features is not None and feature_id not in allowed_features:
            continue
        query_percentile = _percentile(
            representation, feature_id, float(query_values[feature_id])
        )
        candidate_percentile = _percentile(
            representation, feature_id, float(candidate_values[feature_id])
        )
        shared_strength = min(query_percentile, candidate_percentile)
        evidence.append(
            {
                "feature_key": f"{namespace}:{feature_id}",
                "feature_id": feature_id,
                "query_activation_percentile": query_percentile,
                "candidate_activation_percentile": candidate_percentile,
                "shared_strength": shared_strength,
                "selection_score": shared_strength * float(representation.idf[feature_id]),
            }
        )
    evidence.sort(key=lambda row: (-row["selection_score"], row["feature_id"]))
    return evidence[:limit]


def _candidate_global(direction: str, candidate_index: int, n_rows: int) -> int:
    return n_rows + candidate_index if direction == "a_to_b" else candidate_index


def _query_global(direction: str, row_index: int, n_rows: int) -> int:
    return row_index if direction == "a_to_b" else n_rows + row_index


def _candidate_side(direction: str) -> str:
    return "b" if direction == "a_to_b" else "a"


def _query_side(direction: str) -> str:
    return "a" if direction == "a_to_b" else "b"


def _candidate_ids(indices: Sequence[int], rows: Sequence[dict[str, Any]], direction: str) -> list[str]:
    side = _candidate_side(direction)
    return [system_id(side, rows[int(index)]) for index in indices]


def _candidate_index(candidate_id: str, row_index_by_id: dict[int, int]) -> int:
    _, raw_id = candidate_id.split(":", 1)
    return row_index_by_id[int(raw_id)]


def _load_and_validate_inputs(
    data_path: Path,
    embeddings_path: Path,
    complementarity_path: Path,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    expected = protocol["sources"]
    observed = {
        "scar_sha256": sha256_file(data_path),
        "embedding_cache_sha256": sha256_file(embeddings_path),
        "complementarity_queries_sha256": sha256_file(complementarity_path),
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise ValueError(f"{key} mismatch: expected {expected[key]}, observed {value}")
    rows = load_jsonl(data_path)
    validate_rows(rows)
    if len(rows) != 400:
        raise ValueError(f"Expected 400 SCAR rows, found {len(rows)}")
    raw_embeddings = load_raw_openai_embeddings(embeddings_path, len(rows))
    complementarity = load_jsonl(complementarity_path)
    if len(complementarity) != 566:
        raise ValueError("Expected 566 frozen complementarity query records")
    return rows, raw_embeddings, complementarity


def _checkpoint_representation(
    name: str, raw_embeddings: np.ndarray, protocol: dict[str, Any]
) -> Representation:
    config = CHECKPOINTS[name]
    observed = sha256_file(config["path"])
    expected = protocol["sources"][f"{name}_sae_sha256"]
    if observed != expected or observed != config["sha256"]:
        raise ValueError(f"{name} checkpoint hash mismatch: {observed}")
    return encode_sae(raw_embeddings, config["path"])


def _random_source_unions(
    raw_embeddings: np.ndarray,
    dense_scores: dict[str, np.ndarray],
    cross_domain_row_ids: np.ndarray,
    *,
    n_rows: int,
) -> list[dict[str, list[list[int]]]]:
    """Materialize 64 random source pools while retaining no projection matrices."""

    dense_rankings = {
        direction: [_topk(row, TOP_K) for row in dense_scores[direction]]
        for direction in ("a_to_b", "b_to_a")
    }
    unions: list[dict[str, list[list[int]]]] = []
    for seed_pair in RANDOM_SEED_PAIRS:
        component_rankings: list[dict[str, list[list[int]]]] = []
        for seed in seed_pair:
            random_unit = l2_normalize(
                _random_sparse_values(raw_embeddings, seed=seed)
            )
            scores = score_from_embeddings(
                random_unit[:n_rows],
                random_unit[n_rows:],
                cross_domain_row_ids,
            )
            component_rankings.append(
                {
                    direction: [_topk(row, TOP_K) for row in scores[direction]]
                    for direction in ("a_to_b", "b_to_a")
                }
            )
            del scores, random_unit
        unions.append(
            {
                direction: [
                    stable_union(
                        dense_rankings[direction][position],
                        component_rankings[0][direction][position],
                        component_rankings[1][direction][position],
                    )
                    for position in range(len(cross_domain_row_ids))
                ]
                for direction in ("a_to_b", "b_to_a")
            }
        )
    return unions


def _query_rows(
    rows: Sequence[dict[str, Any]],
    cross_domain_row_ids: np.ndarray,
    score_matrices: dict[str, dict[str, np.ndarray]],
    random_source_unions: Sequence[dict[str, Sequence[Sequence[int]]]],
    complementarity: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(random_source_unions) != RANDOM_SEED_PAIR_COUNT:
        raise ValueError(
            f"Expected {RANDOM_SEED_PAIR_COUNT} random source pools, "
            f"found {len(random_source_unions)}"
        )
    frozen_by_id = {str(row["query_id"]): row for row in complementarity}
    prepared: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    directions = {
        name: build_direction(rows, cross_domain_row_ids, name)
        for name in ("a_to_b", "b_to_a")
    }

    for direction in ("a_to_b", "b_to_a"):
        retrieval = directions[direction]
        for position, raw_row_index in enumerate(cross_domain_row_ids):
            row_index = int(raw_row_index)
            row = rows[row_index]
            query_id = f"{direction}:{row['id']}"
            frozen = frozen_by_id.get(query_id)
            if frozen is None:
                raise ValueError(f"Missing frozen query record {query_id}")

            rankings = {
                name: _topk(scores[direction][position], TOP_K)
                for name, scores in score_matrices.items()
            }
            dense30 = _topk(score_matrices["dense"][direction][position], DENSE_POOL_SIZE)
            sae_union = stable_union(
                rankings["dense"], rankings["cslg"], rankings["astroph"]
            )
            random_unions = [
                [int(value) for value in source[direction][position]]
                for source in random_source_unions
            ]
            if len(sae_union) > 30 or any(
                len(random_union) > 30 for random_union in random_unions
            ):
                raise AssertionError(
                    "Three top-10 lists cannot yield more than 30 candidates"
                )

            sae_padded = pad_pool_to_size(sae_union, dense30)
            verified_random_padded = [
                pad_pool_to_size(random_unions[index], dense30)
                for index in VERIFIED_RANDOM_PAIR_INDICES
            ]

            pools_as_indices = {
                "dense_ranking": rankings["dense"],
                "dense30_structure": dense30,
                "sae_union_padded30_structure": sae_padded,
                "random_union_padded30_structure_1": verified_random_padded[0],
                "random_union_padded30_structure_2": verified_random_padded[1],
                "random_union_padded30_structure_3": verified_random_padded[2],
                "sae_union_padded30_activation_only": sae_padded,
                "sae_union_padded30_aligned_description": sae_padded,
                "sae_union_padded30_shuffled_description": sae_padded,
            }
            if tuple(pools_as_indices) != ARM_NAMES:
                raise AssertionError(
                    "Candidate arm order drifted from the frozen contract"
                )
            for arm, indices in pools_as_indices.items():
                expected_size = TOP_K if arm == "dense_ranking" else DENSE_POOL_SIZE
                if len(indices) != expected_size or len(set(indices)) != expected_size:
                    raise AssertionError(
                        f"{arm} must contain exactly {expected_size} unique candidates"
                    )
            superpool = stable_union(
                dense30,
                sae_padded,
                *verified_random_padded,
            )
            side = _candidate_side(direction)
            dense_scores = score_matrices["dense"][direction][position]
            prepared.append(
                {
                    "query_id": query_id,
                    "pair_id": int(row["id"]),
                    "pair_group": canonical_pair_group(
                        str(row["system_a"]), str(row["system_b"])
                    ),
                    "direction": direction,
                    "query_system_id": system_id(_query_side(direction), row),
                    "pools": {
                        arm: _candidate_ids(indices, rows, direction)
                        for arm, indices in pools_as_indices.items()
                    },
                    "source_pools": {
                        "sae_union": _candidate_ids(sae_union, rows, direction),
                        "random_unions": [
                            _candidate_ids(indices, rows, direction)
                            for indices in random_unions
                        ],
                    },
                    "superpool": _candidate_ids(superpool, rows, direction),
                    "dense_scores": {
                        system_id(side, rows[index]): float(dense_scores[index])
                        for index in superpool
                    },
                }
            )
            gold_ids = _candidate_ids(retrieval.gold_indices[position], rows, direction)
            qrels.append(
                {
                    "query_id": query_id,
                    "pair_id": int(row["id"]),
                    "pair_group": prepared[-1]["pair_group"],
                    "gold_candidate_ids": gold_ids,
                    "dense_top10": bool(frozen["dense_top10"]),
                    "known_sae_rescue": bool(frozen["sae_rescue"]),
                }
            )
    return prepared, qrels


def _preflight(prepared: Sequence[dict[str, Any]], qrels: Sequence[dict[str, Any]]) -> dict[str, Any]:
    qrels_by_id = {row["query_id"]: row for row in qrels}

    def hit(query: dict[str, Any], arm: str) -> bool:
        gold = set(qrels_by_id[query["query_id"]]["gold_candidate_ids"])
        return bool(gold.intersection(query["pools"][arm]))

    def source_hit(query: dict[str, Any], candidates: Sequence[str]) -> bool:
        gold = set(qrels_by_id[query["query_id"]]["gold_candidate_ids"])
        return bool(gold.intersection(map(str, candidates)))

    def distribution(values: Sequence[int]) -> dict[str, Any]:
        array = np.asarray(values, dtype=float)
        return {
            "draws": len(values),
            "values": [int(value) for value in values],
            "min": int(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "max": int(array.max()),
        }

    dense10_hits = sum(hit(row, "dense_ranking") for row in prepared)
    dense30_hits = sum(hit(row, "dense30_structure") for row in prepared)
    sae_union_hits = sum(
        source_hit(row, row["source_pools"]["sae_union"]) for row in prepared
    )
    sae_padded_hits = sum(
        hit(row, "sae_union_padded30_structure") for row in prepared
    )
    random_source_oracle_hits = [
        sum(
            source_hit(row, row["source_pools"]["random_unions"][pair_index])
            for row in prepared
        )
        for pair_index in range(RANDOM_SEED_PAIR_COUNT)
    ]
    random_padded_hits = [
        sum(hit(row, f"random_union_padded30_structure_{offset}") for row in prepared)
        for offset in range(1, len(VERIFIED_RANDOM_PAIR_INDICES) + 1)
    ]
    known_rescues = sum(row["known_sae_rescue"] for row in qrels)
    rescue_beyond_dense30 = sum(
        row["known_sae_rescue"] and not hit(query, "dense30_structure")
        for query, row in zip(prepared, qrels)
    )
    expected = {
        "queries": 566,
        "dense_top10_hits": 146,
        "dense_top30_hits": 262,
        "sae_union_hits": 200,
        "known_sae_rescues": 54,
        "known_sae_rescues_beyond_dense30": 19,
    }
    observed = {
        "queries": len(prepared),
        "dense_top10_hits": dense10_hits,
        "dense_top30_hits": dense30_hits,
        "sae_union_hits": sae_union_hits,
        "known_sae_rescues": known_rescues,
        "known_sae_rescues_beyond_dense30": rescue_beyond_dense30,
    }
    if observed != expected:
        raise ValueError(f"Frozen retrieval preflight drifted: {observed} != {expected}")
    sae_sizes = np.asarray(
        [len(row["source_pools"]["sae_union"]) for row in prepared], dtype=float
    )
    random_sizes = np.asarray(
        [
            len(source_pool)
            for row in prepared
            for source_pool in row["source_pools"]["random_unions"]
        ],
        dtype=float,
    )
    exact_padded_arms = tuple(arm for arm in ARM_NAMES if arm != "dense_ranking")
    if any(
        len(row["pools"][arm]) != DENSE_POOL_SIZE
        or len(set(row["pools"][arm])) != DENSE_POOL_SIZE
        for row in prepared
        for arm in exact_padded_arms
    ):
        raise AssertionError(
            "Every verifier arm must have exactly 30 unique candidates"
        )
    random_oracle_distribution = distribution(random_source_oracle_hits)
    random_oracle_array = np.asarray(random_source_oracle_hits, dtype=float)
    random_oracle_distribution.update(
        {
            "q05_higher": int(np.quantile(random_oracle_array, 0.05, method="higher")),
            "q95_higher": int(np.quantile(random_oracle_array, 0.95, method="higher")),
            "sae_source_oracle_hits": sae_union_hits,
            "draws_at_least_sae": int(
                np.count_nonzero(random_oracle_array >= sae_union_hits)
            ),
            "plus_one_tail_probability": float(
                (1 + np.count_nonzero(random_oracle_array >= sae_union_hits))
                / (1 + len(random_oracle_array))
            ),
        }
    )
    return {
        **observed,
        "sae_union_padded30_hits": sae_padded_hits,
        "random_source_oracle_hit_distribution": random_oracle_distribution,
        "verified_random_source_oracle_hits": [
            random_source_oracle_hits[index]
            for index in VERIFIED_RANDOM_PAIR_INDICES
        ],
        "verified_random_padded30_hits": random_padded_hits,
        "sae_source_union_size": {
            "min": int(sae_sizes.min()),
            "median": float(np.median(sae_sizes)),
            "mean": float(sae_sizes.mean()),
            "max": int(sae_sizes.max()),
        },
        "random_source_union_size": {
            "min": int(random_sizes.min()),
            "median": float(np.median(random_sizes)),
            "mean": float(random_sizes.mean()),
            "max": int(random_sizes.max()),
        },
        "verifier_candidate_pool_size": DENSE_POOL_SIZE,
        "all_verifier_candidate_pools_exactly_30": True,
    }


def development_screen(qrels: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Outcome-stratified SCAR screen; never represented as held out."""

    rescues = [row for row in qrels if bool(row["known_sae_rescue"])]
    if len(rescues) != 54:
        raise ValueError(f"Expected 54 known SAE rescues, found {len(rescues)}")
    rescue_groups = {str(row["pair_group"]) for row in rescues}
    dense_eligible = [
        row
        for row in qrels
        if bool(row["dense_top10"])
        and str(row["pair_group"]) not in rescue_groups
    ]
    dense_eligible.sort(
        key=lambda row: (
            hashlib.sha256(
                f"structural-rescue-dense-control\0{row['query_id']}".encode()
            ).hexdigest(),
            str(row["query_id"]),
        )
    )
    controls = dense_eligible[:SCREEN_DENSE_CONTROL_COUNT]
    if len(controls) != SCREEN_DENSE_CONTROL_COUNT:
        raise ValueError("Not enough non-overlapping dense-hit controls")
    query_ids = sorted(
        [str(row["query_id"]) for row in rescues + controls]
    )
    if len(query_ids) != 108 or len(set(query_ids)) != 108:
        raise AssertionError("Development screen must contain 108 unique queries")
    return {
        "selection": "outcome_stratified_exploratory_screen",
        "query_ids": query_ids,
        "known_sae_rescue_queries": 54,
        "dense_retention_control_queries": 54,
        "rescue_pair_groups": len(rescue_groups),
        "control_pair_groups": len({str(row["pair_group"]) for row in controls}),
        "confirmatory_or_population_recall_claim_allowed": False,
    }


def _feature_catalog_and_pair_evidence(
    prepared: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    representations: dict[str, Representation],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_index_by_id = {int(row["id"]): index for index, row in enumerate(rows)}
    n_rows = len(rows)
    tallies: dict[str, defaultdict[int, float]] = {
        name: defaultdict(float) for name in representations
    }

    for query in prepared:
        direction = str(query["direction"])
        query_index = row_index_by_id[int(str(query["query_system_id"]).split(":", 1)[1])]
        query_global = _query_global(direction, query_index, n_rows)
        # Feature-description selection is based only on the original SAE
        # candidate source; dense padding must not change which features exist.
        for candidate_id in query["source_pools"]["sae_union"]:
            candidate_index = _candidate_index(candidate_id, row_index_by_id)
            candidate_global = _candidate_global(direction, candidate_index, n_rows)
            for namespace, representation in representations.items():
                for evidence in shared_feature_rows(
                    representation,
                    query_global,
                    candidate_global,
                    namespace=namespace,
                    limit=SHARED_FEATURES_PER_PAIR,
                ):
                    tallies[namespace][int(evidence["feature_id"])] += float(
                        evidence["selection_score"]
                    )

    selected: dict[str, set[int]] = {}
    catalog: list[dict[str, Any]] = []
    all_system_ids = [
        *(system_id("a", row) for row in rows),
        *(system_id("b", row) for row in rows),
    ]
    for namespace, representation in representations.items():
        ordered = sorted(
            tallies[namespace].items(), key=lambda item: (-item[1], item[0])
        )[:DESCRIBED_FEATURES_PER_REPRESENTATION]
        selected[namespace] = {feature_id for feature_id, _ in ordered}
        for feature_id, aggregate_score in ordered:
            values = representation.values[:, feature_id]
            active_indices = np.flatnonzero(values > 0)
            top_indices = sorted(
                active_indices.astype(int).tolist(),
                key=lambda index: (-float(values[index]), index),
            )[:DESCRIPTION_EXAMPLES_PER_FEATURE]
            catalog.append(
                {
                    "feature_key": f"{namespace}:{feature_id}",
                    "representation": namespace,
                    "feature_id": feature_id,
                    "corpus_active_count": int(len(active_indices)),
                    "aggregate_pair_selection_score": float(aggregate_score),
                    "top_examples": [
                        {
                            "system_id": all_system_ids[index],
                            "activation_percentile": _percentile(
                                representation, feature_id, float(values[index])
                            ),
                        }
                        for index in top_indices
                    ],
                }
            )

    pair_rows: list[dict[str, Any]] = []
    for query in prepared:
        direction = str(query["direction"])
        query_index = row_index_by_id[int(str(query["query_system_id"]).split(":", 1)[1])]
        query_global = _query_global(direction, query_index, n_rows)
        # Pair evidence must cover the full pool that the three SAE evidence
        # arms score, including candidates appended from dense top-30.
        for candidate_id in query["pools"]["sae_union_padded30_structure"]:
            candidate_index = _candidate_index(candidate_id, row_index_by_id)
            candidate_global = _candidate_global(direction, candidate_index, n_rows)
            evidence: list[dict[str, Any]] = []
            for namespace, representation in representations.items():
                evidence.extend(
                    shared_feature_rows(
                        representation,
                        query_global,
                        candidate_global,
                        namespace=namespace,
                        limit=SHARED_FEATURES_PER_PAIR,
                        allowed_features=selected[namespace],
                    )
                )
            evidence.sort(
                key=lambda row: (-row["selection_score"], row["feature_key"])
            )
            pair_rows.append(
                {
                    "query_id": query["query_id"],
                    "candidate_id": candidate_id,
                    "shared_features": evidence[:SHARED_FEATURES_PER_PAIR],
                }
            )
    return catalog, pair_rows


def prepare_development(
    *,
    data_path: Path = DEFAULT_DATA,
    embeddings_path: Path = DEFAULT_EMBEDDINGS,
    complementarity_path: Path = DEFAULT_COMPLEMENTARITY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    protocol = _strict_protocol()
    rows, raw_embeddings, complementarity = _load_and_validate_inputs(
        data_path, embeddings_path, complementarity_path, protocol
    )
    cross_domain_row_ids = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["system_a_domain"] != row["system_b_domain"]
        ],
        dtype=np.int64,
    )
    if len(cross_domain_row_ids) != 283:
        raise ValueError(f"Expected 283 cross-domain pairs, found {len(cross_domain_row_ids)}")

    dense = l2_normalize(raw_embeddings)
    cslg = _checkpoint_representation("cslg", raw_embeddings, protocol)
    astroph = _checkpoint_representation("astroph", raw_embeddings, protocol)

    n_rows = len(rows)
    score_matrices: dict[str, dict[str, np.ndarray]] = {
        "dense": score_from_embeddings(dense[:n_rows], dense[n_rows:], cross_domain_row_ids),
        "cslg": score_from_embeddings(
            cslg.unit[:n_rows], cslg.unit[n_rows:], cross_domain_row_ids
        ),
        "astroph": score_from_embeddings(
            astroph.unit[:n_rows], astroph.unit[n_rows:], cross_domain_row_ids
        ),
    }
    random_source_unions = _random_source_unions(
        raw_embeddings,
        score_matrices["dense"],
        cross_domain_row_ids,
        n_rows=n_rows,
    )

    prepared, qrels = _query_rows(
        rows,
        cross_domain_row_ids,
        score_matrices,
        random_source_unions,
        complementarity,
    )
    preflight = _preflight(prepared, qrels)
    screen = development_screen(qrels)
    screen_ids = set(map(str, screen["query_ids"]))
    capacity_selection = largest_batch_preflight(
        [row for row in prepared if str(row["query_id"]) in screen_ids],
        query_count=1,
    )
    batch_plan = verifier_batch_plan(prepared, screen)
    catalog, pair_evidence = _feature_catalog_and_pair_evidence(
        prepared, rows, {"cslg": cslg, "astroph": astroph}
    )
    description_shuffle = feature_description_shuffle_map(catalog)

    paths = {
        "candidate_manifest": output_dir / "candidate_manifest.jsonl",
        "qrels_sidecar": output_dir / "qrels_sidecar.jsonl",
        "feature_catalog": output_dir / "feature_catalog.jsonl",
        "feature_description_shuffle_map": (
            output_dir / "feature_description_shuffle_map.json"
        ),
        "pair_feature_evidence": output_dir / "pair_feature_evidence.jsonl",
        "screen_selection": output_dir / "screen_selection.json",
        "capacity_smoke_selection": output_dir / "capacity_smoke_selection.json",
        "verifier_batch_plan": output_dir / "verifier_batch_plan.json",
    }
    write_jsonl(paths["candidate_manifest"], prepared, overwrite=overwrite)
    write_jsonl(paths["qrels_sidecar"], qrels, overwrite=overwrite)
    write_jsonl(paths["feature_catalog"], catalog, overwrite=overwrite)
    write_json(
        paths["feature_description_shuffle_map"],
        description_shuffle,
        overwrite=overwrite,
    )
    write_jsonl(paths["pair_feature_evidence"], pair_evidence, overwrite=overwrite)
    write_json(paths["screen_selection"], screen, overwrite=overwrite)
    write_json(
        paths["capacity_smoke_selection"],
        capacity_selection,
        overwrite=overwrite,
    )
    write_json(paths["verifier_batch_plan"], batch_plan, overwrite=overwrite)

    report = {
        "study_id": protocol["study_id"],
        "status": "prepared_no_llm_scoring",
        "development_only": True,
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "source_sha256": {
            "scar": sha256_file(data_path),
            "embedding_cache": sha256_file(embeddings_path),
            "complementarity_queries": sha256_file(complementarity_path),
            "cslg_sae": sha256_file(CHECKPOINTS["cslg"]["path"]),
            "astroph_sae": sha256_file(CHECKPOINTS["astroph"]["path"]),
        },
        "preflight": preflight,
        "random_source_oracle": {
            "seed_namespace": RANDOM_SEED_NAMESPACE,
            "seed_pair_count": RANDOM_SEED_PAIR_COUNT,
            "seed_pairs_sha256": RANDOM_SEED_PAIRS_SHA256,
            "verified_pair_indices": list(VERIFIED_RANDOM_PAIR_INDICES),
        },
        "feature_catalog_rows": len(catalog),
        "feature_description_shuffle_map": {
            "mapping_count": description_shuffle["mapping_count"],
            "frequency_bins_per_representation": description_shuffle[
                "frequency_bins_per_representation"
            ],
            "scheme": description_shuffle["scheme"],
        },
        "pair_feature_evidence_rows": len(pair_evidence),
        "verifier_screen": {
            key: value for key, value in screen.items() if key != "query_ids"
        },
        "capacity_smoke_selection": capacity_selection,
        "verifier_batch_plan": {
            "batch_size": batch_plan["batch_size"],
            "query_count": batch_plan["query_count"],
            "batch_count": batch_plan["batch_count"],
        },
        "artifacts": {
            key: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for key, path in paths.items()
        },
        "claim_boundary": (
            "Candidate preparation is not a verifier result, an SAE causal test, "
            "or evidence of serendipity."
        ),
    }
    write_json(output_dir / "prepare_report.json", report, overwrite=overwrite)
    return report
