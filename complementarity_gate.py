#!/usr/bin/env python3
"""Pre-specified complementarity/oracle-union gate for SAE retrieval.

This script intentionally does not fit a hybrid. It asks whether either corrected
scientific SAE retrieves enough dense misses to justify building one.

Gate (fixed before inspecting corrected results):
  * at least 20 unique SAE rescues across 566 query-directions; and
  * at least +4.0 percentage points oracle-union Recall@10 over dense.

If either condition fails, retrieval is abandoned rather than retuned.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from sae_smoke_test import (
    FastAutoencoder,
    SCAR_URL,
    best_positive_ranks,
    build_direction,
    jaccard,
    l2_normalize,
    load_jsonl,
    score_from_embeddings,
    sha256_file,
    validate_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "scar_system_analogy_en.jsonl"
DEFAULT_EMBEDDINGS = ROOT / ".cache" / "openai_text_embedding_3_small_scar.npz"
DEFAULT_WEIGHTS = ROOT / "weights"
DEFAULT_SUMMARY = ROOT / "results" / "complementarity_gate.json"
DEFAULT_QUERIES = ROOT / "results" / "complementarity_queries.jsonl"

CHECKPOINTS = {
    "sae_cslg_k64_n9216": {
        "filename": "csLG_64_9216.pth",
        "domain": "cs.LG",
        "expected_sha256": "29073be46ce5ddceee53f7e9ebf46449e239c1bc29f57dfebced041833698752",
        "url": (
            "https://huggingface.co/datasets/charlieoneill/saerchModels/resolve/"
            "b2cbb184b58880b77a546511e11d8fd214c40556/"
            "csLG_64_9216.pth?download=true"
        ),
    },
    "sae_astroph_k64_n9216": {
        "filename": "astroPH_64_9216.pth",
        "domain": "astro.PH",
        "expected_sha256": "112e8a006ff0cc8e3b4439e1ef28df816564c5d9054974a763eaa69804cf02ed",
        "url": (
            "https://huggingface.co/datasets/charlieoneill/saerchModels/resolve/"
            "b2cbb184b58880b77a546511e11d8fd214c40556/"
            "astroPH_64_9216.pth?download=true"
        ),
    },
}

MIN_UNIQUE_RESCUES = 20
MIN_ORACLE_IMPROVEMENT = 0.04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    return parser.parse_args()


def load_raw_openai_embeddings(path: Path, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached raw OpenAI embeddings: {path}. Run sae_smoke_test.py first."
        )
    with np.load(path) as cached:
        embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
    expected_shape = (expected_rows * 2, 1536)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Expected raw embedding shape {expected_shape}, found {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Raw embedding cache contains non-finite values")
    return embeddings


def load_exact_sae(checkpoint_path: Path) -> FastAutoencoder:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_shapes = {
        "pre_bias": (1536,),
        "latent_bias": (9216,),
        "encoder.weight": (9216, 1536),
        "decoder.weight": (1536, 9216),
    }
    actual_shapes = {key: tuple(value.shape) for key, value in state_dict.items()}
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"Unexpected checkpoint structure in {checkpoint_path}: {actual_shapes}"
        )
    model = FastAutoencoder(n_dirs=9216, d_model=1536, k=64)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def encode_exact_sae(raw_embeddings: np.ndarray, checkpoint_path: Path) -> np.ndarray:
    """Feed raw API vectors to the SAE, then normalize only the sparse output."""

    model = load_exact_sae(checkpoint_path)
    encoded_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(raw_embeddings), 64):
            batch = torch.from_numpy(raw_embeddings[start : start + 64]).float()
            encoded_batches.append(model.encode(batch).cpu().numpy().astype(np.float32))
    return l2_normalize(np.concatenate(encoded_batches, axis=0))


def flatten_by_direction(values: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([values["a_to_b"], values["b_to_a"]])


def finite_spearman(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    result = spearmanr(a, b)
    coefficient = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(coefficient):
        raise ValueError("Rank correlation is not finite")
    return {"spearman_rho": coefficient, "p_value": p_value}


def exact_topk_successes(
    scores: np.ndarray,
    gold_indices: Sequence[np.ndarray],
    *,
    k: int = 10,
) -> tuple[np.ndarray, int]:
    """Use score-descending, candidate-index-ascending deterministic top-k sets."""

    successes = np.zeros(scores.shape[0], dtype=bool)
    cutoff_ties = 0
    for query_index, positives in enumerate(gold_indices):
        row = scores[query_index]
        ordering = np.argsort(-row, kind="stable")
        topk = ordering[:k]
        successes[query_index] = bool(np.intersect1d(topk, positives).size)
        if len(ordering) > k and np.isclose(
            row[ordering[k - 1]], row[ordering[k]], atol=1e-8, rtol=1e-6
        ):
            cutoff_ties += 1
    return successes, cutoff_ties


def summarize_single_sae(
    dense_success: np.ndarray,
    sae_success: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    dense = dense_success[mask]
    sae = sae_success[mask]
    shared = dense & sae
    rescues = ~dense & sae
    dense_rescues = dense & ~sae
    oracle = dense | sae
    n_queries = int(mask.sum())
    return {
        "queries": n_queries,
        "sae_top10_successes": int(sae.sum()),
        "shared_successes": int(shared.sum()),
        "sae_unique_rescues": int(rescues.sum()),
        "dense_unique_rescues": int(dense_rescues.sum()),
        "oracle_union_successes": int(oracle.sum()),
        "oracle_union_recall_at_10": float(oracle.mean()),
        "oracle_improvement_over_dense": float(oracle.mean() - dense.mean()),
        "oracle_improvement_percentage_points": float(
            100.0 * (oracle.mean() - dense.mean())
        ),
    }


def summarize_successes(
    dense_success: np.ndarray,
    cslg_success: np.ndarray,
    astroph_success: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    dense = dense_success[mask]
    cslg = cslg_success[mask]
    astroph = astroph_success[mask]
    sae_any = cslg | astroph
    shared = dense & sae_any
    sae_rescues = ~dense & sae_any
    dense_rescues = dense & ~sae_any
    neither = ~dense & ~sae_any
    oracle = dense | sae_any
    n_queries = int(mask.sum())

    counts = {
        "queries": n_queries,
        "dense_top10_successes": int(dense.sum()),
        "cslg_sae_top10_successes": int(cslg.sum()),
        "astroph_sae_top10_successes": int(astroph.sum()),
        "either_sae_top10_successes": int(sae_any.sum()),
        "shared_dense_and_either_sae": int(shared.sum()),
        "sae_unique_rescues": int(sae_rescues.sum()),
        "dense_unique_rescues": int(dense_rescues.sum()),
        "neither_succeeds": int(neither.sum()),
        "oracle_union_successes": int(oracle.sum()),
        "cslg_unique_rescues_over_dense": int((~dense & cslg).sum()),
        "astroph_unique_rescues_over_dense": int((~dense & astroph).sum()),
        "rescued_by_both_saes": int((~dense & cslg & astroph).sum()),
    }
    assert counts["oracle_union_successes"] == (
        counts["dense_top10_successes"] + counts["sae_unique_rescues"]
    )
    assert n_queries == (
        counts["shared_dense_and_either_sae"]
        + counts["sae_unique_rescues"]
        + counts["dense_unique_rescues"]
        + counts["neither_succeeds"]
    )

    dense_recall = counts["dense_top10_successes"] / n_queries
    sae_recall = counts["either_sae_top10_successes"] / n_queries
    oracle_recall = counts["oracle_union_successes"] / n_queries
    return {
        **counts,
        "dense_recall_at_10": dense_recall,
        "either_sae_oracle_recall_at_10": sae_recall,
        "oracle_union_recall_at_10": oracle_recall,
        "oracle_improvement_over_dense": oracle_recall - dense_recall,
        "oracle_improvement_percentage_points": 100.0 * (oracle_recall - dense_recall),
    }


def query_records(
    rows: Sequence[dict[str, Any]],
    cross_domain_row_ids: np.ndarray,
    directions: dict[str, Any],
    ranks: dict[str, dict[str, np.ndarray]],
    successes: dict[str, dict[str, np.ndarray]],
    low_overlap_row_ids: set[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for direction_name in ("a_to_b", "b_to_a"):
        direction = directions[direction_name]
        if direction_name == "a_to_b":
            query_key, gold_key = "system_a", "system_b"
            source_domain_key, target_domain_key = (
                "system_a_domain",
                "system_b_domain",
            )
        else:
            query_key, gold_key = "system_b", "system_a"
            source_domain_key, target_domain_key = (
                "system_b_domain",
                "system_a_domain",
            )
        for position, row_id_value in enumerate(cross_domain_row_ids):
            row_id = int(row_id_value)
            dense_rank = float(ranks["openai_dense"][direction_name][position])
            cslg_rank = float(ranks["sae_cslg_k64_n9216"][direction_name][position])
            astroph_rank = float(ranks["sae_astroph_k64_n9216"][direction_name][position])
            dense_success = bool(successes["openai_dense"][direction_name][position])
            cslg_success = bool(
                successes["sae_cslg_k64_n9216"][direction_name][position]
            )
            astroph_success = bool(
                successes["sae_astroph_k64_n9216"][direction_name][position]
            )
            sae_success = cslg_success or astroph_success
            records.append(
                {
                    "query_id": f"{direction_name}:{rows[row_id]['id']}",
                    "pair_id": rows[row_id]["id"],
                    "direction": direction_name,
                    "query": rows[row_id][query_key],
                    "gold": rows[row_id][gold_key],
                    "source_domain": rows[row_id][source_domain_key],
                    "target_domain": rows[row_id][target_domain_key],
                    "low_lexical_overlap": row_id in low_overlap_row_ids,
                    "dense_rank": dense_rank,
                    "cslg_sae_rank": cslg_rank,
                    "astroph_sae_rank": astroph_rank,
                    "best_sae_rank": min(cslg_rank, astroph_rank),
                    "dense_top10": dense_success,
                    "cslg_sae_top10": cslg_success,
                    "astroph_sae_top10": astroph_success,
                    "either_sae_top10": sae_success,
                    "shared_success": dense_success and sae_success,
                    "sae_rescue": (not dense_success) and sae_success,
                    "dense_rescue": dense_success and (not sae_success),
                }
            )
    return records


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.data)
    validate_rows(rows)
    cross_domain_row_ids = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["system_a_domain"] != row["system_b_domain"]
        ],
        dtype=np.int64,
    )
    if len(cross_domain_row_ids) != 283:
        raise ValueError(
            f"Expected 283 cross-domain pairs, found {len(cross_domain_row_ids)}"
        )

    lexical_overlap = np.asarray(
        [
            jaccard(rows[int(i)]["system_a_background"], rows[int(i)]["system_b_background"])
            for i in cross_domain_row_ids
        ],
        dtype=np.float64,
    )
    low_overlap_threshold = float(np.median(lexical_overlap))
    low_overlap_pair_positions = np.flatnonzero(lexical_overlap <= low_overlap_threshold)
    low_overlap_row_ids = {
        int(cross_domain_row_ids[position]) for position in low_overlap_pair_positions
    }

    directions = {
        name: build_direction(rows, cross_domain_row_ids, name)
        for name in ("a_to_b", "b_to_a")
    }
    raw_embeddings = load_raw_openai_embeddings(args.embeddings, len(rows))
    raw_a = raw_embeddings[: len(rows)]
    raw_b = raw_embeddings[len(rows) :]

    # Dense cosine uses normalized copies; SAE inference uses raw API vectors.
    dense_a = l2_normalize(raw_a)
    dense_b = l2_normalize(raw_b)
    method_scores: dict[str, dict[str, np.ndarray]] = {
        "openai_dense": score_from_embeddings(
            dense_a, dense_b, cross_domain_row_ids
        )
    }
    checkpoint_metadata: dict[str, Any] = {}
    for method_name, config in CHECKPOINTS.items():
        checkpoint_path = args.weights_dir / str(config["filename"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing exact checkpoint {checkpoint_path}; source: {config['url']}"
            )
        checkpoint_hash = sha256_file(checkpoint_path)
        if checkpoint_hash != config["expected_sha256"]:
            raise ValueError(
                f"Checkpoint hash mismatch for {checkpoint_path}: {checkpoint_hash}"
            )
        print(f"Encoding raw vectors with {method_name}...", flush=True)
        sparse_embeddings = encode_exact_sae(raw_embeddings, checkpoint_path)
        method_scores[method_name] = score_from_embeddings(
            sparse_embeddings[: len(rows)],
            sparse_embeddings[len(rows) :],
            cross_domain_row_ids,
        )
        checkpoint_metadata[method_name] = {
            **config,
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": checkpoint_hash,
            "input_dimensions": 1536,
            "n_latents": 9216,
            "k": 64,
            "auxk_training_config": 128,
        }

    ranks = {
        method_name: {
            direction_name: best_positive_ranks(
                scores[direction_name], directions[direction_name].gold_indices
            )
            for direction_name in ("a_to_b", "b_to_a")
        }
        for method_name, scores in method_scores.items()
    }
    successes: dict[str, dict[str, np.ndarray]] = {}
    cutoff_ties: dict[str, dict[str, int]] = {}
    for method_name, scores in method_scores.items():
        successes[method_name] = {}
        cutoff_ties[method_name] = {}
        for direction_name in ("a_to_b", "b_to_a"):
            success, tie_count = exact_topk_successes(
                scores[direction_name],
                directions[direction_name].gold_indices,
                k=10,
            )
            successes[method_name][direction_name] = success
            cutoff_ties[method_name][direction_name] = tie_count

    flat_ranks = {
        method_name: flatten_by_direction(direction_ranks)
        for method_name, direction_ranks in ranks.items()
    }
    dense_rank = flat_ranks["openai_dense"]
    cslg_rank = flat_ranks["sae_cslg_k64_n9216"]
    astroph_rank = flat_ranks["sae_astroph_k64_n9216"]
    best_sae_rank = np.minimum(cslg_rank, astroph_rank)
    flat_successes = {
        method_name: flatten_by_direction(direction_successes)
        for method_name, direction_successes in successes.items()
    }
    dense_success = flat_successes["openai_dense"]
    cslg_success = flat_successes["sae_cslg_k64_n9216"]
    astroph_success = flat_successes["sae_astroph_k64_n9216"]

    all_query_mask = np.ones(len(dense_rank), dtype=bool)
    low_pair_mask = np.zeros(len(cross_domain_row_ids), dtype=bool)
    low_pair_mask[low_overlap_pair_positions] = True
    low_query_mask = np.concatenate([low_pair_mask, low_pair_mask])
    overall = summarize_successes(
        dense_success, cslg_success, astroph_success, all_query_mask
    )
    low_overlap = summarize_successes(
        dense_success, cslg_success, astroph_success, low_query_mask
    )

    gate_passes = (
        overall["sae_unique_rescues"] >= MIN_UNIQUE_RESCUES
        and overall["oracle_improvement_over_dense"] >= MIN_ORACLE_IMPROVEMENT
    )
    decision = "PROCEED_TO_HYBRID" if gate_passes else "ABANDON_SAE_RETRIEVAL"
    embedding_norms = np.linalg.norm(raw_embeddings, axis=1)

    records = query_records(
        rows,
        cross_domain_row_ids,
        directions,
        ranks,
        successes,
        low_overlap_row_ids,
    )
    assert len(records) == 566
    assert len({record["query_id"] for record in records}) == 566
    assert sum(record["sae_rescue"] for record in records) == overall["sae_unique_rescues"]
    rescued_pair_counts: dict[int, int] = {}
    for record in records:
        if record["sae_rescue"]:
            pair_id = int(record["pair_id"])
            rescued_pair_counts[pair_id] = rescued_pair_counts.get(pair_id, 0) + 1
    overall["rescued_distinct_pairs"] = len(rescued_pair_counts)
    overall["pairs_rescued_in_both_directions"] = sum(
        count == 2 for count in rescued_pair_counts.values()
    )

    summary = {
        "run": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": Path(__file__).name,
        },
        "hypothesis": (
            "Inferior global SAE retrieval may still rescue enough dense top-10 misses "
            "to justify a shortlist reranker."
        ),
        "preprocessing": {
            "openai_model": "text-embedding-3-small",
            "sae_input": "raw cached API vectors; no pre-SAE L2 normalization",
            "dense_similarity": "cosine after L2 normalization",
            "sae_similarity": "cosine over L2-normalized Top-K latent vectors",
            "raw_embedding_norm_min": float(embedding_norms.min()),
            "raw_embedding_norm_mean": float(embedding_norms.mean()),
            "raw_embedding_norm_max": float(embedding_norms.max()),
            "embedding_cache_sha256": sha256_file(args.embeddings),
        },
        "benchmark": {
            "name": "SCAR English bidirectional cross-domain retrieval proxy",
            "source_url": SCAR_URL,
            "source_sha256": sha256_file(args.data),
            "pairs": len(rows),
            "cross_domain_pairs": len(cross_domain_row_ids),
            "query_directions": len(records),
            "candidates_per_direction": len(rows),
            "low_overlap_pairs": len(low_overlap_pair_positions),
            "low_overlap_jaccard_threshold": low_overlap_threshold,
            "top_k": 10,
            "duplicate_label_policy": (
                "all candidates with the same normalized system name count as relevant"
            ),
        },
        "checkpoints": checkpoint_metadata,
        "gate": {
            "pre_specified_min_unique_rescues": MIN_UNIQUE_RESCUES,
            "pre_specified_min_oracle_improvement": MIN_ORACLE_IMPROVEMENT,
            "pre_specified_min_oracle_improvement_percentage_points": (
                100.0 * MIN_ORACLE_IMPROVEMENT
            ),
            "passes": gate_passes,
            "decision": decision,
        },
        "overall": overall,
        "low_lexical_overlap": low_overlap,
        "checkpoint_complementarity": {
            "sae_cslg_k64_n9216": summarize_single_sae(
                dense_success, cslg_success, all_query_mask
            ),
            "sae_astroph_k64_n9216": summarize_single_sae(
                dense_success, astroph_success, all_query_mask
            ),
        },
        "top10_cutoff_ties": cutoff_ties,
        "rank_correlations": {
            "dense_vs_cslg_sae": finite_spearman(dense_rank, cslg_rank),
            "dense_vs_astroph_sae": finite_spearman(dense_rank, astroph_rank),
            "dense_vs_best_of_two_sae_oracle": finite_spearman(
                dense_rank, best_sae_rank
            ),
            "cslg_vs_astroph_sae": finite_spearman(cslg_rank, astroph_rank),
        },
        "artifacts": {
            "per_query_jsonl": str(args.queries.relative_to(ROOT)),
        },
        "scope_caveats": [
            "SCAR contains system descriptions rather than scientific-paper abstracts.",
            "Best-of-two SAE success is an oracle diagnostic, not a deployable fusion rule.",
            "The gate tests direct latent-space complementarity only; it does not test "
            "IDF-weighted feature evidence or a learned reranker.",
        ],
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.queries.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.queries.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"gate": summary["gate"], "overall": overall}, indent=2))
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.queries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
