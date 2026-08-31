#!/usr/bin/env python3
"""Small, reproducible SAE-vs-dense analogy-retrieval smoke test.

The benchmark uses the English SCAR system-analogy pairs.  For each cross-domain
pair, one system description is the query and all descriptions on the opposite
side are the candidate corpus.  The paired system (or any duplicate candidate
with the same normalized system name) is relevant.

The scientifically meaningful comparison is:

    OpenAI text-embedding-3-small cosine
        vs.
    cosine after the matching O'Neill et al. scientific Top-K SAE

The script also includes seeded random, BM25, and a local MiniLM dense baseline.
It never writes the OpenAI key to disk; only the returned embeddings are cached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "scar_system_analogy_en.jsonl"
DEFAULT_WEIGHTS = ROOT / "weights"
DEFAULT_CACHE = ROOT / ".cache"
DEFAULT_RESULTS = ROOT / "results" / "results.json"

SCAR_URL = (
    "https://raw.githubusercontent.com/siyuyuan/scar/"
    "main/release/system_analogy_en.json"
)
CHECKPOINTS = {
    "sae_cslg_k128_n3072": {
        "filename": "csLG_128_3072_256.pth",
        "url": (
            "https://huggingface.co/charlieoneill/embedding-saes/resolve/main/"
            "csLG_128_3072_256.pth?download=true"
        ),
        "domain": "cs.LG",
    },
    "sae_astroph_k128_n3072": {
        "filename": "astroPH_128_3072_256.pth",
        "url": (
            "https://huggingface.co/charlieoneill/embedding-saes/resolve/main/"
            "astroPH_128_3072_256.pth?download=true"
        ),
        "domain": "astro.PH",
    },
}

TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "these", "this", "to", "was",
    "were", "which", "with",
}


@dataclass(frozen=True)
class RetrievalDirection:
    """One side of the bidirectional closed-corpus retrieval task."""

    name: str
    query_texts: list[str]
    candidate_texts: list[str]
    query_row_ids: np.ndarray
    gold_indices: list[np.ndarray]


class FastAutoencoder(nn.Module):
    """Exact inference architecture used by the authors' public SAErch Space."""

    def __init__(self, n_dirs: int = 3072, d_model: int = 1536, k: int = 128):
        super().__init__()
        self.n_dirs = n_dirs
        self.d_model = d_model
        self.k = k
        self.encoder = nn.Linear(d_model, n_dirs, bias=False)
        self.decoder = nn.Linear(n_dirs, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latents_pre_act = self.encoder(x - self.pre_bias) + self.latent_bias
        values, indices = torch.topk(latents_pre_act, k=self.k, dim=-1)
        values = F.relu(values)
        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, indices, values)
        return latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--skip-openai",
        action="store_true",
        help="Run only random/BM25/local-MiniLM baselines.",
    )
    parser.add_argument(
        "--force-openai",
        action="store_true",
        help="Ignore a cached OpenAI embedding file and call the API again.",
    )
    parser.add_argument("--openai-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return rows


def validate_rows(rows: Sequence[dict[str, Any]]) -> None:
    required = {
        "id",
        "system_a",
        "system_b",
        "system_a_domain",
        "system_b_domain",
        "system_a_background",
        "system_b_background",
        "mappings",
    }
    if not rows:
        raise ValueError("SCAR dataset is empty")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"SCAR rows are missing required fields: {missing}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("SCAR pair ids are not unique")
    for row in rows:
        if not row["system_a_background"] or not row["system_b_background"]:
            raise ValueError(f"SCAR pair {row['id']} has an empty background")


def normalize_label(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.lower()))


def content_tokens(value: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(value.lower()) if token not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    tokens_a = content_tokens(a)
    tokens_b = content_tokens(b)
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 0.0


def build_direction(
    rows: Sequence[dict[str, Any]], query_row_ids: np.ndarray, direction: str
) -> RetrievalDirection:
    if direction == "a_to_b":
        query_text_key, candidate_text_key = "system_a_background", "system_b_background"
        gold_label_key, candidate_label_key = "system_b", "system_b"
    elif direction == "b_to_a":
        query_text_key, candidate_text_key = "system_b_background", "system_a_background"
        gold_label_key, candidate_label_key = "system_a", "system_a"
    else:
        raise ValueError(f"Unknown direction: {direction}")

    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for candidate_index, row in enumerate(rows):
        label_to_indices[normalize_label(str(row[candidate_label_key]))].append(candidate_index)

    gold_indices = []
    for row_id in query_row_ids:
        label = normalize_label(str(rows[int(row_id)][gold_label_key]))
        gold_indices.append(np.asarray(label_to_indices[label], dtype=np.int64))

    return RetrievalDirection(
        name=direction,
        query_texts=[str(rows[int(i)][query_text_key]) for i in query_row_ids],
        candidate_texts=[str(row[candidate_text_key]) for row in rows],
        query_row_ids=query_row_ids,
        gold_indices=gold_indices,
    )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def bm25_scores(
    query_texts: Sequence[str],
    candidate_texts: Sequence[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """Compute Okapi BM25 scores without an additional retrieval dependency."""

    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b",
        stop_words="english",
    )
    candidate_counts = vectorizer.fit_transform(candidate_texts).tocsr().astype(np.float32)
    query_counts = vectorizer.transform(query_texts).tocsr().astype(np.float32)
    query_counts.data[:] = 1.0

    n_documents = candidate_counts.shape[0]
    document_frequency = np.asarray((candidate_counts > 0).sum(axis=0)).ravel()
    inverse_document_frequency = np.log1p(
        (n_documents - document_frequency + 0.5) / (document_frequency + 0.5)
    ).astype(np.float32)

    document_lengths = np.asarray(candidate_counts.sum(axis=1)).ravel()
    average_document_length = float(document_lengths.mean())
    weighted = candidate_counts.copy().tocsr()
    row_ids = np.repeat(np.arange(n_documents), np.diff(weighted.indptr))
    term_ids = weighted.indices
    term_frequency = weighted.data.copy()
    length_normalizer = k1 * (
        1.0 - b + b * document_lengths[row_ids] / max(average_document_length, 1e-12)
    )
    weighted.data = (
        inverse_document_frequency[term_ids]
        * term_frequency
        * (k1 + 1.0)
        / (term_frequency + length_normalizer)
    )
    return (query_counts @ weighted.T).toarray().astype(np.float32)


def load_or_encode_minilm(
    all_texts: Sequence[str], cache_path: Path
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        embeddings = cached["embeddings"]
        if embeddings.shape[0] == len(all_texts):
            return l2_normalize(embeddings)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        local_files_only=True,
    )
    embeddings = model.encode(
        list(all_texts),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    return embeddings


def load_or_encode_openai(
    all_texts: Sequence[str],
    cache_path: Path,
    *,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        embeddings = cached["embeddings"]
        if embeddings.shape == (len(all_texts), 1536):
            return l2_normalize(embeddings)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set and no compatible embedding cache exists"
        )

    from openai import OpenAI

    client = OpenAI()
    collected: list[np.ndarray] = []
    for start in range(0, len(all_texts), batch_size):
        batch = list(all_texts[start : start + batch_size])
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        collected.extend(np.asarray(item.embedding, dtype=np.float32) for item in ordered)
        print(
            f"OpenAI embeddings: {min(start + len(batch), len(all_texts))}/{len(all_texts)}",
            flush=True,
        )

    embeddings = np.stack(collected)
    if embeddings.shape != (len(all_texts), 1536):
        raise ValueError(f"Unexpected OpenAI embedding shape: {embeddings.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    return l2_normalize(embeddings)


def load_sae(checkpoint_path: Path) -> FastAutoencoder:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_shapes = {
        "pre_bias": (1536,),
        "latent_bias": (3072,),
        "encoder.weight": (3072, 1536),
        "decoder.weight": (1536, 3072),
    }
    actual_shapes = {key: tuple(value.shape) for key, value in state_dict.items()}
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"Unexpected checkpoint structure in {checkpoint_path}: {actual_shapes}"
        )
    model = FastAutoencoder(n_dirs=3072, d_model=1536, k=128)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def encode_sae(embeddings: np.ndarray, checkpoint_path: Path) -> np.ndarray:
    model = load_sae(checkpoint_path)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), 128):
            batch = torch.from_numpy(embeddings[start : start + 128]).float()
            latents = model.encode(batch)
            batches.append(latents.cpu().numpy().astype(np.float32))
    return l2_normalize(np.concatenate(batches, axis=0))


def score_from_embeddings(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    cross_domain_row_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    # optimize=False avoids spurious floating-point warnings emitted by the
    # macOS Accelerate matmul path on otherwise finite normalized matrices.
    return {
        "a_to_b": np.einsum(
            "ik,jk->ij",
            embeddings_a[cross_domain_row_ids],
            embeddings_b,
            optimize=False,
        ),
        "b_to_a": np.einsum(
            "ik,jk->ij",
            embeddings_b[cross_domain_row_ids],
            embeddings_a,
            optimize=False,
        ),
    }


def best_positive_ranks(
    scores: np.ndarray, gold_indices: Sequence[np.ndarray]
) -> np.ndarray:
    """Return 1-based midranks of the highest-scoring relevant candidate."""

    ranks = np.empty(scores.shape[0], dtype=np.float64)
    for query_index, positives in enumerate(gold_indices):
        row = scores[query_index]
        best_positive_score = float(np.max(row[positives]))
        tie_mask = np.isclose(
            row, best_positive_score, atol=1e-8, rtol=1e-6
        )
        greater = int(np.count_nonzero((row > best_positive_score) & ~tie_mask))
        tied = int(np.count_nonzero(tie_mask))
        ranks[query_index] = greater + (tied + 1.0) / 2.0
    return ranks


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def bootstrap_mean_delta(
    challenger_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if challenger_values.shape != baseline_values.shape:
        raise ValueError("Paired bootstrap arrays have different shapes")
    differences = challenger_values - baseline_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrapped = differences[indices].mean(axis=1)
    return {
        "delta": float(differences.mean()),
        "ci_95_low": float(np.quantile(bootstrapped, 0.025)),
        "ci_95_high": float(np.quantile(bootstrapped, 0.975)),
    }


def evaluate_methods(
    method_scores: dict[str, dict[str, np.ndarray]],
    directions: dict[str, RetrievalDirection],
    low_overlap_positions: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    all_positions = np.arange(len(directions["a_to_b"].query_row_ids))
    scopes = {
        "cross_domain": all_positions,
        "low_lexical_overlap": low_overlap_positions,
    }
    metrics: dict[str, Any] = {}
    pair_values: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    ranks_by_method: dict[str, dict[str, np.ndarray]] = {}

    for method, scores_by_direction in method_scores.items():
        direction_ranks = {
            name: best_positive_ranks(scores_by_direction[name], directions[name].gold_indices)
            for name in ("a_to_b", "b_to_a")
        }
        ranks_by_method[method] = direction_ranks
        metrics[method] = {}
        pair_values[method] = {}
        for scope, positions in scopes.items():
            combined_ranks = np.concatenate(
                [direction_ranks["a_to_b"][positions], direction_ranks["b_to_a"][positions]]
            )
            metrics[method][scope] = metrics_from_ranks(combined_ranks)
            pair_values[method][scope] = {
                "mrr": (
                    1.0 / direction_ranks["a_to_b"][positions]
                    + 1.0 / direction_ranks["b_to_a"][positions]
                )
                / 2.0,
                "recall_at_10": (
                    (direction_ranks["a_to_b"][positions] <= 10).astype(float)
                    + (direction_ranks["b_to_a"][positions] <= 10).astype(float)
                )
                / 2.0,
            }

    comparisons: dict[str, Any] = {}
    if "openai_dense" in method_scores:
        baseline = "openai_dense"
        for challenger in (
            "sae_cslg_k128_n3072",
            "sae_astroph_k128_n3072",
            "sae_dual_domain_ensemble",
        ):
            if challenger not in method_scores:
                continue
            comparisons[f"{challenger}_vs_{baseline}"] = {}
            for scope in scopes:
                comparisons[f"{challenger}_vs_{baseline}"][scope] = {}
                for metric_name in ("mrr", "recall_at_10"):
                    comparisons[f"{challenger}_vs_{baseline}"][scope][metric_name] = (
                        bootstrap_mean_delta(
                            pair_values[challenger][scope][metric_name],
                            pair_values[baseline][scope][metric_name],
                            samples=bootstrap_samples,
                            seed=seed,
                        )
                    )

    return metrics, comparisons, ranks_by_method


def top_retrieval_examples(
    rows: Sequence[dict[str, Any]],
    scores: dict[str, np.ndarray],
    direction: RetrievalDirection,
    ranks: np.ndarray,
    *,
    count: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    if direction.name == "a_to_b":
        query_label_key, candidate_label_key = "system_a", "system_b"
    else:
        query_label_key, candidate_label_key = "system_b", "system_a"

    def build(position: int) -> dict[str, Any]:
        row_id = int(direction.query_row_ids[position])
        top_candidate = int(np.argmax(scores[direction.name][position]))
        gold_candidate = int(direction.gold_indices[position][0])
        return {
            "pair_id": rows[row_id]["id"],
            "query": rows[row_id][query_label_key],
            "gold": rows[row_id][candidate_label_key],
            "retrieved_top1": rows[top_candidate][candidate_label_key],
            "gold_rank": float(ranks[position]),
            "gold_candidate_pair_id": rows[gold_candidate]["id"],
        }

    ordered = np.argsort(ranks)
    return {
        "best": [build(int(i)) for i in ordered[:count]],
        "worst": [build(int(i)) for i in ordered[-count:][::-1]],
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.data.exists():
        raise FileNotFoundError(
            f"Missing SCAR data at {args.data}. Download from {SCAR_URL}."
        )

    rows = load_jsonl(args.data)
    validate_rows(rows)
    if len(rows) != 400:
        print(f"Warning: expected 400 SCAR English pairs, found {len(rows)}", file=sys.stderr)

    cross_domain_row_ids = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["system_a_domain"] != row["system_b_domain"]
        ],
        dtype=np.int64,
    )
    lexical_overlap = np.asarray(
        [
            jaccard(rows[int(i)]["system_a_background"], rows[int(i)]["system_b_background"])
            for i in cross_domain_row_ids
        ],
        dtype=np.float64,
    )
    overlap_threshold = float(np.median(lexical_overlap))
    low_overlap_positions = np.flatnonzero(lexical_overlap <= overlap_threshold)

    directions = {
        name: build_direction(rows, cross_domain_row_ids, name)
        for name in ("a_to_b", "b_to_a")
    }
    all_a = [str(row["system_a_background"]) for row in rows]
    all_b = [str(row["system_b_background"]) for row in rows]
    all_texts = all_a + all_b

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    method_scores: dict[str, dict[str, np.ndarray]] = {}

    rng = np.random.default_rng(args.seed)
    method_scores["random_seeded"] = {
        name: rng.random(
            (len(cross_domain_row_ids), len(rows)), dtype=np.float32
        )
        for name in directions
    }

    print("Computing BM25 baseline...", flush=True)
    method_scores["bm25"] = {
        name: bm25_scores(direction.query_texts, direction.candidate_texts)
        for name, direction in directions.items()
    }

    print("Computing local MiniLM dense baseline...", flush=True)
    minilm = load_or_encode_minilm(
        all_texts, args.cache_dir / "minilm_scar_embeddings.npz"
    )
    method_scores["minilm_dense"] = score_from_embeddings(
        minilm[: len(rows)], minilm[len(rows) :], cross_domain_row_ids
    )

    skipped: dict[str, str] = {}
    openai_embeddings: np.ndarray | None = None
    if args.skip_openai:
        skipped["openai_dense_and_sae"] = "--skip-openai was supplied"
    else:
        try:
            print("Loading or computing OpenAI dense embeddings...", flush=True)
            openai_embeddings = load_or_encode_openai(
                all_texts,
                args.cache_dir / "openai_text_embedding_3_small_scar.npz",
                batch_size=args.openai_batch_size,
                force=args.force_openai,
            )
        except RuntimeError as exc:
            skipped["openai_dense_and_sae"] = str(exc)
            print(f"Skipping exact OpenAI/SAE arm: {exc}", file=sys.stderr)

    checkpoint_metadata: dict[str, Any] = {}
    if openai_embeddings is not None:
        openai_a = openai_embeddings[: len(rows)]
        openai_b = openai_embeddings[len(rows) :]
        method_scores["openai_dense"] = score_from_embeddings(
            openai_a, openai_b, cross_domain_row_ids
        )

        sae_scores = []
        for method_name, config in CHECKPOINTS.items():
            checkpoint_path = args.weights_dir / str(config["filename"])
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing {checkpoint_path}. Public source: {config['url']}"
                )
            print(f"Encoding with {method_name}...", flush=True)
            sae_embeddings = encode_sae(openai_embeddings, checkpoint_path)
            scores = score_from_embeddings(
                sae_embeddings[: len(rows)],
                sae_embeddings[len(rows) :],
                cross_domain_row_ids,
            )
            method_scores[method_name] = scores
            sae_scores.append(scores)
            checkpoint_metadata[method_name] = {
                "domain": config["domain"],
                "filename": config["filename"],
                "url": config["url"],
                "sha256": sha256_file(checkpoint_path),
                "k": 128,
                "n_latents": 3072,
                "input_dimensions": 1536,
            }

        method_scores["sae_dual_domain_ensemble"] = {
            name: np.mean([scores[name] for scores in sae_scores], axis=0)
            for name in directions
        }

    metrics, comparisons, ranks_by_method = evaluate_methods(
        method_scores,
        directions,
        low_overlap_positions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    example_method = (
        "sae_dual_domain_ensemble"
        if "sae_dual_domain_ensemble" in method_scores
        else "minilm_dense"
    )
    examples = top_retrieval_examples(
        rows,
        method_scores[example_method],
        directions["a_to_b"],
        ranks_by_method[example_method]["a_to_b"],
    )

    result = {
        "run": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "python": sys.version,
            "torch": torch.__version__,
        },
        "benchmark": {
            "name": "SCAR English bidirectional closed-corpus retrieval proxy",
            "source_url": SCAR_URL,
            "source_sha256": sha256_file(args.data),
            "total_pairs": len(rows),
            "cross_domain_pairs": int(len(cross_domain_row_ids)),
            "queries_cross_domain_bidirectional": int(2 * len(cross_domain_row_ids)),
            "candidate_count_per_direction": len(rows),
            "low_lexical_overlap_pairs": int(len(low_overlap_positions)),
            "low_lexical_overlap_jaccard_threshold": overlap_threshold,
            "relevance": (
                "paired candidate; duplicate candidates with the same normalized system "
                "name are also counted as relevant"
            ),
        },
        "models": {
            "random_seeded": {"seed": args.seed},
            "bm25": {"k1": 1.5, "b": 0.75},
            "minilm_dense": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "local_files_only": True,
            },
            "openai_dense": {
                "model": "text-embedding-3-small",
                "dimensions": 1536,
            },
            **checkpoint_metadata,
        },
        "metrics": metrics,
        "paired_bootstrap_comparisons": comparisons,
        "example_method": example_method,
        "examples": examples,
        "skipped": skipped,
        "scope_caveats": [
            "SCAR contains system descriptions, not scientific-paper abstracts.",
            "This smoke test evaluates retrieval of known analogues, not whether a retrieved "
            "source improves downstream scientific ideation.",
            "The negative pool is the other 399 SCAR candidates; it is not a purpose-built set "
            "of same-topic/different-mechanism hard negatives.",
            "The two SAEs were trained on different scientific domains, not multiple random "
            "seeds; this is a domain-sensitivity check, not a feature-stability estimate.",
            "Purpose-mechanism, LLM-schema, query-expansion, and random-diverse ideation "
            "baselines are intentionally deferred until this cheap gate is informative.",
        ],
    }

    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
