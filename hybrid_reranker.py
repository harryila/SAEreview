#!/usr/bin/env python3
"""Held-out dense-shortlist reranker for the SAE complementarity follow-up.

The complementarity gate is diagnostic only: it uses an oracle union.  This
script asks the deployable follow-up question with a fixed, compact protocol:

* dense top-100 candidate generation;
* five pair-grouped folds (both directions of a SCAR pair stay together);
* a linear pairwise reranker trained on all nonrelevant shortlist candidates; and
* four controls plus the pre-specified IDF-weighted two-SAE hybrid.

No API call is made.  Raw cached ``text-embedding-3-small`` vectors are fed to
the exact released k=64, n=9,216 scientific SAE checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.random_projection import SparseRandomProjection
from sklearn.exceptions import DataDimensionalityWarning

from complementarity_gate import load_exact_sae, load_raw_openai_embeddings
from sae_smoke_test import (
    SCAR_URL,
    bm25_scores,
    build_direction,
    content_tokens,
    jaccard,
    l2_normalize,
    load_jsonl,
    normalize_label,
    score_from_embeddings,
    sha256_file,
    validate_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "scar_system_analogy_en.jsonl"
DEFAULT_EMBEDDINGS = ROOT / ".cache" / "openai_text_embedding_3_small_scar.npz"
DEFAULT_WEIGHTS = ROOT / "weights"
DEFAULT_SUMMARY = ROOT / "results" / "hybrid_reranker.json"
DEFAULT_QUERIES = ROOT / "results" / "hybrid_queries.jsonl"

SEED = 20260831
SHORTLIST_SIZE = 100
TOP_K = 10
N_FOLDS = 5
BOOTSTRAP_SAMPLES = 5000
MIN_STRONG_HYBRID_GAIN = 0.02

CHECKPOINTS = {
    "cslg": {
        "filename": "csLG_64_9216.pth",
        "domain": "cs.LG",
        "sha256": "29073be46ce5ddceee53f7e9ebf46449e239c1bc29f57dfebced041833698752",
    },
    "astroph": {
        "filename": "astroPH_64_9216.pth",
        "domain": "astro.PH",
        "sha256": "112e8a006ff0cc8e3b4439e1ef28df816564c5d9054974a763eaa69804cf02ed",
    },
}
PINNED_CHECKPOINT_SOURCE = (
    "https://huggingface.co/datasets/charlieoneill/saerchModels/tree/"
    "b2cbb184b58880b77a546511e11d8fd214c40556"
)

FEATURE_NAMES = [
    "dense_cosine",
    "dense_rank",
    "bm25",
    "lexical_jaccard",
    "random_sparse_cosine",
    "random_idf_min_activation",
    "random_overlap_rarity",
    "random_shared_feature_count",
    "random_reconstruction_cosine",
    "cslg_sae_cosine",
    "astroph_sae_cosine",
    "cslg_idf_min_activation",
    "astroph_idf_min_activation",
    "cslg_overlap_rarity",
    "astroph_overlap_rarity",
    "cslg_shared_feature_count",
    "astroph_shared_feature_count",
    "cslg_reconstruction_cosine",
    "astroph_reconstruction_cosine",
]
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

METHOD_FEATURES: dict[str, list[str]] = {
    "dense_bm25": ["dense_cosine", "dense_rank", "bm25"],
    "dense_random_sparse": [
        "dense_cosine",
        "dense_rank",
        "random_sparse_cosine",
        "random_idf_min_activation",
        "random_overlap_rarity",
        "random_shared_feature_count",
        "random_reconstruction_cosine",
    ],
    "dense_unweighted_sae": [
        "dense_cosine",
        "dense_rank",
        "cslg_sae_cosine",
        "astroph_sae_cosine",
    ],
    "dense_idf_sae": [
        "dense_cosine",
        "dense_rank",
        "bm25",
        "lexical_jaccard",
        "cslg_sae_cosine",
        "astroph_sae_cosine",
        "cslg_idf_min_activation",
        "astroph_idf_min_activation",
        "cslg_overlap_rarity",
        "astroph_overlap_rarity",
        "cslg_shared_feature_count",
        "astroph_shared_feature_count",
        "cslg_reconstruction_cosine",
        "astroph_reconstruction_cosine",
    ],
}


@dataclass
class SparseRepresentation:
    """Top-k activations plus inexpensive per-document sparse views."""

    values: np.ndarray
    active_indices: list[np.ndarray]
    active_values: list[np.ndarray]
    norms: np.ndarray
    idf_a_candidates: np.ndarray
    idf_b_candidates: np.ndarray
    reconstruction_unit: np.ndarray | None = None


@dataclass
class CandidateTable:
    """Fixed-width candidate table: one contiguous block per query."""

    features: np.ndarray
    labels: np.ndarray
    candidate_indices: np.ndarray
    query_pair_ids: np.ndarray
    fold_group_ids: np.ndarray
    query_directions: list[str]
    query_row_ids: np.ndarray
    low_overlap: np.ndarray

    @property
    def n_queries(self) -> int:
        return len(self.query_pair_ids)

    def row_slice(self, query_index: int) -> slice:
        start = query_index * SHORTLIST_SIZE
        return slice(start, start + SHORTLIST_SIZE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    return parser.parse_args()


def encode_sae_and_reconstruct(
    raw_embeddings: np.ndarray, checkpoint_path: Path
) -> SparseRepresentation:
    """Apply the released SAE and retain raw activations and reconstructions."""

    model = load_exact_sae(checkpoint_path)
    latent_batches: list[np.ndarray] = []
    reconstruction_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(raw_embeddings), 32):
            inputs = torch.from_numpy(raw_embeddings[start : start + 32]).float()
            latents = model.encode(inputs)
            reconstructions = model.decoder(latents) + model.pre_bias
            latent_batches.append(latents.cpu().numpy().astype(np.float32))
            reconstruction_batches.append(
                reconstructions.cpu().numpy().astype(np.float32)
            )
    values = np.concatenate(latent_batches, axis=0)
    reconstruction_unit = l2_normalize(
        np.concatenate(reconstruction_batches, axis=0)
    )
    del model, latent_batches, reconstruction_batches
    gc.collect()
    return make_sparse_representation(values, reconstruction_unit)


def make_sparse_representation(
    values: np.ndarray,
    reconstruction_unit: np.ndarray | None = None,
) -> SparseRepresentation:
    values = np.asarray(values, dtype=np.float32)
    active_indices: list[np.ndarray] = []
    active_values: list[np.ndarray] = []
    for row in values:
        indices = np.flatnonzero(row > 0).astype(np.int32)
        active_indices.append(indices)
        active_values.append(row[indices])
    if len(values) % 2:
        raise ValueError("Expected equal A/B embedding blocks")
    midpoint = len(values) // 2

    def candidate_idf(candidate_values: np.ndarray) -> np.ndarray:
        document_frequency = np.count_nonzero(candidate_values > 0, axis=0)
        return (
            np.log(
                (len(candidate_values) + 1.0) / (document_frequency + 1.0)
            )
            + 1.0
        ).astype(np.float32)

    norms = np.linalg.norm(values, axis=1).astype(np.float32)
    return SparseRepresentation(
        values=values,
        active_indices=active_indices,
        active_values=active_values,
        norms=norms,
        idf_a_candidates=candidate_idf(values[:midpoint]),
        idf_b_candidates=candidate_idf(values[midpoint:]),
        reconstruction_unit=reconstruction_unit,
    )


def random_sparse_control(raw_embeddings: np.ndarray) -> SparseRepresentation:
    """Seeded random 9,216-D projection followed by the same top-64 ReLU."""

    projector = SparseRandomProjection(
        n_components=9216,
        density="auto",
        random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataDimensionalityWarning)
        projected = np.asarray(
            projector.fit_transform(l2_normalize(raw_embeddings)), dtype=np.float32
        )
    top_indices = np.argpartition(projected, -64, axis=1)[:, -64:]
    rows = np.arange(len(projected))[:, None]
    top_values = np.maximum(projected[rows, top_indices], 0.0)
    sparse_values = np.zeros_like(projected, dtype=np.float32)
    sparse_values[rows, top_indices] = top_values
    reconstruction = np.asarray(
        sparse_values @ projector.components_, dtype=np.float32
    )
    reconstruction_unit = l2_normalize(reconstruction)
    del projected, projector, top_indices, top_values, reconstruction
    gc.collect()
    return make_sparse_representation(sparse_values, reconstruction_unit)


def sparse_pair_features(
    representation: SparseRepresentation,
    query_index: int,
    candidate_indices: np.ndarray,
    idf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cosine, IDF-weighted min activation, rarity, and overlap count."""

    n_candidates = len(candidate_indices)
    cosine = np.zeros(n_candidates, dtype=np.float32)
    idf_min = np.zeros(n_candidates, dtype=np.float32)
    rarity = np.zeros(n_candidates, dtype=np.float32)
    shared_count = np.zeros(n_candidates, dtype=np.float32)
    query_active = representation.active_indices[query_index]
    query_values = representation.active_values[query_index]
    query_norm = float(representation.norms[query_index])
    if query_norm <= 0:
        return cosine, idf_min, rarity, shared_count

    for offset, candidate_index_value in enumerate(candidate_indices):
        candidate_index = int(candidate_index_value)
        candidate_active = representation.active_indices[candidate_index]
        common, query_positions, candidate_positions = np.intersect1d(
            query_active,
            candidate_active,
            assume_unique=True,
            return_indices=True,
        )
        if not len(common):
            continue
        query_shared = query_values[query_positions]
        candidate_shared = representation.active_values[candidate_index][
            candidate_positions
        ]
        denominator = query_norm * float(representation.norms[candidate_index])
        if denominator > 0:
            cosine[offset] = float(
                np.dot(query_shared, candidate_shared) / denominator
            )
        shared_idf = idf[common]
        idf_min[offset] = float(
            np.sum(shared_idf * np.minimum(query_shared, candidate_shared))
        )
        rarity[offset] = float(np.max(shared_idf))
        shared_count[offset] = float(len(common))
    return cosine, idf_min, rarity, shared_count


def lexical_jaccard_scores(query_text: str, candidate_texts: Sequence[str]) -> np.ndarray:
    query_tokens = content_tokens(query_text)
    scores = np.zeros(len(candidate_texts), dtype=np.float32)
    for index, candidate_text in enumerate(candidate_texts):
        candidate_tokens = content_tokens(candidate_text)
        union = query_tokens | candidate_tokens
        if union:
            scores[index] = len(query_tokens & candidate_tokens) / len(union)
    return scores


def build_candidate_table(
    rows: Sequence[dict[str, Any]],
    cross_domain_row_ids: np.ndarray,
    raw_embeddings: np.ndarray,
    representations: dict[str, SparseRepresentation],
) -> CandidateTable:
    n_rows = len(rows)
    dense_a = l2_normalize(raw_embeddings[:n_rows])
    dense_b = l2_normalize(raw_embeddings[n_rows:])
    directions = {
        name: build_direction(rows, cross_domain_row_ids, name)
        for name in ("a_to_b", "b_to_a")
    }
    dense_scores = score_from_embeddings(dense_a, dense_b, cross_domain_row_ids)
    bm25_by_direction = {
        name: bm25_scores(direction.query_texts, direction.candidate_texts)
        for name, direction in directions.items()
    }

    lexical_overlap = np.asarray(
        [
            jaccard(
                rows[int(row_id)]["system_a_background"],
                rows[int(row_id)]["system_b_background"],
            )
            for row_id in cross_domain_row_ids
        ],
        dtype=np.float64,
    )
    low_threshold = float(np.median(lexical_overlap))
    low_by_row_id = {
        int(row_id): bool(value <= low_threshold)
        for row_id, value in zip(cross_domain_row_ids, lexical_overlap)
    }

    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    candidate_blocks: list[np.ndarray] = []
    pair_ids: list[Any] = []
    fold_group_ids: list[str] = []
    query_directions: list[str] = []
    query_row_ids: list[int] = []
    low_flags: list[bool] = []

    for direction_name in ("a_to_b", "b_to_a"):
        direction = directions[direction_name]
        for position, row_id_value in enumerate(cross_domain_row_ids):
            row_id = int(row_id_value)
            score_row = dense_scores[direction_name][position]
            dense_order = np.argsort(-score_row, kind="stable")
            shortlist = dense_order[:SHORTLIST_SIZE].astype(np.int64)
            if direction_name == "a_to_b":
                query_global = row_id
                candidate_global = shortlist + n_rows
            else:
                query_global = n_rows + row_id
                candidate_global = shortlist

            block = np.zeros((SHORTLIST_SIZE, len(FEATURE_NAMES)), dtype=np.float32)
            block[:, FEATURE_INDEX["dense_cosine"]] = score_row[shortlist]
            block[:, FEATURE_INDEX["dense_rank"]] = np.arange(
                1, SHORTLIST_SIZE + 1, dtype=np.float32
            )
            block[:, FEATURE_INDEX["bm25"]] = bm25_by_direction[direction_name][
                position, shortlist
            ]
            shortlisted_texts = [direction.candidate_texts[int(i)] for i in shortlist]
            block[:, FEATURE_INDEX["lexical_jaccard"]] = lexical_jaccard_scores(
                direction.query_texts[position], shortlisted_texts
            )

            random_representation = representations["random"]
            random_cosine, random_idf_min, random_rarity, random_count = (
                sparse_pair_features(
                    random_representation,
                    query_global,
                    candidate_global,
                    (
                        random_representation.idf_b_candidates
                        if direction_name == "a_to_b"
                        else random_representation.idf_a_candidates
                    ),
                )
            )
            block[:, FEATURE_INDEX["random_sparse_cosine"]] = random_cosine
            block[:, FEATURE_INDEX["random_idf_min_activation"]] = random_idf_min
            block[:, FEATURE_INDEX["random_overlap_rarity"]] = random_rarity
            block[:, FEATURE_INDEX["random_shared_feature_count"]] = random_count
            if random_representation.reconstruction_unit is None:
                raise ValueError("Missing random-projection reconstruction")
            block[:, FEATURE_INDEX["random_reconstruction_cosine"]] = np.einsum(
                "ij,j->i",
                random_representation.reconstruction_unit[candidate_global],
                random_representation.reconstruction_unit[query_global],
                optimize=False,
            )

            for key, prefix in (("cslg", "cslg"), ("astroph", "astroph")):
                representation = representations[key]
                cosine, idf_min, rarity, shared_count = sparse_pair_features(
                    representation,
                    query_global,
                    candidate_global,
                    (
                        representation.idf_b_candidates
                        if direction_name == "a_to_b"
                        else representation.idf_a_candidates
                    ),
                )
                block[:, FEATURE_INDEX[f"{prefix}_sae_cosine"]] = cosine
                block[:, FEATURE_INDEX[f"{prefix}_idf_min_activation"]] = idf_min
                block[:, FEATURE_INDEX[f"{prefix}_overlap_rarity"]] = rarity
                block[:, FEATURE_INDEX[f"{prefix}_shared_feature_count"]] = shared_count
                if representation.reconstruction_unit is None:
                    raise ValueError(f"Missing reconstruction for {key}")
                block[:, FEATURE_INDEX[f"{prefix}_reconstruction_cosine"]] = np.einsum(
                    "ij,j->i",
                    representation.reconstruction_unit[candidate_global],
                    representation.reconstruction_unit[query_global],
                    optimize=False,
                )

            labels = np.isin(shortlist, direction.gold_indices[position])
            feature_blocks.append(block)
            label_blocks.append(labels)
            candidate_blocks.append(shortlist)
            pair_ids.append(rows[row_id]["id"])
            canonical_pair = sorted(
                [
                    normalize_label(str(rows[row_id]["system_a"])),
                    normalize_label(str(rows[row_id]["system_b"])),
                ]
            )
            fold_group_ids.append("\u241f".join(canonical_pair))
            query_directions.append(direction_name)
            query_row_ids.append(row_id)
            low_flags.append(low_by_row_id[row_id])

    table = CandidateTable(
        features=np.concatenate(feature_blocks, axis=0),
        labels=np.concatenate(label_blocks).astype(bool),
        candidate_indices=np.concatenate(candidate_blocks),
        query_pair_ids=np.asarray(pair_ids),
        fold_group_ids=np.asarray(fold_group_ids),
        query_directions=query_directions,
        query_row_ids=np.asarray(query_row_ids, dtype=np.int64),
        low_overlap=np.asarray(low_flags, dtype=bool),
    )
    if table.n_queries != 566 or table.features.shape != (56600, len(FEATURE_NAMES)):
        raise ValueError(f"Unexpected candidate table shape: {table.features.shape}")
    if not np.isfinite(table.features).all():
        raise ValueError("Candidate features contain non-finite values")
    dense_hits = np.asarray(
        [table.labels[table.row_slice(index)][:TOP_K].any() for index in range(566)]
    )
    if int(dense_hits.sum()) != 146:
        raise ValueError(
            f"Dense shortlist construction drifted: expected 146 hits, got {dense_hits.sum()}"
        )
    return table


def assign_pair_grouped_folds(
    pair_ids: np.ndarray, fold_group_ids: np.ndarray
) -> np.ndarray:
    unique_ids = np.unique(pair_ids)
    rng = np.random.default_rng(SEED)
    shuffled = unique_ids[rng.permutation(len(unique_ids))]
    pair_to_base_fold = {
        pair_id: index % N_FOLDS for index, pair_id in enumerate(shuffled)
    }
    group_to_fold: dict[str, int] = {}
    for group in np.unique(fold_group_ids):
        member_pair_ids = np.unique(pair_ids[fold_group_ids == group])
        representative = sorted(member_pair_ids.tolist())[0]
        group_to_fold[str(group)] = pair_to_base_fold[representative]
    folds = np.asarray(
        [group_to_fold[str(group)] for group in fold_group_ids], dtype=np.int64
    )
    for fold in range(N_FOLDS):
        test_groups = set(fold_group_ids[folds == fold].tolist())
        train_groups = set(fold_group_ids[folds != fold].tolist())
        if test_groups & train_groups:
            raise AssertionError("Pair-group leakage across folds")
    return folds


def pairwise_training_rows(
    table: CandidateTable,
    query_indices: np.ndarray,
    feature_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    differences: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    trained_queries = 0
    for query_index_value in query_indices:
        query_index = int(query_index_value)
        block_slice = table.row_slice(query_index)
        labels = table.labels[block_slice]
        positive_positions = np.flatnonzero(labels)
        negative_positions = np.flatnonzero(~labels)
        if not len(positive_positions) or not len(negative_positions):
            continue
        trained_queries += 1
        block = table.features[block_slice][:, feature_indices]
        for positive_position in positive_positions:
            positive_differences = block[positive_position] - block[negative_positions]
            differences.append(positive_differences)
            differences.append(-positive_differences)
            per_row_weight = 1.0 / (
                2.0 * len(positive_positions) * len(negative_positions)
            )
            weights.append(
                np.full(len(positive_differences), per_row_weight, dtype=np.float64)
            )
            weights.append(
                np.full(len(positive_differences), per_row_weight, dtype=np.float64)
            )
    if not differences:
        raise ValueError("No pairwise training rows were generated")
    matrix = np.concatenate(differences, axis=0).astype(np.float64)
    labels = np.concatenate(
        [
            np.ones(len(block), dtype=np.int8)
            if index % 2 == 0
            else np.zeros(len(block), dtype=np.int8)
            for index, block in enumerate(differences)
        ]
    )
    return matrix, labels, np.concatenate(weights), trained_queries


def fit_oof_scores(
    table: CandidateTable, folds: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    oof_scores: dict[str, np.ndarray] = {
        "dense_only": table.features[:, FEATURE_INDEX["dense_cosine"]].astype(np.float64)
    }
    diagnostics: dict[str, Any] = {}
    all_query_indices = np.arange(table.n_queries)

    for method_name, feature_names in METHOD_FEATURES.items():
        feature_indices = np.asarray(
            [FEATURE_INDEX[name] for name in feature_names], dtype=np.int64
        )
        predictions = np.full(len(table.features), np.nan, dtype=np.float64)
        fold_coefficients: list[list[float]] = []
        fold_diagnostics: list[dict[str, int]] = []
        for fold in range(N_FOLDS):
            train_queries = all_query_indices[folds != fold]
            test_queries = all_query_indices[folds == fold]
            train_matrix, train_labels, train_weights, trained_queries = (
                pairwise_training_rows(
                    table, train_queries, feature_indices
                )
            )
            weight_sum = float(train_weights.sum())
            feature_mean = np.sum(
                train_matrix * train_weights[:, None], axis=0
            ) / weight_sum
            centered_train = train_matrix - feature_mean
            feature_variance = np.sum(
                centered_train**2 * train_weights[:, None], axis=0
            ) / weight_sum
            feature_scale = np.sqrt(np.maximum(feature_variance, 1e-12))
            scaled_train = centered_train / feature_scale
            model = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                fit_intercept=False,
                max_iter=1000,
                random_state=SEED,
            )
            model.fit(
                scaled_train,
                train_labels,
                sample_weight=train_weights,
            )
            test_rows = np.concatenate(
                [
                    np.arange(
                        query_index * SHORTLIST_SIZE,
                        (query_index + 1) * SHORTLIST_SIZE,
                    )
                    for query_index in test_queries
                ]
            )
            scaled_test = (
                table.features[test_rows][:, feature_indices] - feature_mean
            ) / feature_scale
            predictions[test_rows] = np.einsum(
                "ij,j->i",
                scaled_test,
                model.coef_[0],
                optimize=False,
            )
            fold_coefficients.append(model.coef_[0].astype(float).tolist())
            fold_diagnostics.append(
                {
                    "fold": fold,
                    "train_queries": int(len(train_queries)),
                    "train_queries_with_shortlist_positive": int(trained_queries),
                    "test_queries": int(len(test_queries)),
                    "pairwise_rows": int(len(train_matrix)),
                }
            )
        if not np.isfinite(predictions).all():
            raise ValueError(f"Missing or non-finite OOF predictions for {method_name}")
        oof_scores[method_name] = predictions
        coefficients = np.asarray(fold_coefficients)
        diagnostics[method_name] = {
            "features": feature_names,
            "folds": fold_diagnostics,
            "mean_standardized_pairwise_coefficients": {
                name: float(value)
                for name, value in zip(feature_names, coefficients.mean(axis=0))
            },
        }
    return oof_scores, diagnostics


def rank_method(
    table: CandidateTable, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.full(table.n_queries, np.inf, dtype=np.float64)
    hits = np.zeros(table.n_queries, dtype=bool)
    for query_index in range(table.n_queries):
        block_slice = table.row_slice(query_index)
        block_scores = scores[block_slice]
        block_candidates = table.candidate_indices[block_slice]
        order = np.lexsort((block_candidates, -block_scores))
        ordered_labels = table.labels[block_slice][order]
        positive_positions = np.flatnonzero(ordered_labels)
        if len(positive_positions):
            ranks[query_index] = float(positive_positions[0] + 1)
            hits[query_index] = bool(positive_positions[0] < TOP_K)
    return ranks, hits


def bootstrap_pair_group_delta(
    challenger_hits: np.ndarray,
    dense_hits: np.ndarray,
    fold_group_ids: np.ndarray,
) -> dict[str, float]:
    differences = challenger_hits.astype(float) - dense_hits.astype(float)
    unique_groups = np.unique(fold_group_ids)
    group_sums = np.asarray(
        [differences[fold_group_ids == group].sum() for group in unique_groups]
    )
    group_sizes = np.asarray(
        [np.count_nonzero(fold_group_ids == group) for group in unique_groups]
    )
    rng = np.random.default_rng(SEED)
    indices = rng.integers(
        0, len(unique_groups), size=(BOOTSTRAP_SAMPLES, len(unique_groups))
    )
    bootstrap = group_sums[indices].sum(axis=1) / group_sizes[indices].sum(axis=1)
    point_delta = float(differences.mean())
    return {
        "delta": point_delta,
        "delta_percentage_points": float(100.0 * point_delta),
        "ci_95_low": float(np.quantile(bootstrap, 0.025)),
        "ci_95_high": float(np.quantile(bootstrap, 0.975)),
        "ci_95_low_percentage_points": float(100.0 * np.quantile(bootstrap, 0.025)),
        "ci_95_high_percentage_points": float(100.0 * np.quantile(bootstrap, 0.975)),
    }


def evaluate_oof(
    table: CandidateTable, oof_scores: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    metrics: dict[str, Any] = {}
    ranks_by_method: dict[str, np.ndarray] = {}
    hits_by_method: dict[str, np.ndarray] = {}
    dense_hits: np.ndarray | None = None
    for method_name, scores in oof_scores.items():
        ranks, hits = rank_method(table, scores)
        ranks_by_method[method_name] = ranks
        hits_by_method[method_name] = hits
        finite_ranks = ranks[np.isfinite(ranks)]
        low = table.low_overlap
        metrics[method_name] = {
            "recall_at_1": float(np.mean(ranks <= 1)),
            "recall_at_5": float(np.mean(ranks <= 5)),
            "recall_at_10": float(hits.mean()),
            "recall_at_10_percentage": float(100.0 * hits.mean()),
            "mrr_with_shortlist_misses_as_zero": float(
                np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))
            ),
            "median_rank_when_retrievable": float(np.median(finite_ranks)),
            "low_overlap_recall_at_10": float(hits[low].mean()),
            "top10_successes": int(hits.sum()),
        }
        if method_name == "dense_only":
            dense_hits = hits

    if dense_hits is None or int(dense_hits.sum()) != 146:
        raise AssertionError("Dense OOF baseline does not reproduce the gate")
    for method_name in oof_scores:
        if method_name == "dense_only":
            continue
        hits = hits_by_method[method_name]
        metrics[method_name]["vs_dense"] = {
            **bootstrap_pair_group_delta(
                hits, dense_hits, table.fold_group_ids
            ),
            "rescues": int((~dense_hits & hits).sum()),
            "losses": int((dense_hits & ~hits).sum()),
            "net_successes": int(hits.sum() - dense_hits.sum()),
        }
    return metrics, ranks_by_method, hits_by_method


def query_records(
    rows: Sequence[dict[str, Any]],
    table: CandidateTable,
    folds: np.ndarray,
    ranks_by_method: dict[str, np.ndarray],
    hits_by_method: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for query_index in range(table.n_queries):
        row_id = int(table.query_row_ids[query_index])
        direction = table.query_directions[query_index]
        if direction == "a_to_b":
            query_key, gold_key = "system_a", "system_b"
        else:
            query_key, gold_key = "system_b", "system_a"
        block_labels = table.labels[table.row_slice(query_index)]
        record: dict[str, Any] = {
            "query_id": f"{direction}:{rows[row_id]['id']}",
            "pair_id": rows[row_id]["id"],
            "fold_group_id": str(table.fold_group_ids[query_index]),
            "fold": int(folds[query_index]),
            "direction": direction,
            "query": rows[row_id][query_key],
            "gold": rows[row_id][gold_key],
            "low_lexical_overlap": bool(table.low_overlap[query_index]),
            "gold_in_dense_top100": bool(block_labels.any()),
        }
        for method_name in ranks_by_method:
            rank = ranks_by_method[method_name][query_index]
            record[f"{method_name}_rank"] = int(rank) if np.isfinite(rank) else None
            record[f"{method_name}_top10"] = bool(
                hits_by_method[method_name][query_index]
            )
        records.append(record)
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
    if len(rows) != 400 or len(cross_domain_row_ids) != 283:
        raise ValueError(
            f"Expected 400 SCAR pairs / 283 cross-domain pairs; found "
            f"{len(rows)} / {len(cross_domain_row_ids)}"
        )
    raw_embeddings = load_raw_openai_embeddings(args.embeddings, len(rows))

    representations: dict[str, SparseRepresentation] = {}
    checkpoint_metadata: dict[str, Any] = {}
    for key, config in CHECKPOINTS.items():
        checkpoint_path = args.weights_dir / str(config["filename"])
        actual_hash = sha256_file(checkpoint_path)
        if actual_hash != config["sha256"]:
            raise ValueError(
                f"Checkpoint hash mismatch for {checkpoint_path}: {actual_hash}"
            )
        print(f"Encoding and reconstructing with {key} SAE...", flush=True)
        representations[key] = encode_sae_and_reconstruct(
            raw_embeddings, checkpoint_path
        )
        nonzero_counts = np.asarray(
            [len(indices) for indices in representations[key].active_indices]
        )
        checkpoint_metadata[key] = {
            **config,
            "path": str(checkpoint_path.relative_to(ROOT)),
            "source_revision": PINNED_CHECKPOINT_SOURCE,
            "d_model": 1536,
            "n_latents": 9216,
            "k": 64,
            "auxk_training_config": 128,
            "input_preprocessing": "raw float32 API vector",
            "mean_positive_activations": float(nonzero_counts.mean()),
        }

    print("Building seeded random sparse-projection control...", flush=True)
    representations["random"] = random_sparse_control(raw_embeddings)
    print("Building dense top-100 candidate features...", flush=True)
    table = build_candidate_table(
        rows, cross_domain_row_ids, raw_embeddings, representations
    )
    folds = assign_pair_grouped_folds(
        table.query_pair_ids, table.fold_group_ids
    )
    print("Fitting fixed five-fold pairwise rerankers...", flush=True)
    oof_scores, training_diagnostics = fit_oof_scores(table, folds)
    metrics, ranks_by_method, hits_by_method = evaluate_oof(table, oof_scores)

    shortlist_ceiling_hits = np.asarray(
        [table.labels[table.row_slice(index)].any() for index in range(table.n_queries)]
    )
    records = query_records(
        rows, table, folds, ranks_by_method, hits_by_method
    )
    if len(records) != 566 or len({record["query_id"] for record in records}) != 566:
        raise AssertionError("Per-query record cardinality mismatch")

    primary = metrics["dense_idf_sae"]
    primary_delta = primary["vs_dense"]["delta"]
    non_primary_control_recalls = {
        method_name: method_metrics["recall_at_10"]
        for method_name, method_metrics in metrics.items()
        if method_name != "dense_idf_sae"
    }
    strong_support = bool(
        primary_delta >= MIN_STRONG_HYBRID_GAIN
        and primary["vs_dense"]["ci_95_low"] > 0
        and primary["recall_at_10"] > max(non_primary_control_recalls.values())
    )
    if strong_support:
        hybrid_verdict = "STRONG_SUPPORT_FOR_SPARSE_BRIDGES"
    elif primary_delta > 0:
        hybrid_verdict = "PROMISING_BUT_INCONCLUSIVE"
    else:
        hybrid_verdict = "NO_HELD_OUT_HYBRID_GAIN"
    summary = {
        "run": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": Path(__file__).name,
            "seed": SEED,
        },
        "question": (
            "Can SAE complementarity be converted into held-out top-10 gains by a "
            "dense-shortlist reranker without mean-score fusion?"
        ),
        "primary_method": "dense_idf_sae",
        "primary_result": {
            "verdict": hybrid_verdict,
            "strong_support": strong_support,
            "improves_dense_point_estimate": bool(primary_delta > 0),
            "recall_at_10": primary["recall_at_10"],
            "delta_over_dense": primary_delta,
            "delta_percentage_points": primary["vs_dense"][
                "delta_percentage_points"
            ],
            "canonical_pair_cluster_bootstrap_ci_95": [
                primary["vs_dense"]["ci_95_low"],
                primary["vs_dense"]["ci_95_high"],
            ],
            "pre_specified_strong_support_rule": (
                "at least +2.0 percentage points over raw dense, positive canonical-"
                "pair-cluster bootstrap lower bound, and higher Recall@10 than every "
                "control"
            ),
            "recall_at_10_minus_each_control": {
                method_name: float(primary["recall_at_10"] - recall)
                for method_name, recall in non_primary_control_recalls.items()
            },
            "interpretation_rule": (
                "Treat this as an exploratory bridge test: a positive held-out point "
                "estimate supports the Sparse Bridges direction; the clustered "
                "interval states uncertainty and is not used for post-hoc tuning."
            ),
        },
        "protocol": {
            "benchmark": "SCAR English bidirectional cross-domain retrieval proxy",
            "benchmark_url": SCAR_URL,
            "data_sha256": sha256_file(args.data),
            "embedding_cache_sha256": sha256_file(args.embeddings),
            "query_directions": table.n_queries,
            "pair_groups": int(len(np.unique(table.query_pair_ids))),
            "canonical_fold_groups": int(len(np.unique(table.fold_group_ids))),
            "bootstrap_clusters": int(len(np.unique(table.fold_group_ids))),
            "candidates_per_direction": len(rows),
            "dense_shortlist_size": SHORTLIST_SIZE,
            "evaluation_k": TOP_K,
            "folds": N_FOLDS,
            "fold_assignment": (
                "seeded SCAR pair-id split, then minimally merged by canonical unordered "
                "normalized system pair; both directions and duplicate pair rows share "
                "a fold"
            ),
            "learner": (
                "L2 logistic pairwise linear ranker, C=1, standardized features, "
                "all shortlist negatives, symmetric differences, equal total weight "
                "per train query"
            ),
            "idf_definition": (
                "per retrieval direction, log((400+1)/(candidate-side activation "
                "document frequency+1))+1; fixed-corpus indexing without labels"
            ),
            "relevance": (
                "any candidate with the same normalized target system name is relevant"
            ),
            "tie_breaking": "score descending, candidate index ascending",
            "shortlist_recall_ceiling": float(shortlist_ceiling_hits.mean()),
            "shortlist_recall_ceiling_percentage": float(
                100.0 * shortlist_ceiling_hits.mean()
            ),
            "shortlist_retrievable_queries": int(shortlist_ceiling_hits.sum()),
            "low_overlap_queries": int(table.low_overlap.sum()),
        },
        "checkpoints": checkpoint_metadata,
        "controls": {
            "dense_only": "unchanged dense ordering",
            "dense_bm25": "pairwise dense score/rank plus BM25",
            "dense_random_sparse": (
                "pairwise dense score/rank plus seeded random 9,216-D top-64 ReLU "
                "projection cosine, IDF bridge/rarity/count, and transpose reconstruction"
            ),
            "dense_unweighted_sae": (
                "pairwise dense score/rank plus separate cs.LG and astro.PH SAE cosines"
            ),
            "dense_idf_sae": (
                "primary full bridge model: dense, lexical, separate SAE cosine, "
                "IDF-min activation, overlap rarity/count, and reconstruction features"
            ),
        },
        "metrics": metrics,
        "training_diagnostics": training_diagnostics,
        "artifacts": {
            "per_query_jsonl": str(args.queries.relative_to(ROOT)),
        },
        "scope_caveats": [
            "SCAR uses system descriptions rather than a scientific-paper corpus.",
            "The candidate corpus is fixed and visible, but pair labels and both query "
            "directions are held out together for every fitted prediction.",
            "The random projection matches one SAE's latent dimensionality, top-k "
            "sparsity, and feature families, not empirical activation frequencies or "
            "the full two-dictionary parameter count.",
            "This is a single pre-specified bridge test, not a tuned final reranker.",
        ],
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.queries.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.queries.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    compact = {
        method: {
            "recall_at_10": values["recall_at_10"],
            "top10_successes": values["top10_successes"],
            "delta_pp": values.get("vs_dense", {}).get("delta_percentage_points", 0.0),
        }
        for method, values in metrics.items()
    }
    print(json.dumps({"primary_result": summary["primary_result"], "metrics": compact}, indent=2))
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.queries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
