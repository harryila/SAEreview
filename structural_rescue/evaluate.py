"""Gold-isolated evaluation for Structural Rescue."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sae_smoke_test import load_jsonl

from .core import ARM_NAMES, TOP_K, write_json, write_jsonl
from .llm import MODEL, verdict_score


PRIMARY_COMPARISONS = (
    ("sae_union_structure", "dense30_structure"),
    ("sae_union_structure", "random_union_structure"),
    ("sae_union_feature_grounded", "sae_union_structure"),
)


def rank_candidates(
    candidate_ids: Sequence[str],
    scores: Mapping[str, float],
    dense_scores: Mapping[str, float],
) -> list[str]:
    missing = [candidate for candidate in candidate_ids if candidate not in scores]
    if missing:
        raise ValueError(f"Missing verifier scores for {missing[:3]}")
    return sorted(
        candidate_ids,
        key=lambda candidate: (
            -float(scores[candidate]),
            -float(dense_scores[candidate]),
            candidate,
        ),
    )


def reciprocal_rank(ranking: Sequence[str], gold: set[str]) -> float:
    for index, candidate in enumerate(ranking, start=1):
        if candidate in gold:
            return 1.0 / index
    return 0.0


def optional_rate(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def clustered_bootstrap_delta(
    challenger: np.ndarray,
    baseline: np.ndarray,
    groups: Sequence[str],
    *,
    seed: int = 20260902,
    samples: int = 5000,
) -> dict[str, float]:
    differences = np.asarray(challenger, dtype=float) - np.asarray(baseline, dtype=float)
    group_array = np.asarray(groups)
    unique_groups = np.unique(group_array)
    sums = np.asarray([differences[group_array == group].sum() for group in unique_groups])
    sizes = np.asarray([np.count_nonzero(group_array == group) for group in unique_groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_groups), size=(samples, len(unique_groups)))
    bootstrap = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    delta = float(differences.mean())
    return {
        "delta": delta,
        "delta_percentage_points": 100.0 * delta,
        "ci_95_low": float(np.quantile(bootstrap, 0.025)),
        "ci_95_high": float(np.quantile(bootstrap, 0.975)),
    }


def evaluate_rows(
    candidates: Sequence[dict[str, Any]],
    qrels: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models = {str(row.get("model", "")) for row in predictions}
    if len(models) != 1 or "" in models:
        raise ValueError(f"Predictions must have one explicit model provenance: {models}")
    prediction_model = next(iter(models))
    evidentiary = prediction_model == MODEL
    qrels_by_id = {str(row["query_id"]): row for row in qrels}
    if len(qrels_by_id) != len(qrels):
        raise ValueError("Duplicate query IDs in qrels")

    score_maps: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in predictions:
        key = (str(row["query_id"]), str(row["mode"]))
        candidate_id = str(row["candidate_id"])
        if candidate_id in score_maps[key]:
            raise ValueError(f"Duplicate prediction for {key} / {candidate_id}")
        recomputed = verdict_score(row)
        if float(row["score"]) != float(recomputed):
            raise ValueError("Stored verifier score differs from fixed rubric score")
        if int(row.get("feature_evidence_count", -1)) == 0 and (
            int(row["feature_support"]) != 0
            or bool(row["accidental_feature_overlap"])
        ):
            raise ValueError("Feature rubric fields are nonempty without evidence")
        score = float(recomputed)
        if not np.isfinite(score):
            raise ValueError("Verifier scores must be finite")
        score_maps[key][candidate_id] = score

    per_query: list[dict[str, Any]] = []
    for query in candidates:
        query_id = str(query["query_id"])
        qrel = qrels_by_id.get(query_id)
        if qrel is None:
            raise ValueError(f"Missing qrels for {query_id}")
        gold = set(map(str, qrel["gold_candidate_ids"]))
        dense_scores = {key: float(value) for key, value in query["dense_scores"].items()}
        structure_scores = score_maps[(query_id, "structure")]
        feature_scores = score_maps[(query_id, "feature_grounded")]
        rankings: dict[str, list[str]] = {
            "dense_ranking": list(query["pools"]["dense_ranking"]),
            "dense30_structure": rank_candidates(
                query["pools"]["dense30_structure"], structure_scores, dense_scores
            ),
            "sae_union_structure": rank_candidates(
                query["pools"]["sae_union_structure"], structure_scores, dense_scores
            ),
            "random_union_structure": rank_candidates(
                query["pools"]["random_union_structure"], structure_scores, dense_scores
            ),
            "sae_union_feature_grounded": rank_candidates(
                query["pools"]["sae_union_feature_grounded"], feature_scores, dense_scores
            ),
        }
        record: dict[str, Any] = {
            "query_id": query_id,
            "pair_id": int(qrel["pair_id"]),
            "pair_group": str(qrel["pair_group"]),
            "known_sae_rescue": bool(qrel["known_sae_rescue"]),
            "dense_top10": bool(qrel["dense_top10"]),
            "arms": {},
        }
        for arm in ARM_NAMES:
            ranking = rankings[arm]
            record["arms"][arm] = {
                "pool_size": len(query["pools"][arm]),
                "pool_contains_gold": bool(gold.intersection(query["pools"][arm])),
                "top10_contains_gold": bool(gold.intersection(ranking[:TOP_K])),
                "reciprocal_rank": reciprocal_rank(ranking, gold),
                "top10_candidate_ids": ranking[:TOP_K],
            }
        per_query.append(record)

    if len(per_query) != len(candidates) or len(per_query) != len(qrels):
        raise ValueError("Candidate/qrel/prediction population mismatch")

    metrics: dict[str, Any] = {}
    groups = [row["pair_group"] for row in per_query]
    hit_vectors: dict[str, np.ndarray] = {}
    for arm in ARM_NAMES:
        hits = np.asarray(
            [row["arms"][arm]["top10_contains_gold"] for row in per_query], dtype=bool
        )
        oracles = np.asarray(
            [row["arms"][arm]["pool_contains_gold"] for row in per_query], dtype=bool
        )
        reciprocal = np.asarray(
            [row["arms"][arm]["reciprocal_rank"] for row in per_query], dtype=float
        )
        rescue_mask = np.asarray([row["known_sae_rescue"] for row in per_query], dtype=bool)
        dense_hit_mask = np.asarray([row["dense_top10"] for row in per_query], dtype=bool)
        hit_vectors[arm] = hits
        metrics[arm] = {
            "queries": len(per_query),
            "top10_successes": int(hits.sum()),
            "screen_top10_rate": float(hits.mean()),
            "screen_mean_reciprocal_rank": float(reciprocal.mean()),
            "pool_oracle_successes": int(oracles.sum()),
            "screen_pool_oracle_rate": float(oracles.mean()),
            "known_rescue_queries": int(rescue_mask.sum()),
            "known_rescues_recovered": int((hits & rescue_mask).sum()),
            "known_rescue_recovery_rate": optional_rate(hits[rescue_mask]),
            "dense_hit_queries": int(dense_hit_mask.sum()),
            "dense_hits_retained": int((hits & dense_hit_mask).sum()),
            "dense_hit_retention_rate": optional_rate(hits[dense_hit_mask]),
            "rescues_vs_dense_top10": int((hits & ~dense_hit_mask).sum()),
            "losses_vs_dense_top10": int((~hits & dense_hit_mask).sum()),
        }

    comparisons: dict[str, Any] = {}
    rescue_mask = np.asarray([row["known_sae_rescue"] for row in per_query], dtype=bool)
    retention_mask = np.asarray(
        [row["dense_top10"] and not row["known_sae_rescue"] for row in per_query],
        dtype=bool,
    )
    for challenger, baseline in PRIMARY_COMPARISONS:
        key = f"{challenger}_minus_{baseline}"
        comparisons[key] = {
            "descriptive_screen_delta": float(
                hit_vectors[challenger].mean() - hit_vectors[baseline].mean()
            ),
            "known_rescue_stratum": clustered_bootstrap_delta(
                hit_vectors[challenger][rescue_mask],
                hit_vectors[baseline][rescue_mask],
                np.asarray(groups)[rescue_mask].tolist(),
            ),
            "dense_retention_stratum": clustered_bootstrap_delta(
                hit_vectors[challenger][retention_mask],
                hit_vectors[baseline][retention_mask],
                np.asarray(groups)[retention_mask].tolist(),
            ),
            "net_screen_top10_successes": int(
                hit_vectors[challenger].sum() - hit_vectors[baseline].sum()
            ),
        }

    status = (
        "exploratory_development_result"
        if evidentiary
        else (
            "dry_run_non_evidentiary"
            if prediction_model == "fixture-no-model"
            else "unrecognized_model_non_evidentiary"
        )
    )
    report = {
        "status": status,
        "evidentiary": evidentiary,
        "prediction_model": prediction_model,
        "population_estimate_allowed": False,
        "selection": "outcome_stratified_exploratory_screen",
        "queries": len(per_query),
        "pair_groups": len(set(groups)),
        "metrics": metrics,
        "comparisons": comparisons,
        "claim_boundary": (
            "This inspected SCAR development result cannot establish a confirmatory "
            "SAE retrieval gain, an SAE intervention effect, serendipity, or discovery."
        ),
    }
    return report, per_query


def evaluate_files(
    *,
    candidate_path: Path,
    qrels_path: Path,
    predictions_path: Path,
    report_path: Path,
    per_query_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    report, per_query = evaluate_rows(
        load_jsonl(candidate_path),
        load_jsonl(qrels_path),
        load_jsonl(predictions_path),
    )
    write_json(report_path, report, overwrite=overwrite)
    write_jsonl(per_query_path, per_query, overwrite=overwrite)
    return report
