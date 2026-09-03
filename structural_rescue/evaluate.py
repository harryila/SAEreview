"""Gold-isolated evaluation for Structural Rescue."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sae_smoke_test import load_jsonl

from .core import ARM_NAMES, DEFAULT_PROTOCOL, TOP_K, write_json, write_jsonl
from .llm import MODEL, verdict_score


SAE_STRUCTURE_ARM = "sae_union_padded30_structure"
SAE_ACTIVATION_ARM = "sae_union_padded30_activation_only"
SAE_ALIGNED_ARM = "sae_union_padded30_aligned_description"
SAE_SHUFFLED_ARM = "sae_union_padded30_shuffled_description"
RANDOM_STRUCTURE_ARMS = tuple(
    f"random_union_padded30_structure_{index}" for index in range(1, 4)
)

PRIMARY_COMPARISONS = (
    (SAE_STRUCTURE_ARM, "dense30_structure"),
    *((SAE_STRUCTURE_ARM, arm) for arm in RANDOM_STRUCTURE_ARMS),
    (SAE_ACTIVATION_ARM, SAE_STRUCTURE_ARM),
    (SAE_ALIGNED_ARM, SAE_STRUCTURE_ARM),
    (SAE_ALIGNED_ARM, SAE_ACTIVATION_ARM),
    (SAE_ALIGNED_ARM, SAE_SHUFFLED_ARM),
    (SAE_ALIGNED_ARM, "dense30_structure"),
    *((SAE_ALIGNED_ARM, arm) for arm in RANDOM_STRUCTURE_ARMS),
    (SAE_SHUFFLED_ARM, SAE_ACTIVATION_ARM),
)

ARM_SCORE_MODES = {
    "dense30_structure": "structure",
    SAE_STRUCTURE_ARM: "structure",
    **{arm: "structure" for arm in RANDOM_STRUCTURE_ARMS},
    SAE_ACTIVATION_ARM: "activation_only",
    SAE_ALIGNED_ARM: "aligned_description",
    SAE_SHUFFLED_ARM: "shuffled_description",
}

FEATURE_MODES = (
    "activation_only",
    "aligned_description",
    "shuffled_description",
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
) -> dict[str, float | None]:
    differences = np.asarray(challenger, dtype=float) - np.asarray(baseline, dtype=float)
    if not len(differences):
        return {
            "delta": None,
            "delta_percentage_points": None,
            "ci_95_low": None,
            "ci_95_high": None,
        }
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


def stratified_clustered_bootstrap_utility(
    challenger: np.ndarray,
    baseline: np.ndarray,
    groups: Sequence[str],
    rescue_mask: np.ndarray,
    retention_mask: np.ndarray,
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, float | int]:
    """Bootstrap the sign of the paired net-utility difference.

    On the fixed balanced screen, utility is rescues recovered minus original
    dense hits lost.  Its arm-to-arm difference is the sum of paired top-10 hit
    differences in the two strata.  Pair groups, rather than query directions,
    are resampled independently inside each stratum.
    """

    differences = np.asarray(challenger, dtype=int) - np.asarray(baseline, dtype=int)
    group_array = np.asarray(groups)
    rng = np.random.default_rng(seed)
    boot_total = np.zeros(samples, dtype=float)
    for mask in (np.asarray(rescue_mask, dtype=bool), np.asarray(retention_mask, dtype=bool)):
        masked_groups = group_array[mask]
        masked_differences = differences[mask]
        unique_groups = np.unique(masked_groups)
        if not len(unique_groups):
            raise ValueError("Both development strata must contain pair groups")
        group_sums = np.asarray(
            [masked_differences[masked_groups == group].sum() for group in unique_groups],
            dtype=float,
        )
        draws = rng.integers(0, len(unique_groups), size=(samples, len(unique_groups)))
        boot_total += group_sums[draws].sum(axis=1)
    observed = int(differences[rescue_mask | retention_mask].sum())
    return {
        "delta_net_utility": observed,
        "bootstrap_samples": samples,
        "bootstrap_positive_fraction": float(np.mean(boot_total > 0)),
        "bootstrap_zero_fraction": float(np.mean(boot_total == 0)),
    }


def _arm_utility(metric: Mapping[str, Any]) -> int:
    return int(metric["known_rescues_recovered"]) - int(metric["losses_vs_dense_top10"])


def _development_gate(
    metrics: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    evidence_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen screen decision rule without discretionary interpretation."""

    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    rule = protocol["development_go_no_rule"]

    def comparison(challenger: str, baseline: str) -> Mapping[str, Any]:
        return comparisons[f"{challenger}_minus_{baseline}"]

    def margin_pass(challenger: str, baseline: str) -> bool:
        result = comparison(challenger, baseline)
        observed_delta = int(result["net_utility"]["delta_net_utility"])
        expected_delta = _arm_utility(metrics[challenger]) - _arm_utility(
            metrics[baseline]
        )
        if observed_delta != expected_delta:
            raise ValueError(
                f"Inconsistent net-utility comparison for {challenger} minus "
                f"{baseline}: {observed_delta} != {expected_delta}"
            )
        return (
            observed_delta >= int(rule["minimum_arm_net_utility_margin"])
            and float(result["net_utility"]["bootstrap_positive_fraction"])
            >= float(rule["minimum_bootstrap_positive_fraction"])
        )

    s_metric = metrics[SAE_STRUCTURE_ARM]
    prepare_report_path = DEFAULT_PROTOCOL.with_name("prepare_report.json")
    prepare_report = json.loads(prepare_report_path.read_text(encoding="utf-8"))
    oracle_distribution = prepare_report["preflight"][
        "random_source_oracle_hit_distribution"
    ]
    sae_oracle_hits = int(oracle_distribution["sae_source_oracle_hits"])
    oracle_above_q95 = sae_oracle_hits > int(oracle_distribution["q95_higher"])
    oracle_tail_pass = float(
        oracle_distribution["plus_one_tail_probability"]
    ) <= float(rule["random_source_tail_alpha"])
    candidate_checks = {
        "sae_source_oracle_above_random_95th_percentile": oracle_above_q95,
        "sae_source_oracle_tail_probability_at_most_alpha": oracle_tail_pass,
        "minimum_known_rescues": int(s_metric["known_rescues_recovered"])
        >= int(rule["minimum_known_rescues_recovered"]),
        "maximum_dense_losses": int(s_metric["losses_vs_dense_top10"])
        <= int(rule["maximum_dense_hits_lost"]),
        "minimum_net_utility": _arm_utility(s_metric)
        >= int(rule["minimum_net_utility"]),
        "beats_dense30": margin_pass(SAE_STRUCTURE_ARM, "dense30_structure"),
        **{
            f"beats_random_{index}": margin_pass(SAE_STRUCTURE_ARM, random_arm)
            for index, random_arm in enumerate(RANDOM_STRUCTURE_ARMS, start=1)
        },
    }
    candidate_pass = all(candidate_checks.values())

    f_metric = metrics[SAE_ALIGNED_ARM]
    evidence_comparators = (
        SAE_STRUCTURE_ARM,
        SAE_ACTIVATION_ARM,
        SAE_SHUFFLED_ARM,
    )
    feature_checks = {
        "minimum_rescue_queries_with_evidence": int(
            evidence_coverage["known_rescue_queries_with_usable_evidence"]
        )
        >= int(rule["minimum_queries_with_usable_evidence_per_stratum"]),
        "minimum_retention_queries_with_evidence": int(
            evidence_coverage["dense_retention_queries_with_usable_evidence"]
        )
        >= int(rule["minimum_queries_with_usable_evidence_per_stratum"]),
        "minimum_known_rescues": int(f_metric["known_rescues_recovered"])
        >= int(rule["minimum_known_rescues_recovered"]),
        "maximum_dense_losses": int(f_metric["losses_vs_dense_top10"])
        <= int(rule["maximum_dense_hits_lost"]),
        "minimum_net_utility": _arm_utility(f_metric)
        >= int(rule["minimum_net_utility"]),
    }
    for baseline in evidence_comparators:
        baseline_metric = metrics[baseline]
        label = baseline.removeprefix("sae_union_padded30_")
        feature_checks[f"beats_{label}"] = margin_pass(SAE_ALIGNED_ARM, baseline)
        feature_checks[f"rescues_not_below_{label}"] = int(
            f_metric["known_rescues_recovered"]
        ) >= int(baseline_metric["known_rescues_recovered"])
        feature_checks[f"losses_within_one_of_{label}"] = int(
            f_metric["losses_vs_dense_top10"]
        ) <= int(baseline_metric["losses_vs_dense_top10"]) + int(
            rule["maximum_incremental_dense_losses"]
        )
    feature_pass = all(feature_checks.values())

    narrow_feature_checks = {
        "feature_gate": feature_pass,
        "beats_dense30": margin_pass(SAE_ALIGNED_ARM, "dense30_structure"),
        **{
            f"beats_random_{index}": margin_pass(SAE_ALIGNED_ARM, random_arm)
            for index, random_arm in enumerate(RANDOM_STRUCTURE_ARMS, start=1)
        },
    }
    narrow_feature_pass = all(narrow_feature_checks.values())

    if candidate_pass and feature_pass:
        decision = "full_go_freeze_candidate_and_feature_method_externally"
    elif candidate_pass:
        decision = "narrow_go_structure_only_drop_feature_grounding_claim"
    elif narrow_feature_pass:
        decision = "narrow_go_feature_evidence_only_no_candidate_specificity_claim"
    else:
        decision = "no_go_stop_structural_rescue_on_scar"
    return {
        "decision": decision,
        "candidate_source_gate_passed": candidate_pass,
        "feature_grounding_gate_passed": feature_pass,
        "narrow_feature_only_gate_passed": narrow_feature_pass,
        "candidate_source_checks": candidate_checks,
        "feature_grounding_checks": feature_checks,
        "narrow_feature_only_checks": narrow_feature_checks,
        "random_source_oracle_diagnostic": oracle_distribution,
        "frozen_thresholds": rule,
        "interpretation": (
            "This is a development-screen decision only. A go licenses method "
            "freezing and one outcome-independent external evaluation; it is not "
            "a confirmatory result on SCAR."
        ),
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
    evidence_counts_by_pair_mode: dict[tuple[str, str, str], int] = {}
    for row in predictions:
        query_id = str(row["query_id"])
        mode = str(row["mode"])
        if mode not in {"structure", *FEATURE_MODES}:
            raise ValueError(f"Unknown verifier mode: {mode}")
        key = (query_id, mode)
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
        evidence_counts_by_pair_mode[(query_id, candidate_id, mode)] = int(
            row.get("feature_evidence_count", -1)
        )

    pair_keys = {(query_id, candidate_id) for query_id, candidate_id, _ in evidence_counts_by_pair_mode}
    for query_id, candidate_id in pair_keys:
        structure_count = evidence_counts_by_pair_mode.get(
            (query_id, candidate_id, "structure")
        )
        if structure_count != 0:
            raise ValueError("Structure-only predictions must have zero feature evidence")
        feature_counts = {
            evidence_counts_by_pair_mode.get((query_id, candidate_id, mode))
            for mode in FEATURE_MODES
        }
        if None in feature_counts or len(feature_counts) != 1:
            raise ValueError(
                "Activation, aligned, and shuffled modes must use identical evidence rows"
            )

    per_query: list[dict[str, Any]] = []
    for query in candidates:
        query_id = str(query["query_id"])
        qrel = qrels_by_id.get(query_id)
        if qrel is None:
            raise ValueError(f"Missing qrels for {query_id}")
        gold = set(map(str, qrel["gold_candidate_ids"]))
        dense_scores = {key: float(value) for key, value in query["dense_scores"].items()}
        rankings: dict[str, list[str]] = {
            "dense_ranking": list(query["pools"]["dense_ranking"])
        }
        for arm, mode in ARM_SCORE_MODES.items():
            rankings[arm] = rank_candidates(
                query["pools"][arm], score_maps[(query_id, mode)], dense_scores
            )
        record: dict[str, Any] = {
            "query_id": query_id,
            "pair_id": int(qrel["pair_id"]),
            "pair_group": str(qrel["pair_group"]),
            "known_sae_rescue": bool(qrel["known_sae_rescue"]),
            "dense_top10": bool(qrel["dense_top10"]),
            "arms": {},
            "source_pool_oracles": {},
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
        source_pools = query.get("source_pools", {})
        sae_source = list(map(str, source_pools.get("sae_union", [])))
        if sae_source:
            record["source_pool_oracles"]["sae_union"] = bool(
                gold.intersection(sae_source)
            )
        random_sources = source_pools.get("random_unions", [])
        for index, source_pool in enumerate(random_sources[: len(RANDOM_STRUCTURE_ARMS)], start=1):
            record["source_pool_oracles"][f"random_union_{index}"] = bool(
                gold.intersection(map(str, source_pool))
            )
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
        metrics[arm]["net_utility_rescues_minus_dense_losses"] = _arm_utility(
            metrics[arm]
        )

    source_pool_oracles: dict[str, Any] = {}
    source_names = sorted(
        {
            name
            for row in per_query
            for name in row["source_pool_oracles"]
        }
    )
    for name in source_names:
        values = [
            bool(row["source_pool_oracles"].get(name, False)) for row in per_query
        ]
        source_pool_oracles[name] = {
            "screen_oracle_successes": int(sum(values)),
            "screen_oracle_rate": float(np.mean(values)),
        }

    comparisons: dict[str, Any] = {}
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    bootstrap_seed_base = int(
        protocol["development_go_no_rule"]["bootstrap_seed_base"]
    )
    rescue_mask = np.asarray([row["known_sae_rescue"] for row in per_query], dtype=bool)
    retention_mask = np.asarray(
        [row["dense_top10"] and not row["known_sae_rescue"] for row in per_query],
        dtype=bool,
    )
    for comparison_index, (challenger, baseline) in enumerate(
        PRIMARY_COMPARISONS, start=1
    ):
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
            "net_utility": stratified_clustered_bootstrap_utility(
                hit_vectors[challenger],
                hit_vectors[baseline],
                groups,
                rescue_mask,
                retention_mask,
                seed=bootstrap_seed_base + comparison_index,
                samples=int(
                    protocol["development_go_no_rule"]["bootstrap_samples"]
                ),
            ),
        }

    aligned_pair_counts = {
        (query_id, candidate_id): count
        for (query_id, candidate_id, mode), count in evidence_counts_by_pair_mode.items()
        if mode == "aligned_description"
    }
    queries_with_evidence: dict[str, bool] = {}
    eligible_pairs = 0
    pairs_with_evidence = 0
    for query in candidates:
        query_id = str(query["query_id"])
        arm_candidates = list(map(str, query["pools"][SAE_ALIGNED_ARM]))
        counts = [aligned_pair_counts[(query_id, candidate)] for candidate in arm_candidates]
        eligible_pairs += len(counts)
        pairs_with_evidence += sum(count > 0 for count in counts)
        queries_with_evidence[query_id] = any(count > 0 for count in counts)
    evidence_coverage = {
        "eligible_pairs": eligible_pairs,
        "pairs_with_usable_evidence": pairs_with_evidence,
        "pair_coverage_rate": (
            float(pairs_with_evidence / eligible_pairs) if eligible_pairs else None
        ),
        "queries_with_usable_evidence": int(sum(queries_with_evidence.values())),
        "query_coverage_rate": float(np.mean(list(queries_with_evidence.values()))),
        "known_rescue_queries_with_usable_evidence": int(
            sum(
                queries_with_evidence[row["query_id"]]
                for row in per_query
                if row["known_sae_rescue"]
            )
        ),
        "dense_retention_queries_with_usable_evidence": int(
            sum(
                queries_with_evidence[row["query_id"]]
                for row in per_query
                if row["dense_top10"] and not row["known_sae_rescue"]
            )
        ),
    }

    complete_screen = (
        len(per_query) == 108
        and int(rescue_mask.sum()) == 54
        and int(retention_mask.sum()) == 54
    )
    development_gate = (
        _development_gate(metrics, comparisons, evidence_coverage)
        if complete_screen
        else {
            "decision": "not_applied_incomplete_screen",
            "complete_screen_required": True,
        }
    )

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
        "source_pool_oracles": source_pool_oracles,
        "feature_evidence_coverage": evidence_coverage,
        "comparisons": comparisons,
        "development_go_no": development_gate,
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
