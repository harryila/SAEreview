from __future__ import annotations

from copy import deepcopy

import pytest

from structural_rescue.evaluate import (
    PRIMARY_COMPARISONS,
    SAE_ACTIVATION_ARM,
    SAE_ALIGNED_ARM,
    SAE_SHUFFLED_ARM,
    SAE_STRUCTURE_ARM,
    _development_gate,
    evaluate_rows,
)
from structural_rescue.llm import MODEL


def _prediction(query_id: str, candidate: str, mode: str, score: int) -> dict:
    if score == 4:
        role_alignment, break_severity = 2, 0
    elif score == -2:
        role_alignment, break_severity = 0, 1
    elif score == 0:
        role_alignment, break_severity = 0, 0
    else:
        raise ValueError("Unsupported test score")
    return {
        "query_id": query_id,
        "candidate_id": candidate,
        "mode": mode,
        "score": score,
        "model": "fixture-no-model",
        "feature_evidence_count": 0,
        "role_alignment": role_alignment,
        "causal_alignment": 0,
        "dynamics_alignment": 0,
        "constraint_alignment": 0,
        "feature_support": 0,
        "lexical_only": False,
        "same_domain_only": False,
        "accidental_feature_overlap": False,
        "break_severity": break_severity,
    }


def test_evaluation_keeps_gold_separate_and_computes_fixed_arms() -> None:
    candidates = [
        {
            "query_id": "q1",
            "dense_scores": {"c1": 0.9, "c2": 0.8, "gold": 0.1},
            "pools": {
                "dense_ranking": ["c1", "c2"],
                "dense30_structure": ["c1", "c2", "gold"],
                "sae_union_padded30_structure": ["c1", "gold"],
                "random_union_padded30_structure_1": ["c1", "c2"],
                "random_union_padded30_structure_2": ["c1", "c2"],
                "random_union_padded30_structure_3": ["c1", "c2"],
                "sae_union_padded30_activation_only": ["c1", "gold"],
                "sae_union_padded30_aligned_description": ["c1", "gold"],
                "sae_union_padded30_shuffled_description": ["c1", "gold"],
            },
            "source_pools": {
                "sae_union": ["c1", "gold"],
                "random_unions": [["c1", "c2"]] * 64,
            },
        },
        {
            "query_id": "q2",
            "dense_scores": {"gold2": 0.9, "d2": 0.2},
            "pools": {
                "dense_ranking": ["gold2", "d2"],
                "dense30_structure": ["gold2", "d2"],
                "sae_union_padded30_structure": ["gold2", "d2"],
                "random_union_padded30_structure_1": ["gold2", "d2"],
                "random_union_padded30_structure_2": ["gold2", "d2"],
                "random_union_padded30_structure_3": ["gold2", "d2"],
                "sae_union_padded30_activation_only": ["gold2", "d2"],
                "sae_union_padded30_aligned_description": ["gold2", "d2"],
                "sae_union_padded30_shuffled_description": ["gold2", "d2"],
            },
            "source_pools": {
                "sae_union": ["gold2", "d2"],
                "random_unions": [["gold2", "d2"]] * 64,
            },
        },
    ]
    qrels = [
        {
            "query_id": "q1",
            "pair_id": 1,
            "pair_group": "g1",
            "gold_candidate_ids": ["gold"],
            "dense_top10": False,
            "known_sae_rescue": True,
        },
        {
            "query_id": "q2",
            "pair_id": 2,
            "pair_group": "g2",
            "gold_candidate_ids": ["gold2"],
            "dense_top10": True,
            "known_sae_rescue": False,
        },
    ]
    predictions = []
    for query_id, scores in (
        ("q1", {"c1": 0, "c2": -2, "gold": 4}),
        ("q2", {"gold2": 4, "d2": 0}),
    ):
        for candidate, score in scores.items():
            predictions.append(_prediction(query_id, candidate, "structure", score))
        for mode in (
            "activation_only",
            "aligned_description",
            "shuffled_description",
        ):
            for candidate, score in scores.items():
                predictions.append(_prediction(query_id, candidate, mode, score))

    report, rows = evaluate_rows(candidates, qrels, predictions)
    assert report["status"] == "dry_run_non_evidentiary"
    assert (
        report["metrics"]["sae_union_padded30_structure"][
            "known_rescues_recovered"
        ]
        == 1
    )
    assert report["metrics"]["sae_union_padded30_structure"]["dense_hits_retained"] == 1
    assert report["development_go_no"]["decision"] == "not_applied_incomplete_screen"
    assert len(rows) == 2

    predictions[0]["score"] = 99
    with pytest.raises(ValueError, match="differs"):
        evaluate_rows(candidates, qrels, predictions)

    predictions[0]["score"] = 0
    predictions[0]["model"] = MODEL
    with pytest.raises(ValueError, match="one explicit model"):
        evaluate_rows(candidates, qrels, predictions)


def _passing_narrow_gate_inputs():
    arm_names = {
        SAE_STRUCTURE_ARM,
        SAE_ACTIVATION_ARM,
        SAE_ALIGNED_ARM,
        SAE_SHUFFLED_ARM,
        "dense30_structure",
        "random_union_padded30_structure_1",
        "random_union_padded30_structure_2",
        "random_union_padded30_structure_3",
    }
    metrics = {
        arm: {
            "known_rescues_recovered": 28,
            "losses_vs_dense_top10": 4,
        }
        for arm in arm_names
    }
    metrics[SAE_ALIGNED_ARM] = {
        "known_rescues_recovered": 33,
        "losses_vs_dense_top10": 4,
    }
    def utility(arm):
        return (
            metrics[arm]["known_rescues_recovered"]
            - metrics[arm]["losses_vs_dense_top10"]
        )

    comparisons = {}
    for challenger, baseline in PRIMARY_COMPARISONS:
        delta = utility(challenger) - utility(baseline)
        comparisons[f"{challenger}_minus_{baseline}"] = {
            "net_utility": {
                "delta_net_utility": delta,
                "bootstrap_positive_fraction": 0.90 if delta > 0 else 0.0,
            }
        }
    coverage = {
        "known_rescue_queries_with_usable_evidence": 41,
        "dense_retention_queries_with_usable_evidence": 41,
    }
    return metrics, comparisons, coverage


def test_frozen_gate_boundaries_allow_only_narrow_feature_path() -> None:
    metrics, comparisons, coverage = _passing_narrow_gate_inputs()
    gate = _development_gate(metrics, comparisons, coverage)
    assert gate["candidate_source_gate_passed"] is False
    assert (
        gate["candidate_source_checks"][
            "sae_source_oracle_above_random_95th_percentile"
        ]
        is False
    )
    assert (
        gate["candidate_source_checks"][
            "sae_source_oracle_tail_probability_at_most_alpha"
        ]
        is False
    )
    assert gate["feature_grounding_gate_passed"] is True
    assert gate["narrow_feature_only_gate_passed"] is True
    assert (
        gate["decision"]
        == "narrow_go_feature_evidence_only_no_candidate_specificity_claim"
    )

    below_margin_metrics = deepcopy(metrics)
    below_margin_metrics[SAE_SHUFFLED_ARM][
        "known_rescues_recovered"
    ] = 29
    below_margin = deepcopy(comparisons)
    key = f"{SAE_ALIGNED_ARM}_minus_{SAE_SHUFFLED_ARM}"
    below_margin[key]["net_utility"]["delta_net_utility"] = 4
    shuffled_activation = (
        f"{SAE_SHUFFLED_ARM}_minus_{SAE_ACTIVATION_ARM}"
    )
    below_margin[shuffled_activation]["net_utility"]["delta_net_utility"] = 1
    assert (
        _development_gate(
            below_margin_metrics, below_margin, coverage
        )["decision"]
        == "no_go_stop_structural_rescue_on_scar"
    )

    below_bootstrap = deepcopy(comparisons)
    below_bootstrap[key]["net_utility"]["bootstrap_positive_fraction"] = 0.8999
    assert (
        _development_gate(metrics, below_bootstrap, coverage)["decision"]
        == "no_go_stop_structural_rescue_on_scar"
    )

    low_coverage = {**coverage, "known_rescue_queries_with_usable_evidence": 40}
    assert (
        _development_gate(metrics, comparisons, low_coverage)["decision"]
        == "no_go_stop_structural_rescue_on_scar"
    )

    excessive_incremental_loss = deepcopy(metrics)
    excessive_incremental_loss[SAE_STRUCTURE_ARM]["losses_vs_dense_top10"] = 2
    excessive_incremental_loss[SAE_STRUCTURE_ARM][
        "known_rescues_recovered"
    ] = 26
    assert (
        _development_gate(
            excessive_incremental_loss, comparisons, coverage
        )["decision"]
        == "no_go_stop_structural_rescue_on_scar"
    )
