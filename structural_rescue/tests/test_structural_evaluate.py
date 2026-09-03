from __future__ import annotations

import pytest

from structural_rescue.evaluate import evaluate_rows
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
                "sae_union_structure": ["c1", "gold"],
                "random_union_structure": ["c1", "c2"],
                "sae_union_feature_grounded": ["c1", "gold"],
            },
        },
        {
            "query_id": "q2",
            "dense_scores": {"gold2": 0.9, "d2": 0.2},
            "pools": {
                "dense_ranking": ["gold2", "d2"],
                "dense30_structure": ["gold2", "d2"],
                "sae_union_structure": ["gold2", "d2"],
                "random_union_structure": ["gold2", "d2"],
                "sae_union_feature_grounded": ["gold2", "d2"],
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
        feature_candidates = {
            "q1": ("c1", "gold"),
            "q2": ("gold2", "d2"),
        }[query_id]
        for candidate in feature_candidates:
            predictions.append(
                _prediction(query_id, candidate, "feature_grounded", scores[candidate])
            )

    report, rows = evaluate_rows(candidates, qrels, predictions)
    assert report["status"] == "dry_run_non_evidentiary"
    assert report["metrics"]["sae_union_structure"]["known_rescues_recovered"] == 1
    assert report["metrics"]["sae_union_structure"]["dense_hits_retained"] == 1
    assert len(rows) == 2

    predictions[0]["score"] = 99
    with pytest.raises(ValueError, match="differs"):
        evaluate_rows(candidates, qrels, predictions)

    predictions[0]["score"] = 0
    predictions[0]["model"] = MODEL
    with pytest.raises(ValueError, match="one explicit model"):
        evaluate_rows(candidates, qrels, predictions)
