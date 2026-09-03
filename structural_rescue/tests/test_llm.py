from __future__ import annotations

import json

import pytest

from structural_rescue.llm import (
    feature_description_payload,
    mechanism_payload,
    pair_payload_hash,
    verdict_score,
    verifier_payload,
)
from structural_rescue.run import (
    _pair_feature_context,
    _validate_fixed_batch,
    validate_verdict_batch,
)


GRAPH = {
    "system_id": "a:1",
    "summary": "A regulated flow",
    "entities_and_roles": [{"entity": "controller", "role": "regulates"}],
    "causal_relations": [
        {"source": "controller", "relation": "limits", "target": "flow"}
    ],
    "dynamics": ["negative feedback"],
    "constraints": ["finite capacity"],
    "boundary_conditions": ["closed system"],
    "model": "ignored provenance",
}


def test_mechanism_payload_rejects_gold_or_pair_fields() -> None:
    with pytest.raises(ValueError, match="exactly"):
        mechanism_payload(
            [
                {
                    "system_id": "a:1",
                    "name": "system",
                    "background": "description",
                    "gold": "answer",
                }
            ]
        )


def test_feature_description_payload_has_no_pair_metadata() -> None:
    features = [
        {
            "feature_key": "cslg:4",
            "top_examples": [{"system_id": "a:1"}],
            "pair_id": 99,
            "aggregate_pair_selection_score": 2.0,
        }
    ]
    examples = {
        "a:1": {
            "system_id": "a:1",
            "name": "system",
            "background": "mechanism",
        }
    }
    payload = feature_description_payload(features, examples)
    serialized = json.dumps(payload)
    assert "pair_id" not in serialized
    assert "aggregate_pair_selection_score" not in serialized
    assert "gold" not in serialized
    assert payload["features"][0]["feature_key"] == "R1:F0004"
    assert "cslg" not in serialized


def test_structure_payload_is_blind_and_pair_hash_is_arm_invariant() -> None:
    payload, aliases = verifier_payload(GRAPH, [("b:2", GRAPH)])
    serialized = json.dumps(payload)
    assert aliases == {"C001": "b:2"}
    for forbidden in ("retrieval", "rank", "gold", "rescue", "b:2", "a:1"):
        assert forbidden not in serialized
    assert pair_payload_hash(GRAPH, GRAPH) == pair_payload_hash(
        {**GRAPH, "retrieval_rank": 1}, {**GRAPH, "gold": True}
    )


def test_feature_grounding_uses_opaque_alias_and_explicit_description() -> None:
    evidence = {
        "b:2": [
            {
                "feature_key": "astroph:12",
                "query_activation_percentile": 0.9,
                "candidate_activation_percentile": 0.8,
            }
        ]
    }
    payload, _ = verifier_payload(
        GRAPH,
        [("b:2", GRAPH)],
        feature_evidence=evidence,
        feature_descriptions={"astroph:12": "oscillatory feedback"},
    )
    feature = payload["pairs"][0]["shared_feature_evidence"][0]
    assert feature["feature_key"] == "R2:F0012"
    assert feature["description"] == "oscillatory feedback"
    assert "astroph" not in json.dumps(payload)


def test_pair_feature_context_excludes_duplicate_content_and_incoherent() -> None:
    evidence = [{"feature_key": "cslg:4"}, {"feature_key": "astroph:12"}]
    descriptions = {
        "cslg:4": {
            "coherent": True,
            "request_example_content_sha256": ["duplicate-content"],
        },
        "astroph:12": {
            "coherent": False,
            "request_example_content_sha256": ["other-content"],
        },
    }
    usable, counts = _pair_feature_context(
        evidence,
        descriptions,
        query_content_sha256="duplicate-content",
        candidate_content_sha256="candidate-content",
    )
    assert usable == []
    assert counts == {"raw": 2, "incoherent": 1, "direct_example": 1, "usable": 0}


def test_empty_evidence_verdict_fields_are_enforced() -> None:
    output = {
        "candidates": [
            {
                "candidate_alias": "C001",
                "feature_support": 1,
                "accidental_feature_overlap": False,
            }
        ]
    }
    with pytest.raises(ValueError, match="without evidence"):
        validate_verdict_batch(
            output,
            aliases={"C001": "b:2"},
            empty_evidence_aliases={"C001"},
        )


def test_resume_rejects_partial_or_stale_fixed_batches() -> None:
    with pytest.raises(ValueError, match="Partial"):
        _validate_fixed_batch(
            [{"request_sha256": "right", "model": "m"}],
            expected_count=2,
            request_sha256="right",
            backend_model="m",
            label="test",
        )
    with pytest.raises(ValueError, match="Stale"):
        _validate_fixed_batch(
            [{"request_sha256": "wrong", "model": "m"}],
            expected_count=1,
            request_sha256="right",
            backend_model="m",
            label="test",
        )


def test_verdict_score_is_fixed_and_penalizes_surface_matches() -> None:
    base = {
        "role_alignment": 3,
        "causal_alignment": 3,
        "dynamics_alignment": 3,
        "constraint_alignment": 2,
        "feature_support": 1,
        "lexical_only": False,
        "same_domain_only": False,
        "accidental_feature_overlap": False,
        "break_severity": 1,
    }
    assert verdict_score({**base, "lexical_only": True}) == verdict_score(base) - 4
