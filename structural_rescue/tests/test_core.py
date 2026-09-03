from __future__ import annotations

import json

import numpy as np
import pytest

from structural_rescue.core import (
    DEFAULT_PROTOCOL,
    _topk,
    canonical_json_sha256,
    make_representation,
    sha256_file,
    shared_feature_rows,
    stable_union,
    system_payload,
    verifier_batch_plan,
)
from structural_rescue.validate_protocol import validate


def test_stable_union_deduplicates_in_source_order() -> None:
    assert stable_union([2, 1, 3], [3, 4], [1, 5]) == [2, 1, 3, 4, 5]


def test_topk_ties_break_by_candidate_index() -> None:
    scores = np.asarray([0.1, 0.9, 0.9, 0.2])
    assert _topk(scores, 3) == [1, 2, 3]


def test_verifier_batch_plan_freezes_aliases_and_candidate_context() -> None:
    prepared = [{"query_id": "q1", "superpool": ["b:2", "b:1", "b:2"]}]
    plan = verifier_batch_plan(
        prepared,
        {"selection": "outcome_stratified_exploratory_screen", "query_ids": ["q1"]},
    )
    batch = plan["queries"][0]["batches"][0]
    assert sorted(batch["candidate_ids"]) == ["b:1", "b:2"]
    aliases = {
        f"C{index:03d}": candidate
        for index, candidate in enumerate(batch["candidate_ids"], start=1)
    }
    assert batch["alias_map_sha256"] == canonical_json_sha256(aliases)


def test_namespaced_shared_features_are_ordered_and_bounded() -> None:
    values = np.zeros((3, 9216), dtype=np.float32)
    values[0, [4, 7]] = [2.0, 1.0]
    values[1, [4, 7]] = [1.0, 3.0]
    values[2, [4, 7]] = [3.0, 2.0]
    representation = make_representation(values)
    evidence = shared_feature_rows(
        representation, 0, 1, namespace="cslg", limit=1
    )
    assert len(evidence) == 1
    assert evidence[0]["feature_key"].startswith("cslg:")
    assert 0.0 <= evidence[0]["query_activation_percentile"] <= 1.0
    assert 0.0 <= evidence[0]["candidate_activation_percentile"] <= 1.0


def test_system_payload_excludes_answer_bearing_fields() -> None:
    row = {
        "id": 1,
        "system_a": "source",
        "system_b": "target",
        "system_a_background": "source mechanism",
        "system_b_background": "target mechanism",
        "system_a_domain": "one",
        "system_b_domain": "two",
        "mappings": [["x", "y"]],
        "Explanation": ["answer"],
    }
    payload = system_payload("a:1", {1: row})
    assert set(payload) == {"system_id", "name", "background"}
    serialized = json.dumps(payload)
    assert "target mechanism" not in serialized
    assert "answer" not in serialized


def test_protocol_is_development_only_and_matches_code() -> None:
    summary = validate()
    assert summary["status"] == "exploratory_development_only"
    assert summary["latent_choice_test_prompts_allowed"] is False


def test_protocol_rejects_latent_choice_reuse(tmp_path) -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    protocol["scope"]["latent_choice_test_prompts_may_be_used"] = True
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Latent Choice"):
        validate(path)


def test_committed_prepare_report_reproduces_frozen_candidate_counts() -> None:
    report_path = DEFAULT_PROTOCOL.with_name("prepare_report.json")
    report = json.loads(report_path.read_text())
    assert report["protocol_sha256"] == sha256_file(DEFAULT_PROTOCOL)
    assert report["status"] == "prepared_no_llm_scoring"
    assert report["verifier_screen"]["known_sae_rescue_queries"] == 54
    assert report["verifier_screen"]["dense_retention_control_queries"] == 54
    assert (
        report["verifier_screen"][
            "confirmatory_or_population_recall_claim_allowed"
        ]
        is False
    )
    assert report["verifier_batch_plan"] == {
        "batch_size": 64,
        "query_count": 108,
        "batch_count": 108,
    }
    assert report["preflight"] == {
        "queries": 566,
        "dense_top10_hits": 146,
        "dense_top30_hits": 262,
        "sae_union_hits": 200,
        "known_sae_rescues": 54,
        "known_sae_rescues_beyond_dense30": 19,
        "random_union_hits": 193,
        "sae_union_size": {"min": 11, "median": 19.0, "mean": 19.10600706713781, "max": 29},
        "random_union_size": {"min": 13, "median": 22.0, "mean": 22.09540636042403, "max": 30},
    }
    serialized = json.dumps(report).lower()
    assert "system_a_background" not in serialized
    assert "system_b_background" not in serialized


def test_committed_live_smoke_report_is_non_evidentiary_and_source_free() -> None:
    report = json.loads(DEFAULT_PROTOCOL.with_name("smoke_report.json").read_text())
    assert report["status"] == "live_smoke_passed_non_evidentiary"
    assert report["protocol_sha256"] == sha256_file(DEFAULT_PROTOCOL)
    assert report["scope"]["queries"] == 2
    assert report["scope"]["population_estimate_allowed"] is False
    assert report["pipeline"]["paired_verifier_predictions"] == "166/166"
    serialized = json.dumps(report).lower()
    for forbidden in ("system_a_background", "system_b_background", "openai_api_key"):
        assert forbidden not in serialized
