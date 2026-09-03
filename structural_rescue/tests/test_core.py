from __future__ import annotations

import json

import numpy as np
import pytest

from structural_rescue.core import (
    ARM_NAMES,
    DEFAULT_PROTOCOL,
    RANDOM_SEED_PAIRS,
    RANDOM_SEED_PAIRS_SHA256,
    RANDOM_SEED_PAIR_COUNT,
    VERIFIED_RANDOM_PAIR_INDICES,
    _feature_catalog_and_pair_evidence,
    _query_rows,
    _topk,
    canonical_json_sha256,
    feature_description_shuffle_map,
    largest_batch_preflight,
    make_representation,
    pad_pool_to_size,
    sha256_file,
    shared_feature_rows,
    stable_union,
    system_payload,
    verifier_batch_plan,
)
from structural_rescue.validate_protocol import validate


def test_stable_union_deduplicates_in_source_order() -> None:
    assert stable_union([2, 1, 3], [3, 4], [1, 5]) == [2, 1, 3, 4, 5]


def test_frozen_arm_contract_has_equal_sized_verifier_controls() -> None:
    assert ARM_NAMES == (
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


def test_hash_derived_random_seed_pairs_are_frozen_and_unique() -> None:
    assert len(RANDOM_SEED_PAIRS) == RANDOM_SEED_PAIR_COUNT == 64
    assert VERIFIED_RANDOM_PAIR_INDICES == (0, 1, 2)
    assert RANDOM_SEED_PAIRS[:3] == (
        (2026090201, 2026090202),
        (2065533481, 2401522923),
        (4014805828, 2701039346),
    )
    assert len({seed for pair in RANDOM_SEED_PAIRS for seed in pair}) == 128
    assert (
        RANDOM_SEED_PAIRS_SHA256
        == "5c53305128f85ec37ca5efc299395200569295a67ee754bde0714067cdf53d72"
    )


def test_padding_preserves_source_and_appends_next_unused_dense_candidates() -> None:
    source = [9, 2, 7]
    dense = [2, 4, 7, 1, 8, 3]
    assert pad_pool_to_size(source, dense, size=6) == [9, 2, 7, 4, 1, 8]


def test_padding_rejects_invalid_or_insufficient_sources() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        pad_pool_to_size([1, 1], [1, 2, 3], size=3)
    with pytest.raises(ValueError, match="exceeds"):
        pad_pool_to_size([1, 2, 3], [4], size=2)
    with pytest.raises(ValueError, match="unique candidates"):
        pad_pool_to_size([1], [1, 2], size=3)


def test_query_rows_preserve_sources_and_emit_exact_padded_arm_pools() -> None:
    rows = [
        {
            "id": index,
            "system_a": f"a{index}",
            "system_b": f"b{index}",
            "system_a_background": f"a background {index}",
            "system_b_background": f"b background {index}",
        }
        for index in range(32)
    ]

    def scores(order: list[int]) -> np.ndarray:
        values = np.empty(32, dtype=float)
        values[order] = np.arange(32, 0, -1, dtype=float)
        return values[None, :]

    dense_order = list(range(32))
    cslg_order = [*range(9), 10, 9, *range(11, 32)]
    astroph_order = [*range(9), 11, 9, 10, *range(12, 32)]
    matrices = {
        name: {
            direction: scores(order)
            for direction in ("a_to_b", "b_to_a")
        }
        for name, order in (
            ("dense", dense_order),
            ("cslg", cslg_order),
            ("astroph", astroph_order),
        )
    }
    random_sources = [
        {
            direction: [list(range(12))]
            for direction in ("a_to_b", "b_to_a")
        }
        for _ in range(64)
    ]
    complementarity = [
        {
            "query_id": f"{direction}:0",
            "dense_top10": True,
            "sae_rescue": False,
        }
        for direction in ("a_to_b", "b_to_a")
    ]
    prepared, _ = _query_rows(
        rows,
        np.asarray([0]),
        matrices,
        random_sources,
        complementarity,
    )
    for query in prepared:
        assert list(query["pools"]) == list(ARM_NAMES)
        assert len(query["source_pools"]["sae_union"]) == 12
        assert len(query["source_pools"]["random_unions"]) == 64
        for arm in ARM_NAMES[1:]:
            assert len(query["pools"][arm]) == len(set(query["pools"][arm])) == 30
        sae_source = query["source_pools"]["sae_union"]
        assert query["pools"]["sae_union_padded30_structure"][:12] == sae_source


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


def test_largest_batch_preflight_is_deterministic_and_qrels_free() -> None:
    prepared = [
        {"query_id": "q-small", "superpool": ["c1"]},
        {"query_id": "q-large-b", "superpool": ["c1", "c2", "c3"]},
        {"query_id": "q-large-a", "superpool": ["c3", "c2", "c1"]},
    ]
    first = largest_batch_preflight(prepared, query_count=2)
    second = largest_batch_preflight(list(reversed(prepared)), query_count=2)
    assert first == second
    assert set(first["query_ids"]) == {"q-large-a", "q-large-b"}
    assert first["superpool_sizes"] == [3, 3]
    assert first["qrels_used"] is False
    assert first["population_estimate_allowed"] is False
    screen_ids = {"q-small", "q-large-a"}
    capacity = largest_batch_preflight(
        [row for row in prepared if row["query_id"] in screen_ids],
        query_count=1,
    )
    assert capacity["query_ids"] == ["q-large-a"]


def test_feature_description_shuffle_is_binned_deterministic_derangement() -> None:
    catalog = [
        {
            "feature_key": f"{representation}:{feature_id}",
            "representation": representation,
            "feature_id": feature_id,
            "corpus_active_count": feature_id + 1,
        }
        for representation in ("astroph", "cslg")
        for feature_id in range(128)
    ]
    first = feature_description_shuffle_map(catalog)
    second = feature_description_shuffle_map(list(reversed(catalog)))
    assert first == second
    assert first["mapping_count"] == 256
    assert first["bin_feature_counts"] == {
        "astroph": [16] * 8,
        "cslg": [16] * 8,
    }
    mappings = first["mappings"]
    assert all(
        row["source_feature_key"] != row["donor_feature_key"] for row in mappings
    )
    assert {row["source_feature_key"] for row in mappings} == {
        row["donor_feature_key"] for row in mappings
    }
    by_key = {row["feature_key"]: row for row in catalog}
    assert all(
        row["representation"]
        == by_key[row["donor_feature_key"]]["representation"]
        for row in mappings
    )


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


def test_feature_selection_uses_source_but_evidence_covers_padded_pool() -> None:
    rows = [
        {"id": 0, "system_a": "a0", "system_b": "b0"},
        {"id": 1, "system_a": "a1", "system_b": "b1"},
    ]
    values = np.zeros((4, 9216), dtype=np.float32)
    values[0, [11, 22]] = 1.0
    values[2, 11] = 1.0
    values[3, 22] = 1.0
    representation = make_representation(values)
    prepared = [
        {
            "query_id": "a_to_b:0",
            "direction": "a_to_b",
            "query_system_id": "a:0",
            "source_pools": {"sae_union": ["b:0"]},
            "pools": {
                "sae_union_padded30_structure": ["b:0", "b:1"],
            },
        }
    ]
    catalog, evidence = _feature_catalog_and_pair_evidence(
        prepared, rows, {"cslg": representation}
    )
    assert [row["feature_id"] for row in catalog] == [11]
    assert [row["candidate_id"] for row in evidence] == ["b:0", "b:1"]
    assert [row["feature_id"] for row in evidence[0]["shared_features"]] == [11]
    assert evidence[1]["shared_features"] == []


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
        "batch_count": 132,
    }
    preflight = report["preflight"]
    assert preflight["queries"] == 566
    assert preflight["dense_top10_hits"] == 146
    assert preflight["dense_top30_hits"] == 262
    assert preflight["sae_union_hits"] == 200
    assert preflight["sae_union_padded30_hits"] == 258
    assert preflight["known_sae_rescues"] == 54
    assert preflight["known_sae_rescues_beyond_dense30"] == 19
    assert preflight["verified_random_source_oracle_hits"] == [193, 191, 192]
    assert preflight["verified_random_padded30_hits"] == [251, 247, 248]
    assert preflight["all_verifier_candidate_pools_exactly_30"] is True
    random_oracle = preflight["random_source_oracle_hit_distribution"]
    assert random_oracle["draws"] == 64
    assert random_oracle["values"][0] == 193
    assert random_oracle["mean"] == 195.53125
    assert random_oracle["q95_higher"] == 204
    assert random_oracle["draws_at_least_sae"] == 18
    assert random_oracle["plus_one_tail_probability"] == pytest.approx(
        19 / 65
    )
    assert report["random_source_oracle"] == {
        "seed_namespace": "structural-rescue-random-source-oracle-v1",
        "seed_pair_count": 64,
        "seed_pairs_sha256": RANDOM_SEED_PAIRS_SHA256,
        "verified_pair_indices": [0, 1, 2],
    }
    assert report["feature_description_shuffle_map"]["mapping_count"] == 256
    assert report["capacity_smoke_selection"]["qrels_used"] is False
    assert report["capacity_smoke_selection"]["superpool_sizes"] == [85]
    assert "feature_description_shuffle_map" in report["artifacts"]
    assert "capacity_smoke_selection" in report["artifacts"]
    serialized = json.dumps(report).lower()
    assert "system_a_background" not in serialized
    assert "system_b_background" not in serialized


def test_committed_revision4_live_smoke_is_historical_non_evidentiary_and_source_free() -> None:
    report = json.loads(DEFAULT_PROTOCOL.with_name("smoke_report.json").read_text())
    assert report["status"] == "live_smoke_passed_non_evidentiary"
    assert (
        report["protocol_sha256"]
        == "e7401d11823d76c846a58c9f88192268682b0f6bdc447b743d89fe3e1389181d"
    )
    assert report["protocol_sha256"] != sha256_file(DEFAULT_PROTOCOL)
    assert report["scope"]["queries"] == 2
    assert report["scope"]["population_estimate_allowed"] is False
    assert report["pipeline"]["paired_verifier_predictions"] == "166/166"
    serialized = json.dumps(report).lower()
    for forbidden in ("system_a_background", "system_b_background", "openai_api_key"):
        assert forbidden not in serialized
