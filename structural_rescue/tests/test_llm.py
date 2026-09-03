from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from structural_rescue.core import DEFAULT_PROTOCOL, sha256_file
from structural_rescue.llm import (
    MODEL,
    PROMPT_VERSION,
    VERIFIER_EVIDENCE_MODES,
    feature_description_payload,
    mechanism_payload,
    pair_payload_hash,
    response_usage_payload,
    strip_verifier_feature_descriptions,
    strip_verifier_feature_evidence,
    verdict_score,
    verifier_payload,
)
from structural_rescue.run import (
    _pair_feature_context,
    _validate_fixed_batch,
    capacity_smoke_report,
    evaluate_command,
    FixtureBackend,
    frozen_real_selection_kind,
    PREPARED_FILENAMES,
    extract_mechanisms,
    matched_feature_contexts,
    normalize_empty_evidence_verdicts,
    validate_capacity_smoke_report,
    validate_coverage_report,
    validate_feature_description_batch,
    validate_verdict_batch,
    validate_prepared_bundle,
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
        evidence_mode="aligned_description",
        feature_evidence=evidence,
        feature_descriptions={"astroph:12": "oscillatory feedback"},
    )
    feature = payload["pairs"][0]["shared_feature_evidence"][0]
    assert feature["feature_key"] == "R2:F0012"
    assert feature["description"] == "oscillatory feedback"
    assert "astroph" not in json.dumps(payload)


def test_incoherent_description_is_preserved_then_normalized() -> None:
    output = {
        "features": [
            {
                "feature_key": "R1:F0001",
                "description": "The examples do not support one clear pattern.",
                "coherent": False,
            },
            {
                "feature_key": "R1:F0002",
                "description": "negative feedback",
                "coherent": True,
            },
        ]
    }
    assert (
        validate_feature_description_batch(
            output, expected_aliases=["R1:F0001", "R1:F0002"]
        )
        == 1
    )
    incoherent = output["features"][0]
    assert (
        incoherent["raw_description"]
        == "The examples do not support one clear pattern."
    )
    assert incoherent["incoherent_description_normalized"] is True
    assert incoherent["description"] == "no coherent mechanistic interpretation"
    assert output["features"][1]["description"] == "negative feedback"


def test_verifier_evidence_modes_preserve_pairs_and_isolate_description_text() -> None:
    candidates = [("b:2", GRAPH), ("b:3", {**GRAPH, "summary": "Another flow"})]
    evidence = {
        "b:2": [
            {
                "feature_key": "astroph:12",
                "query_activation_percentile": 0.9,
                "candidate_activation_percentile": 0.8,
            }
        ],
        "b:3": [],
    }
    structure, structure_aliases = verifier_payload(GRAPH, candidates)
    activation, activation_aliases = verifier_payload(
        GRAPH,
        candidates,
        evidence_mode="activation_only",
        feature_evidence=evidence,
    )
    aligned, aligned_aliases = verifier_payload(
        GRAPH,
        candidates,
        evidence_mode="aligned_description",
        feature_evidence=evidence,
        feature_descriptions={"astroph:12": "oscillatory feedback"},
    )
    shuffled, shuffled_aliases = verifier_payload(
        GRAPH,
        candidates,
        evidence_mode="shuffled_description",
        feature_evidence=evidence,
        feature_descriptions={"astroph:12": "material phase transition"},
    )

    assert tuple(VERIFIER_EVIDENCE_MODES) == (
        "structure",
        "activation_only",
        "aligned_description",
        "shuffled_description",
    )
    assert structure_aliases == activation_aliases == aligned_aliases == shuffled_aliases
    assert [row["candidate_alias"] for row in structure["pairs"]] == ["C001", "C002"]
    assert strip_verifier_feature_evidence(activation) == structure
    assert strip_verifier_feature_evidence(aligned) == structure
    assert strip_verifier_feature_evidence(shuffled) == structure
    assert strip_verifier_feature_descriptions(aligned) == activation
    assert strip_verifier_feature_descriptions(shuffled) == activation

    activation_feature = activation["pairs"][0]["shared_feature_evidence"][0]
    aligned_feature = aligned["pairs"][0]["shared_feature_evidence"][0]
    shuffled_feature = shuffled["pairs"][0]["shared_feature_evidence"][0]
    assert set(activation_feature) == {
        "feature_key",
        "query_activation_percentile",
        "candidate_activation_percentile",
    }
    assert set(aligned_feature) == set(shuffled_feature) == {
        *activation_feature,
        "description",
    }
    assert aligned_feature["description"] == "oscillatory feedback"
    assert shuffled_feature["description"] == "material phase transition"


def test_verifier_evidence_modes_reject_incompatible_inputs() -> None:
    evidence = {"b:2": []}
    with pytest.raises(ValueError, match="Unknown"):
        verifier_payload(GRAPH, [("b:2", GRAPH)], evidence_mode="unknown")
    with pytest.raises(ValueError, match="cannot receive"):
        verifier_payload(
            GRAPH,
            [("b:2", GRAPH)],
            evidence_mode="structure",
            feature_evidence=evidence,
        )
    with pytest.raises(ValueError, match="must omit"):
        verifier_payload(
            GRAPH,
            [("b:2", GRAPH)],
            evidence_mode="activation_only",
            feature_evidence=evidence,
            feature_descriptions={},
        )


def test_response_usage_payload_is_optional_and_json_safe() -> None:
    class Usage:
        def model_dump(self, *, mode: str):
            assert mode == "json"
            return {
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            }

    assert response_usage_payload(SimpleNamespace(usage=None)) is None
    assert response_usage_payload(SimpleNamespace(usage=Usage())) == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
    }
    assert response_usage_payload(SimpleNamespace(usage={"bad": object()})) is None


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


def test_matched_feature_contexts_hold_rows_fixed_across_evidence_controls() -> None:
    evidence = [
        {
            "feature_key": "cslg:4",
            "query_activation_percentile": 0.9,
            "candidate_activation_percentile": 0.8,
        },
        {
            "feature_key": "cslg:7",
            "query_activation_percentile": 0.7,
            "candidate_activation_percentile": 0.6,
        },
    ]
    descriptions = {
        "cslg:4": {
            "description": "negative feedback",
            "coherent": True,
            "request_example_content_sha256": ["example-four"],
        },
        "cslg:7": {
            "description": "resource bottleneck",
            "coherent": True,
            "request_example_content_sha256": ["example-seven"],
        },
    }
    usable, aligned, shuffled, counts = matched_feature_contexts(
        evidence,
        descriptions,
        {"cslg:4": "cslg:7", "cslg:7": "cslg:4"},
        query_content_sha256="query",
        candidate_content_sha256="candidate",
    )
    assert usable == evidence
    assert aligned == {
        "cslg:4": "negative feedback",
        "cslg:7": "resource bottleneck",
    }
    assert shuffled == {
        "cslg:4": "resource bottleneck",
        "cslg:7": "negative feedback",
    }
    assert counts["usable"] == 2

    descriptions["cslg:7"]["request_example_content_sha256"] = ["query"]
    usable, _, _, counts = matched_feature_contexts(
        evidence,
        descriptions,
        {"cslg:4": "cslg:7", "cslg:7": "cslg:4"},
        query_content_sha256="query",
        candidate_content_sha256="candidate",
    )
    assert usable == []
    assert counts["source_direct_example"] == 1
    assert counts["donor_direct_example"] == 1


def test_empty_evidence_verdict_fields_are_normalized_and_preserved() -> None:
    output = {
        "candidates": [
            {
                "candidate_alias": "C001",
                "feature_support": 1,
                "accidental_feature_overlap": False,
            }
        ]
    }
    validate_verdict_batch(
        output,
        aliases={"C001": "b:2"},
        empty_evidence_aliases={"C001"},
    )
    assert normalize_empty_evidence_verdicts(
        output, empty_evidence_aliases={"C001"}
    ) == 1
    verdict = output["candidates"][0]
    assert verdict["raw_feature_support"] == 1
    assert verdict["feature_support"] == 0
    assert verdict["empty_evidence_normalized"] is True


def test_resume_rejects_partial_or_stale_fixed_batches() -> None:
    with pytest.raises(ValueError, match="Partial"):
        _validate_fixed_batch(
            [{"request_sha256": "right", "model": "m"}],
            expected_count=2,
            request_sha256="right",
            backend_model="m",
            current_git_commit="commit",
            label="test",
        )
    with pytest.raises(ValueError, match="Stale code provenance"):
        _validate_fixed_batch(
            [
                {
                    "request_sha256": "right",
                    "model": MODEL,
                    "generation_git_commit": "old",
                    "generation_git_worktree_dirty": False,
                }
            ],
            expected_count=1,
            request_sha256="right",
            backend_model=MODEL,
            current_git_commit="new",
            label="test",
        )
    with pytest.raises(ValueError, match="Stale"):
        _validate_fixed_batch(
            [{"request_sha256": "wrong", "model": "m"}],
            expected_count=1,
            request_sha256="right",
            backend_model="m",
            current_git_commit="commit",
            label="test",
        )


def test_subset_mechanism_smoke_resumes_into_full_frozen_batches(tmp_path) -> None:
    data_path = tmp_path / "scar.jsonl"
    data_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": index,
                    "system_a": f"a{index}",
                    "system_b": f"b{index}",
                    "system_a_background": f"a background {index}",
                    "system_b_background": f"b background {index}",
                }
            )
            + "\n"
            for index in range(400)
        ),
        encoding="utf-8",
    )
    candidate_path = tmp_path / "candidate_manifest.jsonl"
    candidate_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "query_id": "q1",
                    "query_system_id": "a:0",
                    "superpool": [f"b:{index}" for index in range(6)],
                },
                {
                    "query_id": "q2",
                    "query_system_id": "a:1",
                    "superpool": [f"b:{index}" for index in range(6, 12)],
                },
            )
        ),
        encoding="utf-8",
    )
    screen_path = tmp_path / "screen_selection.json"
    screen_path.write_text(json.dumps({"query_ids": ["q1", "q2"]}), encoding="utf-8")
    capacity_path = tmp_path / "capacity.json"
    capacity_path.write_text(json.dumps({"query_ids": ["q1"]}), encoding="utf-8")
    output_path = tmp_path / "mechanisms.jsonl"

    smoke = extract_mechanisms(
        data_path=data_path,
        candidate_path=candidate_path,
        selection_path=capacity_path,
        output_path=output_path,
        backend=FixtureBackend(),
        limit_queries=None,
        overwrite=False,
    )
    assert smoke["required_systems"] == 7
    assert smoke["completed_systems"] == 7
    assert smoke["stable_batch_universe_systems"] == 14

    full = extract_mechanisms(
        data_path=data_path,
        candidate_path=candidate_path,
        selection_path=screen_path,
        output_path=output_path,
        backend=FixtureBackend(),
        limit_queries=None,
        overwrite=False,
    )
    assert full["required_systems"] == full["completed_systems"] == 14


def test_coverage_report_is_bound_to_current_inputs(tmp_path) -> None:
    selection_path = tmp_path / "screen_selection.json"
    descriptions_path = tmp_path / "feature_descriptions.jsonl"
    shuffle_path = tmp_path / "feature_description_shuffle_map.json"
    coverage_path = tmp_path / "coverage_report.json"
    selection_path.write_text('{"query_ids":["q1"]}\n', encoding="utf-8")
    descriptions_path.write_text('{"feature_key":"f1"}\n', encoding="utf-8")
    shuffle_path.write_text('{"mappings":[]}\n', encoding="utf-8")
    coverage_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "qrels_read": False,
                "selection_sha256": sha256_file(selection_path),
                "feature_descriptions_sha256": sha256_file(descriptions_path),
                "feature_description_shuffle_map_sha256": sha256_file(shuffle_path),
            }
        ),
        encoding="utf-8",
    )
    validate_coverage_report(
        coverage_path,
        frozen_selection_path=selection_path,
        feature_descriptions_path=descriptions_path,
        feature_shuffle_path=shuffle_path,
    )
    descriptions_path.write_text('{"feature_key":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="stale.*feature_descriptions_sha256"):
        validate_coverage_report(
            coverage_path,
            frozen_selection_path=selection_path,
            feature_descriptions_path=descriptions_path,
            feature_shuffle_path=shuffle_path,
        )


def test_capacity_smoke_report_is_frozen_and_required(tmp_path) -> None:
    selection_path = tmp_path / "capacity_smoke_selection.json"
    batch_plan_path = tmp_path / "verifier_batch_plan.json"
    descriptions_path = tmp_path / "feature_descriptions.jsonl"
    shuffle_path = tmp_path / "feature_description_shuffle_map.json"
    for path, content in (
        (selection_path, '{"query_ids":["q1"]}\n'),
        (batch_plan_path, '{"queries":[]}\n'),
        (descriptions_path, '{"feature_key":"f1"}\n'),
        (shuffle_path, '{"mappings":[]}\n'),
    ):
        path.write_text(content, encoding="utf-8")
    summary = capacity_smoke_report(
        {
            "required_predictions": 340,
            "completed_predictions": 340,
            "maximum_batch_size_exercised": 64,
            "mode_request_workers": 4,
            "backend": MODEL,
            "prompt_version": PROMPT_VERSION,
            "generation_git_commit": "frozen-commit",
            "generation_git_worktree_dirty": False,
            "mechanisms_sha256": "a" * 64,
        },
        selection_path=selection_path,
        batch_plan_path=batch_plan_path,
        feature_descriptions_path=descriptions_path,
        feature_shuffle_path=shuffle_path,
    )
    assert summary["status"] == "passed"
    report_path = tmp_path / "capacity_smoke_report.json"
    report_path.write_text(json.dumps(summary), encoding="utf-8")
    validate_capacity_smoke_report(
        report_path,
        capacity_selection_path=selection_path,
        batch_plan_path=batch_plan_path,
        feature_descriptions_path=descriptions_path,
        feature_shuffle_path=shuffle_path,
        current_git_commit="frozen-commit",
    )
    shuffle_path.write_text('{"mappings":["changed"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="stale.*shuffle"):
        validate_capacity_smoke_report(
            report_path,
            capacity_selection_path=selection_path,
            batch_plan_path=batch_plan_path,
            feature_descriptions_path=descriptions_path,
            feature_shuffle_path=shuffle_path,
            current_git_commit="frozen-commit",
        )


def test_real_partial_evaluation_is_rejected_before_qrels_are_needed(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"query_id": "q1", "model": MODEL}) + "\n",
        encoding="utf-8",
    )
    selection_path = tmp_path / "screen_selection.json"
    selection_path.write_text(
        json.dumps({"query_ids": ["q1", "q2"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Refusing to open qrels"):
        evaluate_command(
            candidate_path=tmp_path / "missing-candidates.jsonl",
            qrels_path=tmp_path / "missing-qrels.jsonl",
            selection_path=selection_path,
            predictions_path=predictions_path,
            output_dir=tmp_path,
            overwrite=False,
        )


def test_real_selection_is_classified_by_frozen_hash_not_path(tmp_path) -> None:
    screen = tmp_path / "screen_selection.json"
    capacity = tmp_path / "capacity_smoke_selection.json"
    copied_screen = tmp_path / "copied-screen.json"
    unknown = tmp_path / "unknown.json"
    screen.write_text('{"query_ids":["q1","q2"]}\n', encoding="utf-8")
    capacity.write_text('{"query_ids":["q1"]}\n', encoding="utf-8")
    copied_screen.write_text(screen.read_text(encoding="utf-8"), encoding="utf-8")
    unknown.write_text('{"query_ids":["q3"]}\n', encoding="utf-8")
    assert (
        frozen_real_selection_kind(copied_screen, output_dir=tmp_path)
        == "screen"
    )
    assert frozen_real_selection_kind(capacity, output_dir=tmp_path) == "capacity"
    assert frozen_real_selection_kind(unknown, output_dir=tmp_path) is None


def test_prepared_bundle_rejects_tampered_artifact(tmp_path) -> None:
    artifacts = {}
    for key, filename in PREPARED_FILENAMES.items():
        path = tmp_path / filename
        path.write_text(f"{key}\n", encoding="utf-8")
        artifacts[key] = {"sha256": sha256_file(path), "path": filename}
    report = {
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "source_sha256": {"source": "fixed"},
        "preflight": {"queries": 1},
        "feature_catalog_rows": 1,
        "pair_feature_evidence_rows": 1,
        "verifier_screen": {"queries": 1},
        "verifier_batch_plan": {"queries": 1},
        "artifacts": artifacts,
    }
    canonical = tmp_path / "canonical.json"
    canonical.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "prepare_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    validate_prepared_bundle(tmp_path, canonical_report_path=canonical)
    (tmp_path / PREPARED_FILENAMES["candidate_manifest"]).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="candidate_manifest"):
        validate_prepared_bundle(tmp_path, canonical_report_path=canonical)


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
