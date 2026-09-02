"""Focused regressions for the frozen structural-quality scope."""

from __future__ import annotations

import hashlib
import json

import pytest

from latent_escape.evaluate import (
    arm_metrics,
    compare_population,
    join_key,
    load_quality,
    quality_assignment_mapping,
    quality_rating_blind_id,
    requires_structural_quality,
    select_quality_generations,
    validate_generation_test_config_binding,
    validate_quality_plan_config,
)


def test_quality_is_required_only_for_baseline_and_full_targeted() -> None:
    assert requires_structural_quality({"condition": "baseline"})
    assert requires_structural_quality(
        {
            "condition": "targeted_feature_suppression",
            "intervention": {"strength": 1.0},
        }
    )
    assert not requires_structural_quality(
        {
            "condition": "targeted_feature_suppression",
            "intervention": {"strength": 0.5},
        }
    )
    assert not requires_structural_quality(
        {"condition": "matched_random_feature_suppression", "feature_id": 7}
    )


def _prompt_metrics(
    selected_rate: float,
    quality: float | None,
    entropy: float,
    distinct: float,
) -> dict[str, float | None]:
    return {
        "selected_domain_rate": selected_rate,
        "domain_entropy_nats": entropy,
        "domain_entropy_normalized": entropy / 3.0,
        "distinct_domain_count": distinct,
        "json_validity_rate": 1.0,
        "structural_quality": quality,
        "source_target_semantic_distance": 0.5,
    }


def test_quality_contrast_does_not_require_random_arm_ratings() -> None:
    prompts = ["p1", "p2", "p3"]
    baseline = "baseline"
    targeted = "targeted_feature_suppression"
    random_arms = [f"matched_random_feature_suppression:{index}" for index in range(5)]
    metrics = {
        baseline: {
            prompt: _prompt_metrics(1.0, 4.0, 0.5, 1.0) for prompt in prompts
        },
        targeted: {
            prompt: _prompt_metrics(0.0, 3.9, 1.0, 2.0) for prompt in prompts
        },
        **{
            arm: {
                prompt: _prompt_metrics(0.8, None, 0.6, 1.2)
                for prompt in prompts
            }
            for arm in random_arms
        },
    }

    result = compare_population(
        "test",
        prompts,
        metrics,
        baseline,
        targeted,
        random_arms,
        resamples=100,
        seed=7,
        quality_margin=-0.25,
        json_margin=-0.02,
    )

    comparisons = result["paired_comparisons"]
    assert comparisons["structural_quality"]["targeted_minus_baseline"][
        "estimate"
    ] == pytest.approx(-0.1)
    assert comparisons["structural_quality"]["targeted_minus_random_mean"][
        "estimate"
    ] is None
    assert comparisons["selected_domain_rate"]["targeted_minus_random_mean"][
        "estimate"
    ] == pytest.approx(-0.8)
    assert result["claim_boundary"]["causal_target_domain_selection_supported"]
    assert result["claim_boundary"]["reduced_homogeneity_supported"]
    assert result["claim_boundary"]["serendipity_evaluated"] is False


def _quality_protocol() -> dict[str, object]:
    return {
        "generation": {"test_paired_samples_per_prompt_per_condition": 8},
        "development_intervention_gate": {"paired_samples_per_prompt": 4},
        "quality_guardrail_sampling": {
            "pair_selection_seed": "latent-escape-quality-pair-v1",
            "reliability_selection_seed": "latent-escape-quality-reliability-v1",
            "samples_per_prompt": 1,
            "reliability_prompt_fraction": 0.10,
            "required_arms": [
                "baseline",
                "targeted_feature_suppression",
            ],
            "primary_rater_estimand": "primary only",
            "duplicate_rater_use": "reliability only",
            "expected_workload": {
                "development_gate": {
                    "prompt_count": 24,
                    "unique_generation_ratings": 48,
                    "reliability_prompt_count": 3,
                    "duplicate_ratings": 6,
                    "total_rating_tasks": 54,
                },
                "test": {
                    "prompt_count": 120,
                    "unique_generation_ratings": 240,
                    "reliability_prompt_count": 12,
                    "duplicate_ratings": 24,
                    "total_rating_tasks": 264,
                },
            },
        },
    }


def _quality_generations(
    split: str, prompt_count: int, samples_per_prompt: int
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for prompt_number in range(prompt_count):
        prompt_id = f"p{prompt_number:03d}"
        for condition in ("baseline", "targeted_feature_suppression"):
            for sample_index in range(samples_per_prompt):
                seed = 100_000 + 100 * prompt_number + sample_index
                records.append(
                    {
                        "protocol_id": "latent-escape-mvp-v1",
                        "run_id": f"run-{condition}",
                        "prompt_id": prompt_id,
                        "split": split,
                        "condition": condition,
                        "sample_index": sample_index,
                        "seed": seed,
                        "generated_text": (
                            f"outcome-{condition}-{prompt_id}-{sample_index}"
                        ),
                        "analogy_schema_valid": sample_index % 2 == 0,
                        "parsed_output": {"target_domain": f"outcome-{sample_index}"},
                        "intervention": {"strength": 1.0}
                        if condition == "targeted_feature_suppression"
                        else None,
                    }
                )
    return records


@pytest.mark.parametrize(
    ("split", "prompt_count", "samples", "unique_count", "duplicate_count"),
    [
        ("development", 24, 4, 48, 6),
        ("test", 120, 8, 240, 24),
    ],
)
def test_frozen_quality_workload_selects_one_paired_sample_per_prompt(
    split: str,
    prompt_count: int,
    samples: int,
    unique_count: int,
    duplicate_count: int,
) -> None:
    selected, reliability_keys, provenance = select_quality_generations(
        _quality_generations(split, prompt_count, samples),
        split,
        _quality_protocol(),
    )

    assert len(selected) == unique_count
    assert len(reliability_keys) == duplicate_count
    assert provenance["total_rating_task_count"] == unique_count + duplicate_count
    assert len(provenance["quality_sampling_plan_sha256"]) == 64
    assert len(provenance["quality_reliability_subset_sha256"]) == 64
    by_prompt: dict[str, set[tuple[int, int]]] = {}
    for row in selected:
        by_prompt.setdefault(str(row["prompt_id"]), set()).add(
            (int(row["sample_index"]), int(row["seed"]))
        )
    assert len(by_prompt) == prompt_count
    assert all(len(pairs) == 1 for pairs in by_prompt.values())


def test_quality_membership_is_independent_of_generated_outcomes() -> None:
    records = _quality_generations("test", 120, 8)
    first, _, first_provenance = select_quality_generations(
        records, "test", _quality_protocol()
    )
    mutated = []
    for row in reversed(records):
        mutated.append(
            {
                **row,
                "run_id": f"changed-{row['run_id']}",
                "generated_text": f"changed-{row['generated_text']}",
                "analogy_schema_valid": not bool(row["analogy_schema_valid"]),
                "parsed_output": {"target_domain": "changed"},
            }
        )
    second, _, second_provenance = select_quality_generations(
        mutated, "test", _quality_protocol()
    )

    def identity(row: dict[str, object]) -> tuple[object, object, object, object]:
        return (
            row["prompt_id"],
            row["condition"],
            row["sample_index"],
            row["seed"],
        )
    assert {identity(row) for row in first} == {identity(row) for row in second}
    assert (
        first_provenance["quality_sampling_plan_sha256"]
        == second_provenance["quality_sampling_plan_sha256"]
    )
    assert (
        first_provenance["quality_reliability_subset_sha256"]
        == second_provenance["quality_reliability_subset_sha256"]
    )


def test_quality_assignment_ids_are_unique_and_slot_is_not_needed_in_ratings() -> None:
    selected, reliability_keys, _ = select_quality_generations(
        _quality_generations("development", 24, 4),
        "development",
        _quality_protocol(),
    )
    mapping = quality_assignment_mapping(selected, reliability_keys)

    assert len(mapping) == 54
    assert len(set(mapping)) == 54
    for generation in selected:
        primary = quality_rating_blind_id(generation, "primary")
        assert primary in mapping
        if join_key(generation) in reliability_keys:
            reliability = quality_rating_blind_id(generation, "reliability")
            assert reliability in mapping
            assert reliability != primary


def test_duplicate_rating_is_reliability_only_and_requires_distinct_rater(
    tmp_path,
) -> None:
    selected, reliability_keys, _ = select_quality_generations(
        _quality_generations("development", 24, 4),
        "development",
        _quality_protocol(),
    )
    mapping = quality_assignment_mapping(selected, reliability_keys)
    primary_keys = {join_key(row) for row in selected}
    primary_score = {
        key: float(1 + index % 5) for index, key in enumerate(sorted(primary_keys))
    }
    rows = []
    for blind_id, (key, slot) in mapping.items():
        score = primary_score[key]
        if slot == "reliability" and int(key[3]) % 2:
            score = max(1.0, score - 1.0)
        rows.append(
            {
                "blind_quality_id": blind_id,
                "structural_quality": score,
                "rater_id": "rater-a" if slot == "primary" else "rater-b",
                "judge_protocol_id": "judge-v1",
                "judge_prompt_sha256": "rubric-hash",
                "blinded": True,
            }
        )
    ratings_path = tmp_path / "ratings.jsonl"
    ratings_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    quality_map, provenance = load_quality(
        [ratings_path],
        False,
        mapping,
        "judge-v1",
        "rubric-hash",
        primary_keys,
        reliability_keys,
    )

    assert quality_map == primary_score
    assert provenance["rating_count"] == 54
    assert provenance["primary_rating_count"] == 48
    assert provenance["duplicate_rating_count"] == 6
    assert provenance["ratings_per_generation"] == {1: 42, 2: 6}
    assert provenance["primary_endpoint_uses_duplicate_ratings"] is False
    reliability = provenance["duplicate_rater_reliability"]
    assert reliability["prompt_count"] == 3
    assert reliability["item_count"] == 6
    assert 0.0 <= reliability["exact_agreement"] <= 1.0
    assert reliability["within_one_point_agreement"] == 1.0
    assert (
        reliability["linear_weighted_cohen_kappa"] is None
        or -1.0 <= reliability["linear_weighted_cohen_kappa"] <= 1.0
    )

    for row in rows:
        assignment = mapping[row["blind_quality_id"]]
        if assignment[1] == "reliability":
            row["rater_id"] = "rater-a"
            break
    invalid_path = tmp_path / "same-rater.jsonl"
    invalid_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="distinct rater IDs"):
        load_quality(
            [invalid_path],
            False,
            mapping,
            "judge-v1",
            "rubric-hash",
            primary_keys,
            reliability_keys,
        )

    missing_path = tmp_path / "missing-rating.jsonl"
    missing_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:-1])
    )
    with pytest.raises(ValueError, match="frozen primary/reliability assignments"):
        load_quality(
            [missing_path],
            False,
            mapping,
            "judge-v1",
            "rubric-hash",
            primary_keys,
            reliability_keys,
        )


def test_quality_plan_hashes_must_match_frozen_test_config() -> None:
    _, _, provenance = select_quality_generations(
        _quality_generations("test", 120, 8), "test", _quality_protocol()
    )
    config = {
        "quality_sampling_plan_sha256": provenance[
            "quality_sampling_plan_sha256"
        ],
        "quality_reliability_subset_sha256": provenance[
            "quality_reliability_subset_sha256"
        ],
    }
    validate_quality_plan_config(config, provenance)
    config["quality_sampling_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sampling plan hash drift"):
        validate_quality_plan_config(config, provenance)


def test_test_generation_records_are_bound_to_frozen_config(tmp_path) -> None:
    config_path = tmp_path / "test_frozen.json"
    config_path.write_text('{"frozen":true}\n')
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    records = _quality_generations("test", 120, 8)
    for row in records:
        row["test_config_sha256"] = config_hash
    validate_generation_test_config_binding(records, config_path)
    records[0]["test_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="test-config hash drift"):
        validate_generation_test_config_binding(records, config_path)


def test_arm_metrics_uses_all_domain_samples_and_one_quality_sample() -> None:
    records = [
        {
            "prompt_id": "p1",
            "domain_label": "selected" if index < 4 else "other",
            "quality": 4.25 if index == 3 else None,
            "semantic_distance": 0.5,
            "json_valid": True,
        }
        for index in range(8)
    ]

    metrics = arm_metrics(records, "selected", ["selected", "other"])["p1"]

    assert metrics["selected_domain_rate"] == 0.5
    assert metrics["distinct_domain_count"] == 2.0
    assert metrics["structural_quality"] == 4.25
