"""Regressions for the immutable base protocol and Amendment 4 bindings."""

from __future__ import annotations

from copy import deepcopy

import pytest

from latent_escape.protocol_amendment import (
    AMENDMENT_PATH,
    BASE_PROTOCOL_PATH,
    BASE_PROTOCOL_SHA256,
    EXACT_TAXONOMY,
    TEST_CONFIG_TEMPLATE_PATH,
    TEST_QUALITY_RELIABILITY_SUBSET_SHA256,
    TEST_QUALITY_SAMPLING_PLAN_SHA256,
    amendment_sha256,
    load_protocol_amendment,
    read_json_object,
    sha256_file,
    validate_protocol_amendment,
    validate_test_config_amendment_bindings,
)


def test_amendment_binds_immutable_base_and_residual_domain_policy() -> None:
    protocol = read_json_object(BASE_PROTOCOL_PATH)
    amendment = load_protocol_amendment(protocol)

    assert sha256_file(BASE_PROTOCOL_PATH) == BASE_PROTOCOL_SHA256
    assert tuple(protocol["target_domain_taxonomy"]) == EXACT_TAXONOMY
    assert amendment["base_protocol"]["sha256"] == BASE_PROTOCOL_SHA256
    assert amendment["domain_selection"]["minimum_development_output_rate"] == 0.10
    assert amendment["domain_selection"]["primary_selected_domain_exclusions"] == [
        "other"
    ]
    assert "other" in protocol["target_domain_taxonomy"]


def test_quality_workload_is_exact_and_reliability_only() -> None:
    quality = load_protocol_amendment()["quality_guardrail_sampling"]

    assert quality["samples_per_prompt"] == 1
    assert quality["reliability_prompt_fraction"] == 0.10
    assert quality["required_arms"] == [
        "baseline",
        "targeted_feature_suppression",
    ]
    assert quality["expected_workload"]["development_gate"] == {
        "prompt_count": 24,
        "unique_generation_ratings": 48,
        "reliability_prompt_count": 3,
        "duplicate_ratings": 6,
        "total_rating_tasks": 54,
    }
    assert quality["expected_workload"]["test"] == {
        "prompt_count": 120,
        "unique_generation_ratings": 240,
        "reliability_prompt_count": 12,
        "duplicate_ratings": 24,
        "total_rating_tasks": 264,
    }
    assert "only for inter-rater reliability" in quality["duplicate_rater_use"]


def test_validator_rejects_relaxed_threshold_or_taxonomy_exclusion() -> None:
    protocol = read_json_object(BASE_PROTOCOL_PATH)
    amendment = read_json_object(AMENDMENT_PATH)

    relaxed = deepcopy(amendment)
    relaxed["domain_selection"]["minimum_development_output_rate"] = 0.05
    with pytest.raises(ValueError, match="10 percent"):
        validate_protocol_amendment(protocol, relaxed)

    excluded = deepcopy(amendment)
    excluded["domain_selection"]["primary_selected_domain_exclusions"] = [
        "other",
        "biology/ecology",
    ]
    with pytest.raises(ValueError, match="exactly"):
        validate_protocol_amendment(protocol, excluded)


def test_test_config_template_binds_amendment_and_guide_hashes() -> None:
    amendment = load_protocol_amendment()
    template = read_json_object(TEST_CONFIG_TEMPLATE_PATH)
    validate_test_config_amendment_bindings(template, amendment)
    assert template["protocol_amendment_sha256"] == amendment_sha256()
    assert template["effective_protocol_revision"] == 4
    assert (
        template["quality_sampling_plan_sha256"]
        == TEST_QUALITY_SAMPLING_PLAN_SHA256
    )
    assert (
        template["quality_reliability_subset_sha256"]
        == TEST_QUALITY_RELIABILITY_SUBSET_SHA256
    )

    drifted = deepcopy(template)
    drifted["protocol_amendment_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bindings drifted"):
        validate_test_config_amendment_bindings(drifted, amendment)
