"""Focused regressions for the frozen structural-quality scope."""

from __future__ import annotations

import pytest

from latent_escape.evaluate import compare_population, requires_structural_quality


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
