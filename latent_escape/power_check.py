#!/usr/bin/env python3
"""Deterministic design-stage MDE calculation for the prompt-clustered test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "latent_escape" / "power_report.json"


def paired_prompt_mde(
    *,
    prompts: int,
    samples: int,
    target_rate: float,
    prompt_icc: float,
    cross_condition_correlation: float,
    alpha: float,
    power: float,
) -> float:
    """Normal-approximation MDE for a paired mean of prompt-level proportions."""

    if prompts <= 1 or samples <= 0:
        raise ValueError("prompts must exceed one and samples must be positive")
    for name, value in {
        "target_rate": target_rate,
        "prompt_icc": prompt_icc,
        "cross_condition_correlation": cross_condition_correlation,
        "alpha": alpha,
        "power": power,
    }.items():
        if not 0 < value < 1:
            raise ValueError(f"{name} must be strictly between zero and one")

    design_effect = 1.0 + (samples - 1) * prompt_icc
    variance_per_condition = target_rate * (1.0 - target_rate) * design_effect / samples
    paired_variance = 2.0 * variance_per_condition * (
        1.0 - cross_condition_correlation
    )
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    return float((z_alpha + z_power) * math.sqrt(paired_variance / prompts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=int, default=120)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--target-rate", type=float, default=0.20)
    parser.add_argument("--prompt-icc", type=float, default=0.20)
    parser.add_argument("--cross-condition-correlation", type=float, default=0.50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    common = {
        "prompts": args.prompts,
        "samples": args.samples,
        "prompt_icc": args.prompt_icc,
        "cross_condition_correlation": args.cross_condition_correlation,
        "alpha": args.alpha,
        "power": args.power,
    }
    assumed_mde = paired_prompt_mde(target_rate=args.target_rate, **common)
    worst_case_mde = paired_prompt_mde(target_rate=0.50, **common)
    report = {
        "schema_version": 1,
        "protocol_id": "latent-escape-mvp-v1",
        "method": (
            "Normal approximation for a paired mean of prompt-level domain-rate "
            "differences with beta-binomial-style design effect"
        ),
        "design": {
            "test_prompts": args.prompts,
            "paired_samples_per_prompt_per_condition": args.samples,
            "two_sided_alpha": args.alpha,
            "target_power": args.power,
        },
        "assumptions": {
            "selected_domain_rate": args.target_rate,
            "within_prompt_output_icc": args.prompt_icc,
            "cross_condition_prompt_correlation": args.cross_condition_correlation,
        },
        "minimum_detectable_absolute_rate_change": assumed_mde,
        "worst_case_rate_0_5_mde": worst_case_mde,
        "interpretation": (
            "Under the stated clustering and pairing assumptions, the full test is "
            f"designed for about a {100.0 * assumed_mde:.1f}-point effect at a 20% "
            f"baseline rate; the variance-maximizing 50% case is {100.0 * worst_case_mde:.1f} points."
        ),
        "caveat": (
            "This is a design sensitivity calculation, not post-hoc observed power. "
            "The confirmatory interval still resamples prompts nonparametrically."
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
