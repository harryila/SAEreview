#!/usr/bin/env python3
"""Validate the frozen Latent Escape protocol and optional prompt manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "latent_escape" / "protocol.json"
MANIFEST = ROOT / "latent_escape" / "artifacts" / "prompt_manifest.jsonl"
POWER_REPORT = ROOT / "latent_escape" / "power_report.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_protocol(protocol: dict[str, Any]) -> None:
    require(protocol["schema_version"] == 1, "unsupported schema_version")
    require(protocol["protocol_id"] == "latent-escape-mvp-v1", "wrong protocol_id")
    require(protocol["protocol_revision"] == 2, "unexpected protocol revision")
    require(protocol["status"] == "development_not_run", "unexpected run status")

    artifacts = protocol["artifacts"]
    require(HEX40.fullmatch(artifacts["model"]["revision"]) is not None, "bad model revision")
    require(HEX40.fullmatch(artifacts["sae"]["revision"]) is not None, "bad SAE revision")
    require(HEX64.fullmatch(artifacts["sae"]["sha256"]) is not None, "bad SAE hash")
    require(artifacts["sae"]["layer_zero_indexed"] == 20, "protocol layer drift")
    require(artifacts["sae"]["width"] == 16384, "protocol SAE width drift")
    require(
        artifacts["model"]["execution"].startswith("single CUDA device cuda:0"),
        "model placement is not frozen",
    )
    require(
        artifacts["semantic_distance_model"]["revision"]
        == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "semantic-distance model revision drift",
    )

    stimuli = protocol["stimuli"]
    require(
        stimuli["development_count"] + stimuli["test_count"]
        == stimuli["prompt_count"]
        == 200,
        "prompt counts must be 80 development + 120 test",
    )
    require(HEX64.fullmatch(stimuli["source_sha256"]) is not None, "bad source hash")
    expected_manifest = stimuli["expected_manifest_sha256"]
    require(
        expected_manifest is not None and HEX64.fullmatch(expected_manifest) is not None,
        "bad expected manifest hash",
    )

    labeling = protocol["domain_labeling"]
    require(
        labeling["classifier_revision"]
        == "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",
        "domain classifier revision drift",
    )
    require(
        labeling["generated_target_domain_policy"].startswith("Never use"),
        "self-reported domain must be excluded",
    )
    quality = protocol["outcomes"]["quality_guardrail"]
    require(
        hashlib.sha256(quality["rubric"].encode("utf-8")).hexdigest()
        == quality["rubric_sha256"]
        and quality["ratings_per_generation"] == 1,
        "quality rubric hash or rating count drifted",
    )

    required_conditions = {
        "baseline",
        "targeted_feature_suppression",
        "matched_random_feature_suppression",
        "l2_matched_activation_noise",
        "diversity_instruction",
        "higher_temperature",
    }
    require(required_conditions <= set(protocol["conditions"]), "missing control condition")
    require(
        protocol["feature_discovery"]["development_only"] is True,
        "feature selection must be development-only",
    )
    multiple_testing = protocol["feature_discovery"]["multiple_testing"]
    require(multiple_testing["permutations"] >= 1000, "too few max-stat permutations")
    require(multiple_testing["permutation_unit"].startswith("shuffle complete prompt-level"), "wrong permutation unit")
    require(
        protocol["feature_discovery"]["matched_random_features"]["count"] == 5,
        "exactly five matched random features are required",
    )
    timing = protocol["feature_discovery"]["pre_domain_evidence"]
    require(
        timing["minimum_boundary_resolution"] == 0.9
        and timing["minimum_active_fraction_of_resolved"] == 0.1,
        "pre-domain evidence thresholds drifted",
    )
    require(
        {
            "development_gate_report_path",
            "development_gate_report_sha256",
        }
        <= set(protocol["required_before_test"]),
        "test freeze must require a passing development-gate artifact",
    )
    require(protocol["analysis"]["resampling_unit"] == "source prompt", "wrong resampling unit")
    require(
        protocol["development_intervention_gate"]["prompt_count"] == 24
        and protocol["development_intervention_gate"]["paired_samples_per_prompt"] == 4,
        "development gate must remain 24 prompts x 4 paired samples",
    )
    require(len(protocol["target_domain_taxonomy"]) >= 10, "domain taxonomy is underspecified")
    power = protocol["power"]
    require(power["test_prompts"] == 120, "power design test count drift")
    require(POWER_REPORT.exists(), "missing frozen power report")
    actual_power_hash = hashlib.sha256(POWER_REPORT.read_bytes()).hexdigest()
    require(actual_power_hash == power["power_report_sha256"], "power report hash drift")


def validate_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    payload = MANIFEST.read_bytes()
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    stimuli = protocol["stimuli"]
    require(len(records) == stimuli["prompt_count"], "manifest prompt count mismatch")
    require(len({row["prompt_id"] for row in records}) == len(records), "duplicate prompt_id")
    require(len({row["source_id"] for row in records}) == len(records), "duplicate source_id")
    require(
        len({row["source_group_id"] for row in records}) == len(records),
        "both sides of a SCAR pair appear in the manifest",
    )
    split_counts = Counter(row["split"] for row in records)
    require(split_counts == {"development": 80, "test": 120}, "manifest split mismatch")
    forbidden = {"system_b", "target_background", "paired_target", "mappings", "Explanation"}
    for row in records:
        require(not (forbidden & set(row)), f"paired-target leakage in {row['prompt_id']}")
        require(row["prompt_text"].startswith("You are given a source mechanism."), "prompt template drift")
    manifest_hash = hashlib.sha256(payload).hexdigest()
    expected = stimuli["expected_manifest_sha256"]
    if expected is not None:
        require(manifest_hash == expected, f"manifest SHA-256 is {manifest_hash}; expected {expected}")
    return {"sha256": manifest_hash, "split_counts": dict(sorted(split_counts.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--show-summary", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    manifest_summary: dict[str, Any] | None = None
    if MANIFEST.exists():
        manifest_summary = validate_manifest(protocol)
    elif args.require_manifest:
        raise FileNotFoundError(f"Missing {MANIFEST}")

    if args.show_summary:
        print(
            json.dumps(
                {
                    "protocol_id": protocol["protocol_id"],
                    "status": protocol["status"],
                    "model": protocol["artifacts"]["model"]["repo_id"],
                    "sae": protocol["artifacts"]["sae"]["sae_lens_id"],
                    "prompts": protocol["stimuli"]["prompt_count"],
                    "manifest": manifest_summary or "not generated",
                },
                indent=2,
            )
        )
    else:
        print("protocol OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
