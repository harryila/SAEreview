"""Validate the Structural Rescue development contract without external assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import (
    ARM_NAMES,
    DEFAULT_PROTOCOL,
    FEATURE_DESCRIPTION_BATCH_SIZE,
    RANDOM_SEED_NAMESPACE,
    RANDOM_SEED_PAIR_COUNT,
    RANDOM_SEED_PAIRS_SHA256,
    VERIFIED_RANDOM_PAIR_INDICES,
    VERIFIER_BATCH_SIZE,
)
from .llm import (
    MAX_ATTEMPTS,
    MODEL,
    PROMPT_VERSION,
    REASONING_EFFORT,
    TEMPERATURE,
    VERIFIER_EVIDENCE_MODES,
)


def validate(path: Path = DEFAULT_PROTOCOL) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("protocol_revision") != 7:
        raise ValueError("Unexpected Structural Rescue protocol revision")
    if protocol["status"] != "exploratory_development_only":
        raise ValueError("Protocol must remain exploratory development-only")
    if protocol["scope"]["confirmatory_claim_allowed"] is not False:
        raise ValueError("SCAR cannot support a confirmatory claim here")
    if protocol["scope"]["latent_choice_test_prompts_may_be_used"] is not False:
        raise ValueError("Latent Choice prompts must remain excluded")
    if tuple(protocol["arms"]) != ARM_NAMES:
        raise ValueError("Protocol arm order/names differ from implementation")
    candidate_generation = protocol["candidate_generation"]
    if candidate_generation["random_seed_namespace"] != RANDOM_SEED_NAMESPACE:
        raise ValueError("Random seed namespace differs from implementation")
    if int(candidate_generation["random_seed_pair_count"]) != RANDOM_SEED_PAIR_COUNT:
        raise ValueError("Random seed-pair count differs from implementation")
    if candidate_generation["random_seed_pairs_sha256"] != RANDOM_SEED_PAIRS_SHA256:
        raise ValueError("Random seed pairs differ from implementation")
    if tuple(candidate_generation["verified_random_pair_indices"]) != (
        VERIFIED_RANDOM_PAIR_INDICES
    ):
        raise ValueError("Verifier random controls differ from implementation")
    if protocol["verifier"]["model"] != MODEL:
        raise ValueError("Verifier model differs from implementation")
    verifier = protocol["verifier"]
    if verifier["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Verifier prompt version differs from implementation")
    if verifier["reasoning_effort"] != REASONING_EFFORT:
        raise ValueError("Verifier reasoning effort differs from implementation")
    if float(verifier["temperature"]) != TEMPERATURE:
        raise ValueError("Verifier temperature differs from implementation")
    if int(verifier["maximum_attempts"]) != MAX_ATTEMPTS:
        raise ValueError("Verifier retry policy differs from implementation")
    if int(verifier["batch_size"]) != VERIFIER_BATCH_SIZE:
        raise ValueError("Verifier batch size differs from implementation")
    if int(verifier["mode_request_workers"]) != 4:
        raise ValueError("Verifier mode-request worker count differs from implementation")
    expected_modes = (
        "structure",
        "activation_only",
        "aligned_description",
        "shuffled_description",
    )
    if tuple(VERIFIER_EVIDENCE_MODES) != expected_modes:
        raise ValueError("Verifier evidence modes differ from the frozen contract")
    if (
        int(protocol["feature_evidence"]["description_batch_size"])
        != FEATURE_DESCRIPTION_BATCH_SIZE
    ):
        raise ValueError("Feature-description batch size differs from implementation")
    serialized = json.dumps(protocol, sort_keys=True).lower()
    if "improved serendipity" in serialized or "causal intervention result" in serialized:
        raise ValueError("Protocol overstates the possible claim")
    gate = protocol["development_go_no_rule"]
    expected_gate = {
        "minimum_known_rescues_recovered": 33,
        "maximum_dense_hits_lost": 4,
        "minimum_net_utility": 29,
        "minimum_arm_net_utility_margin": 5,
        "minimum_bootstrap_positive_fraction": 0.90,
        "bootstrap_samples": 10000,
        "bootstrap_seed_base": 2026090200,
        "random_source_tail_alpha": 0.05,
        "minimum_queries_with_usable_evidence_per_stratum": 41,
        "maximum_incremental_dense_losses": 1,
    }
    for key, expected in expected_gate.items():
        if gate.get(key) != expected:
            raise ValueError(f"Development gate threshold drifted: {key}")
    return {
        "study_id": protocol["study_id"],
        "status": protocol["status"],
        "arms": list(protocol["arms"]),
        "verifier_model": protocol["verifier"]["model"],
        "latent_choice_test_prompts_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--show-summary", action="store_true")
    args = parser.parse_args()
    summary = validate(args.protocol)
    if args.show_summary:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
