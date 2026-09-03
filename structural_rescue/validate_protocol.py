"""Validate the Structural Rescue development contract without external assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ARM_NAMES, DEFAULT_PROTOCOL, RANDOM_SEEDS, VERIFIER_BATCH_SIZE
from .llm import MAX_ATTEMPTS, MODEL, PROMPT_VERSION, REASONING_EFFORT, TEMPERATURE


def validate(path: Path = DEFAULT_PROTOCOL) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("protocol_revision") != 2:
        raise ValueError("Unexpected Structural Rescue protocol revision")
    if protocol["status"] != "exploratory_development_only":
        raise ValueError("Protocol must remain exploratory development-only")
    if protocol["scope"]["confirmatory_claim_allowed"] is not False:
        raise ValueError("SCAR cannot support a confirmatory claim here")
    if protocol["scope"]["latent_choice_test_prompts_may_be_used"] is not False:
        raise ValueError("Latent Choice prompts must remain excluded")
    if tuple(protocol["arms"]) != ARM_NAMES:
        raise ValueError("Protocol arm order/names differ from implementation")
    if tuple(protocol["candidate_generation"]["random_seeds"]) != RANDOM_SEEDS:
        raise ValueError("Random seeds differ from implementation")
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
    serialized = json.dumps(protocol, sort_keys=True).lower()
    if "improved serendipity" in serialized or "causal intervention result" in serialized:
        raise ValueError("Protocol overstates the possible claim")
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
