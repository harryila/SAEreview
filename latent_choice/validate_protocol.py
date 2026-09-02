#!/usr/bin/env python3
"""Validate the frozen Latent Choice v1 protocol without accessing test content."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from latent_choice.choice_endpoint import (
    CODE_COMPLETIONS,
    CODE_SYMBOLS,
    DEFAULT_DRAW_SEED,
    DEFAULT_MAPPING_SEED,
    DOMAINS,
    build_choice_instruction,
    mapping_for_prompt,
    mapping_sha256,
    render_choice_prompt,
    resolve_code_token_ids,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "latent_choice" / "protocol.json"
TEMPLATE_PATH = ROOT / "latent_choice" / "test_config.template.json"
CODE_TOKEN_MANIFEST_PATH = ROOT / "latent_choice" / "code_token_manifest.json"

EXPECTED_MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
EXPECTED_SAE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
EXPECTED_SAE_SHA256 = "bbd770b6f8b92a2fe7498e05bd6274c6cfa89ebc08fb972c0e842840737f1a82"
EXPECTED_MANIFEST_SHA256 = "18100b8a28777539737e5a33b1c00bbccd5da35dd3325828399a5d5e426d5b98"
EXPECTED_PREDECESSOR_COMMIT = "487ff9d76c7f5187b5dfc7582f146ea66d351648"
EXPECTED_AUDIT_SHA256 = "fcd7e2b7de340b4ad6972ba8bd7c977df24d2373641c05b4ea2b8b41d2e55b98"
EXPECTED_CODE_TOKEN_MANIFEST_SHA256 = "ab93e0c3a236163ae5bf643d10f0337110bc8ebfbfb65c4cf326f957967fa16f"
EXPECTED_PROTOCOL_SHA256 = "06023e7551795726753787e1531fade5f48fafbfb1ddf278c8760fb5dfa8924f"
EXPECTED_MAPPING_SEED = "latent-choice-domain-code-map-v1"
EXPECTED_DRAW_SEED = "latent-choice-paired-draw-v1"
EXPECTED_DOMAINS = [
    "biology/ecology",
    "medicine/public health",
    "physics",
    "chemistry/materials",
    "engineering/control",
    "computer science/software",
    "AI/neural networks",
    "economics/markets",
    "organizations/governance",
    "sociology/culture",
    "psychology/cognition",
    "education/learning",
    "law/policy",
    "history",
    "arts/literature",
    "sports/games",
    "geography/earth/environment",
    "everyday/household",
]
EXPECTED_CODES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
]
EXPECTED_COMPLETIONS = [
    " A", " B", " C", " D", " E", " F", " G", " H", " I",
    " J", " K", " L", " M", " N", " O", " P", " Q", " R",
]
EXPECTED_TOKEN_IDS = {
    "A": 586,
    "B": 599,
    "C": 585,
    "D": 608,
    "E": 637,
    "F": 633,
    "G": 653,
    "H": 640,
    "I": 590,
    "J": 713,
    "K": 747,
    "L": 629,
    "M": 595,
    "N": 646,
    "O": 687,
    "P": 596,
    "Q": 1274,
    "R": 625,
}
GOLDEN_PROMPT_IDS = [f"golden-{index:02d}" for index in range(18)]
GOLDEN_PROMPT_ID = "golden-07"
EXPECTED_GOLDEN_MAPPING = {
    "biology/ecology": "L",
    "medicine/public health": "M",
    "physics": "N",
    "chemistry/materials": "O",
    "engineering/control": "P",
    "computer science/software": "Q",
    "AI/neural networks": "R",
    "economics/markets": "A",
    "organizations/governance": "B",
    "sociology/culture": "C",
    "psychology/cognition": "D",
    "education/learning": "E",
    "law/policy": "F",
    "history": "G",
    "arts/literature": "H",
    "sports/games": "I",
    "geography/earth/environment": "J",
    "everyday/household": "K",
}
EXPECTED_GOLDEN_MAPPING_SHA256 = "9f7329dcd21d00624bc520324f891c4e3f35353f2a492f407853c4862caccaa9"
EXPECTED_RECORD_FIELDS = [
    "prompt_id",
    "split",
    "condition",
    "feature_id",
    "strength",
    "candidate_token_ids",
    "domain_to_code",
    "domain_logits",
    "domain_probabilities",
    "code_logits",
    "full_vocab_candidate_mass",
    "mapping_sha256",
    "activation_row_index",
    "activation_sha256",
    "decision_forward_count",
    "capture_hook_calls",
    "paired_realized_choices",
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_tokenizer(protocol: dict[str, Any], local_files_only: bool) -> list[int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional real-stack check
        raise RuntimeError("--check-tokenizer requires transformers") from exc

    model = protocol["artifacts"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["repo_id"],
        revision=model["revision"],
        token=os.environ.get("HF_TOKEN"),
        local_files_only=local_files_only,
    )
    prompt_ids = ["synthetic-token-boundary-prompt"]
    mapping = mapping_for_prompt(
        prompt_ids[0],
        prompt_ids,
        seed=protocol["choice_endpoint"]["mapping_seed"],
    )
    rendered = render_choice_prompt(
        tokenizer,
        source_name="Synthetic source",
        source_domain="synthetic source domain",
        source_description="A mechanism used only to validate token boundaries.",
        mapping=mapping,
    )
    resolved = resolve_code_token_ids(tokenizer, rendered)
    require(resolved == EXPECTED_TOKEN_IDS, "pinned tokenizer IDs drift from manifest")
    return [resolved[code] for code in CODE_SYMBOLS]


def validate(
    protocol: dict[str, Any],
    template: dict[str, Any],
    code_manifest: dict[str, Any],
) -> None:
    require(
        sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA256,
        "frozen protocol hash drift",
    )
    require(protocol.get("schema_version") == 1, "wrong schema_version")
    require(protocol.get("protocol_id") == "latent-choice-v1", "wrong protocol_id")
    require(protocol.get("protocol_revision") == 1, "wrong protocol_revision")
    require(protocol.get("status") == "predevelopment_frozen", "protocol is not frozen")

    predecessor = protocol["predecessor"]
    require(
        predecessor["frozen_stop_commit"] == EXPECTED_PREDECESSOR_COMMIT,
        "predecessor commit drift",
    )
    require(
        predecessor["domain_audit_report_sha256"] == EXPECTED_AUDIT_SHA256,
        "predecessor audit binding drift",
    )

    artifacts = protocol["artifacts"]
    require(
        artifacts["model"]["revision"] == EXPECTED_MODEL_REVISION,
        "model revision drift",
    )
    require(artifacts["model"]["quantization"] == "none", "quantization forbidden")
    require(artifacts["sae"]["revision"] == EXPECTED_SAE_REVISION, "SAE revision drift")
    require(artifacts["sae"]["sha256"] == EXPECTED_SAE_SHA256, "SAE hash drift")
    require(artifacts["sae"]["layer_zero_indexed"] == 20, "SAE layer drift")
    require(artifacts["sae"]["width"] == 16384, "SAE width drift")
    code_artifact = artifacts["choice_code_tokens"]
    require(
        code_artifact["sha256"] == EXPECTED_CODE_TOKEN_MANIFEST_SHA256,
        "protocol code-token manifest binding drift",
    )
    require(
        sha256_file(CODE_TOKEN_MANIFEST_PATH) == EXPECTED_CODE_TOKEN_MANIFEST_SHA256,
        "committed code-token manifest hash drift",
    )
    require(code_manifest["code_symbols"] == EXPECTED_CODES, "manifest code symbols drift")
    require(
        code_manifest["code_completions"] == EXPECTED_COMPLETIONS,
        "manifest code completions drift",
    )
    require(code_manifest["code_token_ids"] == EXPECTED_TOKEN_IDS, "manifest token IDs drift")
    require(code_manifest["protocol_id"] == "latent-choice-v1", "manifest protocol drift")
    require(code_manifest["protocol_revision"] == 1, "manifest revision drift")
    require(
        code_manifest["tokenizer"]["revision"] == EXPECTED_MODEL_REVISION,
        "manifest tokenizer revision drift",
    )
    require(
        code_manifest["validation"]["development_or_test_content_accessed"] is False,
        "token validation must not access experiment content",
    )

    stimuli = protocol["stimuli"]
    require(
        stimuli["prompt_manifest_sha256"] == EXPECTED_MANIFEST_SHA256,
        "prompt manifest binding drift",
    )
    require(
        (stimuli["prompt_count"], stimuli["development_count"], stimuli["test_count"])
        == (200, 80, 120),
        "prompt counts drift",
    )

    menu = protocol["domain_menu"]
    require(menu["domains"] == EXPECTED_DOMAINS, "domain menu/order drift")
    require(menu["code_surfaces"] == EXPECTED_CODES, "code menu/order drift")
    require(menu["code_completions"] == EXPECTED_COMPLETIONS, "code completions drift")
    require("other" not in menu["domains"], "other must not enter the action space")
    require("excluded by design" in menu["other_policy"], "other policy is ambiguous")
    mapping = menu["mapping"]
    require(tuple(DOMAINS) == tuple(EXPECTED_DOMAINS), "runtime domain constants drift")
    require(tuple(CODE_SYMBOLS) == tuple(EXPECTED_CODES), "runtime code constants drift")
    require(
        [CODE_COMPLETIONS[code] for code in CODE_SYMBOLS] == EXPECTED_COMPLETIONS,
        "runtime completion constants drift",
    )
    require(DEFAULT_MAPPING_SEED == EXPECTED_MAPPING_SEED, "runtime mapping seed drift")
    require(DEFAULT_DRAW_SEED == EXPECTED_DRAW_SEED, "runtime draw seed drift")
    require(mapping["seed"] == EXPECTED_MAPPING_SEED, "mapping seed drift")
    require("(i + r) mod 18" in mapping["algorithm"], "mapping algorithm drift")
    golden_mapping = mapping_for_prompt(
        GOLDEN_PROMPT_ID, GOLDEN_PROMPT_IDS, seed=EXPECTED_MAPPING_SEED
    )
    require(golden_mapping == EXPECTED_GOLDEN_MAPPING, "golden mapping drift")
    require(
        mapping_sha256(golden_mapping) == EXPECTED_GOLDEN_MAPPING_SHA256,
        "golden mapping hash drift",
    )

    choice = protocol["choice_prompt"]
    for placeholder in ("{coded_menu}", "{source_name}", "{source_domain}", "{source_description}"):
        require(placeholder in choice["template"], f"choice template missing {placeholder}")
    require(choice["assistant_prefill"] == "CHOICE:", "assistant prefill drift")
    require(choice["choice_temperature"] == 1.0, "choice temperature drift")
    code_to_domain = {code: domain for domain, code in golden_mapping.items()}
    coded_menu = "\n".join(
        choice["menu_line_template"].format(code=code, domain=code_to_domain[code])
        for code in EXPECTED_CODES
    )
    frozen_instruction = choice["template"].format(
        source_name="Golden source",
        source_domain="golden source domain",
        source_description="Golden mechanism.",
        coded_menu=coded_menu,
    )
    runtime_instruction = build_choice_instruction(
        source_name="Golden source",
        source_domain="golden source domain",
        source_description="Golden mechanism.",
        mapping=golden_mapping,
    )
    require(
        runtime_instruction.encode("utf-8") == frozen_instruction.encode("utf-8"),
        "runtime choice instruction differs byte-for-byte from protocol template",
    )
    endpoint = protocol["choice_endpoint"]
    require(endpoint["mapping_seed"] == EXPECTED_MAPPING_SEED, "endpoint mapping seed drift")
    require(endpoint["paired_draw_seed"] == EXPECTED_DRAW_SEED, "paired draw seed drift")
    require(endpoint["paired_draws_per_prompt"] == 8, "baseline paired draw count drift")
    require(endpoint["decision_prefix"] == "CHOICE:", "endpoint decision prefix drift")
    measurement = protocol["choice_measurement"]
    require("float64" in measurement["conditional_probability"], "q dtype drift")
    compliance = measurement["candidate_mass_compliance"]
    require(compliance["per_prompt_floor"] == 0.5, "candidate-mass floor drift")
    require(compliance["minimum_prompt_fraction"] == 0.9, "candidate-mass fraction drift")
    require(
        measurement["required_record_fields"] == EXPECTED_RECORD_FIELDS,
        "required choice-record schema drift",
    )

    discovery = protocol["feature_discovery"]
    require(discovery["development_only"] is True, "discovery must be development-only")
    require(discovery["multiple_testing"]["permutations"] == 1000, "permutation count drift")
    require(discovery["multiple_testing"]["familywise_alpha"] == 0.05, "alpha drift")
    require(discovery["matched_random_features"]["count"] == 5, "control count drift")

    intervention = protocol["intervention"]
    require(intervention["target_doses"] == [0.25, 0.5, 1.0], "dose grid drift")
    require(intervention["confirmatory_strength"] == 1.0, "test strength drift")
    require(intervention["noise_seed"] == 20260902, "noise seed drift")
    require("exactly one" in intervention["scope"], "choice-only hook scope is ambiguous")
    require(
        "discard every intervened cache/state"
        in intervention["clean_downstream_generation"]["rule"],
        "clean generation does not forbid cache reuse",
    )

    human = protocol["human_evaluation"]
    require(human["rater_type"] == "human_only", "human-only policy drift")
    require(human["gates"]["minimum_consistency_rate_each_arm"] == 0.9, "consistency gate drift")
    require(human["gates"]["quality_noninferiority_margin"] == -0.25, "quality margin drift")

    guard = protocol["test_access_guard"]
    require(guard["confirmation_flag"] == "--confirm-test", "test flag drift")
    required = protocol["required_before_test"]
    require(len(required) == len(set(required)), "duplicate required-before-test fields")
    require(
        set(required).issubset(template),
        "test template lacks required-before-test fields",
    )
    require(template["protocol_id"] == protocol["protocol_id"], "template protocol drift")
    require(template["protocol_revision"] == protocol["protocol_revision"], "template revision drift")
    require(template["protocol_sha256"] == sha256_file(PROTOCOL_PATH), "template protocol hash drift")
    require(
        template["choice_prompt_sha256"]
        == sha256_bytes(choice["template"].encode("utf-8")),
        "template choice-prompt hash drift",
    )
    require(template["prompt_manifest_sha256"] == EXPECTED_MANIFEST_SHA256, "template manifest drift")
    require(
        template["choice_code_token_manifest_sha256"]
        == EXPECTED_CODE_TOKEN_MANIFEST_SHA256,
        "template code-token manifest hash drift",
    )
    require(template["choice_code_token_ids"] == EXPECTED_TOKEN_IDS, "template token IDs drift")
    require(template["domain_mapping_seed"] == mapping["seed"], "template mapping seed drift")
    require(
        template["domain_mapping_algorithm_sha256"] == canonical_hash(mapping),
        "template mapping-algorithm hash drift",
    )
    require(
        template["noise_rule_sha256"]
        == canonical_hash(
            {
                "control": intervention["activation_noise_control"],
                "seed": intervention["noise_seed"],
            }
        ),
        "template noise-rule hash drift",
    )
    require(
        template["human_rater_protocol_sha256"] == canonical_hash(human),
        "template human-rater protocol hash drift",
    )
    require(template["intervention_strength"] == 1.0, "template test strength drift")
    require(template["test_prompt_count"] == 120, "template test prompt count drift")
    require(template["test_samples_per_prompt"] == 8, "template test sample count drift")
    require(len(template["confirmatory_arms"]) == 8, "template must freeze eight test arms")
    require(template["frozen_at_utc"] is None, "template must not masquerade as frozen")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-summary", action="store_true")
    parser.add_argument("--check-tokenizer", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    protocol = load_json(PROTOCOL_PATH)
    template = load_json(TEMPLATE_PATH)
    code_manifest = load_json(CODE_TOKEN_MANIFEST_PATH)
    validate(protocol, template, code_manifest)
    token_ids = (
        validate_tokenizer(protocol, args.local_files_only)
        if args.check_tokenizer
        else None
    )
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_revision": protocol["protocol_revision"],
        "status": protocol["status"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "choice_prompt_sha256": sha256_bytes(
            protocol["choice_prompt"]["template"].encode("utf-8")
        ),
        "domain_count": len(protocol["domain_menu"]["domains"]),
        "other_in_action_space": "other" in protocol["domain_menu"]["domains"],
        "development_prompts": protocol["stimuli"]["development_count"],
        "test_prompts": protocol["stimuli"]["test_count"],
        "test_content_accessed": False,
        "tokenizer_checked": token_ids is not None,
        "code_token_manifest_checked": True,
        "choice_code_token_ids": token_ids or EXPECTED_TOKEN_IDS,
    }
    if args.show_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Latent Choice protocol validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
