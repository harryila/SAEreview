#!/usr/bin/env python3
"""Generate paired Latent Escape samples from the pinned Gemma/SAE bundle.

The default is the development baseline.  Test generation is deliberately
guarded by a complete ``test_frozen.json`` and an explicit confirmation flag.
``--dry-run`` exercises the manifest, pairing, resume, and output contracts
without importing Torch, Transformers, or downloading model weights.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Keep both ``python -m latent_escape.generate`` and direct script execution
    # working; intervene.py uses package-qualified imports for auditability.
    sys.path.insert(0, str(ROOT))
DEFAULT_PROTOCOL = ROOT / "latent_escape" / "protocol.json"
DEFAULT_MANIFEST = ROOT / "latent_escape" / "artifacts" / "prompt_manifest.jsonl"
DEFAULT_TEST_CONFIG = ROOT / "latent_escape" / "test_frozen.json"
OUTPUT_DIR = ROOT / "latent_escape" / "outputs" / "generations"
TEST_LEDGER = ROOT / "latent_escape" / "outputs" / "test_generation_ledger.json"

DIVERSITY_INSTRUCTION = (
    "\n\nCondition instruction: Avoid the most obvious or familiar target domain. "
    "Choose an uncommon target domain while preserving the source mechanism's "
    "causal structure and boundary conditions."
)


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(value)
    return records


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(path, payload)


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in records
    ]
    atomic_write(path, (("\n".join(lines) + "\n") if lines else "").encode("utf-8"))


def git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def paired_seed(seed_base: int, prompt_id: str, sample_index: int) -> int:
    """Return a condition-independent seed for paired generation."""

    suffix = int(
        sha256_bytes(f"{prompt_id}|{sample_index}".encode("utf-8"))[:8], 16
    )
    return (int(seed_base) + suffix) % (2**31 - 1)


def generation_id(prompt_id: str, sample_index: int) -> str:
    return f"{prompt_id}:s{sample_index:03d}"


def filename_number(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def effective_prompt(prompt_text: str, condition: str, protocol: dict[str, Any]) -> str:
    if condition != "diversity_instruction":
        return prompt_text
    instruction = protocol["generation"].get(
        "diversity_instruction_text", DIVERSITY_INSTRUCTION
    )
    return prompt_text.rstrip() + str(instruction)


def analogy_schema_valid(value: dict[str, Any]) -> bool:
    required_strings = ("target_domain", "target_system", "explanation")
    if any(
        not isinstance(value.get(key), str) or not value[key].strip()
        for key in required_strings
    ):
        return False
    mappings = value.get("mappings")
    if not isinstance(mappings, list) or len(mappings) < 3:
        return False
    for mapping in mappings:
        if not isinstance(mapping, dict) or any(
            not isinstance(mapping.get(key), str) or not mapping[key].strip()
            for key in ("source_role", "target_role", "shared_relation")
        ):
            return False
    limitations = value.get("limitations")
    return bool(
        isinstance(limitations, list)
        and len(limitations) >= 2
        and all(isinstance(item, str) and item.strip() for item in limitations)
    )


def parse_generated_json(text: str) -> tuple[bool, dict[str, Any] | None]:
    stripped = text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    start = stripped.find("{")
    if start >= 0:
        try:
            candidate_text = stripped[start:]
            value, end = json.JSONDecoder().raw_decode(candidate_text)
            if isinstance(value, dict) and not candidate_text[end:].strip():
                return analogy_schema_valid(value), value
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return analogy_schema_valid(value), value
    return False, None


def _selection_config(path: Path | None) -> dict[str, Any]:
    return read_json(path) if path is not None and path.exists() else {}


def _resolve_feature(
    condition: str,
    explicit_feature: int | None,
    matched_random_index: int,
    selection: dict[str, Any],
) -> int | None:
    no_feature = {"baseline", "diversity_instruction", "higher_temperature"}
    if condition in no_feature:
        if explicit_feature is not None:
            raise ValueError(f"{condition} does not accept --feature-id")
        return None
    if explicit_feature is not None:
        feature_id = explicit_feature
    elif condition == "matched_random_feature_suppression":
        candidates = selection.get("five_matched_random_feature_ids", [])
        if len(candidates) != 5:
            raise ValueError(
                "matched-random generation needs five IDs in the selection config "
                "or an explicit --feature-id"
            )
        if not 0 <= matched_random_index < len(candidates):
            raise ValueError("--matched-random-index is out of range")
        feature_id = candidates[matched_random_index]
    else:
        feature_id = selection.get("selected_feature_id")
    if feature_id is None:
        raise ValueError(f"{condition} needs a frozen or explicit feature ID")
    feature_id = int(feature_id)
    if not 0 <= feature_id < 16384:
        raise ValueError(f"Feature ID {feature_id} is outside [0, 16384)")
    return feature_id


def _validate_manifest(
    manifest_path: Path, protocol: dict[str, Any], split: str
) -> tuple[list[dict[str, Any]], str]:
    manifest_hash = sha256_file(manifest_path)
    expected = protocol["stimuli"].get("expected_manifest_sha256")
    if expected and manifest_hash != expected:
        raise ValueError(
            f"Manifest SHA-256 is {manifest_hash}; protocol requires {expected}"
        )
    records = [row for row in read_jsonl(manifest_path) if row.get("split") == split]
    expected_count = protocol["stimuli"][f"{split}_count"]
    if len(records) != expected_count:
        raise ValueError(
            f"Manifest has {len(records)} {split} prompts; expected {expected_count}"
        )
    ids = [str(row.get("prompt_id")) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate prompt IDs")
    return records, manifest_hash


def _validate_test_freeze(
    config_path: Path,
    protocol: dict[str, Any],
    manifest_hash: str,
    confirm_test: bool,
) -> dict[str, Any]:
    if not confirm_test:
        raise ValueError("Test access requires --confirm-test")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}; freeze the development choices before test access"
        )
    frozen = read_json(config_path)
    if frozen.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("test_frozen.json has the wrong protocol_id")
    for key in protocol["required_before_test"]:
        value = frozen.get(key)
        if value is None or value == "" or value == []:
            raise ValueError(f"test_frozen.json is missing {key}")
    if len(frozen.get("five_matched_random_feature_ids", [])) != 5:
        raise ValueError("test_frozen.json must contain exactly five matched controls")
    selected_feature = int(frozen["selected_feature_id"])
    random_features = [int(value) for value in frozen["five_matched_random_feature_ids"]]
    if not 0 <= selected_feature < int(protocol["artifacts"]["sae"]["width"]):
        raise ValueError("frozen selected feature is outside the SAE width")
    if len(set(random_features)) != 5 or selected_feature in random_features:
        raise ValueError("frozen matched controls must be unique and exclude the target")
    if any(
        not 0 <= value < int(protocol["artifacts"]["sae"]["width"])
        for value in random_features
    ):
        raise ValueError("a frozen matched control is outside the SAE width")
    if frozen.get("selected_domain") not in protocol["target_domain_taxonomy"]:
        raise ValueError("frozen selected domain is outside the taxonomy")
    if frozen.get("domain_classifier_revision") != protocol["domain_labeling"][
        "classifier_revision"
    ]:
        raise ValueError("frozen domain classifier revision has drifted")
    if frozen.get("power_report_sha256") != protocol["power"][
        "power_report_sha256"
    ]:
        raise ValueError("frozen power-report hash has drifted")
    quality = protocol["outcomes"]["quality_guardrail"]
    if (
        frozen.get("judge_model_or_rater_protocol") != quality["judge_protocol_id"]
        or frozen.get("judge_prompt_sha256") != quality["rubric_sha256"]
    ):
        raise ValueError("frozen quality judge or rubric has drifted")
    if not frozen.get("frozen_at_utc"):
        raise ValueError("test_frozen.json must record frozen_at_utc")
    gate_path = Path(str(frozen["development_gate_report_path"]))
    if not gate_path.is_absolute():
        gate_path = ROOT / gate_path
    if not gate_path.exists():
        raise FileNotFoundError(f"frozen development-gate report is missing: {gate_path}")
    if sha256_file(gate_path) != frozen["development_gate_report_sha256"]:
        raise ValueError("frozen development-gate report hash has drifted")
    gate_report = read_json(gate_path)
    gate = gate_report.get("development_intervention_gate")
    if (
        gate_report.get("protocol_id") != protocol["protocol_id"]
        or gate_report.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
        or gate_report.get("split") != "development"
        or not isinstance(gate, dict)
        or gate.get("status") != "pass"
    ):
        raise ValueError("test access requires a passing development-gate report")
    if (
        gate_report.get("selected_domain") != frozen["selected_domain"]
        or int(gate_report.get("selected_feature_id", -1)) != selected_feature
        or [int(value) for value in gate_report.get("matched_random_feature_ids", [])]
        != random_features
        or not math.isclose(
            float(gate_report.get("eligible_activation_threshold")),
            float(frozen["eligible_activation_threshold"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("test freeze choices differ from the passing development gate")
    if frozen.get("prompt_manifest_sha256") != manifest_hash:
        raise ValueError("Frozen prompt-manifest hash does not match the current manifest")
    current_commit, dirty = git_state()
    if current_commit != frozen.get("generation_code_commit"):
        raise ValueError("Current commit differs from frozen generation_code_commit")
    if dirty:
        raise ValueError("Test generation requires a clean worktree")
    lock_path = ROOT / "uv.lock"
    if not lock_path.exists() or sha256_file(lock_path) != frozen.get(
        "environment_lock_sha256"
    ):
        raise ValueError("Current uv.lock differs from the frozen environment hash")
    return frozen


def _condition_settings(
    protocol: dict[str, Any],
    condition: str,
    feature_id: int | None,
    strength: float,
    promotion_target: float | None,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    generation = protocol["generation"]
    decoding = {
        "do_sample": True,
        "max_new_tokens": int(generation["max_new_tokens"]),
        "temperature": float(generation["baseline_temperature"]),
        "top_p": float(generation["baseline_top_p"]),
        "batch_size": 1,
    }
    if condition == "higher_temperature":
        decoding["temperature"] = float(generation["higher_temperature"])
        decoding["top_p"] = float(generation["higher_temperature_top_p"])

    intervention: dict[str, Any] | None = None
    if condition in {
        "targeted_feature_suppression",
        "matched_random_feature_suppression",
    }:
        intervention = {
            "mode": "suppress",
            "feature_id": feature_id,
            "strength": strength,
            "promotion_target": None,
            "noise_seed": seed,
        }
    elif condition == "l2_matched_activation_noise":
        intervention = {
            "mode": "noise",
            "feature_id": feature_id,
            "strength": strength,
            "promotion_target": None,
            "noise_seed": seed,
        }
    elif condition == "targeted_feature_promotion_secondary":
        if promotion_target is None:
            raise ValueError("Promotion requires --promotion-target")
        intervention = {
            "mode": "promote",
            "feature_id": feature_id,
            "strength": strength,
            "promotion_target": promotion_target,
            "noise_seed": seed,
        }
    return decoding, intervention


def _set_runtime_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bundle_input_device(bundle: Any) -> Any:
    try:
        return bundle.model.get_input_embeddings().weight.device
    except AttributeError:
        return next(bundle.model.parameters()).device


def _generate_real(
    bundle: Any,
    prompt_text: str,
    decoding: dict[str, Any],
    intervention_data: dict[str, Any] | None,
    seed: int,
) -> tuple[str, list[int], int, dict[str, Any]]:
    import torch

    try:
        from .intervene import InterventionSpec, intervention_context
    except ImportError:
        from intervene import InterventionSpec, intervention_context

    _set_runtime_seed(seed)
    tokenizer = bundle.tokenizer
    messages = [{"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = _bundle_input_device(bundle)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])

    hook_context: Any = contextlib.nullcontext(None)
    if intervention_data is not None:
        spec = InterventionSpec(**intervention_data)
        hook_context = intervention_context(
            bundle.model,
            bundle.sae,
            spec,
            layer_index=bundle.layer_index,
        )
    editor = None
    with hook_context as installed_editor:
        editor = installed_editor
        with torch.inference_mode():
            output_ids = bundle.model.generate(
                **inputs,
                do_sample=bool(decoding["do_sample"]),
                max_new_tokens=int(decoding["max_new_tokens"]),
                temperature=float(decoding["temperature"]),
                top_p=float(decoding["top_p"]),
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
    generated_ids = output_ids[0, input_length:].detach().cpu().tolist()
    hook_calls = int(editor.call_count) if editor is not None else 0
    if editor is not None and intervention_data is not None:
        if float(intervention_data["strength"]) > 0 and hook_calls != len(generated_ids):
            raise RuntimeError(
                f"Intervention hook ran {hook_calls} times for "
                f"{len(generated_ids)} generated tokens"
            )
    hook_diagnostics = {
        "hook_calls": hook_calls,
        "expected_hook_calls": len(generated_ids) if editor is not None else 0,
        "last_feature_activation": (
            editor.last_feature_activation.detach().float().cpu().reshape(-1).tolist()
            if editor is not None and editor.last_feature_activation is not None
            else None
        ),
        "last_delta_norm": (
            editor.last_delta_norm.detach().float().cpu().reshape(-1).tolist()
            if editor is not None and editor.last_delta_norm is not None
            else None
        ),
        "last_requested_delta_norm": (
            editor.last_requested_delta_norm.detach().float().cpu().reshape(-1).tolist()
            if editor is not None and editor.last_requested_delta_norm is not None
            else None
        ),
    }
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return (
        text,
        [int(token) for token in generated_ids],
        input_length,
        hook_diagnostics,
    )


def _zero_strength_preflight(bundle: Any) -> dict[str, Any]:
    """Prove once per real process that a zero-strength hook preserves logits."""

    import torch

    try:
        from .intervene import InterventionSpec, intervention_context
    except ImportError:
        from intervene import InterventionSpec, intervention_context

    text = "Return exactly OK."
    inputs = bundle.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = _bundle_input_device(bundle)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    spec = InterventionSpec(mode="suppress", feature_id=0, strength=0.0)
    with torch.inference_mode():
        plain = bundle.model(**inputs, use_cache=False).logits
        with intervention_context(
            bundle.model,
            bundle.sae,
            spec,
            layer_index=bundle.layer_index,
        ):
            hooked = bundle.model(**inputs, use_cache=False).logits
    exact = bool(torch.equal(plain, hooked))
    maximum_difference = float((plain.float() - hooked.float()).abs().max().item())
    if not exact:
        raise RuntimeError(
            "Zero-strength intervention changed logits "
            f"(max absolute difference {maximum_difference})"
        )
    return {
        "prompt_sha256": sha256_bytes(text.encode("utf-8")),
        "exact_logits_equal": exact,
        "max_absolute_logit_difference": maximum_difference,
        "feature_id": 0,
        "strength": 0.0,
    }


def _dry_generation(prompt_id: str, sample_index: int, condition: str) -> str:
    payload = {
        "target_domain": "other",
        "target_system": f"offline dry-run {prompt_id} sample {sample_index}",
        "mappings": [
            {
                "source_role": "source element",
                "target_role": "placeholder element",
                "shared_relation": "dry-run contract only",
            }
        ]
        * 3,
        "explanation": f"No model was loaded for condition {condition}.",
        "limitations": ["Synthetic dry-run record", "Not valid evidence"],
    }
    return json.dumps(payload, sort_keys=True)


def _meta_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".meta.json")


def _load_resume(
    output: Path, meta_path: Path, run_id: str, resume: bool, overwrite: bool
) -> list[dict[str, Any]]:
    if not output.exists() and not meta_path.exists():
        return []
    if overwrite:
        return []
    if not resume:
        raise FileExistsError(f"{output} already exists; use --resume or --overwrite")
    if meta_path.exists() and not output.exists():
        metadata = read_json(meta_path)
        if metadata.get("run_id") != run_id:
            raise ValueError("Existing metadata belongs to a different run configuration")
        return []
    if not output.exists() or not meta_path.exists():
        raise ValueError("Generation output and metadata sidecar are inconsistent")
    metadata = read_json(meta_path)
    if metadata.get("run_id") != run_id:
        raise ValueError("Existing output belongs to a different run configuration")
    records = read_jsonl(output)
    if any(record.get("run_id") != run_id for record in records):
        raise ValueError("Existing JSONL mixes generation runs")
    return records


def _register_test_run(
    *,
    config_path: Path,
    run_id: str,
    condition: str,
    feature_id: int | None,
    matched_random_index: int,
    strength: float,
    promotion_target: float | None,
    output: Path,
) -> None:
    """Atomically reserve one output path for each frozen confirmatory arm."""

    config_hash = sha256_file(config_path)
    arm_key = canonical_hash(
        {
            "condition": condition,
            "feature_id": feature_id,
            "matched_random_index": (
                matched_random_index
                if condition == "matched_random_feature_suppression"
                else None
            ),
            "strength": strength,
            "promotion_target": promotion_target,
        }
    )
    ledger = (
        read_json(TEST_LEDGER)
        if TEST_LEDGER.exists()
        else {
            "schema_version": 1,
            "record_type": "confirmatory_generation_ledger",
            "test_config_sha256": config_hash,
            "arms": {},
        }
    )
    if ledger.get("test_config_sha256") != config_hash:
        raise ValueError(
            "confirmatory ledger belongs to a different test_frozen.json; preserve "
            "the original test run rather than replacing it"
        )
    arms = ledger.setdefault("arms", {})
    existing = arms.get(arm_key)
    reservation = {
        "run_id": run_id,
        "condition": condition,
        "feature_id": feature_id,
        "matched_random_index": (
            matched_random_index
            if condition == "matched_random_feature_suppression"
            else None
        ),
        "strength": strength,
        "promotion_target": promotion_target,
        "output": str(output.resolve()),
    }
    if existing is not None and existing != reservation:
        raise ValueError(
            "this confirmatory arm is already reserved; only resume its exact run_id "
            "and output path"
        )
    if existing is None:
        arms[arm_key] = reservation
        atomic_write_json(TEST_LEDGER, ledger)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--samples-per-prompt", type=int)
    parser.add_argument("--feature-id", type=int)
    parser.add_argument("--matched-random-index", type=int, default=0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--promotion-target", type=float)
    parser.add_argument("--selection-config", type=Path)
    parser.add_argument(
        "--development-plan",
        type=Path,
        help="passing discover_feature.py artifact; selects the frozen 24 gate prompts",
    )
    parser.add_argument("--test-config", type=Path, default=DEFAULT_TEST_CONFIG)
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--prompt-id", action="append", default=[])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment-variable name containing the Hugging Face token",
    )
    args = parser.parse_args()

    if args.overwrite and args.resume:
        args.resume = False
    if args.limit_prompts is not None and args.limit_prompts < 1:
        raise ValueError("--limit-prompts must be positive")
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must be between zero and one")

    protocol = read_json(args.protocol)
    if args.condition not in protocol["conditions"]:
        raise ValueError(f"Unknown condition: {args.condition}")
    prompts, manifest_hash = _validate_manifest(args.manifest, protocol, args.split)
    development_plan: dict[str, Any] = {}
    if args.development_plan:
        if args.split != "development":
            raise ValueError("--development-plan is valid only for development generation")
        if args.limit_prompts is not None or args.prompt_id:
            raise ValueError("development-plan prompts cannot be overridden")
        development_plan = read_json(args.development_plan)
        if (
            development_plan.get("protocol_id") != protocol["protocol_id"]
            or development_plan.get("protocol_sha256") != sha256_file(DEFAULT_PROTOCOL)
            or development_plan.get("development_gate_ready") is not True
        ):
            raise ValueError("development discovery artifact is not gate-ready")
    if args.split == "test":
        if args.protocol.resolve() != DEFAULT_PROTOCOL.resolve():
            raise ValueError("confirmatory generation requires the repository protocol")
        if args.limit_prompts is not None or args.prompt_id:
            raise ValueError("Confirmatory test generation must include all frozen prompts")
        if args.overwrite:
            raise ValueError("Confirmatory test outputs cannot be overwritten; use resume")
        if args.feature_id is not None:
            raise ValueError("Test feature IDs must come from test_frozen.json")
        if args.selection_config is not None:
            raise ValueError("Test choices must come only from test_frozen.json")
        if args.promotion_target is not None:
            raise ValueError("Test promotion target must come from test_frozen.json")
        if args.strength != 1.0:
            raise ValueError("Confirmatory test intervention strength is frozen at 1.0")
        if args.dry_run:
            raise ValueError("Use development --dry-run; confirmatory test runs must be real")
    selected_ids = (
        {
            str(value)
            for value in development_plan.get("development_gate_plan", {}).get(
                "prompt_ids", []
            )
        }
        if development_plan
        else set(args.prompt_id)
    )
    if development_plan and len(selected_ids) != int(
        protocol["development_intervention_gate"]["prompt_count"]
    ):
        raise ValueError("development plan does not contain exactly 24 gate prompts")
    if selected_ids:
        prompts = [row for row in prompts if row["prompt_id"] in selected_ids]
        missing = selected_ids - {row["prompt_id"] for row in prompts}
        if missing:
            raise ValueError(f"Unknown {args.split} prompt IDs: {sorted(missing)}")
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    if not prompts:
        raise ValueError("No prompts selected")

    selection_path = args.selection_config
    if args.split == "test":
        frozen = _validate_test_freeze(
            args.test_config, protocol, manifest_hash, args.confirm_test
        )
        if selection_path is None:
            selection_path = args.test_config
        selection = frozen
    else:
        if development_plan and selection_path is not None and (
            selection_path.resolve() != args.development_plan.resolve()
        ):
            raise ValueError("development gate choices must come from its discovery plan")
        if development_plan:
            selection_path = args.development_plan
        selection = _selection_config(selection_path)

    promotion_target = args.promotion_target
    if promotion_target is None and selection.get("promotion_target") is not None:
        promotion_target = float(selection["promotion_target"])

    feature_id = _resolve_feature(
        args.condition,
        args.feature_id,
        args.matched_random_index,
        selection,
    )
    generation = protocol["generation"]
    if args.split == "development" and development_plan:
        default_samples = protocol["development_intervention_gate"][
            "paired_samples_per_prompt"
        ]
    elif args.split == "development" and args.condition == "baseline":
        default_samples = generation["development_baseline_samples_per_prompt"]
    elif args.split == "development":
        default_samples = protocol["development_intervention_gate"][
            "paired_samples_per_prompt"
        ]
    else:
        default_samples = generation["test_paired_samples_per_prompt_per_condition"]
    samples_per_prompt = args.samples_per_prompt or int(default_samples)
    if samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be positive")
    if args.split == "test" and samples_per_prompt != int(default_samples):
        raise ValueError("Confirmatory test sample count differs from the frozen protocol")

    protocol_hash = sha256_file(args.protocol)
    commit, dirty = git_state()
    implementation_hashes = {
        name: sha256_file(ROOT / "latent_escape" / name)
        for name in ("generate.py", "model_sae.py", "intervene.py")
        if (ROOT / "latent_escape" / name).exists()
    }
    run_spec = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "manifest_sha256": manifest_hash,
        "split": args.split,
        "condition": args.condition,
        "samples_per_prompt": samples_per_prompt,
        "prompt_ids": [row["prompt_id"] for row in prompts],
        "feature_id": feature_id,
        "matched_random_index": (
            args.matched_random_index
            if args.condition == "matched_random_feature_suppression"
            else None
        ),
        "strength": args.strength,
        "promotion_target": promotion_target,
        "seed_base": generation["seed_base"],
        "model_revision": protocol["artifacts"]["model"]["revision"],
        "sae_revision": protocol["artifacts"]["sae"]["revision"],
        "implementation_sha256": implementation_hashes,
        "dry_run": args.dry_run,
        "test_config_sha256": (
            sha256_file(args.test_config) if args.split == "test" else None
        ),
        "development_plan_sha256": (
            sha256_file(args.development_plan) if args.development_plan else None
        ),
    }
    run_id = canonical_hash(run_spec)
    suffix = ".dry-run.jsonl" if args.dry_run else ".jsonl"
    strength_suffix = (
        f"-s{filename_number(args.strength)}" if args.strength != 1.0 else ""
    )
    output_prefix = (
        f"{args.split}_gate" if development_plan else f"{args.split}"
    )
    output = args.output or (
        OUTPUT_DIR
        / f"{output_prefix}_{args.condition}"
        f"{('-f' + str(feature_id)) if feature_id is not None else ''}"
        f"{strength_suffix}{suffix}"
    )
    if args.split == "test":
        _register_test_run(
            config_path=args.test_config,
            run_id=run_id,
            condition=args.condition,
            feature_id=feature_id,
            matched_random_index=args.matched_random_index,
            strength=args.strength,
            promotion_target=promotion_target,
            output=output,
        )
    meta_path = _meta_path(output)
    records = _load_resume(output, meta_path, run_id, args.resume, args.overwrite)
    completed = {
        (row["prompt_id"], int(row["sample_index"])) for row in records
    }
    expected_keys = {
        (row["prompt_id"], sample_index)
        for row in prompts
        for sample_index in range(samples_per_prompt)
    }
    if not completed <= expected_keys:
        raise ValueError("Existing output contains unexpected prompt/sample keys")

    metadata = {
        **run_spec,
        "record_type": "generation_run",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_code_commit": commit,
        "git_dirty_at_start": dirty,
        "output": str(output),
        "analysis_unit": "source prompt; samples are paired within prompt",
    }
    if not meta_path.exists() or args.overwrite:
        atomic_write_json(meta_path, metadata)

    bundle = None
    zero_strength_preflight = None
    if not args.dry_run and completed != expected_keys:
        try:
            from .model_sae import load_pinned_bundle
        except ImportError:
            from model_sae import load_pinned_bundle

        token = os.environ.get(args.hf_token_env)
        bundle = load_pinned_bundle(
            token=token,
            device_map={"": "cuda:0"},
            local_files_only=args.local_files_only,
        )
        if str(bundle.model_revision) != run_spec["model_revision"]:
            raise ValueError("Loaded model revision differs from the protocol")
        if str(bundle.sae_revision) != run_spec["sae_revision"]:
            raise ValueError("Loaded SAE revision differs from the protocol")
        zero_strength_preflight = _zero_strength_preflight(bundle)
        metadata["zero_strength_preflight"] = zero_strength_preflight
        import torch

        metadata["runtime"] = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(),
            "device_map": {"": "cuda:0"},
        }
        atomic_write_json(meta_path, metadata)

    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        prompt_text = effective_prompt(prompt["prompt_text"], args.condition, protocol)
        for sample_index in range(samples_per_prompt):
            key = (prompt_id, sample_index)
            if key in completed:
                continue
            seed = paired_seed(generation["seed_base"], prompt_id, sample_index)
            decoding, intervention_data = _condition_settings(
                protocol,
                args.condition,
                feature_id,
                args.strength,
                promotion_target,
                seed,
            )
            if args.dry_run:
                generated_text = _dry_generation(
                    prompt_id, sample_index, args.condition
                )
                generated_token_ids: list[int] = []
                input_token_count = None
                hook_diagnostics = {
                    "hook_calls": 0,
                    "expected_hook_calls": 0,
                    "last_feature_activation": None,
                    "last_delta_norm": None,
                    "last_requested_delta_norm": None,
                }
            else:
                assert bundle is not None
                (
                    generated_text,
                    generated_token_ids,
                    input_token_count,
                    hook_diagnostics,
                ) = _generate_real(
                    bundle, prompt_text, decoding, intervention_data, seed
                )
            json_valid, parsed_output = parse_generated_json(generated_text)
            record = {
                "schema_version": 1,
                "record_type": "generation",
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_hash,
                "manifest_sha256": manifest_hash,
                "run_id": run_id,
                "prompt_id": prompt_id,
                "cluster_id": prompt_id,
                "split": args.split,
                "condition": args.condition,
                "sample_index": sample_index,
                "seed": seed,
                "generation_id": generation_id(prompt_id, sample_index),
                "prompt_text_sha256": sha256_bytes(prompt_text.encode("utf-8")),
                "generated_text": generated_text,
                "parsed_output": parsed_output,
                "json_syntax_valid": parsed_output is not None,
                "json_valid": json_valid,
                "analogy_schema_valid": json_valid,
                "model_revision": run_spec["model_revision"],
                "sae_revision": run_spec["sae_revision"],
                "feature_id": feature_id,
                "intervention": intervention_data,
                "decoding": decoding,
                "input_token_count": input_token_count,
                "generated_token_ids": generated_token_ids,
                "hook_diagnostics": hook_diagnostics,
                "zero_strength_preflight_passed": (
                    bool(zero_strength_preflight["exact_logits_equal"])
                    if zero_strength_preflight is not None
                    else None
                ),
                "dry_run": args.dry_run,
            }
            records.append(record)
            completed.add(key)
        # A crash can lose at most the current prompt; completed prompts are atomic.
        records.sort(key=lambda row: (row["prompt_id"], int(row["sample_index"])))
        atomic_write_jsonl(output, records)
        print(
            f"{prompt_id}: {sum(key[0] == prompt_id for key in completed)}/"
            f"{samples_per_prompt} samples complete"
        )

    print(f"wrote {len(records)} records to {output}")
    print(f"run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
