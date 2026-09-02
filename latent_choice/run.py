#!/usr/bin/env python3
"""Run the development-only explicit-domain baseline for Latent Choice.

Each prompt receives exactly one model forward.  The output records the full
18-way restricted choice distribution and paired inverse-CDF choices; the
prompt-level SAE activation matrix is written separately.  This command cannot
open the confirmatory test split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Iterable

import numpy as np
import torch

from latent_choice.choice_endpoint import (
    CODE_SYMBOLS,
    DOMAINS,
    inverse_cdf_domain,
    mapping_for_prompt,
    paired_uniform,
    render_choice_prompt,
    resolve_code_token_ids,
    score_choice_logits,
)
from latent_escape.generate import atomic_write_json, atomic_write_jsonl, git_state, sha256_file
from latent_escape.model_sae import (
    MODEL_REPO_ID,
    MODEL_REVISION,
    SAE_REPO_ID,
    SAE_REVISION,
    SAE_SHA256,
    load_pinned_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "latent_choice" / "protocol.json"
DEFAULT_TEMPLATE = ROOT / "latent_choice" / "test_config.template.json"
DEFAULT_CODE_TOKENS = ROOT / "latent_choice" / "code_token_manifest.json"
DEFAULT_MANIFEST = ROOT / "latent_escape" / "artifacts" / "prompt_manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "latent_choice" / "outputs" / "development_choice_baseline.jsonl"
DEFAULT_ACTIVATIONS = (
    ROOT / "latent_choice" / "outputs" / "development_choice_baseline.activations.npz"
)
DEFAULT_DRY_OUTPUT = (
    ROOT / "latent_choice" / "outputs" / "development_choice_baseline.dry-run.jsonl"
)
DEFAULT_DRY_ACTIVATIONS = (
    ROOT
    / "latent_choice"
    / "outputs"
    / "development_choice_baseline.dry-run.activations.npz"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_canonical_protocol(protocol_path: Path, protocol: dict[str, Any]) -> None:
    if protocol_path.resolve() != DEFAULT_PROTOCOL.resolve():
        raise ValueError("evidentiary runs require the canonical latent_choice/protocol.json")
    from latent_choice.validate_protocol import load_json, validate

    validate(
        protocol,
        load_json(DEFAULT_TEMPLATE),
        load_json(DEFAULT_CODE_TOKENS),
    )


def _load_code_token_manifest(
    path: Path, *, protocol_path: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    manifest = _read_json(path)
    if manifest.get("artifact") != "latent_choice_code_token_manifest":
        raise ValueError("wrong choice-code token manifest artifact")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("choice-code manifest protocol mismatch")
    if manifest.get("protocol_revision") != protocol["protocol_revision"]:
        raise ValueError("choice-code manifest protocol revision drift")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or tokenizer.get("repo_id") != MODEL_REPO_ID or tokenizer.get(
        "revision"
    ) != MODEL_REVISION:
        raise ValueError("choice-code manifest model drift")
    raw_ids = manifest.get("code_token_ids")
    if not isinstance(raw_ids, dict):
        raise ValueError("choice-code manifest lacks code_token_ids")
    resolved = {str(code): int(token_id) for code, token_id in raw_ids.items()}
    if set(resolved) != set(CODE_SYMBOLS) or len(set(resolved.values())) != len(
        resolved
    ):
        raise ValueError("choice-code manifest codes or token IDs are invalid")
    expected_hash = protocol["artifacts"]["choice_code_tokens"]["sha256"]
    if sha256_file(path) != expected_hash:
        raise ValueError("choice-code manifest hash drift")
    return manifest, resolved


def _development_rows(path: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    expected_hash = protocol["stimuli"]["prompt_manifest_sha256"]
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"prompt manifest hash {observed_hash} does not match {expected_hash}"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")
        if value.get("split") == "development":
            rows.append(value)
    expected_count = int(protocol["stimuli"]["development_count"])
    if len(rows) != expected_count:
        raise ValueError(f"found {len(rows)} development prompts, expected {expected_count}")
    prompt_ids = [str(row.get("prompt_id", "")) for row in rows]
    if any(not prompt_id for prompt_id in prompt_ids) or len(prompt_ids) != len(
        set(prompt_ids)
    ):
        raise ValueError("development prompt IDs must be nonempty and unique")
    return rows


def _hidden_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        tensor = next((value for value in output if isinstance(value, torch.Tensor)), None)
        if tensor is not None:
            return tensor
    if isinstance(output, dict):
        tensor = next(
            (value for value in output.values() if isinstance(value, torch.Tensor)), None
        )
        if tensor is not None:
            return tensor
    raise TypeError(f"unsupported decoder output type: {type(output).__name__}")


@torch.no_grad()
def _score_real_prompt(bundle: Any, prompt: str) -> tuple[Any, np.ndarray, int]:
    """Perform exactly one clean choice forward and capture its pre-choice SAE vector."""

    captured: list[torch.Tensor] = []

    def capture_hook(module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        hidden = _hidden_tensor(output)
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("choice endpoint requires batch-one decoder residuals")
        captured.append(bundle.sae.encode(hidden[:, -1, :]).detach().cpu())

    handle = bundle.layer.register_forward_hook(capture_hook)
    try:
        encoded = bundle.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        device = bundle.model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = bundle.model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"choice capture hook ran {len(captured)} times, expected one")
    activation = captured[0]
    if activation.shape != (1, 16384):
        raise RuntimeError(f"unexpected SAE activation shape {tuple(activation.shape)}")
    if not torch.isfinite(activation).all().item() or (activation < 0).any().item():
        raise RuntimeError("choice-position SAE activations must be finite and nonnegative")
    return output.logits[0, -1].detach().cpu(), activation[0].numpy(), 1


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _synthetic_logits(mapping: dict[str, str], prompt_id: str) -> tuple[torch.Tensor, dict[str, int]]:
    code_token_ids = {code: 20 + index for index, code in enumerate(CODE_SYMBOLS)}
    logits = torch.full((128,), -7.0, dtype=torch.float32)
    offset = int(hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()[:8], 16)
    for index, domain in enumerate(DOMAINS):
        logits[code_token_ids[mapping[domain]]] = float((offset + index * 7) % 23) / 10
    return logits, code_token_ids


def run_baseline(
    *,
    protocol_path: Path,
    manifest_path: Path,
    output_path: Path,
    activations_path: Path,
    dry_run: bool,
    limit_prompts: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    if protocol.get("protocol_id") != "latent-choice-v1":
        raise ValueError("wrong Latent Choice protocol")
    if not dry_run:
        _validate_canonical_protocol(protocol_path, protocol)
        if overwrite:
            raise ValueError("evidentiary outputs are immutable; --overwrite is dry-run only")
        pre_run_commit, pre_run_dirty = git_state()
        if pre_run_commit is None or pre_run_dirty is not False:
            raise ValueError("an evidentiary baseline requires a clean Git worktree")
    else:
        pre_run_commit, pre_run_dirty = git_state()
    rows = _development_rows(manifest_path, protocol)
    if limit_prompts is not None:
        if not dry_run:
            raise ValueError("--limit-prompts is allowed only for a synthetic dry run")
        if not 1 <= limit_prompts <= len(rows):
            raise ValueError("--limit-prompts is outside the development split")
        rows = rows[:limit_prompts]
    if (output_path.exists() or activations_path.exists()) and not overwrite:
        raise FileExistsError("output exists; pass --overwrite only for a non-evidentiary rerun")

    full_development_ids = [
        str(row["prompt_id"]) for row in _development_rows(manifest_path, protocol)
    ]
    code_manifest: dict[str, Any] | None = None
    expected_token_ids: dict[str, int] | None = None
    prepared_prompts: dict[str, tuple[str, dict[str, int]]] = {}
    if not dry_run:
        code_manifest, expected_token_ids = _load_code_token_manifest(
            DEFAULT_CODE_TOKENS, protocol_path=protocol_path, protocol=protocol
        )
    bundle = None if dry_run else load_pinned_bundle(token=os.environ.get("HF_TOKEN"))
    if not dry_run:
        assert bundle is not None
        # Complete all prompt/token-boundary validation before the first model
        # forward so a late tokenizer mismatch cannot create a partial run.
        for row in rows:
            prompt_id = str(row["prompt_id"])
            mapping = mapping_for_prompt(
                prompt_id,
                full_development_ids,
                seed=str(protocol["choice_endpoint"]["mapping_seed"]),
            )
            prompt = render_choice_prompt(
                bundle.tokenizer,
                source_name=str(row["source_name"]),
                source_domain=str(row["source_domain"]),
                source_description=str(row["source_description"]),
                mapping=mapping,
            )
            observed_token_ids = resolve_code_token_ids(bundle.tokenizer, prompt)
            if observed_token_ids != expected_token_ids:
                raise ValueError(
                    f"choice-code token IDs drifted for development prompt {prompt_id}"
                )
            prepared_prompts[prompt_id] = (prompt, observed_token_ids)
    records: list[dict[str, Any]] = []
    activations: list[np.ndarray] = []
    draws_per_prompt = int(protocol["choice_endpoint"]["paired_draws_per_prompt"])

    for row in rows:
        prompt_id = str(row["prompt_id"])
        mapping = mapping_for_prompt(
            prompt_id,
            full_development_ids,
            seed=str(protocol["choice_endpoint"]["mapping_seed"]),
        )
        if dry_run:
            prompt = None
            logits, token_ids = _synthetic_logits(mapping, prompt_id)
            activation = np.zeros(16384, dtype=np.float32)
            hook_calls = 0
        else:
            assert bundle is not None
            prompt, token_ids = prepared_prompts[prompt_id]
            logits, activation, hook_calls = _score_real_prompt(bundle, prompt)
        score = score_choice_logits(logits, mapping=mapping, code_token_ids=token_ids)
        realized = []
        for sample_index in range(draws_per_prompt):
            uniform = paired_uniform(
                prompt_id,
                sample_index,
                seed=str(protocol["choice_endpoint"]["paired_draw_seed"]),
            )
            realized.append(
                {
                    "sample_index": sample_index,
                    "uniform": uniform,
                    "declared_domain": inverse_cdf_domain(
                        score.domain_probabilities, uniform=uniform
                    ),
                }
            )
        prompt_hash = (
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt is not None else None
        )
        activation_array = np.asarray(activation, dtype="<f4")
        activation_sha256 = hashlib.sha256(activation_array.tobytes()).hexdigest()
        records.append(
            {
                "schema_version": "1.0",
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": sha256_file(protocol_path),
                "record_type": "latent_choice_baseline",
                "evidentiary": not dry_run,
                "run_mode": "synthetic_dry_run" if dry_run else "real_model",
                "split": "development",
                "condition": "baseline",
                "feature_id": None,
                "strength": 0.0,
                "prompt_id": prompt_id,
                "choice_prompt_sha256": prompt_hash,
                "domain_to_code": mapping,
                **score.to_dict(),
                "activation_row_index": len(activations),
                "activation_sha256": activation_sha256,
                "paired_realized_choices": realized,
                "decision_forward_count": 1 if not dry_run else 0,
                "capture_hook_calls": hook_calls,
                "intervention_hook_calls": 0,
            }
        )
        activations.append(activation_array)

    activation_matrix = np.stack(activations, axis=0)
    prompt_ids_array = np.asarray([record["prompt_id"] for record in records])
    atomic_write_jsonl(output_path, records)
    _atomic_npz(
        activations_path,
        prompt_ids=prompt_ids_array,
        activations=activation_matrix,
    )
    commit, dirty = git_state()
    masses = [float(record["full_vocab_candidate_mass"]) for record in records]
    compliance = protocol["choice_measurement"]["candidate_mass_compliance"]
    per_prompt_floor = float(compliance["per_prompt_floor"])
    minimum_fraction = float(compliance["minimum_prompt_fraction"])
    passing_fraction = sum(mass >= per_prompt_floor for mass in masses) / len(masses)
    compliance_pass = passing_fraction >= minimum_fraction
    lock_path = ROOT / "uv.lock"
    report = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "split": "development",
        "condition": "baseline",
        "evidentiary": not dry_run,
        "run_mode": "synthetic_dry_run" if dry_run else "real_model",
        "prompt_count": len(records),
        "decision_forward_count": 0 if dry_run else len(records),
        "activation_shape": list(activation_matrix.shape),
        "output_path": _display_path(output_path),
        "output_sha256": sha256_file(output_path),
        "activations_path": _display_path(activations_path),
        "activations_sha256": sha256_file(activations_path),
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "sae_repo_id": SAE_REPO_ID,
        "sae_revision": SAE_REVISION,
        "sae_sha256": SAE_SHA256,
        "environment_lock_sha256": sha256_file(lock_path),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if not dry_run else None,
        },
        "choice_code_token_manifest_path": (
            _display_path(DEFAULT_CODE_TOKENS) if code_manifest is not None else None
        ),
        "choice_code_token_manifest_sha256": (
            sha256_file(DEFAULT_CODE_TOKENS) if code_manifest is not None else None
        ),
        "choice_code_token_ids": expected_token_ids,
        "candidate_mass_compliance": {
            "per_prompt_floor": per_prompt_floor,
            "minimum_prompt_fraction": minimum_fraction,
            "observed_prompt_fraction": passing_fraction,
            "minimum_observed_mass": min(masses),
            "median_observed_mass": float(np.median(np.asarray(masses))),
            "status": "pass" if compliance_pass else "stop",
        },
        "feature_discovery_allowed": bool(not dry_run and compliance_pass),
        "pre_run_git_commit": pre_run_commit,
        "pre_run_working_tree_dirty": pre_run_dirty,
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "test_split_generated": False,
    }
    report_path = output_path.with_suffix(".report.json")
    atomic_write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("baseline",))
    parser.add_argument("--split", default="development", choices=("development", "test"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split != "development":
        raise SystemExit(
            "Latent Choice test access is disabled until a passing development gate "
            "and immutable test_frozen.json exist"
        )
    output_path = args.output
    activations_path = args.activations
    if args.dry_run:
        if output_path == DEFAULT_OUTPUT:
            output_path = DEFAULT_DRY_OUTPUT
        if activations_path == DEFAULT_ACTIVATIONS:
            activations_path = DEFAULT_DRY_ACTIVATIONS
    report = run_baseline(
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        output_path=output_path,
        activations_path=activations_path,
        dry_run=bool(args.dry_run),
        limit_prompts=args.limit_prompts,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
