#!/usr/bin/env python3
"""Capture prompt-level and pre-domain SAE activations for feature discovery.

The dense NPZ has one and only one row per source prompt.  Eight generations
from a prompt are outcomes attached to that prompt, never eight independent
feature observations.  A companion JSONL nests sparse per-generation
pre-domain activations under their prompt cluster.  The model's self-reported
``target_domain`` field is used only to locate a token boundary; it is never
used as the domain label.

``--dry-run`` creates deterministic synthetic arrays and validates the full
file/resume contract without importing or downloading model dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .generate import (
        DEFAULT_MANIFEST,
        DEFAULT_PROTOCOL,
        atomic_write,
        atomic_write_json,
        atomic_write_jsonl,
        canonical_hash,
        effective_prompt,
        git_state,
        read_json,
        read_jsonl,
        sha256_bytes,
        sha256_file,
        _validate_test_freeze,
    )
except ImportError:  # Support ``python latent_escape/capture_activations.py``.
    from generate import (
        DEFAULT_MANIFEST,
        DEFAULT_PROTOCOL,
        atomic_write,
        atomic_write_json,
        atomic_write_jsonl,
        canonical_hash,
        effective_prompt,
        git_state,
        read_json,
        read_jsonl,
        sha256_bytes,
        sha256_file,
        _validate_test_freeze,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATIONS = (
    ROOT / "latent_escape" / "outputs" / "generations" / "development_baseline.jsonl"
)
DEFAULT_DEVELOPMENT_OUTPUT = (
    ROOT
    / "latent_escape"
    / "artifacts"
    / "activations"
    / "development_baseline_prompt.npz"
)
TARGET_DOMAIN_FIELD = re.compile(r'"target_domain"\s*:\s*"', re.IGNORECASE)


def _atomic_write_npz(
    path: Path,
    *,
    prompt_ids: list[str],
    activations: np.ndarray,
    decoder_norms: np.ndarray,
    protocol_id: str,
    protocol_sha256: str,
    manifest_sha256: str,
    capture_run_id: str,
    split: str,
    model_revision: str,
    sae_revision: str,
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            prompt_ids=np.asarray(prompt_ids, dtype=np.str_),
            activations=np.asarray(activations, dtype=np.float32),
            decoder_norms=np.asarray(decoder_norms, dtype=np.float32),
            protocol_id=np.asarray(protocol_id, dtype=np.str_),
            protocol_sha256=np.asarray(protocol_sha256, dtype=np.str_),
            manifest_sha256=np.asarray(manifest_sha256, dtype=np.str_),
            capture_run_id=np.asarray(capture_run_id, dtype=np.str_),
            splits=np.asarray([split] * len(prompt_ids), dtype=np.str_),
            capture_split=np.asarray(split, dtype=np.str_),
            model_revision=np.asarray(model_revision, dtype=np.str_),
            sae_revision=np.asarray(sae_revision, dtype=np.str_),
            dry_run=np.asarray(dry_run, dtype=np.bool_),
        )
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _meta_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".meta.json")


def _token_output_path(output: Path) -> Path:
    return output.with_suffix(".pre_domain.jsonl")


def _manifest_by_id(
    path: Path, protocol: dict[str, Any], split: str
) -> tuple[dict[str, dict[str, Any]], str]:
    manifest_hash = sha256_file(path)
    expected = protocol["stimuli"].get("expected_manifest_sha256")
    if expected and manifest_hash != expected:
        raise ValueError(
            f"Manifest SHA-256 is {manifest_hash}; protocol requires {expected}"
        )
    rows = [row for row in read_jsonl(path) if row.get("split") == split]
    expected_count = int(protocol["stimuli"][f"{split}_count"])
    if len(rows) != expected_count:
        raise ValueError(
            f"Manifest has {len(rows)} {split} prompts; expected {expected_count}"
        )
    mapping = {str(row["prompt_id"]): row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("Manifest contains duplicate prompt IDs")
    return mapping, manifest_hash


def _canonical_generation_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record["prompt_id"]),
        str(record["condition"]),
        int(record["sample_index"]),
        int(record["seed"]),
    )


def _load_generations(
    path: Path,
    *,
    protocol: dict[str, Any],
    manifest_hash: str,
    split: str,
    condition: str,
    allow_dry_run: bool,
) -> tuple[dict[str, list[dict[str, Any]]], str, str]:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"No generation records in {path}")
    selected = [
        row
        for row in records
        if row.get("split") == split and row.get("condition") == condition
    ]
    if not selected:
        raise ValueError(f"No {split}/{condition} generation records in {path}")
    keys = [_canonical_generation_key(row) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Generation input contains duplicate canonical keys")
    run_ids = {str(row.get("run_id")) for row in selected}
    if len(run_ids) != 1:
        raise ValueError("Generation input mixes run IDs")
    if any(row.get("protocol_id") != protocol["protocol_id"] for row in selected):
        raise ValueError("Generation input has the wrong protocol_id")
    if any(row.get("manifest_sha256") != manifest_hash for row in selected):
        raise ValueError("Generation input has a different prompt manifest")
    model_revision = protocol["artifacts"]["model"]["revision"]
    sae_revision = protocol["artifacts"]["sae"]["revision"]
    if any(row.get("model_revision") != model_revision for row in selected):
        raise ValueError("Generation input has a different model revision")
    if any(row.get("sae_revision") != sae_revision for row in selected):
        raise ValueError("Generation input has a different SAE revision")
    if any(bool(row.get("dry_run")) for row in selected) and not allow_dry_run:
        raise ValueError("Dry-run generations require capture --dry-run")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row["prompt_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
    return grouped, next(iter(run_ids)), sha256_file(path)


def _dry_sparse_vector(namespace: str, width: int) -> tuple[list[int], list[float]]:
    """Small deterministic positive vector used only to test serialization."""

    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    count = 12
    indices: set[int] = set()
    cursor = 0
    while len(indices) < count:
        block = hashlib.sha256(digest + cursor.to_bytes(4, "big")).digest()
        for offset in range(0, len(block), 2):
            indices.add(int.from_bytes(block[offset : offset + 2], "big") % width)
            if len(indices) == count:
                break
        cursor += 1
    ordered = sorted(indices)
    values = [
        round(0.1 + digest[index % len(digest)] / 255.0, 7)
        for index, _ in enumerate(ordered)
    ]
    return ordered, values


def _dense_from_sparse(
    indices: list[int], values: list[float], width: int
) -> np.ndarray:
    vector = np.zeros(width, dtype=np.float32)
    vector[np.asarray(indices, dtype=np.int64)] = np.asarray(values, dtype=np.float32)
    return vector


def _sparse_from_array(
    vector: np.ndarray, epsilon: float
) -> tuple[list[int], list[float]]:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    active = np.flatnonzero(np.abs(flat) > epsilon)
    return active.astype(int).tolist(), flat[active].astype(float).tolist()


def _extract_hidden(output: Any) -> Any:
    """Extract the residual tensor from a decoder-layer hook output."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - real runs require torch
        raise RuntimeError("Activation capture requires torch") from exc
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, dict):
        for key in ("last_hidden_state", "hidden_states"):
            if isinstance(output.get(key), torch.Tensor):
                return output[key]
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported decoder-layer output type: {type(output).__name__}")


def _input_device(bundle: Any) -> Any:
    try:
        return bundle.model.get_input_embeddings().weight.device
    except AttributeError:
        return next(bundle.model.parameters()).device


def _chat_inputs(bundle: Any, prompt_text: str) -> dict[str, Any]:
    inputs = bundle.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = _input_device(bundle)
    return {key: value.to(device) for key, value in inputs.items()}


def _capture_hidden(bundle: Any, model_inputs: dict[str, Any]) -> Any:
    import torch

    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["hidden"] = _extract_hidden(output).detach()

    handle = bundle.layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            # The base decoder avoids materializing the large vocabulary logits;
            # the layer hook still observes the exact same residual stream.
            base_model = getattr(bundle.model, "model", bundle.model)
            base_model(**model_inputs, use_cache=False)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("Decoder-layer hook did not capture hidden states")
    return captured["hidden"]


def _encode_hidden(bundle: Any, hidden: Any) -> np.ndarray:
    activations = bundle.sae.encode(hidden)
    vector = activations.detach().to(dtype=activations.dtype).float().cpu().numpy()
    return np.asarray(vector, dtype=np.float32).reshape(-1)


def _capture_prompt_vector(bundle: Any, prompt_text: str) -> np.ndarray:
    inputs = _chat_inputs(bundle, prompt_text)
    hidden = _capture_hidden(bundle, inputs)
    return _encode_hidden(bundle, hidden[..., -1, :])


def _domain_token_index(tokenizer: Any, generated_text: str) -> tuple[int | None, str]:
    """Locate the first token of the self-reported field value.

    This is a timing locator only.  The independently blinded classifier's
    label, produced elsewhere, remains the sole analysis outcome.
    """

    match = TARGET_DOMAIN_FIELD.search(generated_text)
    if match is None:
        return None, "target_domain_field_not_found"
    value_start = match.end()
    try:
        encoded = tokenizer(
            generated_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
    except (TypeError, ValueError, NotImplementedError):
        prefix_ids = tokenizer.encode(
            generated_text[:value_start], add_special_tokens=False
        )
        return len(prefix_ids), "prefix_retokenization"
    for index, (start, end) in enumerate(offsets):
        if int(end) > value_start and int(start) <= value_start:
            return index, "self_reported_field_offset"
        if int(start) >= value_start:
            return index, "self_reported_field_offset"
    return None, "target_domain_value_has_no_token"


def _generated_token_ids(bundle: Any, record: dict[str, Any]) -> tuple[list[int], str]:
    stored = record.get("generated_token_ids")
    if isinstance(stored, list) and stored:
        return [int(value) for value in stored], "generation_record"
    values = bundle.tokenizer.encode(
        str(record.get("generated_text", "")), add_special_tokens=False
    )
    return [int(value) for value in values], "retokenized_generated_text"


def _align_generated_ids_at_boundary(
    generated_ids: list[int],
    independently_tokenized: list[int],
    domain_index: int,
    special_ids: set[int],
) -> tuple[list[int] | None, str]:
    """Verify the stored-token index at the domain boundary.

    Decoding and then re-encoding a generated sequence is not guaranteed to
    reproduce tokens after the boundary.  Causal capture only needs the stored
    and offset-tokenized prefixes to agree through the first domain-value token.
    """

    if not 0 <= domain_index < len(independently_tokenized):
        return None, "domain_token_index_out_of_range"
    if generated_ids == independently_tokenized:
        return generated_ids, "exact_match"

    trailing = generated_ids[len(independently_tokenized) :]
    if (
        generated_ids[: len(independently_tokenized)] == independently_tokenized
        and trailing
        and all(value in special_ids for value in trailing)
    ):
        return independently_tokenized, "terminal_specials_stripped"

    required_prefix = domain_index + 1
    if (
        len(generated_ids) >= required_prefix
        and generated_ids[:required_prefix]
        == independently_tokenized[:required_prefix]
    ):
        return generated_ids, "domain_boundary_prefix_verified"
    return None, "stored_token_ids_do_not_match_boundary_tokenization"


def _capture_pre_domain(
    bundle: Any,
    prompt_text: str,
    generation: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    text = str(generation.get("generated_text", ""))
    domain_index, locator = _domain_token_index(bundle.tokenizer, text)
    if domain_index is None:
        return None, {
            "status": "unresolved",
            "locator": locator,
            "outcome_label_source": "none",
        }
    generated_ids, token_source = _generated_token_ids(bundle, generation)
    independently_tokenized = [
        int(value)
        for value in bundle.tokenizer(text, add_special_tokens=False)["input_ids"]
    ]
    special_ids = {
        int(value) for value in getattr(bundle.tokenizer, "all_special_ids", [])
    }
    generated_ids, token_alignment = _align_generated_ids_at_boundary(
        generated_ids,
        independently_tokenized,
        domain_index,
        special_ids,
    )
    if generated_ids is None:
        return None, {
            "status": "unresolved",
            "locator": token_alignment,
            "outcome_label_source": "none",
        }
    if token_alignment != "exact_match":
        token_source += "+" + token_alignment
    if not 0 <= domain_index < len(generated_ids):
        return None, {
            "status": "unresolved",
            "locator": "domain_token_index_out_of_range",
            "outcome_label_source": "none",
        }

    prompt_inputs = _chat_inputs(bundle, prompt_text)
    input_ids = prompt_inputs["input_ids"]
    import torch

    generated_tensor = torch.tensor(
        [generated_ids], dtype=input_ids.dtype, device=input_ids.device
    )
    full_ids = torch.cat((input_ids, generated_tensor), dim=-1)
    attention_mask = torch.ones_like(full_ids)
    hidden = _capture_hidden(
        bundle, {"input_ids": full_ids, "attention_mask": attention_mask}
    )
    pre_position = int(input_ids.shape[-1]) + domain_index - 1
    vector = _encode_hidden(bundle, hidden[..., pre_position, :])
    return vector, {
        "status": "captured",
        "locator": locator,
        "locator_field": "generated target_domain (timing only, never outcome label)",
        "generated_token_source": token_source,
        "boundary_token_alignment": token_alignment,
        "domain_token_index": domain_index,
        "pre_domain_sequence_position": pre_position,
        "outcome_label_source": "independent blinded classifier, joined later",
    }


def _load_existing(
    output: Path,
    token_output: Path,
    meta_path: Path,
    run_id: str,
    width: int,
    resume: bool,
    overwrite: bool,
) -> tuple[list[str], list[np.ndarray], list[dict[str, Any]], np.ndarray | None]:
    exists = (output.exists(), token_output.exists(), meta_path.exists())
    if not any(exists) or overwrite:
        return [], [], [], None
    if not resume:
        raise FileExistsError(f"{output} already exists; use --resume or --overwrite")
    if meta_path.exists() and not all(exists):
        metadata = read_json(meta_path)
        if metadata.get("capture_run_id") != run_id:
            raise ValueError("Existing activation metadata belongs to another run")
        # Atomic replacement prevents corrupt files. If interruption occurred
        # between the two artifact replacements, safely rebuild the last unit.
        return [], [], [], None
    if not all(exists):
        raise ValueError("Activation NPZ, sparse JSONL, and metadata are inconsistent")
    metadata = read_json(meta_path)
    if metadata.get("capture_run_id") != run_id:
        raise ValueError("Existing activation output belongs to another run")
    with np.load(output, allow_pickle=False) as archive:
        prompt_ids = [str(value) for value in archive["prompt_ids"].tolist()]
        activations = np.asarray(archive["activations"], dtype=np.float32)
        decoder_norms = np.asarray(archive["decoder_norms"], dtype=np.float32)
        stored_run_id = str(archive["capture_run_id"].item())
    if stored_run_id != run_id:
        raise ValueError("NPZ capture_run_id does not match its metadata")
    if activations.shape != (len(prompt_ids), width):
        raise ValueError(
            f"Existing activations have shape {activations.shape}; expected "
            f"({len(prompt_ids)}, {width})"
        )
    if decoder_norms.shape != (width,):
        raise ValueError(
            f"Existing decoder_norms have shape {decoder_norms.shape}; expected ({width},)"
        )
    rows = read_jsonl(token_output)
    if [row.get("prompt_id") for row in rows] != prompt_ids:
        raise ValueError("Sparse JSONL prompt order differs from the dense NPZ")
    return prompt_ids, [row.copy() for row in activations], rows, decoder_norms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--token-output", type=Path)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--prompt-id", action="append", default=[])
    parser.add_argument("--sparsity-epsilon", type=float, default=0.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--test-config",
        type=Path,
        default=ROOT / "latent_escape" / "test_frozen.json",
    )
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment-variable name containing the Hugging Face token",
    )
    args = parser.parse_args()

    if args.condition != "baseline":
        raise ValueError(
            "Feature-discovery capture must use untreated development baseline generations"
        )
    if args.limit_prompts is not None and args.limit_prompts < 1:
        raise ValueError("--limit-prompts must be positive")
    if args.sparsity_epsilon < 0:
        raise ValueError("--sparsity-epsilon must be nonnegative")
    if args.overwrite and args.resume:
        args.resume = False

    protocol = read_json(args.protocol)
    manifest, manifest_hash = _manifest_by_id(args.manifest, protocol, args.split)
    if args.split == "test":
        if args.protocol.resolve() != DEFAULT_PROTOCOL.resolve():
            raise ValueError("confirmatory capture requires the repository protocol")
        if args.limit_prompts is not None or args.prompt_id:
            raise ValueError("Confirmatory test capture must include all frozen test prompts")
        _validate_test_freeze(
            args.test_config, protocol, manifest_hash, args.confirm_test
        )
    grouped, generation_run_id, generation_sha256 = _load_generations(
        args.generations,
        protocol=protocol,
        manifest_hash=manifest_hash,
        split=args.split,
        condition=args.condition,
        allow_dry_run=args.dry_run,
    )
    unknown = set(grouped) - set(manifest)
    if unknown:
        raise ValueError(f"Generation input has unknown prompt IDs: {sorted(unknown)}")
    prompt_ids = [prompt_id for prompt_id in manifest if prompt_id in grouped]
    selected = set(args.prompt_id)
    if selected:
        prompt_ids = [prompt_id for prompt_id in prompt_ids if prompt_id in selected]
        missing = selected - set(prompt_ids)
        if missing:
            raise ValueError(f"Unknown generated prompt IDs: {sorted(missing)}")
    if args.limit_prompts is not None:
        prompt_ids = prompt_ids[: args.limit_prompts]
    if not prompt_ids:
        raise ValueError("No prompts selected")

    protocol_sha256 = sha256_file(args.protocol)
    width = int(protocol["artifacts"]["sae"]["width"])
    implementation_hashes = {
        name: sha256_file(ROOT / "latent_escape" / name)
        for name in ("capture_activations.py", "model_sae.py")
        if (ROOT / "latent_escape" / name).exists()
    }
    run_spec = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_hash,
        "generation_sha256": generation_sha256,
        "generation_run_id": generation_run_id,
        "split": args.split,
        "condition": args.condition,
        "prompt_ids": prompt_ids,
        "sae_width": width,
        "sparsity_epsilon": args.sparsity_epsilon,
        "model_revision": protocol["artifacts"]["model"]["revision"],
        "sae_revision": protocol["artifacts"]["sae"]["revision"],
        "implementation_sha256": implementation_hashes,
        "dry_run": args.dry_run,
    }
    capture_run_id = canonical_hash(run_spec)
    output = args.output or (
        DEFAULT_DEVELOPMENT_OUTPUT
        if args.split == "development"
        else DEFAULT_DEVELOPMENT_OUTPUT.with_name("test_baseline_prompt.npz")
    )
    if args.dry_run and args.output is None:
        output = output.with_name(output.stem + ".dry-run.npz")
    token_output = args.token_output or _token_output_path(output)
    meta_path = _meta_path(output)
    existing_ids, dense_rows, sparse_rows, decoder_norms = _load_existing(
        output,
        token_output,
        meta_path,
        capture_run_id,
        width,
        args.resume,
        args.overwrite,
    )
    completed = set(existing_ids)
    if not completed <= set(prompt_ids):
        raise ValueError("Existing activation output contains unexpected prompts")

    commit, dirty = git_state()
    metadata = {
        **run_spec,
        "record_type": "activation_capture_run",
        "capture_run_id": capture_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_code_commit": commit,
        "git_dirty_at_start": dirty,
        "dense_output": str(output),
        "sparse_pre_domain_output": str(token_output),
        "observation_unit": "one source prompt",
        "pseudoreplication_guard": (
            "NPZ rows are prompts; generation samples are nested outcomes and must "
            "be aggregated to prompt-level domain frequencies before feature tests"
        ),
        "domain_label_policy": (
            "self-reported target_domain locates timing only; independent blinded "
            "labels are joined downstream"
        ),
    }
    if not meta_path.exists() or args.overwrite:
        atomic_write_json(meta_path, metadata)

    bundle = None
    if not args.dry_run and completed != set(prompt_ids):
        try:
            from .model_sae import load_pinned_bundle
        except ImportError:
            from model_sae import load_pinned_bundle

        bundle = load_pinned_bundle(
            token=os.environ.get(args.hf_token_env),
            device_map={"": "cuda:0"},
            local_files_only=args.local_files_only,
        )
        if str(bundle.model_revision) != run_spec["model_revision"]:
            raise ValueError("Loaded model revision differs from the protocol")
        if str(bundle.sae_revision) != run_spec["sae_revision"]:
            raise ValueError("Loaded SAE revision differs from the protocol")
        decoder_norms = (
            bundle.sae.W_dec.float().norm(dim=-1).detach().cpu().numpy().astype(np.float32)
        )
    elif args.dry_run and decoder_norms is None:
        # Serialization-only placeholder. Dry-run artifacts are barred from analysis.
        decoder_norms = np.ones(width, dtype=np.float32)
    if decoder_norms is None or decoder_norms.shape != (width,):
        raise ValueError(f"decoder_norms must have shape ({width},)")

    for prompt_id in prompt_ids:
        if prompt_id in completed:
            continue
        source = manifest[prompt_id]
        prompt_text = effective_prompt(source["prompt_text"], args.condition, protocol)
        expected_prompt_hash = sha256_bytes(prompt_text.encode("utf-8"))
        if any(
            row.get("prompt_text_sha256") != expected_prompt_hash
            for row in grouped[prompt_id]
        ):
            raise ValueError(f"Generation prompt hash mismatch for {prompt_id}")
        if args.dry_run:
            indices, values = _dry_sparse_vector(f"prompt|{prompt_id}", width)
            prompt_vector = _dense_from_sparse(indices, values, width)
        else:
            assert bundle is not None
            prompt_vector = _capture_prompt_vector(bundle, prompt_text)
            if prompt_vector.shape != (width,):
                raise ValueError(
                    f"Prompt {prompt_id} activation shape {prompt_vector.shape} != ({width},)"
                )

        generation_entries: list[dict[str, Any]] = []
        for generation in grouped[prompt_id]:
            if args.dry_run:
                token_indices, token_values = _dry_sparse_vector(
                    "pre-domain|"
                    + "|".join(map(str, _canonical_generation_key(generation))),
                    width,
                )
                boundary = {
                    "status": "dry_run",
                    "locator": "synthetic_contract_only",
                    "outcome_label_source": "none",
                }
            else:
                assert bundle is not None
                vector, boundary = _capture_pre_domain(
                    bundle, prompt_text, generation
                )
                if vector is None:
                    token_indices, token_values = [], []
                else:
                    if vector.shape != (width,):
                        raise ValueError(
                            f"Pre-domain activation shape {vector.shape} != ({width},)"
                        )
                    token_indices, token_values = _sparse_from_array(
                        vector, args.sparsity_epsilon
                    )
            generation_entries.append(
                {
                    "prompt_id": prompt_id,
                    "condition": str(generation["condition"]),
                    "sample_index": int(generation["sample_index"]),
                    "seed": int(generation["seed"]),
                    "generation_id": generation.get("generation_id"),
                    "boundary": boundary,
                    "pre_domain_activation": {
                        "encoding": "sparse_indices_values",
                        "width": width,
                        "indices": token_indices,
                        "values": token_values,
                    },
                }
            )

        row_index = len(existing_ids)
        sparse_rows.append(
            {
                "schema_version": 1,
                "record_type": "prompt_activation_bundle",
                "protocol_id": protocol["protocol_id"],
                "capture_run_id": capture_run_id,
                "generation_run_id": generation_run_id,
                "prompt_id": prompt_id,
                "cluster_id": prompt_id,
                "split": args.split,
                "condition": args.condition,
                "prompt_activation": {
                    "npz": output.name,
                    "row_index": row_index,
                    "position": "last_prompt_token_pre_treatment",
                },
                "generations": generation_entries,
                "dry_run": args.dry_run,
            }
        )
        existing_ids.append(prompt_id)
        dense_rows.append(prompt_vector)
        completed.add(prompt_id)

        dense = np.stack(dense_rows, axis=0).astype(np.float32, copy=False)
        _atomic_write_npz(
            output,
            prompt_ids=existing_ids,
            activations=dense,
            decoder_norms=decoder_norms,
            protocol_id=protocol["protocol_id"],
            protocol_sha256=protocol_sha256,
            manifest_sha256=manifest_hash,
            capture_run_id=capture_run_id,
            split=args.split,
            model_revision=run_spec["model_revision"],
            sae_revision=run_spec["sae_revision"],
            dry_run=args.dry_run,
        )
        atomic_write_jsonl(token_output, sparse_rows)
        print(f"{prompt_id}: prompt activation and {len(generation_entries)} outcomes captured")

    print(f"wrote {len(existing_ids)} prompt rows with shape ({len(existing_ids)}, {width})")
    print(f"dense={output}")
    print(f"sparse={token_output}")
    print(f"capture_run_id={capture_run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
