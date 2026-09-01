"""Pinned Gemma 2 / Gemma Scope loading for the Latent Escape experiment.

The SAE implementation is intentionally small and mirrors the official Gemma
Scope conversion used by SAELens: the checkpoint contains ``W_enc``, ``W_dec``,
``b_enc``, ``b_dec``, and per-feature JumpReLU ``threshold`` arrays.  Gemma
Scope residual SAEs do not subtract ``b_dec`` before encoding and do not apply
an activation-norm scaling factor.

Heavy dependencies are imported normally, but model and Hub imports are lazy so
that the tensor-only hook tests can run offline without loading Transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


MODEL_REPO_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"

SAE_REPO_ID = "google/gemma-scope-9b-it-res"
SAE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
SAE_FILENAME = "layer_20/width_16k/average_l0_91/params.npz"
SAE_SHA256 = "bbd770b6f8b92a2fe7498e05bd6274c6cfa89ebc08fb972c0e842840737f1a82"

LAYER_INDEX = 20
MODEL_WIDTH = 3584
SAE_WIDTH = 16384


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for *path* without loading it whole."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GemmaScopeSAE(nn.Module):
    """Dependency-light, inference-only Gemma Scope JumpReLU SAE.

    Shapes follow the published checkpoint layout:

    - ``W_enc``: ``[d_model, d_sae]``
    - ``W_dec``: ``[d_sae, d_model]``
    - ``b_enc`` and ``threshold``: ``[d_sae]``
    - ``b_dec``: ``[d_model]``

    All buffers are kept in float32, as fixed in the protocol.  A coordinate
    intervention should use :meth:`decoder_vector`; it must not replace the
    model residual with :meth:`decode`.
    """

    def __init__(
        self,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: torch.Tensor,
    ) -> None:
        super().__init__()

        W_enc = torch.as_tensor(W_enc, dtype=torch.float32).contiguous()
        W_dec = torch.as_tensor(W_dec, dtype=torch.float32).contiguous()
        b_enc = torch.as_tensor(b_enc, dtype=torch.float32).contiguous()
        b_dec = torch.as_tensor(b_dec, dtype=torch.float32).contiguous()
        threshold = torch.as_tensor(threshold, dtype=torch.float32).contiguous()

        if W_enc.ndim != 2 or W_dec.ndim != 2:
            raise ValueError("W_enc and W_dec must both be rank-2")
        d_model, d_sae = W_enc.shape
        if W_dec.shape != (d_sae, d_model):
            raise ValueError(
                f"W_dec shape {tuple(W_dec.shape)} does not match "
                f"W_enc shape {tuple(W_enc.shape)}"
            )
        if b_enc.shape != (d_sae,):
            raise ValueError(f"b_enc must have shape ({d_sae},)")
        if b_dec.shape != (d_model,):
            raise ValueError(f"b_dec must have shape ({d_model},)")
        if threshold.shape != (d_sae,):
            raise ValueError(f"threshold must have shape ({d_sae},)")
        if not all(
            torch.isfinite(tensor).all().item()
            for tensor in (W_enc, W_dec, b_enc, b_dec, threshold)
        ):
            raise ValueError("SAE parameters must all be finite")

        self.register_buffer("W_enc", W_enc)
        self.register_buffer("W_dec", W_dec)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("b_dec", b_dec)
        self.register_buffer("threshold", threshold)
        self.requires_grad_(False)

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[1])

    def encode_pre(self, residual: torch.Tensor) -> torch.Tensor:
        """Compute float32 encoder pre-activations without thresholding."""

        if residual.shape[-1] != self.d_model:
            raise ValueError(
                f"residual width {residual.shape[-1]} != SAE input width {self.d_model}"
            )
        if residual.device != self.W_enc.device:
            raise ValueError(
                f"residual is on {residual.device}, but SAE is on {self.W_enc.device}; "
                "load the SAE on the hooked decoder layer's device"
            )
        residual_f32 = residual.to(dtype=torch.float32)
        return residual_f32 @ self.W_enc + self.b_enc

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """Encode residuals with Gemma Scope's per-feature JumpReLU."""

        pre = self.encode_pre(residual)
        # The official loader fixes activation_fn=ReLU, then masks values at or
        # below the learned threshold.  Writing both operations explicitly
        # preserves that behavior even for synthetic negative thresholds.
        relu = torch.relu(pre)
        return relu * (pre > self.threshold).to(dtype=relu.dtype)

    def decode(self, feature_activations: torch.Tensor) -> torch.Tensor:
        """Reconstruct residuals; provided for audits, not intervention hooks."""

        if feature_activations.shape[-1] != self.d_sae:
            raise ValueError(
                f"feature width {feature_activations.shape[-1]} != SAE width {self.d_sae}"
            )
        if feature_activations.device != self.W_dec.device:
            raise ValueError("feature activations and SAE must be on the same device")
        return feature_activations.to(torch.float32) @ self.W_dec + self.b_dec

    def decoder_vector(self, feature_id: int) -> torch.Tensor:
        """Return the float32 residual-space vector for one SAE coordinate."""

        if isinstance(feature_id, bool) or not isinstance(feature_id, int):
            raise TypeError("feature_id must be an integer")
        if not 0 <= feature_id < self.d_sae:
            raise IndexError(f"feature_id {feature_id} outside [0, {self.d_sae})")
        return self.W_dec[feature_id]

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        expected_d_model: int | None = None,
        expected_d_sae: int | None = None,
        device: str | torch.device = "cpu",
    ) -> "GemmaScopeSAE":
        """Load and validate a Gemma Scope ``params.npz`` checkpoint."""

        path = Path(path)
        if expected_sha256 is not None:
            observed = sha256_file(path)
            if observed != expected_sha256.lower():
                raise ValueError(
                    f"SAE SHA-256 mismatch for {path}: expected "
                    f"{expected_sha256.lower()}, observed {observed}"
                )

        required = {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"}
        with np.load(path, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            extra = set(archive.files).difference(required | {"scaling_factor"})
            if missing:
                raise ValueError(f"SAE checkpoint missing arrays: {sorted(missing)}")
            if extra:
                raise ValueError(f"SAE checkpoint has unexpected arrays: {sorted(extra)}")
            if "scaling_factor" in archive.files and not np.allclose(
                archive["scaling_factor"], 1.0
            ):
                raise ValueError("non-unit SAE scaling_factor is not supported")
            # Copies detach tensors from the lifetime of the zip-backed archive.
            arrays = {
                name: torch.from_numpy(np.array(archive[name], dtype=np.float32, copy=True))
                for name in required
            }

        sae = cls(**arrays)
        if expected_d_model is not None and sae.d_model != expected_d_model:
            raise ValueError(
                f"SAE input width {sae.d_model} != expected {expected_d_model}"
            )
        if expected_d_sae is not None and sae.d_sae != expected_d_sae:
            raise ValueError(f"SAE width {sae.d_sae} != expected {expected_d_sae}")
        return sae.to(device=device, dtype=torch.float32).eval()


def resolve_decoder_layer(model: nn.Module, layer_index: int = LAYER_INDEX) -> nn.Module:
    """Resolve a Hugging Face Gemma decoder layer without relying on class names."""

    candidate_paths = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("base_model", "model", "layers"),
    )
    for path in candidate_paths:
        value: Any = model
        for name in path:
            if not hasattr(value, name):
                break
            value = getattr(value, name)
        else:
            try:
                layer = value[layer_index]
            except (IndexError, KeyError, TypeError):
                continue
            if isinstance(layer, nn.Module):
                return layer
    raise ValueError(
        f"could not resolve decoder layer {layer_index}; expected a Hugging Face "
        "causal LM exposing model.layers"
    )


def _module_device(module: nn.Module) -> torch.device:
    for tensor in tuple(module.parameters(recurse=True)) + tuple(
        module.buffers(recurse=True)
    ):
        if tensor.device.type != "meta":
            return tensor.device
    raise ValueError("cannot infer a real device for the selected decoder layer")


@dataclass(frozen=True)
class ModelSAEBundle:
    """The loaded, pinned experiment artifacts and their intervention site."""

    model: nn.Module
    tokenizer: Any
    sae: GemmaScopeSAE
    layer: nn.Module
    layer_index: int = LAYER_INDEX
    model_revision: str = MODEL_REVISION
    sae_revision: str = SAE_REVISION
    sae_sha256: str = SAE_SHA256


def load_pinned_sae(
    *,
    token: str | None = None,
    device: str | torch.device = "cpu",
    local_files_only: bool = False,
) -> GemmaScopeSAE:
    """Download by immutable revision, hash-check, and load the frozen SAE."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised only in minimal envs
        raise RuntimeError(
            "loading pinned artifacts requires the `huggingface-hub` dependency"
        ) from exc

    path = hf_hub_download(
        repo_id=SAE_REPO_ID,
        filename=SAE_FILENAME,
        revision=SAE_REVISION,
        token=token,
        local_files_only=local_files_only,
    )
    return GemmaScopeSAE.from_npz(
        path,
        expected_sha256=SAE_SHA256,
        expected_d_model=MODEL_WIDTH,
        expected_d_sae=SAE_WIDTH,
        device=device,
    )


def load_pinned_bundle(
    *,
    token: str | None = None,
    device_map: str | dict[str, Any] | None = None,
    local_files_only: bool = False,
) -> ModelSAEBundle:
    """Load the exact Gemma 2 model and Gemma Scope SAE frozen in the protocol.

    The model is BF16, eager, unquantized, evaluation-only, uncompiled, and kept
    wholly on one CUDA device.  Offloading is deliberately rejected: a hook that
    silently moves layer 20 while leaving the SAE elsewhere is not a frozen
    intervention.
    """

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only in minimal envs
        raise RuntimeError(
            "loading the pinned model requires the `transformers` dependency"
        ) from exc

    frozen_device_map = {"": "cuda:0"}
    if device_map is None:
        device_map = frozen_device_map
    if device_map != frozen_device_map:
        raise ValueError(
            "the frozen run requires device_map={'': 'cuda:0'}; automatic "
            "placement and CPU/disk offload are not permitted"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the real Latent Escape run requires a CUDA GPU; use --dry-run to "
            "exercise the pipeline without loading weights"
        )

    common = {
        "revision": MODEL_REVISION,
        "token": token,
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID, **common)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO_ID,
        **common,
        dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    config = model.config
    if getattr(config, "model_type", None) != "gemma2":
        raise ValueError(f"expected Gemma 2, got model_type={config.model_type!r}")
    if int(getattr(config, "hidden_size", -1)) != MODEL_WIDTH:
        raise ValueError("loaded model hidden size does not match the frozen SAE")
    if bool(getattr(model, "is_loaded_in_4bit", False)) or bool(
        getattr(model, "is_loaded_in_8bit", False)
    ):
        raise ValueError("the frozen protocol forbids a quantized model")

    model.eval()
    model.requires_grad_(False)
    layer = resolve_decoder_layer(model, LAYER_INDEX)
    expected_device = torch.device("cuda:0")
    observed_devices = {
        tensor.device
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.device.type != "meta"
    }
    if observed_devices != {expected_device}:
        raise ValueError(
            "the model is not wholly resident on the frozen CUDA device: "
            f"{sorted(map(str, observed_devices))}"
        )
    sae = load_pinned_sae(
        token=token,
        device=expected_device,
        local_files_only=local_files_only,
    )
    if _module_device(layer) != expected_device or sae.W_dec.device != expected_device:
        raise ValueError("decoder layer and SAE are not colocated on cuda:0")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return ModelSAEBundle(model=model, tokenizer=tokenizer, sae=sae, layer=layer)


__all__ = [
    "GemmaScopeSAE",
    "LAYER_INDEX",
    "MODEL_REPO_ID",
    "MODEL_REVISION",
    "MODEL_WIDTH",
    "ModelSAEBundle",
    "SAE_FILENAME",
    "SAE_REPO_ID",
    "SAE_REVISION",
    "SAE_SHA256",
    "SAE_WIDTH",
    "load_pinned_bundle",
    "load_pinned_sae",
    "resolve_decoder_layer",
    "sha256_file",
]
