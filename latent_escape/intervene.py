"""Auditable SAE-coordinate interventions for cached autoregressive generation.

The hook edits only the final sequence position.  On the first generation pass
that is the final prompt token; with the KV cache on subsequent passes it is the
single newly generated token.  Crucially, it adds one decoder-coordinate delta
to the original residual stream and never substitutes a full SAE reconstruction.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass
import hashlib
from typing import Any, Iterator, Literal, Mapping, Sequence

import torch
from torch import nn

from latent_escape.model_sae import GemmaScopeSAE, LAYER_INDEX, resolve_decoder_layer


InterventionMode = Literal["suppress", "promote", "noise"]


@dataclass(frozen=True)
class InterventionSpec:
    """One frozen last-token intervention condition.

    ``strength`` is an interpolation fraction in ``[0, 1]``. Suppression
    subtracts the decoder contribution implied by moving the measured selected
    activation toward zero; it does not claim the edited residual re-encodes to
    an exactly zero latent. Promotion adds the corresponding decoder direction
    toward ``promotion_target`` only when below that target. Noise uses the L2
    norm of the corresponding suppression delta in a deterministic random
    residual-space direction.
    """

    mode: InterventionMode
    feature_id: int
    strength: float = 1.0
    promotion_target: float | None = None
    noise_seed: int = 20260831

    def __post_init__(self) -> None:
        if self.mode not in {"suppress", "promote", "noise"}:
            raise ValueError(f"unsupported intervention mode: {self.mode!r}")
        if isinstance(self.feature_id, bool) or not isinstance(self.feature_id, int):
            raise TypeError("feature_id must be an integer")
        if not 0.0 <= float(self.strength) <= 1.0:
            raise ValueError("strength must be between 0 and 1")
        if self.mode == "promote":
            if self.promotion_target is None:
                raise ValueError("promotion requires promotion_target")
            if not torch.isfinite(torch.tensor(float(self.promotion_target))).item():
                raise ValueError("promotion_target must be finite")
            if float(self.promotion_target) < 0.0:
                raise ValueError("promotion_target must be nonnegative")
        elif self.promotion_target is not None:
            raise ValueError("promotion_target is valid only for promote mode")
        if isinstance(self.noise_seed, bool) or not isinstance(self.noise_seed, int):
            raise TypeError("noise_seed must be an integer")


def deterministic_random_feature(
    candidate_feature_ids: Sequence[int],
    *,
    seed: int,
    control_index: int = 0,
) -> int:
    """Select a stable hash-ordered control feature without global RNG state."""

    if control_index < 0:
        raise ValueError("control_index must be nonnegative")
    candidates = sorted(set(candidate_feature_ids))
    if not candidates:
        raise ValueError("candidate_feature_ids cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in candidates):
        raise ValueError("candidate feature IDs must be nonnegative integers")

    ordered = sorted(
        candidates,
        key=lambda feature_id: hashlib.sha256(
            f"latent-escape-control-v1\0{seed}\0{feature_id}".encode("ascii")
        ).digest(),
    )
    if control_index >= len(ordered):
        raise IndexError("control_index exceeds the number of unique candidates")
    return ordered[control_index]


def last_token_activations(
    hidden_states: torch.Tensor, sae: GemmaScopeSAE
) -> torch.Tensor:
    """Return float32 SAE activations for only the final sequence position."""

    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 2:
        raise ValueError("hidden_states must have shape [..., sequence, d_model]")
    return sae.encode(hidden_states[..., -1, :])


def coordinate_delta(
    sae: GemmaScopeSAE,
    feature_activation: torch.Tensor,
    *,
    feature_id: int,
    replacement_activation: torch.Tensor,
) -> torch.Tensor:
    """Calculate ``(replacement - original) * W_dec[feature_id]`` in FP32."""

    decoder = sae.decoder_vector(feature_id)
    original = feature_activation.to(device=decoder.device, dtype=torch.float32)
    replacement = replacement_activation.to(device=decoder.device, dtype=torch.float32)
    if original.shape != replacement.shape:
        raise ValueError("original and replacement activation shapes must match")
    return (replacement - original).unsqueeze(-1) * decoder


def _extract_hidden(output: Any) -> tuple[torch.Tensor, Any]:
    """Return the residual tensor and a structure-preserving replacement closure."""

    if isinstance(output, torch.Tensor):
        return output, lambda replacement: replacement

    if isinstance(output, tuple):
        tensor_index = next(
            (index for index, value in enumerate(output) if isinstance(value, torch.Tensor)),
            None,
        )
        if tensor_index is None:
            raise TypeError("hook output tuple contains no tensor")

        def replace_tuple(replacement: torch.Tensor) -> tuple[Any, ...]:
            values = list(output)
            values[tensor_index] = replacement
            if hasattr(output, "_fields"):  # namedtuple
                return type(output)(*values)
            return tuple(values)

        return output[tensor_index], replace_tuple

    if isinstance(output, list):
        tensor_index = next(
            (index for index, value in enumerate(output) if isinstance(value, torch.Tensor)),
            None,
        )
        if tensor_index is None:
            raise TypeError("hook output list contains no tensor")

        def replace_list(replacement: torch.Tensor) -> list[Any]:
            values = list(output)
            values[tensor_index] = replacement
            return values

        return output[tensor_index], replace_list

    if isinstance(output, Mapping):
        preferred = ("last_hidden_state", "hidden_states")
        tensor_key = next(
            (key for key in preferred if isinstance(output.get(key), torch.Tensor)),
            None,
        )
        if tensor_key is None:
            tensor_key = next(
                (key for key, value in output.items() if isinstance(value, torch.Tensor)),
                None,
            )
        if tensor_key is None:
            raise TypeError("hook output mapping contains no tensor")

        def replace_mapping(replacement: torch.Tensor) -> Mapping[str, Any]:
            values = copy.copy(output)
            values[tensor_key] = replacement
            return values

        return output[tensor_key], replace_mapping

    raise TypeError(f"unsupported decoder-layer output type: {type(output).__name__}")


class SAELastTokenEditor:
    """Stateful callable suitable for ``register_forward_hook``.

    A new instance should be installed for each generation call.  The only
    state is a call counter used to derive deterministic, distinct noise at each
    decoding step plus small detached diagnostics for audit logs.
    """

    def __init__(self, sae: GemmaScopeSAE, spec: InterventionSpec) -> None:
        # Validate the coordinate before registering a hook on a large model.
        sae.decoder_vector(spec.feature_id)
        self.sae = sae
        self.spec = spec
        self.call_count = 0
        self.last_feature_activation: torch.Tensor | None = None
        self.last_requested_delta_norm: torch.Tensor | None = None
        self.last_delta_norm: torch.Tensor | None = None

    def _noise_like(self, reference: torch.Tensor) -> torch.Tensor:
        seed = (self.spec.noise_seed + self.call_count * 1_000_003) % (2**63 - 1)
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            reference.shape,
            dtype=torch.float32,
            device=reference.device,
            generator=generator,
        )
        norms = torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
        # The probability of an all-zero normal draw is effectively zero, but a
        # deterministic basis-vector fallback makes the norm contract total.
        zero = norms == 0
        if zero.any():
            noise = noise.clone()
            noise[zero.expand_as(noise)] = 0.0
            noise[..., 0] = torch.where(
                zero.squeeze(-1), torch.ones_like(noise[..., 0]), noise[..., 0]
            )
            norms = torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
        return noise / norms

    @torch.no_grad()
    def __call__(
        self, module: nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> Any:
        del module, inputs
        # Returning the exact same object is important for the zero-strength
        # invariant: no SAE matmul, clone, cast, or roundoff can affect logits.
        if self.spec.strength == 0.0:
            return output

        hidden, replace_hidden = _extract_hidden(output)
        if hidden.ndim < 2:
            raise ValueError("decoder residual must have a sequence and model dimension")

        last = hidden[..., -1, :]
        activations = self.sae.encode(last)
        feature = activations[..., self.spec.feature_id]
        self.last_feature_activation = feature.detach()

        if self.spec.mode == "suppress":
            replacement = feature * (1.0 - self.spec.strength)
            delta_f32 = coordinate_delta(
                self.sae,
                feature,
                feature_id=self.spec.feature_id,
                replacement_activation=replacement,
            )
        elif self.spec.mode == "promote":
            target = torch.full_like(feature, float(self.spec.promotion_target))
            desired = torch.maximum(feature, target)
            replacement = feature + self.spec.strength * (desired - feature)
            delta_f32 = coordinate_delta(
                self.sae,
                feature,
                feature_id=self.spec.feature_id,
                replacement_activation=replacement,
            )
        else:
            suppressed = feature * (1.0 - self.spec.strength)
            targeted_delta = coordinate_delta(
                self.sae,
                feature,
                feature_id=self.spec.feature_id,
                replacement_activation=suppressed,
            )
            target_norm = torch.linalg.vector_norm(
                targeted_delta, dim=-1, keepdim=True
            )
            delta_f32 = self._noise_like(targeted_delta) * target_norm

        self.last_requested_delta_norm = torch.linalg.vector_norm(
            delta_f32, dim=-1
        ).detach()
        edited = hidden.clone()
        edited[..., -1, :] = last + delta_f32.to(dtype=hidden.dtype)
        realized_delta = edited[..., -1, :].float() - last.float()
        self.last_delta_norm = torch.linalg.vector_norm(
            realized_delta, dim=-1
        ).detach()
        self.call_count += 1
        return replace_hidden(edited)


def install_intervention_on_layer(
    layer: nn.Module, sae: GemmaScopeSAE, spec: InterventionSpec
) -> tuple[torch.utils.hooks.RemovableHandle, SAELastTokenEditor]:
    """Install on an already-resolved decoder layer and return handle + editor."""

    editor = SAELastTokenEditor(sae, spec)
    handle = layer.register_forward_hook(editor)
    return handle, editor


def install_intervention(
    model: nn.Module,
    sae: GemmaScopeSAE,
    spec: InterventionSpec,
    *,
    layer_index: int = LAYER_INDEX,
) -> tuple[torch.utils.hooks.RemovableHandle, SAELastTokenEditor]:
    """Resolve the frozen decoder layer and install the last-token editor."""

    layer = resolve_decoder_layer(model, layer_index)
    return install_intervention_on_layer(layer, sae, spec)


@contextmanager
def intervention_context(
    model: nn.Module,
    sae: GemmaScopeSAE,
    spec: InterventionSpec,
    *,
    layer_index: int = LAYER_INDEX,
) -> Iterator[SAELastTokenEditor]:
    """Install an intervention for a block and always remove it afterward."""

    handle, editor = install_intervention(
        model, sae, spec, layer_index=layer_index
    )
    try:
        yield editor
    finally:
        handle.remove()


__all__ = [
    "InterventionMode",
    "InterventionSpec",
    "SAELastTokenEditor",
    "coordinate_delta",
    "deterministic_random_feature",
    "install_intervention",
    "install_intervention_on_layer",
    "intervention_context",
    "last_token_activations",
]
