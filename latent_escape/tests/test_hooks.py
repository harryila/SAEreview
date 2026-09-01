"""Offline invariants for the Latent Escape residual-stream hooks."""

from __future__ import annotations

import torch
from torch import nn

from latent_escape.intervene import (
    InterventionSpec,
    SAELastTokenEditor,
    deterministic_random_feature,
    install_intervention,
)
from latent_escape.model_sae import GemmaScopeSAE


class _IdentityDecoderLayer(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, str]:
        # A sentinel makes structure preservation observable.
        return hidden_states, "untouched-cache"


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_IdentityDecoderLayer()])


class _TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.lm_head = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, -1.0, 0.5],
                    ]
                )
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, cache = self.model.layers[0](hidden_states)
        assert cache == "untouched-cache"
        return self.lm_head(hidden_states)


def _fake_sae() -> GemmaScopeSAE:
    # feature 0 reads residual coordinate 0; feature 1 reads coordinate 1.
    return GemmaScopeSAE(
        W_enc=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        W_dec=torch.tensor(
            [
                [1.5, -0.5, 0.25],
                [-0.25, 1.0, 0.75],
            ]
        ),
        b_enc=torch.zeros(2),
        b_dec=torch.tensor([99.0, 99.0, 99.0]),
        threshold=torch.zeros(2),
    )


def _hidden() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [5.0, 4.0, 3.0],
                [2.0, 1.0, 7.0],
            ]
        ]
    )


def test_zero_strength_hook_reproduces_identical_logits() -> None:
    model = _TinyCausalLM().eval()
    hidden = _hidden()
    baseline_logits = model(hidden)

    handle, editor = install_intervention(
        model,
        _fake_sae(),
        InterventionSpec(mode="suppress", feature_id=0, strength=0.0),
        layer_index=0,
    )
    try:
        hooked_logits = model(hidden)
    finally:
        handle.remove()

    assert torch.equal(hooked_logits, baseline_logits)
    assert editor.call_count == 0


def test_suppression_adds_exactly_one_decoder_coordinate_to_last_token() -> None:
    sae = _fake_sae()
    layer = _IdentityDecoderLayer()
    hidden = _hidden()
    baseline_output = layer(hidden)
    baseline = baseline_output[0]

    editor = SAELastTokenEditor(
        sae, InterventionSpec(mode="suppress", feature_id=0, strength=1.0)
    )
    edited_output = editor(layer, (hidden,), baseline_output)
    edited = edited_output[0]

    old_activation = sae.encode(hidden[:, -1, :])[:, 0]
    expected_delta = -old_activation.unsqueeze(-1) * sae.decoder_vector(0)

    assert type(edited_output) is tuple
    assert edited_output[1] == "untouched-cache"
    assert torch.equal(edited[:, :-1, :], baseline[:, :-1, :])
    assert torch.equal(edited[:, -1, :] - baseline[:, -1, :], expected_delta)
    # b_dec=99 makes a full SAE reconstruction obviously different; the hook
    # has changed only the selected feature's decoder coordinate.
    assert not torch.equal(edited[:, -1, :], sae.decode(sae.encode(hidden[:, -1, :])))


def test_bfloat16_suppression_realized_delta_stays_on_decoder_direction() -> None:
    sae = _fake_sae()
    # Non-dyadic values force visible BF16 rounding in the residual update.
    sae.W_dec[0].copy_(torch.tensor([1.37, -0.43, 0.23]))
    hidden = _hidden().to(torch.bfloat16)
    editor = SAELastTokenEditor(
        sae, InterventionSpec(mode="suppress", feature_id=0, strength=1.0)
    )

    edited = editor(
        _IdentityDecoderLayer(), (hidden,), (hidden, "untouched-cache")
    )[0]
    realized = (
        edited[:, -1, :].float() - hidden[:, -1, :].float()
    ).squeeze(0)
    decoder = sae.decoder_vector(0)
    projection = (realized @ decoder) / (decoder @ decoder) * decoder
    relative_orthogonal_error = torch.linalg.vector_norm(
        realized - projection
    ) / torch.linalg.vector_norm(realized)

    # The FP32 intervention is exactly collinear; allow 2% for BF16 cast/add
    # quantization while still rejecting a perturbation in another direction.
    assert relative_orthogonal_error.item() < 0.02
    assert torch.dot(realized, decoder).item() < 0.0
    assert torch.equal(edited[:, :-1, :], hidden[:, :-1, :])


def test_noise_is_deterministic_last_token_only_and_l2_matched() -> None:
    sae = _fake_sae()
    hidden = _hidden()
    output = (hidden, "untouched-cache")
    spec = InterventionSpec(mode="noise", feature_id=0, strength=0.5, noise_seed=7)

    first = SAELastTokenEditor(sae, spec)
    second = SAELastTokenEditor(sae, spec)
    edited_a = first(_IdentityDecoderLayer(), (hidden,), output)[0]
    edited_b = second(_IdentityDecoderLayer(), (hidden,), output)[0]

    assert torch.equal(edited_a, edited_b)
    assert torch.equal(edited_a[:, :-1, :], hidden[:, :-1, :])

    feature = sae.encode(hidden[:, -1, :])[:, 0]
    targeted_delta = -0.5 * feature.unsqueeze(-1) * sae.decoder_vector(0)
    observed_delta = edited_a[:, -1, :] - hidden[:, -1, :]
    assert torch.allclose(
        torch.linalg.vector_norm(observed_delta, dim=-1),
        torch.linalg.vector_norm(targeted_delta, dim=-1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_bfloat16_noise_realized_l2_matches_targeted_realized_l2() -> None:
    sae = _fake_sae()
    sae.W_dec[0].copy_(torch.tensor([1.37, -0.43, 0.23]))
    hidden = _hidden().to(torch.bfloat16)
    output = (hidden, "untouched-cache")

    targeted = SAELastTokenEditor(
        sae, InterventionSpec(mode="suppress", feature_id=0, strength=0.5)
    )(_IdentityDecoderLayer(), (hidden,), output)[0]
    noise = SAELastTokenEditor(
        sae,
        InterventionSpec(mode="noise", feature_id=0, strength=0.5, noise_seed=7),
    )(_IdentityDecoderLayer(), (hidden,), output)[0]

    targeted_norm = torch.linalg.vector_norm(
        targeted[:, -1, :].float() - hidden[:, -1, :].float(), dim=-1
    )
    noise_norm = torch.linalg.vector_norm(
        noise[:, -1, :].float() - hidden[:, -1, :].float(), dim=-1
    )

    # Compare the realized BF16 residual changes, not the ideal FP32 deltas.
    assert torch.allclose(noise_norm, targeted_norm, rtol=0.02, atol=0.0)
    assert torch.equal(noise[:, :-1, :], hidden[:, :-1, :])


def test_promotion_only_moves_activation_up_to_target() -> None:
    sae = _fake_sae()
    hidden = _hidden()
    editor = SAELastTokenEditor(
        sae,
        InterventionSpec(
            mode="promote",
            feature_id=0,
            strength=0.5,
            promotion_target=6.0,
        ),
    )
    edited = editor(_IdentityDecoderLayer(), (hidden,), (hidden, "cache"))[0]

    # Original activation is 2; halfway to 6 means a +2 coordinate change.
    expected = 2.0 * sae.decoder_vector(0)
    assert torch.equal(edited[:, -1, :] - hidden[:, -1, :], expected.unsqueeze(0))
    assert torch.equal(edited[:, :-1, :], hidden[:, :-1, :])


def test_random_control_selection_is_stable_and_without_replacement() -> None:
    candidates = [11, 7, 5, 11, 3]
    selected = [
        deterministic_random_feature(candidates, seed=20260831, control_index=index)
        for index in range(4)
    ]
    repeated = [
        deterministic_random_feature(candidates, seed=20260831, control_index=index)
        for index in range(4)
    ]
    assert selected == repeated
    assert len(set(selected)) == 4
    assert set(selected) == {3, 5, 7, 11}
