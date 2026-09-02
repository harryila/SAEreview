from __future__ import annotations

from types import SimpleNamespace

import hashlib
import json
import pytest
import torch
from torch import nn

from latent_choice.run import (
    DEFAULT_CODE_TOKENS,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    _load_code_token_manifest,
    _read_json,
    _score_real_prompt,
    main,
    run_baseline,
)


class _FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        del text, kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }


class _FakeSAE:
    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = (*hidden.shape[:-1], 16384)
        result = torch.zeros(shape, dtype=torch.float32, device=hidden.device)
        result[..., 7] = 1.25
        return result


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.layer = nn.Linear(4, 4)
        self.head = nn.Linear(4, 32)
        self.calls = 0

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        del kwargs
        self.calls += 1
        hidden = self.layer(self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden))


def test_real_score_is_one_forward_one_capture_and_removes_hook() -> None:
    model = _FakeModel()
    bundle = SimpleNamespace(
        model=model,
        tokenizer=_FakeTokenizer(),
        sae=_FakeSAE(),
        layer=model.layer,
    )
    logits, activation, hook_calls = _score_real_prompt(bundle, "prompt")
    assert model.calls == 1
    assert hook_calls == 1
    assert logits.shape == (32,)
    assert activation.shape == (16384,)
    assert activation[7] == pytest.approx(1.25)
    assert not model.layer._forward_hooks


def test_development_dry_run_is_non_evidentiary(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest_rows = [
        {
            "prompt_id": f"dev-{index:03d}",
            "split": "development",
            "source_name": f"Source {index}",
            "source_domain": "synthetic",
            "source_description": "A synthetic mechanism for a dry-run contract test.",
        }
        for index in range(80)
    ]
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows)
    )
    protocol_value = _read_json(DEFAULT_PROTOCOL)
    protocol_value["stimuli"]["prompt_manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_value, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "baseline.dry-run.jsonl"
    activations = tmp_path / "baseline.dry-run.npz"
    report = run_baseline(
        protocol_path=protocol,
        manifest_path=manifest,
        output_path=output,
        activations_path=activations,
        dry_run=True,
        limit_prompts=2,
        overwrite=False,
    )
    assert report["run_mode"] == "synthetic_dry_run"
    assert report["evidentiary"] is False
    assert report["feature_discovery_allowed"] is False
    assert report["test_split_generated"] is False
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["evidentiary"] is False for row in rows)


def test_real_overwrite_is_rejected_before_model_load(tmp_path) -> None:
    with pytest.raises(ValueError, match="immutable"):
        run_baseline(
            protocol_path=DEFAULT_PROTOCOL,
            manifest_path=DEFAULT_MANIFEST,
            output_path=tmp_path / "out.jsonl",
            activations_path=tmp_path / "out.npz",
            dry_run=False,
            limit_prompts=None,
            overwrite=True,
        )


def test_committed_code_token_manifest_loads_and_test_cli_is_blocked() -> None:
    protocol = _read_json(DEFAULT_PROTOCOL)
    _, token_ids = _load_code_token_manifest(
        DEFAULT_CODE_TOKENS,
        protocol_path=DEFAULT_PROTOCOL,
        protocol=protocol,
    )
    assert len(token_ids) == 18
    assert len(set(token_ids.values())) == 18
    with pytest.raises(SystemExit, match="test access is disabled"):
        main(["baseline", "--split", "test"])
