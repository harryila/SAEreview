from __future__ import annotations

import math
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json

import pytest
import torch

from latent_choice.choice_endpoint import CODE_SYMBOLS, DOMAINS
from latent_choice.exploratory_calibration import (
    CALIBRATION_PROMPT_COUNT,
    DEFAULT_PLAN,
    GROUP_LABELS,
    HIERARCHY_GROUPS,
    ROTATION_COUNT,
    _load_plan,
    _score_jobs,
    combine_hierarchy,
    cyclic_mapping,
    flat_rotation,
    geometric_mean_distribution,
    resolve_candidate_token_ids,
    render_flat_prompt,
    score_candidates,
    select_calibration_rows,
    summarize_rotation_distributions,
)


def test_selection_is_deterministic_and_development_only() -> None:
    rows = [
        {"prompt_id": f"dev-{index:03d}", "split": "development"}
        for index in range(80)
    ]
    selected = select_calibration_rows(rows)
    assert len(selected) == CALIBRATION_PROMPT_COUNT
    assert selected == select_calibration_rows(list(reversed(rows)))
    with pytest.raises(ValueError, match="development"):
        select_calibration_rows([*rows[:-1], {"prompt_id": "test-001", "split": "test"}])


def test_flat_rotations_balance_every_domain_code_pair() -> None:
    counts = {(domain, code): 0 for domain in DOMAINS for code in CODE_SYMBOLS}
    for prompt_rank in range(CALIBRATION_PROMPT_COUNT):
        for rotation_index in range(ROTATION_COUNT):
            mapping = cyclic_mapping(
                DOMAINS,
                CODE_SYMBOLS,
                rotation=flat_rotation(prompt_rank, rotation_index),
            )
            for domain, code in mapping.items():
                counts[(domain, code)] += 1
    assert set(counts.values()) == {8}


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text.endswith(" A"):
            return [1, 2, 10]
        if text.endswith(" B"):
            return [1, 2, 11]
        return [1, 2]


def test_generic_token_contract_and_candidate_scoring() -> None:
    token_ids = resolve_candidate_token_ids(_Tokenizer(), "prompt", {"A": " A", "B": " B"})
    mapping = {"first": "A", "second": "B"}
    logits = torch.full((32,), -3.0)
    logits[token_ids["B"]] = 4.0
    score = score_candidates(logits, mapping=mapping, code_token_ids=token_ids)
    assert score["argmax_label"] == "second"
    assert math.isclose(sum(score["label_probabilities"].values()), 1.0)
    assert 0.0 < score["full_vocab_candidate_mass"] < 1.0


def test_hierarchy_combines_to_canonical_domain_distribution() -> None:
    group_probabilities = {group: 1.0 / len(GROUP_LABELS) for group in GROUP_LABELS}
    branches = {
        group: {domain: 1.0 / 3.0 for domain in domains}
        for group, domains in HIERARCHY_GROUPS
    }
    combined = combine_hierarchy(group_probabilities, branches)
    assert tuple(combined) == DOMAINS
    assert math.isclose(sum(combined.values()), 1.0)
    assert all(value == pytest.approx(1.0 / 18.0) for value in combined.values())


def test_rotation_summary_detects_invariant_semantics() -> None:
    rows = []
    for prompt_index in range(CALIBRATION_PROMPT_COUNT):
        for rotation_index in range(ROTATION_COUNT):
            probabilities = {domain: 0.01 / (len(DOMAINS) - 1) for domain in DOMAINS}
            probabilities[DOMAINS[prompt_index % len(DOMAINS)]] = 0.99
            rows.append(
                {
                    "prompt_id": f"dev-{prompt_index:03d}",
                    "rotation_index": rotation_index,
                    "domain_probabilities": probabilities,
                    "argmax_domain": DOMAINS[prompt_index % len(DOMAINS)],
                }
            )
    summaries, metrics = summarize_rotation_distributions(rows)
    assert len(summaries) == CALIBRATION_PROMPT_COUNT
    assert metrics["mean_modal_agreement"] == 1.0
    assert metrics["leave_one_out_stable_prompt_count"] == CALIBRATION_PROMPT_COUNT
    assert metrics["strong_mapping_invariance"] is True
    aggregate = geometric_mean_distribution(
        [row["domain_probabilities"] for row in rows[:ROTATION_COUNT]], DOMAINS
    )
    assert max(aggregate, key=aggregate.get) == DOMAINS[0]


def test_canonical_plan_is_hash_bound_and_newline_instruction_matches() -> None:
    assert _load_plan(DEFAULT_PLAN)["calibration_id"].endswith("exploratory-v1")

    class InstructionTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return messages[0]["content"] + "\nASSISTANT:"

    mapping = cyclic_mapping(DOMAINS, CODE_SYMBOLS, rotation=0)
    prompt = render_flat_prompt(
        InstructionTokenizer(),
        {
            "source_name": "source",
            "source_domain": "domain",
            "source_description": "mechanism",
        },
        mapping,
        prefill="CHOICE:\n",
    )
    assert "prefilled with CHOICE: followed by a newline" in prompt
    assert prompt.endswith("ASSISTANT:CHOICE:\n")


def test_batched_scoring_uses_each_exact_left_padded_prompt() -> None:
    class BatchTokenizer:
        padding_side = "right"

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            completion = None
            if text.endswith(" A"):
                text, completion = text[:-2], 10
            elif text.endswith(" B"):
                text, completion = text[:-2], 11
            base = [1] if text == "short" else [1, 2]
            return base + ([] if completion is None else [completion])

        def __call__(self, texts, **kwargs):
            del kwargs
            encoded = [self.encode(text, add_special_tokens=False) for text in texts]
            width = max(map(len, encoded))
            input_ids = []
            masks = []
            for values in encoded:
                padding = [0] * (width - len(values))
                input_ids.append(padding + values)
                masks.append([0] * len(padding) + [1] * len(values))
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(masks),
            }

        def decode(self, token_ids, **kwargs):
            del kwargs
            return f"token-{token_ids[0]}"

        def convert_ids_to_tokens(self, token_id):
            return f"piece-{token_id}"

    class BatchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 2)
            self.calls = 0

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
            assert use_cache is False
            assert logits_to_keep == 1
            assert attention_mask[:, -1].all()
            self.calls += 1
            logits = torch.zeros((input_ids.shape[0], 1, 32))
            logits[:, :, 11] = 2.0
            return SimpleNamespace(logits=logits)

    tokenizer = BatchTokenizer()
    model = BatchModel()
    bundle = SimpleNamespace(tokenizer=tokenizer, model=model)
    jobs = [
        {
            "endpoint_arm": "synthetic",
            "stage": "flat",
            "group": None,
            "prompt_id": f"p{index}",
            "selection_rank": index,
            "rotation_index": 0,
            "rotation": 0,
            "label_to_code": {"first": "A", "second": "B"},
            "prompt": prompt,
            "completions": {"A": " A", "B": " B"},
        }
        for index, prompt in enumerate(("short", "longer"))
    ]
    records, batch_calls, boundary = _score_jobs(bundle, jobs, batch_size=2)
    assert model.calls == batch_calls == 1
    assert [record["argmax_label"] for record in records] == ["second", "second"]
    assert boundary["synthetic:flat"]["context_count"] == 2


def test_committed_calibration_is_the_verified_coded_choice_stop() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "results" / "exploratory_calibration"
    report_path = result_dir / "calibration_report.json"
    forwards_path = result_dir / "calibration_forwards.jsonl"
    averages_path = result_dir / "calibration_rotation_averages.jsonl"
    verification = json.loads((result_dir / "calibration_verification.json").read_text())
    report = json.loads(report_path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(report_path) == verification["artifact_sha256"]["calibration_report"]
    assert digest(forwards_path) == verification["artifact_sha256"]["calibration_forwards"]
    assert digest(averages_path) == verification["artifact_sha256"][
        "calibration_rotation_averages"
    ]
    assert sum(1 for line in forwards_path.read_text().splitlines() if line) == 1296
    assert sum(1 for line in averages_path.read_text().splitlines() if line) == 72
    assert report["calibration_recommendation"]["status"] == "stop_coded_choice"
    assert report["test_split_generated_or_scored"] is False
    assert verification["independent_checks"]["test_prompt_overlap_count"] == 0
