from __future__ import annotations

from collections import Counter
import math

import pytest
import torch

from latent_choice.choice_endpoint import (
    CODE_SYMBOLS,
    DOMAINS,
    balanced_rotations,
    build_choice_instruction,
    inverse_cdf_domain,
    mapping_for_prompt,
    paired_uniform,
    resolve_code_token_ids,
    score_choice_logits,
)


def test_balanced_rotations_are_order_independent_and_near_equal() -> None:
    prompt_ids = [f"dev-{index:03d}" for index in range(80)]
    forward = balanced_rotations(prompt_ids)
    reverse = balanced_rotations(list(reversed(prompt_ids)))
    assert forward == reverse
    counts = Counter(forward.values())
    assert set(counts) == set(range(18))
    assert max(counts.values()) - min(counts.values()) == 1

    for domain in DOMAINS:
        code_counts = Counter(
            mapping_for_prompt(prompt_id, prompt_ids)[domain]
            for prompt_id in prompt_ids
        )
        assert set(code_counts) == set(CODE_SYMBOLS)
        assert max(code_counts.values()) - min(code_counts.values()) == 1


def test_mapping_rejects_duplicate_prompt_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        balanced_rotations(["a", "a"])


def test_instruction_uses_18_substantive_domains_and_no_other() -> None:
    prompt_ids = ["p"]
    mapping = mapping_for_prompt("p", prompt_ids)
    instruction = build_choice_instruction(
        source_name="Source",
        source_domain="source science",
        source_description="A causal mechanism.",
        mapping=mapping,
    )
    assert all(domain in instruction for domain in DOMAINS)
    assert "\nother\n" not in instruction
    assert instruction.count(" = ") == 18


class _OneTokenTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        for index, code in enumerate(CODE_SYMBOLS):
            if text.endswith(f" {code}"):
                return [101, 102, 1000 + index]
        return [101, 102]


class _BrokenTokenizer(_OneTokenTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        result = super().encode(text, add_special_tokens=add_special_tokens)
        if text.endswith(" R"):
            return result + [9999]
        return result


def test_code_tokens_are_unique_single_token_continuations() -> None:
    ids = resolve_code_token_ids(_OneTokenTokenizer(), "prompt")
    assert ids == {code: 1000 + index for index, code in enumerate(CODE_SYMBOLS)}
    with pytest.raises(ValueError, match="one continuation token"):
        resolve_code_token_ids(_BrokenTokenizer(), "prompt")


def test_logits_map_back_to_domains_and_probability_contract() -> None:
    prompt_ids = [f"p{index}" for index in range(18)]
    mapping = mapping_for_prompt("p7", prompt_ids)
    token_ids = {code: 20 + index for index, code in enumerate(CODE_SYMBOLS)}
    logits = torch.full((100,), -5.0)
    expected_domain = DOMAINS[4]
    logits[token_ids[mapping[expected_domain]]] = 7.0
    score = score_choice_logits(logits, mapping=mapping, code_token_ids=token_ids)
    assert max(score.domain_probabilities, key=score.domain_probabilities.get) == expected_domain
    assert math.isclose(sum(score.domain_probabilities.values()), 1.0, abs_tol=1e-6)
    assert 0.0 < score.full_vocab_candidate_mass < 1.0
    assert set(score.code_logits) == set(CODE_SYMBOLS)


def test_paired_inverse_cdf_draw_is_deterministic() -> None:
    probabilities = {domain: 0.0 for domain in DOMAINS}
    probabilities[DOMAINS[2]] = 0.25
    probabilities[DOMAINS[11]] = 0.75
    assert inverse_cdf_domain(probabilities, uniform=0.20) == DOMAINS[2]
    assert inverse_cdf_domain(probabilities, uniform=0.25) == DOMAINS[11]
    assert inverse_cdf_domain(probabilities, uniform=0.90) == DOMAINS[11]
    first = paired_uniform("dev-001", 3)
    assert first == paired_uniform("dev-001", 3)
    assert 0.0 < first < 1.0
    assert first != paired_uniform("dev-001", 4)
