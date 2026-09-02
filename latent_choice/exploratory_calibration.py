#!/usr/bin/env python3
"""Development-only exploratory calibration for the Latent Choice endpoint.

This is deliberately separate from the frozen Latent Choice v1 protocol.  It
does not discover SAE features, install intervention hooks, generate analogies,
or access the confirmatory split.  Its only purpose is to diagnose whether a
semantic domain choice is stable when domain-to-code mappings and token
boundaries change.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np
from scipy.stats import spearmanr
import torch

from latent_choice.choice_endpoint import (
    CODE_SYMBOLS,
    DOMAINS,
    build_choice_instruction,
    validate_mapping,
)
from latent_choice.run import (
    DEFAULT_CODE_TOKENS,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    _development_rows,
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
DEFAULT_PLAN = ROOT / "latent_choice" / "exploratory_calibration_plan.json"
DEFAULT_OUTPUT_DIR = ROOT / "latent_choice" / "outputs" / "exploratory_calibration"
CALIBRATION_ID = "latent-choice-endpoint-calibration-exploratory-v1"
SELECTION_SEED = "latent-choice-endpoint-calibration-v1"
CALIBRATION_PROMPT_COUNT = 24
ROTATION_COUNT = 6
TOP_K = 20
BATCH_SIZE = 2
CANONICAL_PROTOCOL_SHA256 = "06023e7551795726753787e1531fade5f48fafbfb1ddf278c8760fb5dfa8924f"
CANONICAL_MANIFEST_SHA256 = "18100b8a28777539737e5a33b1c00bbccd5da35dd3325828399a5d5e426d5b98"
# Updated only when the explicitly post-v1 exploratory plan itself changes.
CANONICAL_PLAN_SHA256 = "5a057102cfb1fc918afec1ada92a39b2060c149eb2e1e313e26e7cd56449fa3c"

FLAT_ARMS: dict[str, dict[str, Any]] = {
    "flat_current": {
        "prefill": "CHOICE:",
        "completions": {code: f" {code}" for code in CODE_SYMBOLS},
    },
    "flat_newline": {
        "prefill": "CHOICE:\n",
        "completions": {code: code for code in CODE_SYMBOLS},
    },
}

HIERARCHY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "life/health",
        ("biology/ecology", "medicine/public health", "psychology/cognition"),
    ),
    (
        "physical/earth sciences",
        ("physics", "chemistry/materials", "geography/earth/environment"),
    ),
    (
        "technology",
        ("engineering/control", "computer science/software", "AI/neural networks"),
    ),
    (
        "institutions/policy",
        ("economics/markets", "organizations/governance", "law/policy"),
    ),
    (
        "society/humanities",
        ("sociology/culture", "history", "arts/literature"),
    ),
    (
        "learning/play/daily life",
        ("education/learning", "sports/games", "everyday/household"),
    ),
)
GROUP_LABELS: tuple[str, ...] = tuple(group for group, _ in HIERARCHY_GROUPS)
GROUP_CODES: tuple[str, ...] = CODE_SYMBOLS[:6]
SUBDOMAIN_CODES: tuple[str, ...] = CODE_SYMBOLS[:3]


def _digest(*parts: str) -> bytes:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).digest()


def select_calibration_rows(
    rows: Sequence[Mapping[str, Any]], *, count: int = CALIBRATION_PROMPT_COUNT
) -> list[dict[str, Any]]:
    """Hash-select a development-only subset without consulting v1 outcomes."""

    if not 1 <= count <= len(rows):
        raise ValueError("calibration count is outside the supplied rows")
    if any(row.get("split") != "development" for row in rows):
        raise ValueError("exploratory calibration accepts development rows only")
    ids = [str(row.get("prompt_id", "")) for row in rows]
    if any(not prompt_id for prompt_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("development prompt IDs must be nonempty and unique")
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (_digest(SELECTION_SEED, str(row["prompt_id"])), str(row["prompt_id"])),
    )
    return ranked[:count]


def cyclic_mapping(
    labels: Sequence[str], codes: Sequence[str], *, rotation: int
) -> dict[str, str]:
    if len(labels) != len(codes) or not labels:
        raise ValueError("labels and codes must have equal nonzero length")
    if len(set(labels)) != len(labels) or len(set(codes)) != len(codes):
        raise ValueError("labels and codes must be unique")
    return {
        str(label): str(codes[(index + int(rotation)) % len(codes)])
        for index, label in enumerate(labels)
    }


def flat_rotation(selection_rank: int, rotation_index: int) -> int:
    """Six rotations whose pooled offsets cover every A-R code exactly 8 times."""

    if not 0 <= selection_rank < CALIBRATION_PROMPT_COUNT:
        raise ValueError("selection rank is outside the calibration subset")
    if not 0 <= rotation_index < ROTATION_COUNT:
        raise ValueError("rotation index must be in [0, 6)")
    return (6 * selection_rank + 5 * rotation_index) % len(DOMAINS)


def hierarchy_group_rotation(selection_rank: int, rotation_index: int) -> int:
    return (selection_rank + rotation_index) % len(GROUP_LABELS)


def hierarchy_subdomain_rotation(
    selection_rank: int, rotation_index: int, group_index: int
) -> int:
    return (selection_rank + 2 * rotation_index + group_index) % 3


def _chat_prompt(tokenizer: Any, instruction: str, prefill: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str):
        raise TypeError("tokenizer.apply_chat_template must return text")
    return prompt + prefill


def render_flat_prompt(
    tokenizer: Any, row: Mapping[str, Any], mapping: Mapping[str, str], *, prefill: str
) -> str:
    validate_mapping(mapping)
    instruction = build_choice_instruction(
        source_name=str(row["source_name"]),
        source_domain=str(row["source_domain"]),
        source_description=str(row["source_description"]),
        mapping=mapping,
    )
    if prefill == "CHOICE:\n":
        frozen_suffix = (
            "Your response is prefilled with CHOICE:. Complete that fixed prefix with "
            "exactly one listed code and nothing else."
        )
        aligned_suffix = (
            "Your response is prefilled with CHOICE: followed by a newline. Complete "
            "that fixed prefix with exactly one listed code and nothing else."
        )
        if frozen_suffix not in instruction:
            raise RuntimeError("the frozen flat instruction suffix changed")
        instruction = instruction.replace(frozen_suffix, aligned_suffix)
    elif prefill != "CHOICE:":
        raise ValueError("unknown flat prefill")
    return _chat_prompt(tokenizer, instruction, prefill)


def _render_hierarchy_prompt(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    stage: str,
    mapping: Mapping[str, str],
    group: str | None = None,
) -> str:
    code_to_label = {code: label for label, code in mapping.items()}
    ordered_codes = tuple(sorted(code_to_label))
    menu = "\n".join(f"{code} = {code_to_label[code]}" for code in ordered_codes)
    source = (
        f"Source system: {str(row['source_name']).strip()}\n"
        f"Source domain: {str(row['source_domain']).strip()}\n"
        f"Source description: {str(row['source_description']).strip()}"
    )
    if stage == "group":
        instruction = (
            "Choose the broad target-domain group in which you would construct a "
            "structurally faithful analogy. Preserve causal roles, relations, and "
            "boundary conditions. Choose only the group; do not write the analogy.\n\n"
            f"{source}\n\nBroad-group menu:\n{menu}\n\n"
            "Your response is prefilled with GROUP: followed by a newline. Complete "
            "that fixed prefix with exactly one listed code and nothing else."
        )
        return _chat_prompt(tokenizer, instruction, "GROUP:\n")
    if stage != "subdomain" or group is None:
        raise ValueError("hierarchy stage must be group or a named subdomain branch")
    instruction = (
        "Assume the broad target-domain group below is fixed. Choose the specific "
        "target domain in which you would construct a structurally faithful analogy. "
        "Preserve causal roles, relations, and boundary conditions. Choose only the "
        "domain; do not write the analogy.\n\n"
        f"{source}\n\nFixed broad group: {group}\nSpecific-domain menu:\n{menu}\n\n"
        "Your response is prefilled with DOMAIN: followed by a newline. Complete "
        "that fixed prefix with exactly one listed code and nothing else."
    )
    return _chat_prompt(tokenizer, instruction, "DOMAIN:\n")


def resolve_candidate_token_ids(
    tokenizer: Any, prompt: str, completions: Mapping[str, str]
) -> dict[str, int]:
    if not completions:
        raise ValueError("candidate completions cannot be empty")
    base = list(tokenizer.encode(prompt, add_special_tokens=False))
    resolved: dict[str, int] = {}
    for code, completion in completions.items():
        extended = list(tokenizer.encode(prompt + str(completion), add_special_tokens=False))
        if extended[: len(base)] != base or len(extended) != len(base) + 1:
            raise ValueError(
                f"candidate {code!r} is not one prefix-preserving continuation token"
            )
        resolved[str(code)] = int(extended[-1])
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("candidate completions resolve to duplicate token IDs")
    return resolved


def score_candidates(
    logits: torch.Tensor,
    *,
    mapping: Mapping[str, str],
    code_token_ids: Mapping[str, int],
) -> dict[str, Any]:
    if set(mapping.values()) != set(code_token_ids):
        raise ValueError("mapping codes and candidate token IDs differ")
    if len(set(mapping)) != len(mapping) or len(set(mapping.values())) != len(mapping):
        raise ValueError("mapping must be one-to-one")
    if logits.ndim != 1 or not torch.isfinite(logits).all().item():
        raise ValueError("logits must be one finite full-vocabulary vector")
    values = logits.detach().to(dtype=torch.float64, device="cpu")
    labels = tuple(mapping)
    candidate_logits = torch.tensor(
        [values[int(code_token_ids[mapping[label]])].item() for label in labels],
        dtype=torch.float64,
    )
    probabilities = torch.softmax(candidate_logits, dim=0)
    candidate_mass = torch.exp(
        torch.logsumexp(candidate_logits, dim=0) - torch.logsumexp(values, dim=0)
    )
    return {
        "label_logits": {
            label: float(candidate_logits[index].item())
            for index, label in enumerate(labels)
        },
        "label_probabilities": {
            label: float(probabilities[index].item())
            for index, label in enumerate(labels)
        },
        "code_logits": {
            code: float(values[int(token_id)].item())
            for code, token_id in sorted(code_token_ids.items())
        },
        "full_vocab_candidate_mass": float(candidate_mass.item()),
        "argmax_label": labels[int(torch.argmax(candidate_logits).item())],
    }


def _token_flags(text: str) -> dict[str, bool]:
    stripped = text.strip()
    return {
        "is_whitespace": bool(text) and not stripped,
        "contains_newline": "\n" in text,
        "is_punctuation_or_symbol": bool(stripped)
        and all(unicodedata.category(character)[0] in {"P", "S"} for character in stripped),
    }


def top_tokens(
    tokenizer: Any,
    logits: torch.Tensor,
    *,
    candidate_token_ids: Mapping[str, int],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    values = logits.detach().to(dtype=torch.float64, device="cpu")
    top_values, top_ids = torch.topk(values, k=min(top_k, values.numel()))
    log_denominator = torch.logsumexp(values, dim=0)
    candidate_lookup = {int(token_id): code for code, token_id in candidate_token_ids.items()}
    rows: list[dict[str, Any]] = []
    for rank, (value, token_id) in enumerate(zip(top_values.tolist(), top_ids.tolist()), start=1):
        decoded = tokenizer.decode(
            [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        piece = tokenizer.convert_ids_to_tokens(int(token_id))
        rows.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token_piece": str(piece),
                "decoded_text": str(decoded),
                "logit": float(value),
                "full_vocab_probability": float(math.exp(float(value - log_denominator))),
                "candidate_code": candidate_lookup.get(int(token_id)),
                **_token_flags(str(decoded)),
            }
        )
    return rows


def geometric_mean_distribution(
    distributions: Sequence[Mapping[str, float]], labels: Sequence[str]
) -> dict[str, float]:
    values = np.asarray(
        [[float(distribution[label]) for label in labels] for distribution in distributions],
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[0] == 0 or np.any(values < 0):
        raise ValueError("distributions must be nonempty and nonnegative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("each distribution must sum to one")
    averaged_log = np.log(np.clip(values, 1e-300, None)).mean(axis=0)
    normalized = np.exp(averaged_log - averaged_log.max())
    normalized /= normalized.sum()
    return {label: float(normalized[index]) for index, label in enumerate(labels)}


def arithmetic_mean_distribution(
    distributions: Sequence[Mapping[str, float]], labels: Sequence[str]
) -> dict[str, float]:
    values = np.asarray(
        [[float(distribution[label]) for label in labels] for distribution in distributions],
        dtype=np.float64,
    ).mean(axis=0)
    values /= values.sum()
    return {label: float(values[index]) for index, label in enumerate(labels)}


def combine_hierarchy(
    group_probabilities: Mapping[str, float],
    branch_probabilities: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    if set(group_probabilities) != set(GROUP_LABELS):
        raise ValueError("group probabilities do not cover the fixed hierarchy")
    combined: dict[str, float] = {}
    for group, domains in HIERARCHY_GROUPS:
        branch = branch_probabilities.get(group)
        if branch is None or set(branch) != set(domains):
            raise ValueError(f"branch probabilities do not cover {group}")
        for domain in domains:
            combined[domain] = float(group_probabilities[group]) * float(branch[domain])
    if set(combined) != set(DOMAINS) or not math.isclose(
        math.fsum(combined.values()), 1.0, abs_tol=1e-8
    ):
        raise ValueError("combined hierarchy distribution is invalid")
    return {domain: combined[domain] for domain in DOMAINS}


def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    p_term = np.zeros_like(p)
    q_term = np.zeros_like(q)
    p_mask = p > 0
    q_mask = q > 0
    p_term[p_mask] = p[p_mask] * np.log(p[p_mask] / midpoint[p_mask])
    q_term[q_mask] = q[q_mask] * np.log(q[q_mask] / midpoint[q_mask])
    return float(0.5 * (p_term.sum() + q_term.sum()))


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if np.all(left_values == left_values[0]) or np.all(right_values == right_values[0]):
        return None
    value = float(spearmanr(left_values, right_values).statistic)
    return value if math.isfinite(value) else None


def summarize_rotation_distributions(
    rotation_rows: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] = DOMAINS,
    minimum_pairwise_spearman: float = 0.70,
    minimum_robust_modal_prompts: int = 20,
    minimum_stable_loo_prompts: int = 22,
    maximum_loo_jsd: float = 0.05,
    minimum_loo_top_matches: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rotation_rows:
        by_prompt[str(row["prompt_id"])].append(row)
    prompt_summaries: list[dict[str, Any]] = []
    pairwise_rhos: list[float] = []
    undefined_pairwise_rhos = 0
    pairwise_jsds: list[float] = []
    for prompt_id, rows in sorted(by_prompt.items()):
        ordered = sorted(rows, key=lambda row: int(row["rotation_index"]))
        if len(ordered) != ROTATION_COUNT or {
            int(row["rotation_index"]) for row in ordered
        } != set(range(ROTATION_COUNT)):
            raise ValueError(f"prompt {prompt_id} does not have six rotations")
        distributions = [row["domain_probabilities"] for row in ordered]
        arrays = [np.asarray([distribution[label] for label in labels]) for distribution in distributions]
        local_rhos: list[float] = []
        local_undefined_rhos = 0
        local_jsds: list[float] = []
        for left, right in itertools.combinations(arrays, 2):
            rho = _safe_spearman(left, right)
            if rho is None:
                local_undefined_rhos += 1
                undefined_pairwise_rhos += 1
            else:
                local_rhos.append(rho)
            local_jsds.append(_js_divergence(left, right))
        pairwise_rhos.extend(local_rhos)
        pairwise_jsds.extend(local_jsds)
        raw_argmax = [str(row["argmax_domain"]) for row in ordered]
        modal_domain, modal_count = Counter(raw_argmax).most_common(1)[0]
        geometric = geometric_mean_distribution(distributions, labels)
        arithmetic = arithmetic_mean_distribution(distributions, labels)
        geometric_top = max(geometric, key=geometric.get)
        leave_one_out = []
        for omitted in range(ROTATION_COUNT):
            estimate = geometric_mean_distribution(
                [distribution for index, distribution in enumerate(distributions) if index != omitted],
                labels,
            )
            leave_one_out.append(
                {
                    "omitted_rotation_index": omitted,
                    "top_domain": max(estimate, key=estimate.get),
                    "jsd_from_full": _js_divergence(
                        [estimate[label] for label in labels],
                        [geometric[label] for label in labels],
                    ),
                }
            )
        stable_loo = (
            max(item["jsd_from_full"] for item in leave_one_out) <= maximum_loo_jsd
            and sum(item["top_domain"] == geometric_top for item in leave_one_out)
            >= minimum_loo_top_matches
        )
        prompt_summaries.append(
            {
                "prompt_id": prompt_id,
                "raw_argmax_domains": raw_argmax,
                "modal_domain": modal_domain,
                "modal_count": modal_count,
                "modal_agreement": modal_count / ROTATION_COUNT,
                "pairwise_spearman_median": (
                    float(np.median(local_rhos)) if local_rhos else None
                ),
                "undefined_pairwise_spearman_count": local_undefined_rhos,
                "pairwise_jsd_median": float(np.median(local_jsds)),
                "geometric_mean_probabilities": geometric,
                "geometric_argmax_domain": geometric_top,
                "arithmetic_mean_probabilities": arithmetic,
                "arithmetic_argmax_domain": max(arithmetic, key=arithmetic.get),
                "geometric_arithmetic_jsd": _js_divergence(
                    [geometric[label] for label in labels],
                    [arithmetic[label] for label in labels],
                ),
                "leave_one_rotation_out": leave_one_out,
                "leave_one_out_stable": stable_loo,
            }
        )
    if len(prompt_summaries) != CALIBRATION_PROMPT_COUNT:
        raise ValueError("an endpoint arm does not cover all 24 calibration prompts")
    stable_count = sum(bool(row["leave_one_out_stable"]) for row in prompt_summaries)
    robust_modal_count = sum(int(row["modal_count"]) >= 4 for row in prompt_summaries)
    metrics = {
        "prompt_count": len(prompt_summaries),
        "mean_modal_agreement": float(
            np.mean([row["modal_agreement"] for row in prompt_summaries])
        ),
        "fraction_unanimous_argmax": sum(
            int(row["modal_count"]) == ROTATION_COUNT for row in prompt_summaries
        )
        / len(prompt_summaries),
        "prompts_with_at_least_4_of_6_modal_agreement": robust_modal_count,
        "median_pairwise_spearman": (
            float(np.median(pairwise_rhos)) if pairwise_rhos else None
        ),
        "undefined_pairwise_spearman_count": undefined_pairwise_rhos,
        "median_pairwise_jsd": float(np.median(pairwise_jsds)),
        "leave_one_out_stable_prompt_count": stable_count,
        "geometric_arithmetic_argmax_agreement": sum(
            row["geometric_argmax_domain"] == row["arithmetic_argmax_domain"]
            for row in prompt_summaries
        )
        / len(prompt_summaries),
    }
    metrics["strong_mapping_invariance"] = bool(
        undefined_pairwise_rhos == 0
        and metrics["median_pairwise_spearman"] is not None
        and metrics["median_pairwise_spearman"] >= minimum_pairwise_spearman
        and robust_modal_count >= minimum_robust_modal_prompts
        and stable_count >= minimum_stable_loo_prompts
    )
    return prompt_summaries, metrics


def _candidate_mass_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = np.asarray([float(row["full_vocab_candidate_mass"]) for row in rows])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _code_bias_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probabilities: dict[str, list[float]] = defaultdict(list)
    centered_logits: dict[str, list[float]] = defaultdict(list)
    argmax_counts: Counter[str] = Counter()
    for row in rows:
        mapping = row["label_to_code"]
        label_probabilities = row["label_probabilities"]
        code_logits = row["code_logits"]
        mean_logit = float(np.mean(list(code_logits.values())))
        inverse = {label: code for label, code in mapping.items()}
        for label, probability in label_probabilities.items():
            code = inverse[label]
            probabilities[code].append(float(probability))
            centered_logits[code].append(float(code_logits[code]) - mean_logit)
        argmax_counts[inverse[str(row["argmax_label"])]] += 1
    probability_means = {code: float(np.mean(probabilities[code])) for code in sorted(probabilities)}
    centered_means = {
        code: float(np.mean(centered_logits[code])) for code in sorted(centered_logits)
    }
    return {
        "mean_conditional_probability_by_code": probability_means,
        "uniform_probability": 1.0 / len(probability_means),
        "centered_logit_mean_by_code": centered_means,
        "centered_logit_code_effect_sd": float(np.std(list(centered_means.values()))),
        "argmax_code_counts": dict(sorted(argmax_counts.items())),
    }


def _top_token_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    top1_candidate = 0
    candidate_in_top20 = 0
    best_ranks: list[int] = []
    token_probability_lower_bound: dict[tuple[int, str, str], float] = defaultdict(float)
    token_occurrences: Counter[tuple[int, str, str]] = Counter()
    category_counts: Counter[str] = Counter()
    for row in rows:
        candidates = [token for token in row["top_20_unmasked_tokens"] if token["candidate_code"]]
        if row["top_20_unmasked_tokens"][0]["candidate_code"]:
            top1_candidate += 1
        if candidates:
            candidate_in_top20 += 1
            best_ranks.append(min(int(token["rank"]) for token in candidates))
        for token in row["top_20_unmasked_tokens"]:
            key = (int(token["token_id"]), str(token["token_piece"]), str(token["decoded_text"]))
            token_probability_lower_bound[key] += float(token["full_vocab_probability"])
            token_occurrences[key] += 1
        top = row["top_20_unmasked_tokens"][0]
        if top["candidate_code"]:
            category_counts["candidate"] += 1
        elif top["is_whitespace"]:
            category_counts["whitespace"] += 1
        elif top["is_punctuation_or_symbol"]:
            category_counts["punctuation_or_symbol"] += 1
        else:
            category_counts["other_token"] += 1
    ranked_tokens = sorted(
        token_probability_lower_bound,
        key=lambda key: (-token_probability_lower_bound[key], key[0]),
    )[:20]
    return {
        "row_count": len(rows),
        "top1_is_candidate_fraction": top1_candidate / len(rows),
        "candidate_appears_in_top20_fraction": candidate_in_top20 / len(rows),
        "median_best_candidate_rank_when_present": (
            float(np.median(best_ranks)) if best_ranks else None
        ),
        "top1_category_counts": dict(sorted(category_counts.items())),
        "top_tokens_by_mean_probability_lower_bound": [
            {
                "token_id": key[0],
                "token_piece": key[1],
                "decoded_text": key[2],
                "top20_occurrence_count": token_occurrences[key],
                "mean_probability_lower_bound": token_probability_lower_bound[key]
                / len(rows),
            }
            for key in ranked_tokens
        ],
    }


def _paired_prompt_bootstrap(
    current_rows: Sequence[Mapping[str, Any]],
    newline_rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = 10_000,
) -> dict[str, float]:
    def per_prompt(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            mass = float(np.clip(float(row["full_vocab_candidate_mass"]), 1e-12, 1.0 - 1e-12))
            values[str(row["prompt_id"])].append(math.log(mass / (1.0 - mass)))
        return {prompt_id: float(np.mean(prompt_values)) for prompt_id, prompt_values in values.items()}

    current = per_prompt(current_rows)
    newline = per_prompt(newline_rows)
    if set(current) != set(newline):
        raise ValueError("prefix arms do not cover the same prompts")
    prompt_ids = sorted(current)
    differences = np.asarray([newline[prompt_id] - current[prompt_id] for prompt_id in prompt_ids])
    rng = np.random.default_rng(20_260_902)
    samples = rng.integers(0, len(differences), size=(draws, len(differences)))
    bootstrapped = differences[samples].mean(axis=1)
    return {
        "prompt_count": len(prompt_ids),
        "mean_newline_minus_current_logit_mass": float(differences.mean()),
        "bootstrap_95_percent_low": float(np.quantile(bootstrapped, 0.025)),
        "bootstrap_95_percent_high": float(np.quantile(bootstrapped, 0.975)),
        "bootstrap_draws": draws,
    }


def _compare_aggregate_arms(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    left_by_prompt = {str(row["prompt_id"]): row for row in left}
    right_by_prompt = {str(row["prompt_id"]): row for row in right}
    if len(left_by_prompt) != len(left) or len(right_by_prompt) != len(right):
        raise ValueError("an aggregate arm contains duplicate prompt IDs")
    if set(left_by_prompt) != set(right_by_prompt):
        raise ValueError("aggregate arms do not cover the same prompts")
    rhos: list[float] = []
    undefined_rhos = 0
    jsds: list[float] = []
    top_matches = 0
    for prompt_id in sorted(left_by_prompt):
        left_q = left_by_prompt[prompt_id]["geometric_mean_probabilities"]
        right_q = right_by_prompt[prompt_id]["geometric_mean_probabilities"]
        left_values = [left_q[domain] for domain in DOMAINS]
        right_values = [right_q[domain] for domain in DOMAINS]
        rho = _safe_spearman(left_values, right_values)
        if rho is None:
            undefined_rhos += 1
        else:
            rhos.append(rho)
        jsds.append(_js_divergence(left_values, right_values))
        top_matches += max(left_q, key=left_q.get) == max(right_q, key=right_q.get)
    return {
        "prompt_count": len(left_by_prompt),
        "top_domain_agreement": top_matches / len(left_by_prompt),
        "median_spearman": float(np.median(rhos)) if rhos else None,
        "undefined_spearman_count": undefined_rhos,
        "median_jsd": float(np.median(jsds)),
    }


def _load_plan(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_PLAN.resolve():
        raise ValueError("the real calibration requires the canonical plan path")
    if sha256_file(path) != CANONICAL_PLAN_SHA256:
        raise ValueError("exploratory calibration plan hash drift")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("calibration_id") != CALIBRATION_ID:
        raise ValueError("wrong exploratory calibration plan")
    if value.get("prompt_sampling", {}).get("count") != CALIBRATION_PROMPT_COUNT:
        raise ValueError("exploratory calibration prompt-count drift")
    if value.get("rotation_design", {}).get("count_per_prompt") != ROTATION_COUNT:
        raise ValueError("exploratory calibration rotation-count drift")
    integrity = value.get("integrity", {})
    if integrity.get("scored_decision_position_count") != 1296:
        raise ValueError("exploratory calibration decision-position count drift")
    if integrity.get("batch_size") != BATCH_SIZE:
        raise ValueError("exploratory calibration batch-size drift")
    if integrity.get("model_batch_forward_count") != 648:
        raise ValueError("exploratory calibration batch-forward count drift")
    planned_groups = value.get("hierarchy_groups")
    observed_groups = {group: list(domains) for group, domains in HIERARCHY_GROUPS}
    if planned_groups != observed_groups:
        raise ValueError("exploratory hierarchy differs from the written plan")
    if value.get("prompt_sampling", {}).get("seed") != SELECTION_SEED:
        raise ValueError("exploratory calibration selection-seed drift")
    if tuple(value.get("decision_rubric", {}).get("arm_order", ())) != (
        "flat_current",
        "flat_newline",
        "hierarchical_newline",
    ):
        raise ValueError("exploratory endpoint arm-order drift")
    expected_arm_names = {"flat_current", "flat_newline", "hierarchical_newline"}
    if set(value.get("endpoint_arms", {})) != expected_arm_names:
        raise ValueError("exploratory endpoint-arm drift")
    return value


def _prepare_jobs(
    tokenizer: Any, selected_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for selection_rank, row in enumerate(selected_rows):
        prompt_id = str(row["prompt_id"])
        for rotation_index in range(ROTATION_COUNT):
            rotation = flat_rotation(selection_rank, rotation_index)
            mapping = cyclic_mapping(DOMAINS, CODE_SYMBOLS, rotation=rotation)
            for arm, configuration in FLAT_ARMS.items():
                prompt = render_flat_prompt(
                    tokenizer, row, mapping, prefill=str(configuration["prefill"])
                )
                jobs.append(
                    {
                        "endpoint_arm": arm,
                        "stage": "flat",
                        "group": None,
                        "prompt_id": prompt_id,
                        "selection_rank": selection_rank,
                        "rotation_index": rotation_index,
                        "rotation": rotation,
                        "label_to_code": mapping,
                        "prompt": prompt,
                        "completions": dict(configuration["completions"]),
                    }
                )

            group_rotation = hierarchy_group_rotation(selection_rank, rotation_index)
            group_mapping = cyclic_mapping(GROUP_LABELS, GROUP_CODES, rotation=group_rotation)
            group_prompt = _render_hierarchy_prompt(
                tokenizer, row, stage="group", mapping=group_mapping
            )
            jobs.append(
                {
                    "endpoint_arm": "hierarchical_newline",
                    "stage": "group",
                    "group": None,
                    "prompt_id": prompt_id,
                    "selection_rank": selection_rank,
                    "rotation_index": rotation_index,
                    "rotation": group_rotation,
                    "label_to_code": group_mapping,
                    "prompt": group_prompt,
                    "completions": {code: code for code in GROUP_CODES},
                }
            )
            for group_index, (group, domains) in enumerate(HIERARCHY_GROUPS):
                branch_rotation = hierarchy_subdomain_rotation(
                    selection_rank, rotation_index, group_index
                )
                branch_mapping = cyclic_mapping(
                    domains, SUBDOMAIN_CODES, rotation=branch_rotation
                )
                branch_prompt = _render_hierarchy_prompt(
                    tokenizer,
                    row,
                    stage="subdomain",
                    mapping=branch_mapping,
                    group=group,
                )
                jobs.append(
                    {
                        "endpoint_arm": "hierarchical_newline",
                        "stage": "subdomain",
                        "group": group,
                        "prompt_id": prompt_id,
                        "selection_rank": selection_rank,
                        "rotation_index": rotation_index,
                        "rotation": branch_rotation,
                        "label_to_code": branch_mapping,
                        "prompt": branch_prompt,
                        "completions": {code: code for code in SUBDOMAIN_CODES},
                    }
                )
    expected = CALIBRATION_PROMPT_COUNT * ROTATION_COUNT * (len(FLAT_ARMS) + 1 + 6)
    if len(jobs) != expected:
        raise RuntimeError(f"prepared {len(jobs)} jobs, expected {expected}")
    return jobs


@torch.no_grad()
def _score_jobs(
    bundle: Any, jobs: list[dict[str, Any]], *, batch_size: int
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    tokenizer = bundle.tokenizer
    tokenizer.padding_side = "left"
    # Validate every exact continuation boundary before the first model forward.
    expected_current_ids = {
        str(code): int(token_id)
        for code, token_id in json.loads(DEFAULT_CODE_TOKENS.read_text())["code_token_ids"].items()
    }
    contract_maps: dict[tuple[str, str], set[tuple[tuple[str, int], ...]]] = defaultdict(set)
    contract_counts: Counter[tuple[str, str]] = Counter()
    for job in jobs:
        if set(job["label_to_code"].values()) != set(job["completions"]):
            raise ValueError("job mapping and completion codes differ")
        completion_values = [str(value) for value in job["completions"].values()]
        if any(not value for value in completion_values) or len(completion_values) != len(
            set(completion_values)
        ):
            raise ValueError("job completion strings must be nonempty and unique")
        job["candidate_token_ids"] = resolve_candidate_token_ids(
            tokenizer, str(job["prompt"]), job["completions"]
        )
        if job["endpoint_arm"] == "flat_current" and job["candidate_token_ids"] != expected_current_ids:
            raise ValueError("the original arm differs from the committed v1 token IDs")
        contract_key = (str(job["endpoint_arm"]), str(job["stage"]))
        contract_maps[contract_key].add(tuple(sorted(job["candidate_token_ids"].items())))
        contract_counts[contract_key] += 1
    records: list[dict[str, Any]] = []
    batch_forward_count = 0
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        encoded = tokenizer(
            [str(job["prompt"]) for job in batch],
            add_special_tokens=False,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if not bool(encoded["attention_mask"][:, -1].all().item()):
            raise RuntimeError("a left-padded calibration prompt does not end in content")
        for row_index, job in enumerate(batch):
            individual = list(tokenizer.encode(str(job["prompt"]), add_special_tokens=False))
            batched = encoded["input_ids"][row_index][
                encoded["attention_mask"][row_index].to(dtype=torch.bool)
            ].tolist()
            if batched != individual:
                raise RuntimeError("batched tokenization differs from the exact preflight prompt")
        device = bundle.model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = bundle.model(**encoded, use_cache=False, logits_to_keep=1)
        if (
            output.logits.ndim != 3
            or output.logits.shape[0] != len(batch)
            or output.logits.shape[1] != 1
        ):
            raise RuntimeError(
                f"unexpected final-logit shape {tuple(output.logits.shape)} for batch {len(batch)}"
            )
        logits_batch = output.logits[:, -1, :].detach().cpu()
        batch_forward_count += 1
        for job, logits in zip(batch, logits_batch):
            score = score_candidates(
                logits,
                mapping=job["label_to_code"],
                code_token_ids=job["candidate_token_ids"],
            )
            records.append(
                {
                    "schema_version": "1.0",
                    "calibration_id": CALIBRATION_ID,
                    "record_type": "exploratory_endpoint_forward",
                    "exploratory": True,
                    "split": "development",
                    "endpoint_arm": job["endpoint_arm"],
                    "stage": job["stage"],
                    "group": job["group"],
                    "prompt_id": job["prompt_id"],
                    "selection_rank": job["selection_rank"],
                    "rotation_index": job["rotation_index"],
                    "rotation": job["rotation"],
                    "choice_prompt_sha256": hashlib.sha256(
                        str(job["prompt"]).encode("utf-8")
                    ).hexdigest(),
                    "label_to_code": job["label_to_code"],
                    "candidate_token_ids": job["candidate_token_ids"],
                    **score,
                    "top_20_unmasked_tokens": top_tokens(
                        tokenizer,
                        logits,
                        candidate_token_ids=job["candidate_token_ids"],
                    ),
                }
            )
    boundary_summary = {
        f"{arm}:{stage}": {
            "context_count": contract_counts[(arm, stage)],
            "distinct_token_id_maps": len(maps),
            "token_ids_constant_across_contexts": len(maps) == 1,
            "token_id_maps": [dict(items) for items in sorted(maps)],
        }
        for (arm, stage), maps in sorted(contract_maps.items())
    }
    return records, batch_forward_count, boundary_summary


def _rotation_rows(forward_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in forward_records:
        if record["stage"] != "flat":
            continue
        probabilities = {domain: float(record["label_probabilities"][domain]) for domain in DOMAINS}
        rows.append(
            {
                "prompt_id": record["prompt_id"],
                "selection_rank": record["selection_rank"],
                "endpoint_arm": record["endpoint_arm"],
                "rotation_index": record["rotation_index"],
                "domain_probabilities": probabilities,
                "argmax_domain": max(probabilities, key=probabilities.get),
                "flat_candidate_mass": record["full_vocab_candidate_mass"],
            }
        )
    hierarchy: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in forward_records:
        if record["endpoint_arm"] != "hierarchical_newline":
            continue
        key = (str(record["prompt_id"]), int(record["rotation_index"]))
        branch_key = "group" if record["stage"] == "group" else str(record["group"])
        hierarchy[key][branch_key] = record
    for (prompt_id, rotation_index), stages in sorted(hierarchy.items()):
        if set(stages) != {"group", *GROUP_LABELS}:
            raise ValueError("a hierarchy rotation is missing a group or branch forward")
        group_record = stages["group"]
        group_probabilities = {
            group: float(group_record["label_probabilities"][group]) for group in GROUP_LABELS
        }
        branch_probabilities = {
            group: {
                domain: float(stages[group]["label_probabilities"][domain])
                for domain in dict(HIERARCHY_GROUPS)[group]
            }
            for group in GROUP_LABELS
        }
        combined = combine_hierarchy(group_probabilities, branch_probabilities)
        weighted_branch_mass = math.fsum(
            group_probabilities[group]
            * float(stages[group]["full_vocab_candidate_mass"])
            for group in GROUP_LABELS
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "selection_rank": group_record["selection_rank"],
                "endpoint_arm": "hierarchical_newline",
                "rotation_index": rotation_index,
                "domain_probabilities": combined,
                "argmax_domain": max(combined, key=combined.get),
                "group_candidate_mass": group_record["full_vocab_candidate_mass"],
                "probability_weighted_subdomain_candidate_mass": weighted_branch_mass,
                "two_stage_candidate_path_mass": float(
                    group_record["full_vocab_candidate_mass"]
                )
                * weighted_branch_mass,
            }
        )
    return rows


def run_calibration(
    *,
    plan_path: Path,
    protocol_path: Path,
    manifest_path: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    plan = _load_plan(plan_path)
    if protocol_path.resolve() != DEFAULT_PROTOCOL.resolve():
        raise ValueError("the real calibration requires canonical latent_choice/protocol.json")
    if manifest_path.resolve() != DEFAULT_MANIFEST.resolve():
        raise ValueError("the real calibration requires the canonical prompt manifest")
    if sha256_file(protocol_path) != CANONICAL_PROTOCOL_SHA256:
        raise ValueError("frozen Latent Choice v1 protocol hash drift")
    if sha256_file(manifest_path) != CANONICAL_MANIFEST_SHA256:
        raise ValueError("frozen development prompt-manifest hash drift")
    if batch_size != BATCH_SIZE:
        raise ValueError(f"the calibration batch size is frozen at {BATCH_SIZE}")
    if output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError("the real calibration requires the canonical ignored output directory")
    from latent_choice.validate_protocol import load_json, validate

    validate(
        load_json(protocol_path),
        load_json(ROOT / "latent_choice" / "test_config.template.json"),
        load_json(DEFAULT_CODE_TOKENS),
    )
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("protocol_id") != "latent-choice-v1":
        raise ValueError("the calibration must reference the frozen v1 protocol")
    pre_run_commit, pre_run_dirty = git_state()
    if pre_run_commit is None or pre_run_dirty is not False:
        raise ValueError("the real exploratory calibration requires a clean Git worktree")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("calibration output directory is not empty")

    development_rows = _development_rows(manifest_path, protocol)
    selected_rows = select_calibration_rows(development_rows)
    bundle = load_pinned_bundle(token=os.environ.get("HF_TOKEN"))
    jobs = _prepare_jobs(bundle.tokenizer, selected_rows)
    forward_records, batch_forward_count, token_boundary_summary = _score_jobs(
        bundle, jobs, batch_size=batch_size
    )
    if len(forward_records) != 1296:
        raise RuntimeError("the calibration did not produce exactly 1,296 decision positions")
    if batch_forward_count != 648:
        raise RuntimeError("the calibration did not produce exactly 648 model batch forwards")
    if {record["split"] for record in forward_records} != {"development"}:
        raise RuntimeError("a non-development record entered calibration")
    if any("source_description" in record or "prompt" in record for record in forward_records):
        raise RuntimeError("source text leaked into a public calibration record")

    rotation_rows = _rotation_rows(forward_records)
    rubric = plan["decision_rubric"]["strong_mapping_invariance"]
    arm_prompt_summaries: dict[str, list[dict[str, Any]]] = {}
    arm_metrics: dict[str, dict[str, Any]] = {}
    aggregate_records: list[dict[str, Any]] = []
    for arm in ("flat_current", "flat_newline", "hierarchical_newline"):
        arm_rows = [row for row in rotation_rows if row["endpoint_arm"] == arm]
        prompt_summaries, metrics = summarize_rotation_distributions(
            arm_rows,
            minimum_pairwise_spearman=float(
                rubric["median_pairwise_spearman_minimum"]
            ),
            minimum_robust_modal_prompts=int(
                rubric["prompts_with_at_least_4_of_6_matching_raw_argmax_minimum"]
            ),
            minimum_stable_loo_prompts=int(
                rubric["leave_one_rotation_out_stable_prompts_minimum"]
            ),
            maximum_loo_jsd=float(rubric["leave_one_rotation_out_jsd_maximum"]),
            minimum_loo_top_matches=int(
                rubric["leave_one_rotation_out_top_domain_matches_minimum"]
            ),
        )
        arm_prompt_summaries[arm] = prompt_summaries
        arm_metrics[arm] = metrics
        for prompt_summary in prompt_summaries:
            aggregate_records.append(
                {
                    "schema_version": "1.0",
                    "calibration_id": CALIBRATION_ID,
                    "record_type": "exploratory_rotation_average",
                    "exploratory": True,
                    "split": "development",
                    "endpoint_arm": arm,
                    **prompt_summary,
                }
            )

    flat_forward = {
        arm: [
            record
            for record in forward_records
            if record["endpoint_arm"] == arm and record["stage"] == "flat"
        ]
        for arm in FLAT_ARMS
    }
    hierarchy_group_forwards = [
        record
        for record in forward_records
        if record["endpoint_arm"] == "hierarchical_newline" and record["stage"] == "group"
    ]
    hierarchy_branch_forwards = [
        record
        for record in forward_records
        if record["endpoint_arm"] == "hierarchical_newline" and record["stage"] == "subdomain"
    ]
    hierarchy_rotation_rows = [
        row for row in rotation_rows if row["endpoint_arm"] == "hierarchical_newline"
    ]

    report = {
        "schema_version": "1.0",
        "artifact": "latent_choice_exploratory_endpoint_calibration",
        "calibration_id": CALIBRATION_ID,
        "status": "complete",
        "exploratory_not_preregistered": True,
        "split": "development",
        "selected_prompt_count": len(selected_rows),
        "selected_prompt_ids": [str(row["prompt_id"]) for row in selected_rows],
        "rotation_count_per_prompt": ROTATION_COUNT,
        "batch_size": batch_size,
        "decision_position_count": len(forward_records),
        "model_batch_forward_count": batch_forward_count,
        "test_split_generated_or_scored": False,
        "sae_feature_association_computed": False,
        "sae_intervention_performed": False,
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "sae_repo_id_loaded_but_not_analyzed": SAE_REPO_ID,
        "sae_revision": SAE_REVISION,
        "sae_sha256": SAE_SHA256,
        "plan_sha256": sha256_file(plan_path),
        "protocol_sha256": sha256_file(protocol_path),
        "prompt_manifest_sha256": sha256_file(manifest_path),
        "environment_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "pre_run_git_commit": pre_run_commit,
        "pre_run_working_tree_dirty": pre_run_dirty,
        "canonical_input_validation": {
            "plan_path_and_sha256": True,
            "v1_protocol_path_sha256_and_structure": True,
            "prompt_manifest_path_and_sha256": True,
            "v1_code_token_manifest_structure": True,
        },
        "token_boundary_validation": token_boundary_summary,
        "forward_counts_by_arm_and_stage": {
            f"{arm}:{stage}": count
            for (arm, stage), count in sorted(
                Counter(
                    (str(record["endpoint_arm"]), str(record["stage"]))
                    for record in forward_records
                ).items()
            )
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "endpoint_metrics": {
            "flat_current": {
                "mapping_invariance": arm_metrics["flat_current"],
                "candidate_mass": _candidate_mass_summary(flat_forward["flat_current"]),
                "code_identity": _code_bias_summary(flat_forward["flat_current"]),
                "top_unmasked_tokens": _top_token_summary(flat_forward["flat_current"]),
            },
            "flat_newline": {
                "mapping_invariance": arm_metrics["flat_newline"],
                "candidate_mass": _candidate_mass_summary(flat_forward["flat_newline"]),
                "code_identity": _code_bias_summary(flat_forward["flat_newline"]),
                "top_unmasked_tokens": _top_token_summary(flat_forward["flat_newline"]),
            },
            "hierarchical_newline": {
                "mapping_invariance": arm_metrics["hierarchical_newline"],
                "group_candidate_mass": _candidate_mass_summary(hierarchy_group_forwards),
                "subdomain_candidate_mass": _candidate_mass_summary(hierarchy_branch_forwards),
                "probability_weighted_subdomain_candidate_mass": _candidate_mass_summary(
                    [
                        {
                            "full_vocab_candidate_mass": row[
                                "probability_weighted_subdomain_candidate_mass"
                            ]
                        }
                        for row in hierarchy_rotation_rows
                    ]
                ),
                "two_stage_candidate_path_mass": _candidate_mass_summary(
                    [
                        {"full_vocab_candidate_mass": row["two_stage_candidate_path_mass"]}
                        for row in hierarchy_rotation_rows
                    ]
                ),
                "group_code_identity": _code_bias_summary(hierarchy_group_forwards),
                "subdomain_code_identity": _code_bias_summary(hierarchy_branch_forwards),
                "group_top_unmasked_tokens": _top_token_summary(hierarchy_group_forwards),
                "subdomain_top_unmasked_tokens": _top_token_summary(
                    hierarchy_branch_forwards
                ),
            },
        },
        "paired_prefix_mass_change": _paired_prompt_bootstrap(
            flat_forward["flat_current"], flat_forward["flat_newline"]
        ),
        "aggregate_endpoint_comparisons": {
            "current_vs_newline": _compare_aggregate_arms(
                arm_prompt_summaries["flat_current"],
                arm_prompt_summaries["flat_newline"],
            ),
            "newline_flat_vs_hierarchy": _compare_aggregate_arms(
                arm_prompt_summaries["flat_newline"],
                arm_prompt_summaries["hierarchical_newline"],
            ),
        },
    }
    simplicity_order = tuple(plan["decision_rubric"]["arm_order"])
    passing = [
        arm
        for arm in simplicity_order
        if bool(arm_metrics[arm]["strong_mapping_invariance"])
    ]
    report["calibration_recommendation"] = {
        "status": "proceed_to_v2_design" if passing else "stop_coded_choice",
        "preferred_endpoint_by_predeclared_simplicity": passing[0] if passing else None,
        "interpretation": (
            "Exploratory mapping invariance met the pre-run descriptive rubric. A separate "
            "v2 protocol may now be frozen before further development analysis."
            if passing
            else "No coded endpoint met the pre-run mapping-invariance rubric; do not create "
            "Latent Choice v2 from these calibration prompts."
        ),
    }

    post_run_commit, post_run_dirty = git_state()
    report["post_run_git_commit"] = post_run_commit
    report["post_run_working_tree_dirty"] = post_run_dirty
    if post_run_commit != pre_run_commit or post_run_dirty is not False:
        raise RuntimeError("Git state changed during exploratory calibration")

    # Refuse Python's permissive non-standard NaN/Infinity JSON before publishing.
    json.dumps(forward_records, allow_nan=False)
    json.dumps(aggregate_records, allow_nan=False)
    json.dumps(report, allow_nan=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    forwards_path = output_dir / "calibration_forwards.jsonl"
    aggregates_path = output_dir / "calibration_rotation_averages.jsonl"
    report_path = output_dir / "calibration_report.json"
    atomic_write_jsonl(forwards_path, forward_records)
    atomic_write_jsonl(aggregates_path, aggregate_records)
    report["artifact_sha256"] = {
        "calibration_forwards": sha256_file(forwards_path),
        "calibration_rotation_averages": sha256_file(aggregates_path),
    }
    atomic_write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, choices=(BATCH_SIZE,))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_calibration(
        plan_path=args.plan,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
