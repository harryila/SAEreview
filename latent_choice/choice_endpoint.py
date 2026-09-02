"""Deterministic explicit-domain choice endpoint for Latent Choice.

The endpoint deliberately separates the causal action (one domain-code token)
from downstream analogy prose.  Domain-to-code assignments rotate across
prompts so a domain effect cannot be identified with one fixed vocabulary token.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch


DOMAINS: tuple[str, ...] = (
    "biology/ecology",
    "medicine/public health",
    "physics",
    "chemistry/materials",
    "engineering/control",
    "computer science/software",
    "AI/neural networks",
    "economics/markets",
    "organizations/governance",
    "sociology/culture",
    "psychology/cognition",
    "education/learning",
    "law/policy",
    "history",
    "arts/literature",
    "sports/games",
    "geography/earth/environment",
    "everyday/household",
)
CODE_SYMBOLS: tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(18))
CODE_COMPLETIONS: dict[str, str] = {code: f" {code}" for code in CODE_SYMBOLS}
DEFAULT_MAPPING_SEED = "latent-choice-domain-code-map-v1"
DEFAULT_DRAW_SEED = "latent-choice-paired-draw-v1"


@dataclass(frozen=True)
class ChoiceScore:
    """Auditable outputs from one next-token domain decision."""

    domain_logits: dict[str, float]
    domain_probabilities: dict[str, float]
    code_logits: dict[str, float]
    candidate_token_ids: dict[str, int]
    full_vocab_candidate_mass: float
    mapping_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _digest(*parts: str) -> bytes:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).digest()


def balanced_rotations(
    prompt_ids: Sequence[str], *, seed: str = DEFAULT_MAPPING_SEED
) -> dict[str, int]:
    """Assign near-perfectly balanced rotations without using input order."""

    ids = [str(prompt_id) for prompt_id in prompt_ids]
    if not ids:
        raise ValueError("prompt_ids cannot be empty")
    if any(not prompt_id for prompt_id in ids):
        raise ValueError("prompt IDs must be nonempty")
    if len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be unique")
    ranked = sorted(ids, key=lambda prompt_id: (_digest(seed, prompt_id), prompt_id))
    return {prompt_id: rank % len(DOMAINS) for rank, prompt_id in enumerate(ranked)}


def mapping_for_prompt(
    prompt_id: str,
    split_prompt_ids: Sequence[str],
    *,
    seed: str = DEFAULT_MAPPING_SEED,
) -> dict[str, str]:
    """Return the canonical-domain to displayed-code mapping for one prompt."""

    rotations = balanced_rotations(split_prompt_ids, seed=seed)
    if prompt_id not in rotations:
        raise KeyError(f"prompt ID {prompt_id!r} is absent from this split")
    rotation = rotations[prompt_id]
    mapping = {
        domain: CODE_SYMBOLS[(domain_index + rotation) % len(CODE_SYMBOLS)]
        for domain_index, domain in enumerate(DOMAINS)
    }
    validate_mapping(mapping)
    return mapping


def validate_mapping(mapping: Mapping[str, str]) -> None:
    if set(mapping) != set(DOMAINS):
        missing = sorted(set(DOMAINS).difference(mapping))
        extra = sorted(set(mapping).difference(DOMAINS))
        raise ValueError(f"domain mapping mismatch; missing={missing}, extra={extra}")
    codes = list(mapping.values())
    if len(codes) != len(set(codes)) or set(codes) != set(CODE_SYMBOLS):
        raise ValueError("mapping must use every choice code exactly once")


def mapping_sha256(mapping: Mapping[str, str]) -> str:
    validate_mapping(mapping)
    payload = json.dumps(
        dict(mapping), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_choice_instruction(
    *,
    source_name: str,
    source_domain: str,
    source_description: str,
    mapping: Mapping[str, str],
) -> str:
    """Build the stage-one instruction; no analogy is generated in this stage."""

    validate_mapping(mapping)
    if (
        not source_name.strip()
        or not source_domain.strip()
        or not source_description.strip()
    ):
        raise ValueError("source_name, source_domain, and source_description must be nonempty")
    code_to_domain = {code: domain for domain, code in mapping.items()}
    menu = "\n".join(
        f"{code} = {code_to_domain[code]}" for code in CODE_SYMBOLS
    )
    return (
        "Choose the target domain in which you would construct a structurally "
        "faithful analogy for the source mechanism below. Preserve causal roles, "
        "relations, and boundary conditions. This step chooses only the domain; "
        "do not write the analogy yet.\n\n"
        f"Source system: {source_name.strip()}\n"
        f"Source domain: {source_domain.strip()}\n"
        f"Source description: {source_description.strip()}\n\n"
        "Domain menu:\n"
        f"{menu}\n\n"
        "Your response is prefilled with CHOICE:. Complete that fixed prefix with "
        "exactly one listed code and nothing else."
    )


def render_choice_prompt(
    tokenizer: Any,
    *,
    source_name: str,
    source_domain: str,
    source_description: str,
    mapping: Mapping[str, str],
) -> str:
    """Render the official chat template and prefill the fixed decision prefix."""

    instruction = build_choice_instruction(
        source_name=source_name,
        source_domain=source_domain,
        source_description=source_description,
        mapping=mapping,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str):
        raise TypeError("tokenizer.apply_chat_template must return text")
    return prompt + "CHOICE:"


def resolve_code_token_ids(
    tokenizer: Any,
    prompt: str,
    *,
    completions: Mapping[str, str] = CODE_COMPLETIONS,
) -> dict[str, int]:
    """Require each code to be one unique continuation token in exact context."""

    if set(completions) != set(CODE_SYMBOLS):
        raise ValueError("completions must cover the frozen code alphabet")
    base = list(tokenizer.encode(prompt, add_special_tokens=False))
    resolved: dict[str, int] = {}
    for code in CODE_SYMBOLS:
        extended = list(
            tokenizer.encode(prompt + completions[code], add_special_tokens=False)
        )
        if extended[: len(base)] != base or len(extended) != len(base) + 1:
            raise ValueError(
                f"choice code {code!r} is not one continuation token in context"
            )
        resolved[code] = int(extended[-1])
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("choice codes do not resolve to unique token IDs")
    return resolved


def score_choice_logits(
    logits: torch.Tensor,
    *,
    mapping: Mapping[str, str],
    code_token_ids: Mapping[str, int],
) -> ChoiceScore:
    """Convert one full-vocabulary logit vector to the frozen choice endpoint."""

    validate_mapping(mapping)
    if set(code_token_ids) != set(CODE_SYMBOLS):
        raise ValueError("code_token_ids must cover every choice code")
    if len(set(code_token_ids.values())) != len(code_token_ids):
        raise ValueError("candidate token IDs must be unique")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional full-vocabulary tensor")
    values = logits.detach().to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(values).all().item():
        raise ValueError("logits must be finite")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or not 0 <= token_id < values.numel()
        for token_id in code_token_ids.values()
    ):
        raise ValueError("a candidate token ID is outside the vocabulary")

    domain_logits_tensor = torch.tensor(
        [values[code_token_ids[mapping[domain]]].item() for domain in DOMAINS],
        dtype=torch.float64,
    )
    probabilities = torch.softmax(domain_logits_tensor, dim=0)
    candidate_mass = torch.exp(
        torch.logsumexp(domain_logits_tensor, dim=0) - torch.logsumexp(values, dim=0)
    )
    domain_logits = {
        domain: float(domain_logits_tensor[index].item())
        for index, domain in enumerate(DOMAINS)
    }
    probability_values = [float(value) for value in probabilities.tolist()]
    serialized_total = math.fsum(probability_values)
    domain_probabilities = {
        domain: probability_values[index] / serialized_total
        for index, domain in enumerate(DOMAINS)
    }
    code_logits = {
        code: float(values[token_id].item())
        for code, token_id in sorted(code_token_ids.items())
    }
    return ChoiceScore(
        domain_logits=domain_logits,
        domain_probabilities=domain_probabilities,
        code_logits=code_logits,
        candidate_token_ids={
            code: int(token_id) for code, token_id in sorted(code_token_ids.items())
        },
        full_vocab_candidate_mass=float(candidate_mass.item()),
        mapping_sha256=mapping_sha256(mapping),
    )


def paired_uniform(
    prompt_id: str,
    sample_index: int,
    *,
    seed: str = DEFAULT_DRAW_SEED,
) -> float:
    """Return a condition-independent uniform variate strictly inside (0, 1)."""

    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise TypeError("sample_index must be an integer")
    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")
    integer = int.from_bytes(
        _digest(seed, str(prompt_id), str(sample_index))[:8], "big"
    )
    return (integer + 0.5) / 2**64


def inverse_cdf_domain(
    domain_probabilities: Mapping[str, float], *, uniform: float
) -> str:
    """Sample in canonical-domain order, making paired draws arm-invariant."""

    if set(domain_probabilities) != set(DOMAINS):
        raise ValueError("domain probabilities must cover the frozen domains")
    if not math.isfinite(float(uniform)) or not 0.0 < float(uniform) < 1.0:
        raise ValueError("uniform must be strictly between zero and one")
    values = [float(domain_probabilities[domain]) for domain in DOMAINS]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("domain probabilities must be finite and nonnegative")
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"domain probabilities sum to {total}, not one")
    cumulative = 0.0
    for domain, probability in zip(DOMAINS, values):
        cumulative += probability / total
        if uniform < cumulative:
            return domain
    return DOMAINS[-1]


__all__ = [
    "CODE_COMPLETIONS",
    "CODE_SYMBOLS",
    "ChoiceScore",
    "DEFAULT_DRAW_SEED",
    "DEFAULT_MAPPING_SEED",
    "DOMAINS",
    "balanced_rotations",
    "build_choice_instruction",
    "inverse_cdf_domain",
    "mapping_for_prompt",
    "mapping_sha256",
    "paired_uniform",
    "render_choice_prompt",
    "resolve_code_token_ids",
    "score_choice_logits",
    "validate_mapping",
]
