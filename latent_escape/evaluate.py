#!/usr/bin/env python3
"""Evaluate paired Latent Escape runs with prompt-clustered inference.

The confirmatory command reports the full prompt population first and a
development-thresholded eligible population second. The ``mde`` command is a
design-stage sensitivity calculation and must not be used as post-hoc power.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np

try:  # Support package imports and direct CLI execution.
    from .protocol_amendment import amendment_sha256, load_protocol_amendment
except ImportError:  # pragma: no cover - direct script execution
    from protocol_amendment import amendment_sha256, load_protocol_amendment


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "latent_escape" / "protocol.json"
POWER_REPORT_PATH = ROOT / "latent_escape" / "power_report.json"
DISTANCE_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
DISTANCE_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
QUALITY_REQUIRED_ARMS = frozenset({"baseline", "targeted_feature_suppression"})
QUALITY_AMENDMENT_PATH = ROOT / "latent_escape" / "protocol_amendment_4.json"
QUALITY_RATING_ID_NAMESPACE = "latent-escape-quality-rating-v2"

GenerationKey = tuple[str, str, str, int, int]
QualityAssignment = tuple[GenerationKey, str]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def amendment_provenance(amendment: dict[str, Any]) -> dict[str, Any]:
    guide = amendment["domain_labeling_guide"]
    return {
        "effective_protocol_revision": int(amendment["effective_protocol_revision"]),
        "protocol_amendment_id": str(amendment["amendment_id"]),
        "protocol_amendment_sha256": amendment_sha256(),
        "domain_labeling_guide_id": str(guide["id"]),
        "domain_labeling_guide_sha256": str(guide["sha256"]),
    }


def validate_amendment_provenance(
    record: dict[str, Any], expected: dict[str, Any], artifact_name: str
) -> None:
    drift = {
        key: {"observed": record.get(key), "expected": value}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if drift:
        raise ValueError(f"{artifact_name} amendment/guide provenance drift: {drift}")


def atomic_write_json(path: Path, value: Any) -> str:
    # Preserve insertion order so the confirmatory full population is rendered
    # before the explicitly secondary eligible-prompt analysis.
    payload = (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_bytes(payload)


def read_jsonl_many(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                row["_source_path"] = str(path)
                rows.append(row)
    if not rows:
        raise ValueError("input JSONL files have no records")
    return rows


def canonical_seed(record: dict[str, Any]) -> int:
    value = record.get("seed", record.get("paired_seed"))
    if value is None:
        raise ValueError("record lacks seed")
    return int(value)


def join_key(record: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(record.get("run_id", "")),
        str(record["prompt_id"]),
        str(record["condition"]),
        int(record["sample_index"]),
        canonical_seed(record),
    )


def pair_key(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["sample_index"]), canonical_seed(record)


def feature_id(record: dict[str, Any]) -> int | None:
    value = record.get("feature_id")
    if value is None and isinstance(record.get("intervention"), dict):
        value = record["intervention"].get("feature_id")
    return None if value is None else int(value)


def generated_text(record: dict[str, Any]) -> str:
    for key in ("generated_text", "raw_text", "completion", "response_text", "text"):
        if isinstance(record.get(key), str):
            return str(record[key])
    parsed = record.get("parsed_output", record.get("parsed_json"))
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    raise ValueError("generation record lacks generated text")


def quality_blind_id(record: dict[str, Any]) -> str:
    identity = json.dumps(join_key(record), ensure_ascii=False, separators=(",", ":"))
    text_hash = sha256_bytes(generated_text(record).encode("utf-8"))
    return sha256_bytes(
        f"latent-escape-quality-v1|{identity}|{text_hash}".encode("utf-8")
    )[:32]


def quality_rating_blind_id(record: dict[str, Any], rating_slot: str) -> str:
    """Return a different opaque ID for primary and reliability assignments."""

    if rating_slot not in {"primary", "reliability"}:
        raise ValueError(f"unknown structural-quality rating slot {rating_slot!r}")
    return sha256_bytes(
        (
            f"{QUALITY_RATING_ID_NAMESPACE}|{rating_slot}|"
            f"{quality_blind_id(record)}"
        ).encode("utf-8")
    )[:32]


def quality_guardrail_sampling(protocol: dict[str, Any]) -> dict[str, Any]:
    """Load the frozen revision-4 quality-sampling amendment."""

    value = protocol.get("quality_guardrail_sampling")
    if value is None and QUALITY_AMENDMENT_PATH.exists():
        amendment = load_protocol_amendment(protocol)
        value = amendment.get("quality_guardrail_sampling")
    if not isinstance(value, dict):
        raise ValueError("missing frozen quality_guardrail_sampling amendment")
    required = {
        "pair_selection_seed",
        "reliability_selection_seed",
        "samples_per_prompt",
        "reliability_prompt_fraction",
        "required_arms",
        "primary_rater_estimand",
        "duplicate_rater_use",
        "expected_workload",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(
            f"quality_guardrail_sampling lacks required fields: {sorted(missing)}"
        )
    if int(value["samples_per_prompt"]) != 1:
        raise ValueError("structural quality must use one paired sample per prompt")
    if list(value["required_arms"]) != [
        "baseline",
        "targeted_feature_suppression",
    ]:
        raise ValueError("quality sampling arms differ from the frozen protocol")
    return value


def quality_pair_hash(
    selection_seed: str,
    split: str,
    prompt_id: str,
    sample_index: int,
    paired_seed: int,
) -> str:
    """Hash a pre-outcome paired generation identity for sample selection."""

    return sha256_bytes(
        (
            f"{selection_seed}|{split}|{prompt_id}|"
            f"{int(sample_index)}|{int(paired_seed)}"
        ).encode("utf-8")
    )


def select_quality_pair(
    selection_seed: str,
    split: str,
    prompt_id: str,
    candidates: set[tuple[int, int]],
) -> tuple[int, int]:
    """Select one sample/seed pair without consulting generated outcomes."""

    if not candidates:
        raise ValueError(f"prompt {prompt_id} has no paired quality candidates")
    return min(
        candidates,
        key=lambda pair: (
            quality_pair_hash(
                selection_seed, split, prompt_id, pair[0], pair[1]
            ),
            pair,
        ),
    )


def select_reliability_prompts(
    selection_seed: str,
    split: str,
    prompt_ids: set[str],
    prompt_count: int,
) -> set[str]:
    """Hash-select the frozen prompt-level duplicate-rater subset."""

    if not 0 <= prompt_count <= len(prompt_ids):
        raise ValueError("invalid duplicate-rater prompt count")
    ordered = sorted(
        prompt_ids,
        key=lambda prompt_id: (
            sha256_bytes(
                f"{selection_seed}|{split}|{prompt_id}".encode("utf-8")
            ),
            prompt_id,
        ),
    )
    return set(ordered[:prompt_count])


def quality_sampling_plan_sha256(
    selected_pairs: dict[str, tuple[int, int]],
) -> str:
    """Hash the canonical pre-outcome prompt/sample/seed quality plan."""

    plan = [
        {
            "prompt_id": prompt_id,
            "sample_index": int(pair[0]),
            "paired_seed": int(pair[1]),
        }
        for prompt_id, pair in sorted(selected_pairs.items())
    ]
    return canonical_json_hash(plan)


def quality_reliability_subset_sha256(prompt_ids: set[str]) -> str:
    """Hash the canonical prompt-level duplicate-rater subset."""

    return canonical_json_hash(sorted(prompt_ids))


def validate_quality_plan_config(
    config: dict[str, Any], sampling_provenance: dict[str, Any]
) -> None:
    """Bind a test run to the exact deterministic quality-rating plan."""

    expected = {
        "quality_sampling_plan_sha256": sampling_provenance[
            "quality_sampling_plan_sha256"
        ],
        "quality_reliability_subset_sha256": sampling_provenance[
            "quality_reliability_subset_sha256"
        ],
    }
    drift = {
        key: {"observed": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if drift:
        raise ValueError(f"test quality sampling plan hash drift: {drift}")


def validate_generation_test_config_binding(
    generations: list[dict[str, Any]], test_config_path: Path
) -> None:
    """Reject confirmatory records created under a different frozen config."""

    expected = sha256_file(test_config_path)
    mismatched = [
        join_key(generation)
        for generation in generations
        if generation.get("test_config_sha256") != expected
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} test generations have test-config hash drift"
        )


def select_quality_generations(
    generations: list[dict[str, Any]],
    split: str,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[GenerationKey], dict[str, Any]]:
    """Select the frozen paired quality sample and reliability assignments."""

    sampling = quality_guardrail_sampling(protocol)
    workload_name = "test" if split == "test" else "development_gate"
    workload = sampling["expected_workload"].get(workload_name)
    if not isinstance(workload, dict):
        raise ValueError(f"missing quality workload for {workload_name}")
    required_arms = ["baseline", "targeted_feature_suppression"]
    covered = [
        generation
        for generation in generations
        if arm_name(generation) in set(required_arms)
    ]
    if not covered:
        raise ValueError("no frozen structural-quality arms were supplied")

    grouped: dict[tuple[str, str], dict[tuple[int, int], dict[str, Any]]] = (
        defaultdict(dict)
    )
    prompts_by_arm: dict[str, set[str]] = defaultdict(set)
    for generation in covered:
        if str(generation.get("split")) != split:
            raise ValueError("quality generation belongs to the wrong split")
        prompt_id = str(generation["prompt_id"])
        arm = arm_name(generation)
        pair = pair_key(generation)
        bucket = grouped[(prompt_id, arm)]
        if pair in bucket:
            raise ValueError(
                f"duplicate quality candidate for prompt {prompt_id}, arm {arm}, pair {pair}"
            )
        bucket[pair] = generation
        prompts_by_arm[arm].add(prompt_id)

    expected_prompts = int(workload["prompt_count"])
    prompt_ids = prompts_by_arm[required_arms[0]]
    if len(prompt_ids) != expected_prompts:
        raise ValueError(
            f"quality workload expects {expected_prompts} prompts, found {len(prompt_ids)}"
        )
    if any(prompts_by_arm[arm] != prompt_ids for arm in required_arms[1:]):
        raise ValueError("baseline and targeted quality prompt sets differ")
    expected_candidates = (
        int(protocol["generation"]["test_paired_samples_per_prompt_per_condition"])
        if split == "test"
        else int(protocol["development_intervention_gate"]["paired_samples_per_prompt"])
    )

    selected: list[dict[str, Any]] = []
    selected_pairs: dict[str, tuple[int, int]] = {}
    for prompt_id in sorted(prompt_ids):
        arm_pairs = {
            arm: set(grouped[(prompt_id, arm)]) for arm in required_arms
        }
        baseline_pairs = arm_pairs[required_arms[0]]
        if len(baseline_pairs) != expected_candidates:
            raise ValueError(
                f"prompt {prompt_id} requires {expected_candidates} paired candidates; "
                f"found {len(baseline_pairs)}"
            )
        if any(pairs != baseline_pairs for pairs in arm_pairs.values()):
            raise ValueError(
                f"baseline and targeted sample/seed pairs differ for prompt {prompt_id}"
            )
        chosen = select_quality_pair(
            str(sampling["pair_selection_seed"]),
            split,
            prompt_id,
            baseline_pairs,
        )
        selected_pairs[prompt_id] = chosen
        selected.extend(grouped[(prompt_id, arm)][chosen] for arm in required_arms)

    reliability_count = int(workload["reliability_prompt_count"])
    fraction_count = int(
        math.ceil(float(sampling["reliability_prompt_fraction"]) * len(prompt_ids))
    )
    if reliability_count != fraction_count:
        raise ValueError("frozen reliability workload does not match its prompt fraction")
    reliability_prompts = select_reliability_prompts(
        str(sampling["reliability_selection_seed"]),
        split,
        prompt_ids,
        reliability_count,
    )
    reliability_keys = {
        join_key(generation)
        for generation in selected
        if str(generation["prompt_id"]) in reliability_prompts
    }
    unique_count = len(selected)
    duplicate_count = len(reliability_keys)
    expected_counts = {
        "unique_generation_ratings": unique_count,
        "duplicate_ratings": duplicate_count,
        "total_rating_tasks": unique_count + duplicate_count,
    }
    drifted = {
        key: (int(workload[key]), observed)
        for key, observed in expected_counts.items()
        if int(workload[key]) != observed
    }
    if drifted:
        raise ValueError(f"quality workload counts have drifted: {drifted}")

    selected.sort(
        key=lambda row: (
            str(row["prompt_id"]),
            required_arms.index(arm_name(row)),
        )
    )
    plan_hash = quality_sampling_plan_sha256(selected_pairs)
    reliability_hash = quality_reliability_subset_sha256(reliability_prompts)
    return selected, reliability_keys, {
        "split": split,
        "selection_unit": "paired_sample_within_source_prompt",
        "pair_selection_seed": sampling["pair_selection_seed"],
        "samples_per_prompt": int(sampling["samples_per_prompt"]),
        "prompt_count": len(prompt_ids),
        "selected_pairs": {
            prompt_id: {
                "sample_index": pair[0],
                "paired_seed": pair[1],
            }
            for prompt_id, pair in sorted(selected_pairs.items())
        },
        "required_arms": list(sampling["required_arms"]),
        "unique_generation_rating_count": unique_count,
        "reliability_selection_seed": sampling["reliability_selection_seed"],
        "reliability_prompt_fraction": float(
            sampling["reliability_prompt_fraction"]
        ),
        "reliability_prompt_ids": sorted(reliability_prompts),
        "reliability_prompt_count": len(reliability_prompts),
        "duplicate_rating_count": duplicate_count,
        "total_rating_task_count": unique_count + duplicate_count,
        "quality_sampling_plan_sha256": plan_hash,
        "quality_reliability_subset_sha256": reliability_hash,
        "primary_rater_estimand": sampling["primary_rater_estimand"],
        "duplicate_rater_use": sampling["duplicate_rater_use"],
    }


def quality_assignment_mapping(
    generations: list[dict[str, Any]],
    reliability_keys: set[GenerationKey],
) -> dict[str, QualityAssignment]:
    """Map opaque rating IDs to a generation and its prespecified rater slot."""

    mapping: dict[str, QualityAssignment] = {}
    for generation in generations:
        key = join_key(generation)
        slots = ["primary"]
        if key in reliability_keys:
            slots.append("reliability")
        for slot in slots:
            blind_id = quality_rating_blind_id(generation, slot)
            if blind_id in mapping:
                raise ValueError(f"duplicate quality rating blind ID {blind_id}")
            mapping[blind_id] = (key, slot)
    return mapping


def target_only_text(record: dict[str, Any]) -> tuple[str, str]:
    """Extract target-system content without reusing the self-reported domain."""
    parsed = record.get("parsed_output", record.get("parsed_json"))
    if not isinstance(parsed, dict):
        try:
            candidate = json.loads(generated_text(record))
            parsed = candidate if isinstance(candidate, dict) else None
        except json.JSONDecodeError:
            parsed = None
    parts: list[str] = []
    if isinstance(parsed, dict):
        if parsed.get("target_system") is not None:
            parts.append(str(parsed["target_system"]))
        mappings = parsed.get("mappings")
        if isinstance(mappings, list):
            for mapping in mappings:
                if isinstance(mapping, dict) and mapping.get("target_role") is not None:
                    parts.append(str(mapping["target_role"]))
    cleaned = "\n".join(part.strip() for part in parts if part.strip())
    if cleaned:
        return cleaned, "target_system_and_target_roles"
    return generated_text(record), "full_generated_text_fallback"


def arm_name(record: dict[str, Any]) -> str:
    condition = str(record["condition"])
    if condition == "matched_random_feature_suppression":
        current_feature = feature_id(record)
        if current_feature is None:
            raise ValueError("matched-random record lacks feature_id")
        return f"matched_random_feature_suppression:{current_feature}"
    if condition == "targeted_feature_suppression":
        intervention = record.get("intervention")
        strength = (
            float(intervention.get("strength", 1.0))
            if isinstance(intervention, dict)
            else 1.0
        )
        if not math.isclose(strength, 1.0, rel_tol=0.0, abs_tol=1e-12):
            return f"targeted_feature_suppression:dose={format(strength, '.12g')}"
    return condition


def requires_structural_quality(record: dict[str, Any]) -> bool:
    """Return whether the frozen primary quality guardrail covers this output."""
    return arm_name(record) in QUALITY_REQUIRED_ARMS


def structural_quality(row: dict[str, Any]) -> float:
    value = next(
        (
            row[key]
            for key in ("structural_quality", "quality_score", "score")
            if row.get(key) is not None
        ),
        None,
    )
    if value is None:
        raise ValueError("quality row lacks structural_quality/quality_score/score")
    score = float(value)
    if not 1.0 <= score <= 5.0:
        raise ValueError(f"structural quality must be in [1, 5], got {score}")
    return score


def validate_label_audit(
    labels: list[dict[str, Any]],
    allow_unreviewed: bool,
    expected_amendment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if expected_amendment:
        audit_expected = {
            key: expected_amendment[key]
            for key in (
                "protocol_amendment_id",
                "protocol_amendment_sha256",
                "domain_labeling_guide_id",
                "domain_labeling_guide_sha256",
            )
        }
        for row in labels:
            validate_amendment_provenance(
                row, audit_expected, "independent domain-label row"
            )
    selected = [row for row in labels if row.get("audit_selected") is True]
    incomplete = [row for row in selected if row.get("manual_audited") is not True]
    agreements = [
        row.get("domain_label") == row.get("classifier_domain_label") for row in selected
    ]
    by_class: dict[str, list[bool]] = defaultdict(list)
    for row, agreement in zip(selected, agreements):
        by_class[str(row.get("classifier_domain_label"))].append(agreement)
    overall = float(np.mean(agreements)) if agreements else None
    class_summary = {
        domain: {"count": len(values), "exact_agreement": float(np.mean(values))}
        for domain, values in sorted(by_class.items())
    }
    design_errors: list[str] = []
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        by_run[str(row.get("run_id"))].append(row)
    for run_id, rows in by_run.items():
        expected = int(math.ceil(0.10 * len(rows)))
        observed = sum(row.get("audit_selected") is True for row in rows)
        if observed != expected or any(
            not math.isclose(float(row.get("audit_fraction", -1.0)), 0.10)
            or row.get("audit_seed") != "latent-escape-domain-audit-v1"
            for row in rows
        ):
            design_errors.append(run_id)
    passed = bool(
        selected
        and not incomplete
        and not design_errors
        and overall is not None
        and overall >= 0.80
        and all(
            value["count"] < 5 or value["exact_agreement"] >= 0.60
            for value in class_summary.values()
        )
    )
    if not passed and not allow_unreviewed:
        raise ValueError(
            "independent-domain-label audit is incomplete or below its frozen agreement gate"
        )
    return {
        "queued_count": len(selected),
        "incomplete_count": len(incomplete),
        "overall_exact_agreement": overall,
        "by_classifier_domain": class_summary,
        "audit_design_error_run_ids": design_errors,
        "gate_pass": passed,
        "bypassed_for_smoke": bool(not passed and allow_unreviewed),
    }


def load_quality(
    paths: list[Path] | None,
    allow_unblinded: bool,
    blind_id_mapping: dict[str, QualityAssignment],
    expected_judge_protocol: str,
    expected_prompt_sha256: str,
    expected_primary_keys: set[GenerationKey],
    expected_reliability_keys: set[GenerationKey],
) -> tuple[dict[GenerationKey, float], dict[str, Any]]:
    if not paths:
        return {}, {"available": False, "rating_count": 0}
    rows = read_jsonl_many(paths)
    ratings: dict[QualityAssignment, tuple[float, str]] = {}
    for row in rows:
        blind_id = row.get("blind_quality_id")
        if blind_id is not None:
            if str(blind_id) not in blind_id_mapping:
                raise ValueError(f"unknown blind_quality_id {blind_id}")
            key, rating_slot = blind_id_mapping[str(blind_id)]
            derived_blinding = True
        else:
            if not allow_unblinded:
                raise ValueError(
                    "primary quality rows must join through blind_quality_id"
                )
            key = join_key(row)
            rating_slot = str(row.get("rating_slot", "primary"))
            derived_blinding = False
        assignment = (key, rating_slot)
        if assignment in ratings:
            raise ValueError(
                f"duplicate structural-quality {rating_slot} assignment for {key}"
            )
        if not allow_unblinded and not derived_blinding and not (
            row.get("blinded") is True or row.get("condition_hidden") is True
        ):
            raise ValueError("quality rows must record blinded=true or condition_hidden=true")
        if not allow_unblinded and (
            row.get("judge_protocol_id") != expected_judge_protocol
            or row.get("judge_prompt_sha256") != expected_prompt_sha256
        ):
            raise ValueError("quality judge protocol or rubric hash has drifted")
        rater_id = str(row.get("rater_id") or "").strip()
        if not allow_unblinded and not rater_id:
            raise ValueError("blinded structural-quality rows require nonempty rater_id")
        ratings[assignment] = (structural_quality(row), rater_id)

    expected_assignments = {
        (key, "primary") for key in expected_primary_keys
    } | {(key, "reliability") for key in expected_reliability_keys}
    if set(ratings) != expected_assignments:
        missing = len(expected_assignments - set(ratings))
        extra = len(set(ratings) - expected_assignments)
        raise ValueError(
            "quality ratings do not match the frozen primary/reliability assignments: "
            f"{missing} missing, {extra} extra"
        )
    for key in expected_reliability_keys:
        primary_rater = ratings[(key, "primary")][1]
        reliability_rater = ratings[(key, "reliability")][1]
        if not allow_unblinded and primary_rater == reliability_rater:
            raise ValueError(
                "duplicate structural-quality assignments require distinct rater IDs"
            )

    quality_map = {
        key: ratings[(key, "primary")][0] for key in expected_primary_keys
    }
    primary_scores = [
        ratings[(key, "primary")][0] for key in sorted(expected_reliability_keys)
    ]
    reliability_scores = [
        ratings[(key, "reliability")][0] for key in sorted(expected_reliability_keys)
    ]
    if primary_scores:
        exact_agreement = float(
            np.mean(np.asarray(primary_scores) == np.asarray(reliability_scores))
        )
        within_one = float(
            np.mean(
                np.abs(
                    np.asarray(primary_scores, dtype=np.float64)
                    - np.asarray(reliability_scores, dtype=np.float64)
                )
                <= 1.0
            )
        )
        from sklearn.metrics import cohen_kappa_score

        kappa = float(
            cohen_kappa_score(primary_scores, reliability_scores, weights="linear")
        )
        linear_kappa: float | None = kappa if math.isfinite(kappa) else None
    else:
        exact_agreement = None
        within_one = None
        linear_kappa = None
    ratings_per_generation = Counter(
        2 if key in expected_reliability_keys else 1 for key in expected_primary_keys
    )
    return (
        quality_map,
        {
            "available": True,
            "rating_count": len(rows),
            "primary_rating_count": len(expected_primary_keys),
            "duplicate_rating_count": len(expected_reliability_keys),
            "rated_generation_count": len(expected_primary_keys),
            "ratings_per_generation": dict(sorted(ratings_per_generation.items())),
            "primary_endpoint_uses_duplicate_ratings": False,
            "duplicate_rater_reliability": {
                "prompt_count": len(
                    {key[1] for key in expected_reliability_keys}
                ),
                "item_count": len(expected_reliability_keys),
                "exact_agreement": exact_agreement,
                "within_one_point_agreement": within_one,
                "linear_weighted_cohen_kappa": linear_kappa,
            },
        },
    )


def prepare_quality_command(args: argparse.Namespace) -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    amendment = load_protocol_amendment(protocol)
    effective_provenance = amendment_provenance(amendment)
    quality_protocol = protocol["outcomes"]["quality_guardrail"]
    test_config: dict[str, Any] = {}
    if not args.split:
        raise ValueError("quality queue preparation requires an explicit split")
    if args.split == "test":
        if args.test_config is None:
            raise ValueError("test quality preparation requires --test-config")
        test_config = load_test_config(args.test_config, protocol, args.split)
    generations = read_jsonl_many(args.generations)
    generations = [row for row in generations if row.get("split") == args.split]
    if not generations:
        raise ValueError("no generations selected for quality queue")
    if args.split == "test":
        validate_generation_test_config_binding(generations, args.test_config)
    all_generation_count = len(generations)
    quality_generations, reliability_keys, selection = select_quality_generations(
        generations, args.split, protocol
    )
    if test_config:
        validate_quality_plan_config(test_config, selection)
    assignments = quality_assignment_mapping(quality_generations, reliability_keys)
    manifest_path = args.prompt_manifest
    manifest_rows = read_jsonl_many([manifest_path])
    manifest = {str(row["prompt_id"]): row for row in manifest_rows}
    queue: list[dict[str, Any]] = []
    generations_by_key = {
        join_key(generation): generation for generation in quality_generations
    }
    for blind_id, (key, _rating_slot) in assignments.items():
        generation = generations_by_key[key]
        if generation.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError("generation protocol ID mismatch")
        prompt_id = str(generation["prompt_id"])
        if prompt_id not in manifest:
            raise ValueError(f"prompt {prompt_id} is absent from prompt manifest")
        prompt = manifest[prompt_id]
        queue.append(
            {
                "schema_version": 1,
                "record_type": "blinded_structural_quality_rating_assignment",
                "blind_quality_id": blind_id,
                "source_system": prompt.get("source_name"),
                "source_domain": prompt.get("source_domain"),
                "source_description": prompt.get("source_description"),
                "analogy_text": generated_text(generation),
                "rubric": quality_protocol["rubric"],
                "judge_protocol_id": quality_protocol["judge_protocol_id"],
                "judge_prompt_sha256": quality_protocol["rubric_sha256"],
                "structural_quality": None,
                "rater_id": None,
                "blinded": True,
            }
        )
    queue.sort(key=lambda row: row["blind_quality_id"])
    payload = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in queue
        )
        + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    metadata = {
        "schema_version": 1,
        "artifact": "blinded_structural_quality_queue",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        **effective_provenance,
        "manifest_sha256": sha256_file(manifest_path),
        "generation_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.generations
        ],
        "queue_sha256": sha256_bytes(payload),
        "item_count": len(queue),
        "primary_rating_count": len(quality_generations),
        "duplicate_rating_count": len(reliability_keys),
        "included_arms": sorted({arm_name(row) for row in quality_generations}),
        "excluded_generation_count": all_generation_count - len(quality_generations),
        "quality_sampling": selection,
        "judge_protocol_id": quality_protocol["judge_protocol_id"],
        "judge_prompt_sha256": quality_protocol["rubric_sha256"],
        "hidden_fields": [
            "prompt_id",
            "split",
            "condition",
            "seed",
            "feature_id",
            "intervention",
        ],
    }
    atomic_write_json(args.output.with_name(args.output.name + ".meta.json"), metadata)
    print(f"wrote {args.output} ({len(queue)} blinded quality items)")
    return 0


def distance_command(args: argparse.Namespace) -> int:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional model-run dependency
        raise RuntimeError(
            "semantic distance requires sentence-transformers in the locked environment"
        ) from exc
    protocol = json.loads(PROTOCOL_PATH.read_text())
    generations = read_jsonl_many(args.generations)
    if args.split:
        generations = [row for row in generations if row.get("split") == args.split]
    if not generations:
        raise ValueError("no generations selected for semantic distance")
    manifest_rows = read_jsonl_many([args.prompt_manifest])
    manifest = {str(row["prompt_id"]): row for row in manifest_rows}
    source_texts: list[str] = []
    target_texts: list[str] = []
    extraction_methods: list[str] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    for generation in generations:
        if generation.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError("generation protocol ID mismatch")
        key = join_key(generation)
        if key in seen:
            raise ValueError(f"duplicate generation key {key}")
        seen.add(key)
        prompt_id = str(generation["prompt_id"])
        if prompt_id not in manifest:
            raise ValueError(f"prompt {prompt_id} is absent from prompt manifest")
        prompt = manifest[prompt_id]
        source_texts.append(
            f"{prompt.get('source_name', '')}\n{prompt.get('source_description', '')}".strip()
        )
        target_text, extraction = target_only_text(generation)
        target_texts.append(target_text)
        extraction_methods.append(extraction)
    model = SentenceTransformer(
        DISTANCE_MODEL_REPO,
        revision=DISTANCE_MODEL_REVISION,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    embeddings = model.encode(
        source_texts + target_texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=not args.no_progress,
    )
    count = len(generations)
    distances = 1.0 - np.sum(embeddings[:count] * embeddings[count:], axis=1)
    rows: list[dict[str, Any]] = []
    for generation, source_text, target_text, extraction, distance in zip(
        generations, source_texts, target_texts, extraction_methods, distances
    ):
        rows.append(
            {
                "schema_version": 1,
                "record_type": "source_target_semantic_distance",
                "protocol_id": protocol["protocol_id"],
                "run_id": generation.get("run_id"),
                "prompt_id": str(generation["prompt_id"]),
                "split": generation.get("split"),
                "condition": str(generation["condition"]),
                "sample_index": int(generation["sample_index"]),
                "seed": canonical_seed(generation),
                "source_text_sha256": sha256_bytes(source_text.encode("utf-8")),
                "target_text_sha256": sha256_bytes(target_text.encode("utf-8")),
                "target_extraction": extraction,
                "semantic_distance": float(distance),
                "embedding_model": DISTANCE_MODEL_REPO,
                "embedding_revision": DISTANCE_MODEL_REVISION,
            }
        )
    rows.sort(key=lambda row: join_key(row))
    payload = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    metadata = {
        "schema_version": 1,
        "artifact": "source_target_semantic_distance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_repo": DISTANCE_MODEL_REPO,
        "model_revision": DISTANCE_MODEL_REVISION,
        "generation_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.generations
        ],
        "manifest_sha256": sha256_file(args.prompt_manifest),
        "output_sha256": sha256_bytes(payload),
        "record_count": len(rows),
        "target_extraction_counts": dict(sorted(Counter(extraction_methods).items())),
    }
    atomic_write_json(args.output.with_name(args.output.name + ".meta.json"), metadata)
    print(f"wrote {args.output} ({len(rows)} semantic distances)")
    return 0


def load_distances(
    paths: list[Path] | None,
) -> tuple[dict[tuple[str, str, str, int, int], float], dict[str, Any]]:
    if not paths:
        return {}, {"available": False, "record_count": 0}
    rows = read_jsonl_many(paths)
    result: dict[tuple[str, str, str, int, int], float] = {}
    for row in rows:
        if row.get("embedding_model") != DISTANCE_MODEL_REPO or row.get(
            "embedding_revision"
        ) != DISTANCE_MODEL_REVISION:
            raise ValueError("semantic-distance model or revision drift")
        key = join_key(row)
        if key in result:
            raise ValueError(f"duplicate semantic-distance key {key}")
        value = float(row["semantic_distance"])
        if not math.isfinite(value):
            raise ValueError("semantic distance is not finite")
        result[key] = value
    return result, {
        "available": True,
        "record_count": len(rows),
        "model_repo": DISTANCE_MODEL_REPO,
        "model_revision": DISTANCE_MODEL_REVISION,
    }


def entropy(labels: list[str], taxonomy: list[str]) -> float:
    counts = np.asarray([labels.count(domain) for domain in taxonomy], dtype=np.float64)
    probabilities = counts[counts > 0] / len(labels)
    return float(-np.sum(probabilities * np.log(probabilities)))


def clustered_summary(values: np.ndarray, bootstrap_indices: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"estimate": None, "ci95": None, "prompt_count": 0}
    draws = values[bootstrap_indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "prompt_count": len(values),
    }


def bootstrap_indices(prompt_count: int, resamples: int, seed: int) -> np.ndarray:
    if prompt_count < 1 or resamples < 1:
        raise ValueError("prompt count and bootstrap resamples must be positive")
    return np.random.default_rng(seed).integers(
        0, prompt_count, size=(resamples, prompt_count), dtype=np.int32
    )


def load_test_config(path: Path, protocol: dict[str, Any], split: str) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("test config protocol mismatch")
    validate_amendment_provenance(
        config,
        amendment_provenance(load_protocol_amendment(protocol)),
        "test config",
    )
    required = protocol["required_before_test"]
    missing = [
        key
        for key in required
        if config.get(key) is None or (isinstance(config.get(key), list) and not config[key])
    ]
    if missing:
        raise ValueError(f"test config has unfrozen required fields: {missing}")
    if len(config["five_matched_random_feature_ids"]) != 5:
        raise ValueError("test config must freeze exactly five matched-random features")
    if split == "test" and not config.get("frozen_at_utc"):
        raise ValueError("test evaluation requires a timestamped frozen test config")
    expected_power_hash = protocol["power"]["power_report_sha256"]
    if config["power_report_sha256"] != expected_power_hash:
        raise ValueError("test config power-report hash drift")
    if POWER_REPORT_PATH.exists() and sha256_file(POWER_REPORT_PATH) != expected_power_hash:
        raise ValueError("local frozen power report hash drift")
    quality = protocol["outcomes"]["quality_guardrail"]
    if (
        config.get("judge_model_or_rater_protocol") != quality["judge_protocol_id"]
        or config.get("judge_prompt_sha256") != quality["rubric_sha256"]
    ):
        raise ValueError("test quality judge or rubric differs from protocol")
    gate_path = Path(str(config["development_gate_report_path"]))
    if not gate_path.is_absolute():
        gate_path = ROOT / gate_path
    if not gate_path.exists() or sha256_file(gate_path) != config[
        "development_gate_report_sha256"
    ]:
        raise ValueError("frozen development-gate report is missing or has drifted")
    gate_report = json.loads(gate_path.read_text())
    if gate_report.get("development_intervention_gate", {}).get("status") != "pass":
        raise ValueError("test config does not point to a passing development gate")
    if (
        gate_report.get("selected_domain") != config["selected_domain"]
        or int(gate_report.get("selected_feature_id", -1))
        != int(config["selected_feature_id"])
        or [int(value) for value in gate_report.get("matched_random_feature_ids", [])]
        != [int(value) for value in config["five_matched_random_feature_ids"]]
        or not math.isclose(
            float(gate_report.get("eligible_activation_threshold")),
            float(config["eligible_activation_threshold"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("test choices differ from the passing development gate")
    return config


def load_eligible_prompts(
    path: Path,
    feature: int,
    threshold: float,
    population_prompts: set[str],
    protocol: dict[str, Any],
    split: str,
) -> tuple[set[str], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if "prompt_ids" not in archive or "activations" not in archive:
            raise ValueError("eligibility NPZ requires prompt_ids and activations")
        prompt_ids = [
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in archive["prompt_ids"].tolist()
        ]
        activations = np.asarray(archive["activations"], dtype=np.float64)
        ids = (
            np.asarray(archive["feature_ids"], dtype=np.int64)
            if "feature_ids" in archive
            else np.arange(activations.shape[1], dtype=np.int64)
        )
        required = {
            "protocol_id",
            "protocol_sha256",
            "manifest_sha256",
            "model_revision",
            "sae_revision",
            "dry_run",
            "splits",
        }
        missing_provenance = required - set(archive.files)
        if missing_provenance:
            raise ValueError(
                f"eligibility activation provenance is incomplete: {sorted(missing_provenance)}"
            )
        provenance = {key: archive[key].item() for key in required - {"splits"}}
        splits = [str(value) for value in archive["splits"].tolist()]
    expected_provenance = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "manifest_sha256": protocol["stimuli"]["expected_manifest_sha256"],
        "model_revision": protocol["artifacts"]["model"]["revision"],
        "sae_revision": protocol["artifacts"]["sae"]["revision"],
    }
    if any(str(provenance[key]) != value for key, value in expected_provenance.items()):
        raise ValueError("eligibility activation provenance has drifted")
    if bool(provenance["dry_run"]) or set(splits) != {split}:
        raise ValueError("eligibility activations are dry-run or from the wrong split")
    if activations.shape != (len(prompt_ids), len(ids)):
        raise ValueError("eligibility activation shape mismatch")
    locations = np.flatnonzero(ids == feature)
    if len(locations) != 1:
        raise ValueError(f"selected feature {feature} not found exactly once")
    values = activations[:, int(locations[0])]
    lookup = dict(zip(prompt_ids, values))
    missing = population_prompts - set(lookup)
    if missing:
        raise ValueError(f"eligibility artifact lacks {len(missing)} population prompts")
    eligible = {prompt_id for prompt_id in population_prompts if lookup[prompt_id] >= threshold}
    return eligible, {
        "rule": "pre-treatment selected-feature activation >= development-frozen threshold",
        "feature_id": feature,
        "threshold": threshold,
        "activation_path": str(path),
        "activation_sha256": sha256_file(path),
        "eligible_prompt_count": len(eligible),
    }


def arm_metrics(
    records: list[dict[str, Any]], selected_domain: str, taxonomy: list[str]
) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["prompt_id"]].append(row)
    result: dict[str, dict[str, float | None]] = {}
    for prompt_id, rows in grouped.items():
        domains = [str(row["domain_label"]) for row in rows]
        qualities = [float(row["quality"]) for row in rows if row["quality"] is not None]
        distances = [
            float(row["semantic_distance"])
            for row in rows
            if row["semantic_distance"] is not None
        ]
        raw_entropy = entropy(domains, taxonomy)
        result[prompt_id] = {
            "selected_domain_rate": float(np.mean([domain == selected_domain for domain in domains])),
            "domain_entropy_nats": raw_entropy,
            "domain_entropy_normalized": raw_entropy / math.log(len(taxonomy)),
            "distinct_domain_count": float(len(set(domains))),
            "json_validity_rate": float(np.mean([row["json_valid"] for row in rows])),
            # Quality is deliberately measured on one hash-selected paired
            # sample per prompt; domain outcomes still use every row above.
            "structural_quality": float(np.mean(qualities)) if qualities else None,
            "source_target_semantic_distance": float(np.mean(distances))
            if len(distances) == len(rows)
            else None,
        }
    return result


def compare_population(
    name: str,
    prompts: list[str],
    metrics: dict[str, dict[str, dict[str, float | None]]],
    baseline_arm: str,
    targeted_arm: str,
    random_arms: list[str],
    resamples: int,
    seed: int,
    quality_margin: float,
    json_margin: float,
) -> dict[str, Any]:
    ordered = sorted(prompts)
    boot = bootstrap_indices(len(ordered), resamples, seed)
    metric_names = (
        "selected_domain_rate",
        "domain_entropy_nats",
        "domain_entropy_normalized",
        "distinct_domain_count",
        "json_validity_rate",
        "structural_quality",
        "source_target_semantic_distance",
    )
    comparisons: dict[str, Any] = {}
    arm_summaries: dict[str, Any] = {}
    all_arms = [baseline_arm, targeted_arm, *random_arms]
    for arm in all_arms:
        arm_summaries[arm] = {}
        for metric in metric_names:
            values = [metrics[arm][prompt][metric] for prompt in ordered]
            if any(value is None for value in values):
                arm_summaries[arm][metric] = {
                    "estimate": None,
                    "ci95": None,
                    "prompt_count": 0,
                }
            else:
                arm_summaries[arm][metric] = clustered_summary(
                    np.asarray(values, dtype=np.float64), boot
                )

    for metric in metric_names:
        baseline_values = [metrics[baseline_arm][prompt][metric] for prompt in ordered]
        targeted_values = [metrics[targeted_arm][prompt][metric] for prompt in ordered]
        random_values = [
            [metrics[arm][prompt][metric] for arm in random_arms] for prompt in ordered
        ]
        unavailable = {"estimate": None, "ci95": None, "prompt_count": 0}
        comparison = {
            "targeted_minus_baseline": dict(unavailable),
            "random_mean_minus_baseline": dict(unavailable),
            "targeted_minus_random_mean": dict(unavailable),
        }
        baseline_complete = not any(value is None for value in baseline_values)
        targeted_complete = not any(value is None for value in targeted_values)
        random_complete = not any(
            value is None for row in random_values for value in row
        )
        baseline_array = (
            np.asarray(baseline_values, dtype=np.float64)
            if baseline_complete
            else None
        )
        targeted_array = (
            np.asarray(targeted_values, dtype=np.float64)
            if targeted_complete
            else None
        )
        random_array = (
            np.asarray(random_values, dtype=np.float64).mean(axis=1)
            if random_complete
            else None
        )
        if baseline_array is not None and targeted_array is not None:
            comparison["targeted_minus_baseline"] = clustered_summary(
                targeted_array - baseline_array, boot
            )
        if baseline_array is not None and random_array is not None:
            comparison["random_mean_minus_baseline"] = clustered_summary(
                random_array - baseline_array, boot
            )
        if targeted_array is not None and random_array is not None:
            comparison["targeted_minus_random_mean"] = clustered_summary(
                targeted_array - random_array, boot
            )
        comparisons[metric] = comparison

    primary = comparisons["selected_domain_rate"]
    quality = comparisons["structural_quality"]["targeted_minus_baseline"]
    validity = comparisons["json_validity_rate"]["targeted_minus_baseline"]
    primary_pass = bool(
        primary["targeted_minus_baseline"]["ci95"]
        and primary["targeted_minus_baseline"]["ci95"][1] < 0
    )
    specificity_pass = bool(
        primary["targeted_minus_random_mean"]["ci95"]
        and primary["targeted_minus_random_mean"]["ci95"][1] < 0
    )
    quality_pass = bool(quality["ci95"] and quality["ci95"][0] >= quality_margin)
    validity_pass = bool(validity["ci95"] and validity["ci95"][0] >= json_margin)
    entropy_change = comparisons["domain_entropy_nats"]["targeted_minus_baseline"]
    distinct_change = comparisons["distinct_domain_count"]["targeted_minus_baseline"]
    entropy_pass = bool(entropy_change["ci95"] and entropy_change["ci95"][0] > 0)
    distinct_pass = bool(distinct_change["ci95"] and distinct_change["ci95"][0] > 0)
    causal_selection = primary_pass and specificity_pass and validity_pass
    reduced_homogeneity = (
        causal_selection and entropy_pass and distinct_pass and quality_pass
    )
    return {
        "population": name,
        "prompt_count": len(ordered),
        "bootstrap": {"unit": "source_prompt", "resamples": resamples, "seed": seed},
        "arm_summaries": arm_summaries,
        "paired_comparisons": comparisons,
        "decision_checks": {
            "selected_domain_rate_upper_ci_below_zero": primary_pass,
            "specificity_upper_ci_below_zero": specificity_pass,
            "quality_lower_ci_at_least_margin": quality_pass,
            "json_validity_lower_ci_at_least_margin": validity_pass,
            "entropy_lower_ci_above_zero": entropy_pass,
            "distinct_domain_lower_ci_above_zero": distinct_pass,
        },
        "claim_boundary": {
            "causal_target_domain_selection_supported": causal_selection,
            "reduced_homogeneity_supported": reduced_homogeneity,
            "serendipity_evaluated": False,
            "note": (
                "Selected-domain contrasts support causal target-domain selection when "
                "the JSON-validity guardrail passes. Reduced homogeneity additionally "
                "requires entropy and distinct-domain gains with non-inferior structural "
                "quality. Serendipity is not evaluated."
            ),
        },
    }


def evaluate_command(args: argparse.Namespace) -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    amendment = load_protocol_amendment(protocol)
    effective_provenance = amendment_provenance(amendment)
    if args.split == "test":
        bypasses = {
            "allow_incomplete": args.allow_incomplete,
            "allow_smoke_labels": args.allow_smoke_labels,
            "allow_unreviewed_labels": args.allow_unreviewed_labels,
            "allow_unblinded_quality": args.allow_unblinded_quality,
        }
        enabled = [name for name, value in bypasses.items() if value]
        if enabled:
            raise ValueError(
                f"confirmatory evaluation forbids bypass flags: {enabled}"
            )
        frozen_resamples = int(protocol["analysis"]["bootstrap_resamples"])
        if args.bootstrap_resamples not in (None, frozen_resamples):
            raise ValueError("confirmatory bootstrap resamples differ from protocol")
        if not math.isclose(args.json_validity_margin, -0.02, abs_tol=1e-12):
            raise ValueError("confirmatory JSON-validity margin must remain -0.02")
        if not args.quality_labels:
            raise ValueError("confirmatory evaluation requires blinded quality labels")
        if not args.distance_metrics:
            raise ValueError("confirmatory evaluation requires semantic-distance metrics")
    taxonomy = list(protocol["target_domain_taxonomy"])
    generations = [
        row for row in read_jsonl_many(args.generations) if row.get("split") == args.split
    ]
    labels = [row for row in read_jsonl_many(args.labels) if row.get("split") == args.split]
    if not generations or not labels:
        raise ValueError(f"no generation/label rows for split {args.split}")
    if any(row.get("protocol_id") != protocol["protocol_id"] for row in generations + labels):
        raise ValueError("input protocol ID mismatch")
    if not args.allow_smoke_labels and any(not row.get("primary_eligible", False) for row in labels):
        raise ValueError("smoke-test labels cannot feed confirmatory evaluation")
    expected_classifier = (
        f"{protocol['domain_labeling']['classifier_repo_id']}@"
        f"{protocol['domain_labeling']['classifier_revision']}"
    )
    if not args.allow_smoke_labels and any(
        str(row.get("classifier_id")) != expected_classifier for row in labels
    ):
        raise ValueError("labels do not use the protocol-pinned domain classifier")
    label_audit = validate_label_audit(
        labels, args.allow_unreviewed_labels, effective_provenance
    )

    config: dict[str, Any] = {}
    development_plan: dict[str, Any] = {}
    if args.development_plan:
        if args.split != "development":
            raise ValueError("--development-plan is valid only for the development gate")
        if args.test_config:
            raise ValueError("use either --development-plan or --test-config, not both")
        development_plan = json.loads(args.development_plan.read_text())
        validate_amendment_provenance(
            development_plan, effective_provenance, "development plan"
        )
        if (
            development_plan.get("protocol_id") != protocol["protocol_id"]
            or development_plan.get("protocol_sha256") != sha256_file(PROTOCOL_PATH)
            or development_plan.get("development_gate_ready") is not True
        ):
            raise ValueError("development plan is not ready for the intervention gate")
    if args.test_config:
        config = load_test_config(args.test_config, protocol, args.split)
    elif args.split == "test":
        raise ValueError("untouched test evaluation requires --test-config")
    if args.split == "test":
        validate_generation_test_config_binding(generations, args.test_config)
    selection = config or development_plan
    selected_domain = selection.get("selected_domain", args.selected_domain)
    selected_feature = selection.get("selected_feature_id", args.selected_feature_id)
    random_features = selection.get(
        "five_matched_random_feature_ids", args.random_feature_ids
    )
    if selected_domain not in taxonomy:
        raise ValueError("selected domain is missing or outside the frozen taxonomy")
    excluded_primary_domains = set(
        amendment["domain_selection"]["primary_selected_domain_exclusions"]
    )
    if selected_domain in excluded_primary_domains:
        raise ValueError(
            f"selected domain {selected_domain!r} is excluded from primary selection"
        )
    if selected_feature is None:
        raise ValueError("selected feature ID is required")
    selected_feature = int(selected_feature)
    random_features = [int(value) for value in (random_features or [])]
    if len(random_features) != 5 or len(set(random_features)) != 5:
        raise ValueError("exactly five unique matched-random feature IDs are required")
    if selected_feature in random_features:
        raise ValueError("selected feature cannot also be a matched-random control")
    frozen_classifier_revision = selection.get("domain_classifier_revision")
    if frozen_classifier_revision and any(
        frozen_classifier_revision not in str(row.get("classifier_id", ""))
        for row in labels
    ):
        raise ValueError("independent-label classifier revision differs from test config")

    label_map: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for row in labels:
        key = join_key(row)
        if key in label_map:
            raise ValueError(f"duplicate independent label key {key}")
        if row.get("domain_label") not in taxonomy:
            raise ValueError(f"label outside frozen taxonomy: {row.get('domain_label')!r}")
        label_map[key] = row
    (
        quality_generations,
        reliability_quality_keys,
        quality_sampling_provenance,
    ) = select_quality_generations(generations, args.split, protocol)
    if config:
        validate_quality_plan_config(config, quality_sampling_provenance)
    quality_id_mapping = quality_assignment_mapping(
        quality_generations, reliability_quality_keys
    )
    expected_quality_keys = {
        join_key(generation) for generation in quality_generations
    }
    quality_map, quality_provenance = load_quality(
        args.quality_labels,
        args.allow_unblinded_quality,
        quality_id_mapping,
        protocol["outcomes"]["quality_guardrail"]["judge_protocol_id"],
        protocol["outcomes"]["quality_guardrail"]["rubric_sha256"],
        expected_quality_keys,
        reliability_quality_keys,
    )
    distance_map, distance_provenance = load_distances(args.distance_metrics)
    expected_metric_keys = {join_key(generation) for generation in generations}
    if args.quality_labels and set(quality_map) != expected_quality_keys:
        raise ValueError(
            "quality labels must cover the frozen paired quality sample exactly"
        )
    if args.distance_metrics and set(distance_map) != expected_metric_keys:
        raise ValueError("semantic-distance rows do not cover every generation exactly once")

    normalized: list[dict[str, Any]] = []
    seen_generation_keys: set[tuple[str, str, str, int, int]] = set()
    for generation in generations:
        key = join_key(generation)
        if key in seen_generation_keys:
            raise ValueError(f"duplicate generation key {key}")
        seen_generation_keys.add(key)
        if key not in label_map:
            raise ValueError(f"generation lacks independent label: {key}")
        if "analogy_schema_valid" not in generation:
            raise ValueError(f"generation lacks analogy_schema_valid: {key}")
        normalized.append(
            {
                "prompt_id": str(generation["prompt_id"]),
                "arm": arm_name(generation),
                "condition": str(generation["condition"]),
                "feature_id": feature_id(generation),
                "sample_index": int(generation["sample_index"]),
                "seed": canonical_seed(generation),
                "domain_label": str(label_map[key]["domain_label"]),
                "json_valid": bool(generation["analogy_schema_valid"]),
                "quality": quality_map.get(key),
                "semantic_distance": distance_map.get(key),
            }
        )

    baseline_arm = "baseline"
    targeted_arm = "targeted_feature_suppression"
    random_arms = [f"matched_random_feature_suppression:{value}" for value in random_features]
    observed_arms = {row["arm"] for row in normalized}
    required_arms = {baseline_arm, targeted_arm, *random_arms}
    missing_arms = sorted(required_arms - observed_arms)
    if missing_arms:
        raise ValueError(f"missing required evaluation arms: {missing_arms}")
    targeted_ids = {
        row["feature_id"] for row in normalized if row["arm"] == targeted_arm
    }
    if targeted_ids != {selected_feature}:
        raise ValueError(
            f"targeted arm feature IDs {targeted_ids} differ from selected {selected_feature}"
        )
    dose_feature_ids = {
        row["feature_id"]
        for row in normalized
        if row["arm"].startswith("targeted_feature_suppression:dose=")
    }
    if dose_feature_ids and dose_feature_ids != {selected_feature}:
        raise ValueError("a dose-response arm uses a feature other than the selected feature")

    rows_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        rows_by_arm[row["arm"]].append(row)
    baseline_prompts = {row["prompt_id"] for row in rows_by_arm[baseline_arm]}
    dose_arms = sorted(
        arm
        for arm in rows_by_arm
        if arm.startswith("targeted_feature_suppression:dose=")
    )
    paired_arms = required_arms | set(dose_arms)
    for arm in paired_arms:
        prompt_set = {row["prompt_id"] for row in rows_by_arm[arm]}
        if prompt_set != baseline_prompts:
            raise ValueError(
                f"arm {arm} prompt mismatch: {len(baseline_prompts-prompt_set)} missing, "
                f"{len(prompt_set-baseline_prompts)} extra"
            )
    paired_keys: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for row in normalized:
        if row["arm"] in paired_arms:
            paired_keys[(row["arm"], row["prompt_id"])].add(
                (row["sample_index"], row["seed"])
            )
    for prompt_id in baseline_prompts:
        baseline_keys = paired_keys[(baseline_arm, prompt_id)]
        for arm in paired_arms - {baseline_arm}:
            if paired_keys[(arm, prompt_id)] != baseline_keys:
                raise ValueError(f"paired seed/sample mismatch for prompt {prompt_id}, arm {arm}")

    if development_plan:
        if args.allow_incomplete:
            raise ValueError("--allow-incomplete cannot bypass a development gate plan")
        plan_ids = {
            str(value)
            for value in development_plan["development_gate_plan"]["prompt_ids"]
        }
        expected_prompts = int(
            protocol["development_intervention_gate"]["prompt_count"]
        )
        expected_samples = int(
            protocol["development_intervention_gate"]["paired_samples_per_prompt"]
        )
        if len(plan_ids) != expected_prompts or baseline_prompts != plan_ids:
            raise ValueError(
                "development-gate prompts differ from the exact discovery plan"
            )
    else:
        expected_prompts = int(protocol["stimuli"][f"{args.split}_count"])
        expected_samples = (
            int(protocol["generation"]["development_baseline_samples_per_prompt"])
            if args.split == "development"
            else int(protocol["generation"]["test_paired_samples_per_prompt_per_condition"])
        )
    count_histogram = Counter(
        len(paired_keys[(baseline_arm, prompt_id)]) for prompt_id in baseline_prompts
    )
    if not args.allow_incomplete and (
        len(baseline_prompts) != expected_prompts or set(count_histogram) != {expected_samples}
    ):
        raise ValueError(
            f"expected {expected_prompts} prompts x {expected_samples} samples; found "
            f"{len(baseline_prompts)} prompts and sample histogram {dict(count_histogram)}"
        )

    metrics = {
        arm: arm_metrics(rows_by_arm[arm], selected_domain, taxonomy)
        for arm in required_arms
    }
    bootstrap_resamples = args.bootstrap_resamples or int(
        protocol["analysis"]["bootstrap_resamples"]
    )
    bootstrap_seed = int(protocol["analysis"]["bootstrap_seed"])
    quality_margin = float(protocol["outcomes"]["quality_guardrail"]["noninferiority_margin"])
    full = compare_population(
        "full",
        sorted(baseline_prompts),
        metrics,
        baseline_arm,
        targeted_arm,
        random_arms,
        bootstrap_resamples,
        bootstrap_seed,
        quality_margin,
        args.json_validity_margin,
    )

    eligible_result: dict[str, Any]
    eligibility_provenance: dict[str, Any] | None = None
    threshold = selection.get("eligible_activation_threshold", args.eligible_threshold)
    if threshold is not None:
        if args.prompt_activations is None:
            raise ValueError("eligible analysis requires --prompt-activations")
        eligible_prompts, eligibility_provenance = load_eligible_prompts(
            args.prompt_activations,
            selected_feature,
            float(threshold),
            baseline_prompts,
            protocol,
            args.split,
        )
        if not eligible_prompts:
            eligible_result = {
                "population": "eligible_secondary",
                "prompt_count": 0,
                "status": "no_eligible_prompts",
            }
        else:
            eligible_result = compare_population(
                "eligible_secondary",
                sorted(eligible_prompts),
                metrics,
                baseline_arm,
                targeted_arm,
                random_arms,
                bootstrap_resamples,
                bootstrap_seed + 1,
                quality_margin,
                args.json_validity_margin,
            )
    else:
        eligible_result = {
            "population": "eligible_secondary",
            "status": "not_configured",
            "prompt_count": 0,
        }

    control_summaries = {
        arm: arm_metrics(rows, selected_domain, taxonomy)
        for arm, rows in rows_by_arm.items()
        if arm not in required_arms
    }
    control_population_means: dict[str, dict[str, float | None]] = {}
    for arm, prompt_metrics in control_summaries.items():
        control_population_means[arm] = {}
        for metric in (
            "selected_domain_rate",
            "domain_entropy_nats",
            "distinct_domain_count",
            "json_validity_rate",
            "source_target_semantic_distance",
        ):
            values = [item[metric] for item in prompt_metrics.values()]
            control_population_means[arm][metric] = (
                float(np.mean(values))
                if values and all(value is not None for value in values)
                else None
            )

    development_gate: dict[str, Any] | None = None
    if args.split == "development":
        frozen_doses = [
            float(value)
            for value in protocol["development_intervention_gate"][
                "suppression_doses"
            ]
        ]
        dose_rates: dict[float, float] = {
            0.0: float(
                np.mean(
                    [
                        values["selected_domain_rate"]
                        for values in metrics[baseline_arm].values()
                    ]
                )
            ),
            1.0: float(
                np.mean(
                    [
                        values["selected_domain_rate"]
                        for values in metrics[targeted_arm].values()
                    ]
                )
            ),
        }
        for arm in dose_arms:
            dose = float(arm.rsplit("=", 1)[1])
            dose_rates[dose] = float(
                np.mean(
                    [
                        values["selected_domain_rate"]
                        for values in control_summaries[arm].values()
                    ]
                )
            )
        expected_doses = [0.0, *frozen_doses]
        dose_complete = all(dose in dose_rates for dose in expected_doses)
        adjacent_decreases = (
            sum(
                dose_rates[right] < dose_rates[left]
                for left, right in zip(expected_doses[:-1], expected_doses[1:])
            )
            if dose_complete
            else 0
        )
        primary_comparisons = full["paired_comparisons"]
        specificity_estimate = primary_comparisons["selected_domain_rate"][
            "targeted_minus_random_mean"
        ]["estimate"]
        quality_estimate = primary_comparisons["structural_quality"][
            "targeted_minus_baseline"
        ]["estimate"]
        validity_estimate = primary_comparisons["json_validity_rate"][
            "targeted_minus_baseline"
        ]["estimate"]
        gate_checks = {
            "targeted_change_below_matched_random_mean": bool(
                specificity_estimate is not None and specificity_estimate < 0
            ),
            "at_least_two_adjacent_dose_decreases": bool(
                dose_complete and adjacent_decreases >= 2
            ),
            "mean_quality_change_at_least_minus_0_25": bool(
                quality_estimate is not None and quality_estimate >= quality_margin
            ),
            "json_validity_change_at_least_minus_0_02": bool(
                validity_estimate is not None and validity_estimate >= -0.02
            ),
        }
        development_gate = {
            "status": "pass" if all(gate_checks.values()) else "stop",
            "checks": gate_checks,
            "dose_response": {
                "required_doses": expected_doses,
                "observed_selected_domain_rates": {
                    format(dose, ".12g"): dose_rates.get(dose)
                    for dose in expected_doses
                },
                "adjacent_strict_decreases": adjacent_decreases,
            },
            "point_estimates": {
                "targeted_minus_random_mean_domain_rate": specificity_estimate,
                "targeted_minus_baseline_quality": quality_estimate,
                "targeted_minus_baseline_json_validity": validity_estimate,
            },
            "note": "This development-only gate is not confirmatory evidence.",
        }
    result = {
        "schema_version": 1,
        "artifact": "latent_escape_evaluation",
        "protocol_id": protocol["protocol_id"],
        "protocol_revision": protocol.get("protocol_revision"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        **effective_provenance,
        "split": args.split,
        "selected_domain": selected_domain,
        "selected_feature_id": selected_feature,
        "matched_random_feature_ids": random_features,
        "eligible_activation_threshold": threshold,
        "population_order": ["full", "eligible_secondary"],
        "full": full,
        "eligible_secondary": eligible_result,
        "eligibility": eligibility_provenance,
        "other_control_population_means_exploratory": control_population_means,
        "quality": {
            **quality_provenance,
            "sampling": quality_sampling_provenance,
            "noninferiority_margin": quality_margin,
        },
        "semantic_distance": distance_provenance,
        "development_intervention_gate": development_gate,
        "json_validity_margin_exploratory": args.json_validity_margin,
        "domain_label_audit": label_audit,
        "development_plan": {
            "path": str(args.development_plan),
            "sha256": sha256_file(args.development_plan),
        }
        if args.development_plan
        else None,
        "power_design": json.loads(POWER_REPORT_PATH.read_text())
        if POWER_REPORT_PATH.exists()
        else None,
        "claim_policy": protocol["outcomes"]["claim_policy"],
    }
    output_hash = atomic_write_json(args.output, result)
    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        **effective_provenance,
        "generation_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.generations
        ],
        "label_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.labels
        ],
        "quality_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (args.quality_labels or [])
        ],
        "distance_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (args.distance_metrics or [])
        ],
        "test_config": {
            "path": str(args.test_config),
            "sha256": sha256_file(args.test_config),
        }
        if args.test_config
        else None,
        "development_plan": {
            "path": str(args.development_plan),
            "sha256": sha256_file(args.development_plan),
        }
        if args.development_plan
        else None,
        "output_sha256": output_hash,
        "generated_target_domain_used": False,
        "resampling_unit": "source_prompt",
    }
    atomic_write_json(args.output.with_name(args.output.name + ".meta.json"), provenance)
    print(
        json.dumps(
            {
                "split": args.split,
                "full_prompt_count": full["prompt_count"],
                "causal_target_domain_selection_supported": full["claim_boundary"][
                    "causal_target_domain_selection_supported"
                ],
                "reduced_homogeneity_supported": full["claim_boundary"][
                    "reduced_homogeneity_supported"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def paired_prompt_mde(
    prompts: int,
    samples: int,
    target_rate: float,
    prompt_icc: float,
    cross_condition_correlation: float,
    alpha: float,
    power: float,
) -> float:
    if prompts <= 1 or samples < 1:
        raise ValueError("prompts must exceed one and samples must be positive")
    for name, value in {
        "target_rate": target_rate,
        "prompt_icc": prompt_icc,
        "cross_condition_correlation": cross_condition_correlation,
        "alpha": alpha,
        "power": power,
    }.items():
        if not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    design_effect = 1.0 + (samples - 1) * prompt_icc
    variance = target_rate * (1.0 - target_rate) * design_effect / samples
    paired_variance = 2.0 * variance * (1.0 - cross_condition_correlation)
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0) + NormalDist().inv_cdf(power)
    return float(critical * math.sqrt(paired_variance / prompts))


def mde_command(args: argparse.Namespace) -> int:
    mde = paired_prompt_mde(
        args.prompts,
        args.samples,
        args.target_rate,
        args.prompt_icc,
        args.cross_condition_correlation,
        args.alpha,
        args.power,
    )
    worst = paired_prompt_mde(
        args.prompts,
        args.samples,
        0.5,
        args.prompt_icc,
        args.cross_condition_correlation,
        args.alpha,
        args.power,
    )
    result = {
        "schema_version": 1,
        "artifact": "design_stage_mde",
        "method": "normal approximation for paired prompt-level proportions with cluster design effect",
        "prompts": args.prompts,
        "samples_per_prompt_per_condition": args.samples,
        "assumptions": {
            "target_rate": args.target_rate,
            "prompt_icc": args.prompt_icc,
            "cross_condition_correlation": args.cross_condition_correlation,
            "two_sided_alpha": args.alpha,
            "target_power": args.power,
        },
        "minimum_detectable_absolute_rate_change": mde,
        "worst_case_rate_0_5_mde": worst,
        "warning": "Design-stage sensitivity only; do not report as post-hoc observed power.",
    }
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("run", help="evaluate generated conditions")
    evaluate_parser.add_argument("--generations", type=Path, nargs="+", required=True)
    evaluate_parser.add_argument("--labels", type=Path, nargs="+", required=True)
    evaluate_parser.add_argument("--quality-labels", type=Path, nargs="+")
    evaluate_parser.add_argument("--distance-metrics", type=Path, nargs="+")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("development", "test"), required=True)
    evaluate_parser.add_argument("--test-config", type=Path)
    evaluate_parser.add_argument(
        "--development-plan",
        type=Path,
        help="passing discovery artifact that freezes the 24-prompt development gate",
    )
    evaluate_parser.add_argument("--selected-domain")
    evaluate_parser.add_argument("--selected-feature-id", type=int)
    evaluate_parser.add_argument("--random-feature-ids", type=int, nargs="*")
    evaluate_parser.add_argument("--prompt-activations", type=Path)
    evaluate_parser.add_argument("--eligible-threshold", type=float)
    evaluate_parser.add_argument("--bootstrap-resamples", type=int)
    evaluate_parser.add_argument("--json-validity-margin", type=float, default=-0.02)
    evaluate_parser.add_argument("--allow-incomplete", action="store_true")
    evaluate_parser.add_argument("--allow-smoke-labels", action="store_true")
    evaluate_parser.add_argument("--allow-unreviewed-labels", action="store_true")
    evaluate_parser.add_argument("--allow-unblinded-quality", action="store_true")
    evaluate_parser.set_defaults(func=evaluate_command)

    quality_parser = subparsers.add_parser(
        "prepare-quality", help="export a condition-blinded structural-quality queue"
    )
    quality_parser.add_argument("--generations", type=Path, nargs="+", required=True)
    quality_parser.add_argument("--output", type=Path, required=True)
    quality_parser.add_argument(
        "--split", choices=("development", "test"), required=True
    )
    quality_parser.add_argument("--test-config", type=Path)
    quality_parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=ROOT / "latent_escape" / "artifacts" / "prompt_manifest.jsonl",
    )
    quality_parser.set_defaults(func=prepare_quality_command)

    distance_parser = subparsers.add_parser(
        "distance", help="compute pinned source-to-target semantic distances"
    )
    distance_parser.add_argument("--generations", type=Path, nargs="+", required=True)
    distance_parser.add_argument("--output", type=Path, required=True)
    distance_parser.add_argument("--split", choices=("development", "test"))
    distance_parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=ROOT / "latent_escape" / "artifacts" / "prompt_manifest.jsonl",
    )
    distance_parser.add_argument("--device", default="cpu")
    distance_parser.add_argument("--batch-size", type=int, default=64)
    distance_parser.add_argument("--local-files-only", action="store_true")
    distance_parser.add_argument("--no-progress", action="store_true")
    distance_parser.set_defaults(func=distance_command)

    mde_parser = subparsers.add_parser("mde", help="design-stage MDE sensitivity")
    mde_parser.add_argument("--prompts", type=int, default=120)
    mde_parser.add_argument("--samples", type=int, default=8)
    mde_parser.add_argument("--target-rate", type=float, default=0.20)
    mde_parser.add_argument("--prompt-icc", type=float, default=0.20)
    mde_parser.add_argument("--cross-condition-correlation", type=float, default=0.50)
    mde_parser.add_argument("--alpha", type=float, default=0.05)
    mde_parser.add_argument("--power", type=float, default=0.80)
    mde_parser.add_argument("--output", type=Path)
    mde_parser.set_defaults(func=mde_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
