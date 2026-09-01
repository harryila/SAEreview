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


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "latent_escape" / "protocol.json"
POWER_REPORT_PATH = ROOT / "latent_escape" / "power_report.json"
DISTANCE_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
DISTANCE_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
QUALITY_REQUIRED_ARMS = frozenset({"baseline", "targeted_feature_suppression"})


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    labels: list[dict[str, Any]], allow_unreviewed: bool
) -> dict[str, Any]:
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
    blind_id_mapping: dict[str, tuple[str, str, str, int, int]],
    expected_judge_protocol: str,
    expected_prompt_sha256: str,
    expected_ratings_per_generation: int,
) -> tuple[dict[tuple[str, str, str, int, int], float], dict[str, Any]]:
    if not paths:
        return {}, {"available": False, "rating_count": 0}
    rows = read_jsonl_many(paths)
    by_key: dict[tuple[str, str, str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        blind_id = row.get("blind_quality_id")
        if blind_id is not None:
            if str(blind_id) not in blind_id_mapping:
                raise ValueError(f"unknown blind_quality_id {blind_id}")
            key = blind_id_mapping[str(blind_id)]
            derived_blinding = True
        else:
            if not allow_unblinded:
                raise ValueError(
                    "primary quality rows must join through blind_quality_id"
                )
            key = join_key(row)
            derived_blinding = False
        if not allow_unblinded and not derived_blinding and not (
            row.get("blinded") is True or row.get("condition_hidden") is True
        ):
            raise ValueError("quality rows must record blinded=true or condition_hidden=true")
        if not allow_unblinded and (
            row.get("judge_protocol_id") != expected_judge_protocol
            or row.get("judge_prompt_sha256") != expected_prompt_sha256
        ):
            raise ValueError("quality judge protocol or rubric hash has drifted")
        by_key[key].append(structural_quality(row))
    if not allow_unblinded and any(
        len(values) != expected_ratings_per_generation for values in by_key.values()
    ):
        raise ValueError("quality ratings per generation differ from the frozen protocol")
    return (
        {key: float(np.mean(values)) for key, values in by_key.items()},
        {
            "available": True,
            "rating_count": len(rows),
            "rated_generation_count": len(by_key),
            "ratings_per_generation": dict(
                sorted(Counter(len(values) for values in by_key.values()).items())
            ),
        },
    )


def prepare_quality_command(args: argparse.Namespace) -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    quality_protocol = protocol["outcomes"]["quality_guardrail"]
    generations = read_jsonl_many(args.generations)
    if args.split:
        generations = [row for row in generations if row.get("split") == args.split]
    if not generations:
        raise ValueError("no generations selected for quality queue")
    all_generation_count = len(generations)
    generations = [row for row in generations if requires_structural_quality(row)]
    if not generations:
        raise ValueError(
            "no baseline or full-strength targeted outputs selected for quality queue"
        )
    manifest_path = args.prompt_manifest
    manifest_rows = read_jsonl_many([manifest_path])
    manifest = {str(row["prompt_id"]): row for row in manifest_rows}
    queue: list[dict[str, Any]] = []
    seen: set[str] = set()
    for generation in generations:
        if generation.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError("generation protocol ID mismatch")
        prompt_id = str(generation["prompt_id"])
        if prompt_id not in manifest:
            raise ValueError(f"prompt {prompt_id} is absent from prompt manifest")
        blind_id = quality_blind_id(generation)
        if blind_id in seen:
            raise ValueError(f"duplicate quality blind ID {blind_id}")
        seen.add(blind_id)
        prompt = manifest[prompt_id]
        queue.append(
            {
                "schema_version": 1,
                "record_type": "blinded_structural_quality_item",
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
        "manifest_sha256": sha256_file(manifest_path),
        "generation_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.generations
        ],
        "queue_sha256": sha256_bytes(payload),
        "item_count": len(queue),
        "included_arms": sorted({arm_name(row) for row in generations}),
        "excluded_generation_count": all_generation_count - len(generations),
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
            "structural_quality": float(np.mean(qualities))
            if len(qualities) == len(rows)
            else None,
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
    label_audit = validate_label_audit(labels, args.allow_unreviewed_labels)

    config: dict[str, Any] = {}
    development_plan: dict[str, Any] = {}
    if args.development_plan:
        if args.split != "development":
            raise ValueError("--development-plan is valid only for the development gate")
        if args.test_config:
            raise ValueError("use either --development-plan or --test-config, not both")
        development_plan = json.loads(args.development_plan.read_text())
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
    selection = config or development_plan
    selected_domain = selection.get("selected_domain", args.selected_domain)
    selected_feature = selection.get("selected_feature_id", args.selected_feature_id)
    random_features = selection.get(
        "five_matched_random_feature_ids", args.random_feature_ids
    )
    if selected_domain not in taxonomy:
        raise ValueError("selected domain is missing or outside the frozen taxonomy")
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
    quality_generations = [
        generation for generation in generations if requires_structural_quality(generation)
    ]
    quality_id_mapping: dict[str, tuple[str, str, str, int, int]] = {}
    for generation in quality_generations:
        blind_id = quality_blind_id(generation)
        if blind_id in quality_id_mapping:
            raise ValueError(f"duplicate blind quality ID {blind_id}")
        quality_id_mapping[blind_id] = join_key(generation)
    quality_map, quality_provenance = load_quality(
        args.quality_labels,
        args.allow_unblinded_quality,
        quality_id_mapping,
        protocol["outcomes"]["quality_guardrail"]["judge_protocol_id"],
        protocol["outcomes"]["quality_guardrail"]["rubric_sha256"],
        int(protocol["outcomes"]["quality_guardrail"]["ratings_per_generation"]),
    )
    distance_map, distance_provenance = load_distances(args.distance_metrics)
    expected_quality_keys = {
        join_key(generation) for generation in quality_generations
    }
    expected_metric_keys = {join_key(generation) for generation in generations}
    if args.quality_labels and set(quality_map) != expected_quality_keys:
        raise ValueError(
            "quality labels must cover baseline and full-strength targeted outputs exactly once"
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
    quality_parser.add_argument("--split", choices=("development", "test"))
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
