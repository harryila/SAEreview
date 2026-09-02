#!/usr/bin/env python3
"""Select at most one development-only SAE feature/domain pair.

Domain labels are aggregated to a frequency per source prompt, and every prompt
activation enters the analysis exactly once. A prompt-label permutation test
uses the maximum positive association over all eligible features and domains,
protecting the feature search rather than merely testing the winning pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import KFold

try:  # Support both module and direct-script execution.
    from .protocol_amendment import amendment_sha256, load_protocol_amendment
except ImportError:  # pragma: no cover - exercised by CLI invocation
    from protocol_amendment import amendment_sha256, load_protocol_amendment


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "latent_escape" / "protocol.json"
MANIFEST_PATH = ROOT / "latent_escape" / "artifacts" / "prompt_manifest.jsonl"
AUDIT_PROVENANCE_FIELDS = (
    "protocol_amendment_id",
    "protocol_amendment_sha256",
    "domain_labeling_guide_id",
    "domain_labeling_guide_sha256",
)


def audit_provenance(
    amendment: dict[str, Any], amendment_hash: str
) -> dict[str, str]:
    guide = amendment.get("domain_labeling_guide")
    if not isinstance(guide, dict):
        raise ValueError("protocol amendment lacks domain_labeling_guide")
    values = {
        "protocol_amendment_id": amendment.get("amendment_id"),
        "protocol_amendment_sha256": amendment_hash,
        "domain_labeling_guide_id": guide.get("id"),
        "domain_labeling_guide_sha256": guide.get("sha256"),
    }
    missing = [name for name, value in values.items() if not isinstance(value, str) or not value]
    if missing:
        raise ValueError(f"protocol amendment lacks audit provenance fields: {missing}")
    return {name: str(value) for name, value in values.items()}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_bytes(payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no records")
    return rows


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in values.tolist()
    ]


def manifest_splits() -> dict[str, str]:
    if not MANIFEST_PATH.exists():
        return {}
    return {
        str(row["prompt_id"]): str(row["split"])
        for row in read_jsonl(MANIFEST_PATH)
    }


def load_prompt_activations(
    path: Path,
    protocol: dict[str, Any],
    allow_width_mismatch: bool,
    allow_smoke_activations: bool,
) -> tuple[list[str], np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    if path.suffix.casefold() != ".npz":
        raise ValueError("prompt activations must be an NPZ with prompt_ids and activations")
    with np.load(path, allow_pickle=False) as archive:
        if "prompt_ids" not in archive or "activations" not in archive:
            raise ValueError("activation NPZ requires prompt_ids and activations arrays")
        prompt_ids = decode_strings(archive["prompt_ids"])
        activations = np.asarray(archive["activations"], dtype=np.float64)
        feature_ids = (
            np.asarray(archive["feature_ids"], dtype=np.int64)
            if "feature_ids" in archive
            else np.arange(activations.shape[1], dtype=np.int64)
        )
        decoder_norms = (
            np.asarray(archive["decoder_norms"], dtype=np.float64)
            if "decoder_norms" in archive
            else None
        )
        split_key = "splits" if "splits" in archive else "split" if "split" in archive else None
        if split_key:
            raw_splits = archive[split_key]
            if raw_splits.ndim == 0:
                item = raw_splits.item()
                value = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                splits = [value] * len(prompt_ids)
            else:
                splits = decode_strings(raw_splits)
        else:
            splits = []
        provenance_keys = {
            "protocol_id",
            "protocol_sha256",
            "manifest_sha256",
            "model_revision",
            "sae_revision",
            "dry_run",
        }
        missing_provenance = provenance_keys - set(archive.files)
        provenance = {
            key: archive[key].item() for key in provenance_keys if key in archive.files
        }

    if missing_provenance and not allow_smoke_activations:
        raise ValueError(
            f"activation NPZ lacks frozen provenance fields: {sorted(missing_provenance)}"
        )
    if not missing_provenance and not allow_smoke_activations:
        expected_provenance = {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "manifest_sha256": protocol["stimuli"]["expected_manifest_sha256"],
            "model_revision": protocol["artifacts"]["model"]["revision"],
            "sae_revision": protocol["artifacts"]["sae"]["revision"],
        }
        drift = {
            key: {"observed": str(provenance[key]), "expected": value}
            for key, value in expected_provenance.items()
            if str(provenance[key]) != value
        }
        if drift:
            raise ValueError(f"activation provenance drift: {drift}")
        if bool(provenance["dry_run"]):
            raise ValueError("dry-run activations cannot feed primary feature discovery")

    if activations.ndim != 2:
        raise ValueError(f"activations must have shape [prompts, features], got {activations.shape}")
    if activations.shape[0] != len(prompt_ids):
        raise ValueError("prompt_ids and activation rows differ")
    if activations.shape[1] != len(feature_ids):
        raise ValueError("feature_ids and activation columns differ")
    if decoder_norms is not None and decoder_norms.shape != (activations.shape[1],):
        raise ValueError("decoder_norms must have one value per activation column")
    if decoder_norms is not None and (
        not np.isfinite(decoder_norms).all() or np.any(decoder_norms <= 0)
    ):
        raise ValueError("decoder_norms must be finite and positive")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("activation NPZ contains duplicate prompt IDs")
    if len(set(int(value) for value in feature_ids)) != len(feature_ids):
        raise ValueError("activation NPZ contains duplicate feature IDs")
    if not np.isfinite(activations).all():
        raise ValueError("activations contain NaN or infinity")
    expected_width = int(protocol["artifacts"]["sae"]["width"])
    if activations.shape[1] != expected_width and not allow_width_mismatch:
        raise ValueError(
            f"activation width is {activations.shape[1]}; protocol requires {expected_width}"
        )
    if splits and len(splits) != len(prompt_ids):
        raise ValueError("split array and activation rows differ")
    if not splits:
        split_map = manifest_splits()
        unknown = [prompt_id for prompt_id in prompt_ids if prompt_id not in split_map]
        if unknown:
            raise ValueError(
                "activation NPZ has no splits and prompt manifest cannot resolve "
                f"{len(unknown)} prompt IDs"
            )
        splits = [split_map[prompt_id] for prompt_id in prompt_ids]
    if set(splits) != {"development"}:
        raise ValueError(
            f"feature discovery is development-only; activation splits are {sorted(set(splits))}"
        )
    return prompt_ids, activations, feature_ids, splits, decoder_norms


def deterministic_audit_ids(
    rows: list[dict[str, Any]], fraction: float, seed: str
) -> set[str]:
    """Reconstruct the frozen stratified audit membership from BART labels."""
    target = int(np.ceil(len(rows) * fraction))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("classifier_domain_label"))].append(row)
    allocations = {domain: int(len(group) * fraction) for domain, group in groups.items()}
    remaining = target - sum(allocations.values())
    remainder_order = sorted(
        groups,
        key=lambda domain: (
            -(len(groups[domain]) * fraction - allocations[domain]),
            sha256_bytes(f"{seed}|stratum|{domain}".encode("utf-8")),
        ),
    )
    for domain in remainder_order[:remaining]:
        allocations[domain] += 1
    chosen: set[str] = set()
    for domain, group in groups.items():
        ordered = sorted(
            group,
            key=lambda row: sha256_bytes(
                f"{seed}|item|{row['blind_id']}".encode("utf-8")
            ),
        )
        chosen.update(str(row["blind_id"]) for row in ordered[: allocations[domain]])
    return chosen


def load_prompt_domain_frequencies(
    path: Path,
    taxonomy: list[str],
    condition: str,
    expected_samples: int,
    allow_incomplete: bool,
    allow_smoke_labels: bool,
    allow_unaudited_labels: bool,
    expected_classifier_id: str,
    expected_audit_provenance: dict[str, str],
    audit_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, int], int, dict[str, Any]]:
    rows = read_jsonl(path)
    filtered = [row for row in rows if str(row.get("condition")) == condition]
    if not filtered:
        raise ValueError(f"no labels found for condition {condition!r}")
    if any(row.get("split") != "development" for row in filtered):
        observed = sorted({str(row.get("split")) for row in filtered})
        raise ValueError(f"feature discovery is development-only; label splits are {observed}")
    if not allow_smoke_labels and any(not row.get("primary_eligible", False) for row in filtered):
        raise ValueError(
            "smoke-test/heuristic labels cannot select the primary feature; "
            "use independently classified labels or pass --allow-smoke-labels only for tests"
        )
    if not allow_smoke_labels and any(
        str(row.get("classifier_id")) != expected_classifier_id for row in filtered
    ):
        raise ValueError(
            "feature discovery requires the protocol-pinned independent domain classifier"
        )
    if not allow_smoke_labels:
        provenance_drift = [
            (index, field)
            for index, row in enumerate(filtered, start=1)
            for field, expected in expected_audit_provenance.items()
            if str(row.get(field, "")) != expected
        ]
        if provenance_drift:
            first_row, first_field = provenance_drift[0]
            raise ValueError(
                "domain labels are not bound to the frozen manual-labeling guide and "
                f"protocol amendment (first drift: filtered row {first_row}, {first_field})"
            )
        blind_ids = [str(row.get("blind_id", "")) for row in filtered]
        if any(not blind_id for blind_id in blind_ids) or len(set(blind_ids)) != len(blind_ids):
            raise ValueError("domain labels require unique nonempty blind IDs")
        outside_queue = [
            row
            for row in filtered
            if row.get("manual_audited") is True and row.get("audit_selected") is not True
        ]
        if outside_queue:
            raise ValueError("manual domain labels may govern only frozen audit rows")
    audit_rows = [row for row in filtered if row.get("audit_selected") is True]
    if not allow_smoke_labels:
        expected_audit_count = int(np.ceil(audit_fraction * len(filtered)))
        if len(audit_rows) != expected_audit_count or any(
            not np.isclose(float(row.get("audit_fraction", -1.0)), audit_fraction)
            or row.get("audit_seed") != "latent-escape-domain-audit-v1"
            for row in filtered
        ):
            raise ValueError("domain-label audit size or seed differs from protocol")
        expected_ids = deterministic_audit_ids(
            filtered, audit_fraction, "latent-escape-domain-audit-v1"
        )
        observed_ids = {str(row["blind_id"]) for row in audit_rows}
        if observed_ids != expected_ids:
            raise ValueError("domain-label audit membership differs from frozen hash selection")
    unaudited = [row for row in audit_rows if row.get("manual_audited") is not True]
    if (not audit_rows or unaudited) and not allow_unaudited_labels:
        raise ValueError(
            "the deterministic 10% audit must be completed before feature discovery; "
            f"found {len(audit_rows)} queued and {len(unaudited)} without rater labels"
        )
    agreements = [
        row.get("domain_label") == row.get("classifier_domain_label") for row in audit_rows
    ]
    by_class: dict[str, list[bool]] = defaultdict(list)
    for row, agreement in zip(audit_rows, agreements):
        by_class[str(row.get("classifier_domain_label"))].append(agreement)
    overall_agreement = float(np.mean(agreements)) if agreements else None
    class_agreement = {
        domain: {
            "count": len(values),
            "exact_agreement": float(np.mean(values)),
        }
        for domain, values in sorted(by_class.items())
    }
    gate_pass = bool(
        agreements
        and overall_agreement is not None
        and overall_agreement >= 0.80
        and all(
            item["count"] < 5 or item["exact_agreement"] >= 0.60
            for item in class_agreement.values()
        )
    )
    if not gate_pass and not allow_unaudited_labels:
        raise ValueError(
            "domain-label audit gate failed (requires >=80% overall and >=60% "
            "within every classifier class with at least five audited examples)"
        )

    domain_index = {domain: index for index, domain in enumerate(taxonomy)}
    by_prompt: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str, int, int]] = set()
    for row in filtered:
        domain = str(row.get("domain_label"))
        if domain not in domain_index:
            raise ValueError(f"label outside frozen taxonomy: {domain!r}")
        key = (
            str(row["prompt_id"]),
            str(row["condition"]),
            int(row["sample_index"]),
            int(row.get("seed", row.get("paired_seed"))),
        )
        if key in seen:
            raise ValueError(f"duplicate label key {key}")
        seen.add(key)
        by_prompt[key[0]].append(domain)

    counts = {prompt_id: len(values) for prompt_id, values in by_prompt.items()}
    if not allow_incomplete and set(counts.values()) != {expected_samples}:
        histogram = dict(sorted(Counter(counts.values()).items()))
        raise ValueError(
            f"expected {expected_samples} labels per prompt; count histogram is {histogram}"
        )
    frequencies: dict[str, np.ndarray] = {}
    for prompt_id, values in by_prompt.items():
        vector = np.zeros(len(taxonomy), dtype=np.float64)
        for domain in values:
            vector[domain_index[domain]] += 1.0
        frequencies[prompt_id] = vector / len(values)
    audit_summary = {
        "queued_count": len(audit_rows),
        "completed_count": len(audit_rows) - len(unaudited),
        "overall_exact_agreement": overall_agreement,
        "by_classifier_domain": class_agreement,
        "gate_pass": gate_pass,
        "bypassed_for_smoke": bool(not gate_pass and allow_unaudited_labels),
        "audit_fraction": audit_fraction,
        **expected_audit_provenance,
    }
    return frequencies, counts, len(filtered), audit_summary


def primary_domain_columns(
    prompt_frequencies: np.ndarray,
    taxonomy: list[str],
    minimum_rate: float,
    exclusions: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return all descriptive rates and the non-residual primary-search columns."""
    if prompt_frequencies.ndim != 2 or prompt_frequencies.shape[1] != len(taxonomy):
        raise ValueError("prompt-frequency columns must match the frozen taxonomy")
    rates = prompt_frequencies.mean(axis=0)
    exclusion_set = set(exclusions)
    permitted = np.asarray(
        [domain not in exclusion_set for domain in taxonomy], dtype=bool
    )
    columns = np.flatnonzero(
        (rates >= minimum_rate)
        & (np.ptp(prompt_frequencies, axis=0) > 0)
        & permitted
    )
    return rates, columns


def normalized_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    normalized = np.zeros_like(centered, dtype=np.float64)
    valid = norms > np.finfo(np.float64).eps
    normalized[:, valid] = centered[:, valid] / norms[valid]
    return normalized, valid


def average_rank_columns(matrix: np.ndarray) -> np.ndarray:
    """Average ranks down each column, including exact ties from sparse zeros."""
    ranks = np.empty_like(matrix, dtype=np.float64)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        order = np.argsort(values, kind="mergesort")
        ordered = values[order]
        boundaries = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
        ranked = np.empty(len(values), dtype=np.float64)
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            ranked[order[start:stop]] = (start + stop - 1) / 2.0
        ranks[:, column] = ranked
    return ranks


def permutation_maxima(
    x_normalized: np.ndarray,
    y_normalized: np.ndarray,
    permutations: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Return the max feature-by-domain correlation for each prompt permutation."""
    if permutations < 1:
        raise ValueError("at least one permutation is required")
    if batch_size < 1:
        raise ValueError("permutation batch size must be positive")
    rng = np.random.default_rng(seed)
    n_prompts, n_domains = y_normalized.shape
    maxima = np.empty(permutations, dtype=np.float64)
    completed = 0
    x32 = np.asarray(x_normalized, dtype=np.float32)
    y32 = np.asarray(y_normalized, dtype=np.float32)
    while completed < permutations:
        current = min(batch_size, permutations - completed)
        permuted = np.stack(
            [y32[rng.permutation(n_prompts), :] for _ in range(current)], axis=0
        )
        flattened = permuted.transpose(1, 0, 2).reshape(n_prompts, current * n_domains)
        scores = (x32.T @ flattened).reshape(x32.shape[1], current, n_domains)
        maxima[completed : completed + current] = scores.max(axis=(0, 2))
        completed += current
    return maxima


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    return float(np.dot(x_centered, y_centered) / denominator) if denominator else 0.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    paired = np.column_stack((x, y))
    ranks = average_rank_columns(paired)
    return pearson(ranks[:, 0], ranks[:, 1])


def prompt_level_cv(
    x: np.ndarray, y: np.ndarray, folds: int, seed: int
) -> dict[str, float]:
    if not 2 <= folds <= len(x):
        raise ValueError("CV folds must be between 2 and the prompt count")
    predictions = np.empty_like(y, dtype=np.float64)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, validation in splitter.split(x):
        x_train = x[train]
        y_train = y[train]
        centered = x_train - x_train.mean()
        denominator = float(np.dot(centered, centered))
        slope = (
            float(np.dot(centered, y_train - y_train.mean()) / denominator)
            if denominator
            else 0.0
        )
        intercept = float(y_train.mean() - slope * x_train.mean())
        predictions[validation] = intercept + slope * x[validation]
    residual_ss = float(np.sum((y - predictions) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    return {
        "folds": folds,
        "oof_spearman_r": spearman(predictions, y),
        "oof_r2": 1.0 - residual_ss / total_ss if total_ss else 0.0,
        "oof_rmse": float(np.sqrt(np.mean((y - predictions) ** 2))),
    }


def bootstrap_association(
    x: np.ndarray, y: np.ndarray, resamples: int, seed: int
) -> dict[str, Any]:
    if resamples < 1:
        raise ValueError("at least one bootstrap resample is required")
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    n = len(x)
    for index in range(resamples):
        sample = rng.integers(0, n, size=n)
        values[index] = spearman(x[sample], y[sample])
    return {
        "resamples": resamples,
        "seed": seed,
        "positive_sign_fraction": float(np.mean(values > 0)),
        "median_spearman_r": float(np.median(values)),
        "spearman_r_ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def top_candidates(
    correlations: np.ndarray,
    feature_ids: np.ndarray,
    domains: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    flat = correlations.ravel()
    count = min(limit, flat.size)
    indices = np.argpartition(flat, -count)[-count:] if count < flat.size else np.arange(flat.size)
    candidates: list[tuple[float, int, int]] = []
    width = correlations.shape[1]
    for flat_index in indices:
        domain_index, column = divmod(int(flat_index), width)
        candidates.append(
            (float(correlations[domain_index, column]), int(feature_ids[column]), domain_index)
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {
            "feature_id": feature_id,
            "domain": domains[domain_index],
            "prompt_level_spearman_r": score,
        }
        for score, feature_id, domain_index in candidates
    ]


def external_decoder_norms(path: Path, feature_ids: np.ndarray) -> np.ndarray:
    if path.suffix.casefold() == ".npy":
        norms = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        norm_feature_ids = feature_ids
    elif path.suffix.casefold() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = "decoder_norms" if "decoder_norms" in archive else "norms"
            if key not in archive:
                raise ValueError("decoder-norm NPZ requires decoder_norms or norms")
            norms = np.asarray(archive[key], dtype=np.float64)
            norm_feature_ids = (
                np.asarray(archive["feature_ids"], dtype=np.int64)
                if "feature_ids" in archive
                else feature_ids
            )
    else:
        raise ValueError("decoder norms must be .npy or .npz")
    if norms.shape != norm_feature_ids.shape:
        raise ValueError("decoder norms and their feature IDs differ")
    lookup = {int(feature): float(norm) for feature, norm in zip(norm_feature_ids, norms)}
    missing = [int(feature) for feature in feature_ids if int(feature) not in lookup]
    if missing:
        raise ValueError(f"decoder norm artifact is missing {len(missing)} feature IDs")
    aligned = np.asarray([lookup[int(feature)] for feature in feature_ids], dtype=np.float64)
    if not np.isfinite(aligned).all() or np.any(aligned <= 0):
        raise ValueError("decoder norms must be finite and positive")
    return aligned


def relative_match(value: np.ndarray, target: float, tolerance: float) -> np.ndarray:
    scale = abs(target)
    if scale <= np.finfo(np.float64).eps:
        return np.isclose(value, target, rtol=0.0, atol=np.finfo(np.float64).eps)
    return np.abs(value - target) <= tolerance * scale


def choose_matched_controls(
    feature_ids: np.ndarray,
    activations: np.ndarray,
    decoder_norms: np.ndarray | None,
    primary_feature_id: int,
    activation_epsilon: float,
    seed: int,
    count: int,
) -> dict[str, Any]:
    if decoder_norms is None:
        return {
            "status": "missing_decoder_norms",
            "feature_ids": [],
            "required_count": count,
            "eligible_count": 0,
        }
    active = activations > activation_epsilon
    active_counts = active.sum(axis=0)
    frequencies = active_counts / activations.shape[0]
    positive_sums = np.where(active, activations, 0.0).sum(axis=0)
    means = np.divide(
        positive_sums,
        active_counts,
        out=np.zeros_like(positive_sums, dtype=np.float64),
        where=active_counts > 0,
    )
    locations = np.flatnonzero(feature_ids == primary_feature_id)
    if len(locations) != 1:
        raise ValueError(f"primary feature {primary_feature_id} not found exactly once")
    primary = int(locations[0])
    mask = (
        relative_match(frequencies, float(frequencies[primary]), 0.10)
        & relative_match(means, float(means[primary]), 0.20)
        & relative_match(decoder_norms, float(decoder_norms[primary]), 0.10)
        & (feature_ids != primary_feature_id)
    )
    columns = np.flatnonzero(mask)
    ordered = sorted(
        (int(column) for column in columns),
        key=lambda column: hashlib.sha256(
            f"{seed}|matched-random|{int(feature_ids[column])}".encode("utf-8")
        ).hexdigest(),
    )
    chosen = ordered[:count]
    return {
        "status": "matched" if len(chosen) == count else "insufficient_matches",
        "feature_ids": [int(feature_ids[column]) for column in chosen],
        "required_count": count,
        "eligible_count": len(ordered),
        "seed": seed,
        "matching_tolerances_relative": {
            "activation_frequency": 0.10,
            "mean_positive_activation": 0.20,
            "decoder_norm": 0.10,
        },
        "primary": {
            "activation_frequency": float(frequencies[primary]),
            "mean_positive_activation": float(means[primary]),
            "decoder_norm": float(decoder_norms[primary]),
        },
        "controls": [
            {
                "feature_id": int(feature_ids[column]),
                "activation_frequency": float(frequencies[column]),
                "mean_positive_activation": float(means[column]),
                "decoder_norm": float(decoder_norms[column]),
            }
            for column in chosen
        ],
    }


def interpretation_review(
    path: Path | None,
    feature_id: int,
    domain: str,
    top_prompt_ids: list[str],
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "pending",
            "required_consistent_examples": 4,
            "top_activating_prompt_ids": top_prompt_ids,
            "development_gate_entry_pass": False,
        }
    review = json.loads(path.read_text())
    if int(review.get("feature_id", -1)) != feature_id or review.get("domain") != domain:
        raise ValueError("semantic review feature/domain does not match the selected candidate")
    if review.get("condition_hidden") is not True:
        raise ValueError("semantic review must record condition_hidden=true")
    items = review.get("items")
    if not isinstance(items, list) or len(items) != 5:
        raise ValueError("semantic review requires exactly five top-activation items")
    reviewed_ids = [str(item.get("prompt_id")) for item in items]
    if set(reviewed_ids) != set(top_prompt_ids):
        raise ValueError("semantic review prompt IDs differ from the frozen top five")
    consistent = sum(item.get("semantically_consistent") is True for item in items)
    passed = consistent >= 4
    return {
        "status": "passed" if passed else "failed",
        "review_path": str(path),
        "review_sha256": sha256_file(path),
        "condition_hidden": True,
        "consistent_examples": consistent,
        "required_consistent_examples": 4,
        "top_activating_prompt_ids": top_prompt_ids,
        "development_gate_entry_pass": passed,
    }


def pre_domain_evidence(
    path: Path | None,
    feature_id: int,
    expected_prompt_ids: list[str],
    expected_samples: int,
    activation_epsilon: float,
    minimum_resolution: float,
    minimum_active_fraction: float,
) -> dict[str, Any]:
    if path is None:
        return {
            "status": "pending",
            "development_gate_entry_pass": False,
            "minimum_boundary_resolution": minimum_resolution,
            "minimum_active_fraction": minimum_active_fraction,
        }
    rows = read_jsonl(path)
    by_prompt = {str(row.get("prompt_id")): row for row in rows}
    if len(by_prompt) != len(rows) or set(by_prompt) != set(expected_prompt_ids):
        raise ValueError("pre-domain activation prompts differ from prompt activations")
    total = 0
    resolved = 0
    active = 0
    for prompt_id in expected_prompt_ids:
        row = by_prompt[prompt_id]
        if row.get("split") != "development" or row.get("condition") != "baseline":
            raise ValueError("pre-domain evidence must be untreated development baseline")
        if row.get("dry_run") is True:
            raise ValueError("dry-run pre-domain activations cannot select a feature")
        generations = row.get("generations")
        if not isinstance(generations, list) or len(generations) != expected_samples:
            raise ValueError(
                f"pre-domain evidence for {prompt_id} must contain {expected_samples} outcomes"
            )
        for generation in generations:
            total += 1
            if generation.get("boundary", {}).get("status") != "captured":
                continue
            resolved += 1
            sparse = generation.get("pre_domain_activation", {})
            indices = [int(value) for value in sparse.get("indices", [])]
            values = [float(value) for value in sparse.get("values", [])]
            if len(indices) != len(values) or len(indices) != len(set(indices)):
                raise ValueError("malformed sparse pre-domain activation vector")
            lookup = dict(zip(indices, values))
            if lookup.get(feature_id, 0.0) > activation_epsilon:
                active += 1
    resolution = resolved / total if total else 0.0
    active_fraction = active / resolved if resolved else 0.0
    passed = resolution >= minimum_resolution and active_fraction >= minimum_active_fraction
    return {
        "status": "passed" if passed else "failed",
        "path": str(path),
        "sha256": sha256_file(path),
        "total_generation_positions": total,
        "resolved_boundary_count": resolved,
        "boundary_resolution_fraction": resolution,
        "selected_feature_active_count": active,
        "selected_feature_active_fraction_of_resolved": active_fraction,
        "minimum_boundary_resolution": minimum_resolution,
        "minimum_active_fraction": minimum_active_fraction,
        "development_gate_entry_pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-activations", type=Path, required=True)
    parser.add_argument(
        "--decoder-norms",
        type=Path,
        help="optional .npy/.npz override when decoder_norms are not embedded",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-review", type=Path)
    parser.add_argument(
        "--pre-domain-activations",
        type=Path,
        help="capture_activations.py sparse pre-domain JSONL",
    )
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--min-domain-rate", type=float, default=0.10)
    parser.add_argument("--min-feature-active-prompts", type=int)
    parser.add_argument("--max-feature-active-fraction", type=float, default=0.90)
    parser.add_argument("--activation-epsilon", type=float, default=0.0)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--permutation-batch-size", type=int, default=8)
    parser.add_argument("--permutation-seed", type=int, default=20260831)
    parser.add_argument("--corrected-alpha", type=float, default=0.05)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=20260831)
    parser.add_argument("--min-cv-correlation", type=float, default=0.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--min-bootstrap-positive", type=float, default=0.90)
    parser.add_argument("--top-candidates", type=int, default=20)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-smoke-labels", action="store_true")
    parser.add_argument("--allow-unaudited-labels", action="store_true")
    parser.add_argument("--allow-width-mismatch", action="store_true")
    parser.add_argument(
        "--allow-smoke-activations",
        action="store_true",
        help="permit synthetic activation NPZ files without frozen provenance",
    )
    parser.add_argument(
        "--allow-analysis-override",
        action="store_true",
        help="permit non-frozen settings for synthetic smoke tests only",
    )
    args = parser.parse_args()

    if not 0 < args.min_domain_rate <= 1:
        parser.error("--min-domain-rate must be in (0, 1]")
    if not 0 < args.corrected_alpha < 1:
        parser.error("--corrected-alpha must be in (0, 1)")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    amendment = load_protocol_amendment(protocol)
    amendment_hash = amendment_sha256()
    audit_binding = audit_provenance(amendment, amendment_hash)
    domain_selection = amendment["domain_selection"]
    frozen_minimum_domain_rate = float(
        domain_selection["minimum_development_output_rate"]
    )
    primary_domain_exclusions = list(
        domain_selection["primary_selected_domain_exclusions"]
    )
    frozen_multiple = protocol["feature_discovery"]["multiple_testing"]
    frozen_settings = {
        "condition": "baseline",
        "permutations": int(frozen_multiple["permutations"]),
        "permutation_seed": int(frozen_multiple["seed"]),
        "corrected_alpha": float(frozen_multiple["familywise_alpha"]),
        "cv_folds": 5,
        "cv_seed": int(frozen_multiple["seed"]),
        "min_cv_correlation": 0.0,
        "bootstrap_resamples": int(protocol["analysis"]["bootstrap_resamples"]),
        "bootstrap_seed": int(protocol["analysis"]["bootstrap_seed"]),
        "min_domain_rate": frozen_minimum_domain_rate,
        "max_feature_active_fraction": 0.90,
        "activation_epsilon": 0.0,
        "min_feature_active_prompts": None,
        "min_bootstrap_positive": 0.90,
    }
    drift = {
        name: {"observed": getattr(args, name), "frozen": expected}
        for name, expected in frozen_settings.items()
        if getattr(args, name) != expected
    }
    if drift and not args.allow_analysis_override:
        raise ValueError(
            f"analysis settings drift from protocol: {drift}; overrides are smoke-test-only"
        )
    smoke_or_override = bool(
        drift
        or args.allow_analysis_override
        or args.allow_incomplete
        or args.allow_smoke_labels
        or args.allow_unaudited_labels
        or args.allow_width_mismatch
        or args.allow_smoke_activations
        or args.decoder_norms is not None
    )
    taxonomy = list(protocol["target_domain_taxonomy"])
    unknown_exclusions = sorted(set(primary_domain_exclusions) - set(taxonomy))
    if unknown_exclusions:
        raise ValueError(
            f"amendment excludes domains outside the frozen taxonomy: {unknown_exclusions}"
        )
    expected_samples = int(protocol["generation"]["development_baseline_samples_per_prompt"])
    expected_prompts = int(protocol["stimuli"]["development_count"])

    prompt_ids, activations, feature_ids, _, embedded_decoder_norms = load_prompt_activations(
        args.prompt_activations,
        protocol,
        args.allow_width_mismatch,
        args.allow_smoke_activations,
    )
    decoder_norms = (
        external_decoder_norms(args.decoder_norms, feature_ids)
        if args.decoder_norms
        else embedded_decoder_norms
    )
    frequencies, sample_counts, label_observations, audit_summary = load_prompt_domain_frequencies(
        args.labels,
        taxonomy,
        args.condition,
        expected_samples,
        args.allow_incomplete,
        args.allow_smoke_labels,
        args.allow_unaudited_labels,
        (
            f"{protocol['domain_labeling']['classifier_repo_id']}@"
            f"{protocol['domain_labeling']['classifier_revision']}"
        ),
        audit_binding,
        0.10,
    )
    activation_set = set(prompt_ids)
    label_set = set(frequencies)
    if activation_set != label_set:
        raise ValueError(
            f"activation/label prompt mismatch: {len(activation_set-label_set)} without labels, "
            f"{len(label_set-activation_set)} without activations"
        )
    if len(prompt_ids) != expected_prompts and not args.allow_incomplete:
        raise ValueError(
            f"expected {expected_prompts} development prompts, found {len(prompt_ids)}"
        )
    y = np.stack([frequencies[prompt_id] for prompt_id in prompt_ids], axis=0)
    domain_rates, eligible_domain_columns = primary_domain_columns(
        y,
        taxonomy,
        args.min_domain_rate,
        primary_domain_exclusions,
    )
    if not len(eligible_domain_columns):
        raise ValueError(
            "no non-excluded domain meets the frozen minimum development output rate"
        )
    eligible_domains = [taxonomy[index] for index in eligible_domain_columns]
    y_eligible = y[:, eligible_domain_columns]

    minimum_active = args.min_feature_active_prompts or int(np.ceil(0.10 * len(prompt_ids)))
    maximum_active = int(np.floor(args.max_feature_active_fraction * len(prompt_ids)))
    active_counts = np.count_nonzero(np.abs(activations) > args.activation_epsilon, axis=0)
    activation_ranks = average_rank_columns(activations)
    x_normalized_all, nonconstant = normalized_columns(activation_ranks)
    eligible_feature_columns = np.flatnonzero(
        nonconstant & (active_counts >= minimum_active) & (active_counts <= maximum_active)
    )
    if not len(eligible_feature_columns):
        raise ValueError("no nonconstant feature meets the activation-frequency threshold")
    x_selected = activations[:, eligible_feature_columns]
    x_normalized = x_normalized_all[:, eligible_feature_columns]
    selected_feature_ids = feature_ids[eligible_feature_columns]
    y_ranks = average_rank_columns(y_eligible)
    y_normalized, nonconstant_domains = normalized_columns(y_ranks)
    if not nonconstant_domains.all():
        raise AssertionError("nonconstant domain filter failed")

    correlations = y_normalized.T @ x_normalized
    observed_max = float(correlations.max())
    tied = np.argwhere(np.isclose(correlations, observed_max, rtol=0.0, atol=1e-14))
    winner_domain_col, winner_feature_col = min(
        ((int(row[0]), int(row[1])) for row in tied),
        key=lambda pair: (int(selected_feature_ids[pair[1]]), pair[0]),
    )
    winner_domain = eligible_domains[winner_domain_col]
    winner_feature_id = int(selected_feature_ids[winner_feature_col])
    winner_x = x_selected[:, winner_feature_col]
    winner_y = y_eligible[:, winner_domain_col]

    null_maxima = permutation_maxima(
        x_normalized,
        y_normalized,
        args.permutations,
        args.permutation_seed,
        args.permutation_batch_size,
    )
    corrected_p = float((1 + np.sum(null_maxima >= observed_max)) / (args.permutations + 1))
    cv = prompt_level_cv(winner_x, winner_y, args.cv_folds, args.cv_seed)
    bootstrap = bootstrap_association(
        winner_x, winner_y, args.bootstrap_resamples, args.bootstrap_seed
    )
    positive_values = winner_x[winner_x > args.activation_epsilon]
    eligible_activation_threshold = (
        float(np.median(positive_values)) if len(positive_values) else None
    )
    above_threshold_ids = (
        [
            prompt_ids[index]
            for index in range(len(prompt_ids))
            if winner_x[index] >= float(eligible_activation_threshold)
        ]
        if eligible_activation_threshold is not None
        else []
    )
    below_threshold_ids = [
        prompt_id for prompt_id in prompt_ids if prompt_id not in set(above_threshold_ids)
    ]

    def gate_order(prompt_id: str, stratum: str) -> str:
        return hashlib.sha256(
            f"{args.permutation_seed}|development-gate|{stratum}|{prompt_id}".encode(
                "utf-8"
            )
        ).hexdigest()

    above_take = min(12, len(above_threshold_ids))
    below_take = min(24 - above_take, len(below_threshold_ids))
    if above_take + below_take < 24:
        above_take = min(24 - below_take, len(above_threshold_ids))
    gate_prompt_ids = sorted(
        sorted(above_threshold_ids, key=lambda value: gate_order(value, "above"))[
            :above_take
        ]
        + sorted(below_threshold_ids, key=lambda value: gate_order(value, "below"))[
            :below_take
        ]
    )
    top_prompt_ids = [
        prompt_ids[index]
        for index in sorted(
            range(len(prompt_ids)), key=lambda index: (-winner_x[index], prompt_ids[index])
        )[:5]
    ]
    diagnostics = {
        "feature_id": winner_feature_id,
        "domain": winner_domain,
        "domain_output_rate": float(domain_rates[taxonomy.index(winner_domain)]),
        "prompt_level_spearman_r": observed_max,
        "permutation_max_statistic_p": corrected_p,
        "activation_frequency": float(np.mean(winner_x > args.activation_epsilon)),
        "active_prompt_count": int(np.count_nonzero(winner_x > args.activation_epsilon)),
        "mean_positive_activation": float(positive_values.mean()) if len(positive_values) else 0.0,
        "eligible_activation_threshold": eligible_activation_threshold,
        "cv": cv,
        "prompt_bootstrap": bootstrap,
    }
    passes = {
        "domain_rate": diagnostics["domain_output_rate"] >= args.min_domain_rate,
        "max_statistic_correction": corrected_p <= args.corrected_alpha,
        "positive_cross_validated_association": cv["oof_spearman_r"] > args.min_cv_correlation,
        "bootstrap_sign_stability": bootstrap["positive_sign_fraction"] >= args.min_bootstrap_positive,
    }
    statistically_selected = all(passes.values())
    matched_count = int(protocol["feature_discovery"]["matched_random_features"]["count"])
    matched_seed = int(protocol["feature_discovery"]["matched_random_features"]["seed"])
    matched_controls = choose_matched_controls(
        feature_ids,
        activations,
        decoder_norms,
        winner_feature_id,
        args.activation_epsilon,
        matched_seed,
        matched_count,
    )
    review = interpretation_review(
        args.semantic_review, winner_feature_id, winner_domain, top_prompt_ids
    )
    pre_domain_rule = protocol["feature_discovery"]["pre_domain_evidence"]
    timing_evidence = pre_domain_evidence(
        args.pre_domain_activations,
        winner_feature_id,
        prompt_ids,
        expected_samples,
        args.activation_epsilon,
        float(pre_domain_rule["minimum_boundary_resolution"]),
        float(pre_domain_rule["minimum_active_fraction_of_resolved"]),
    )
    development_gate_ready = bool(
        not smoke_or_override
        and statistically_selected
        and matched_controls["status"] == "matched"
        and review["development_gate_entry_pass"]
        and timing_evidence["development_gate_entry_pass"]
    )
    if development_gate_ready:
        selection_status = "ready_for_development_gate"
    elif statistically_selected and matched_controls["status"] == "matched":
        pending = []
        if not review["development_gate_entry_pass"]:
            pending.append("interpretation_review")
        if not timing_evidence["development_gate_entry_pass"]:
            pending.append("pre_domain_evidence")
        selection_status = "candidate_pending_" + "_and_".join(pending)
    elif statistically_selected:
        selection_status = "candidate_without_matched_controls"
    else:
        selection_status = "no_stable_pair"
    result = {
        "schema_version": 1,
        "artifact": "development_feature_discovery",
        "protocol_id": protocol["protocol_id"],
        "protocol_revision": protocol.get("protocol_revision"),
        "effective_protocol_revision": amendment["effective_protocol_revision"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "protocol_amendment_id": amendment["amendment_id"],
        "protocol_amendment_sha256": amendment_hash,
        "domain_labeling_guide_id": audit_binding["domain_labeling_guide_id"],
        "domain_labeling_guide_sha256": audit_binding[
            "domain_labeling_guide_sha256"
        ],
        "domain_classifier_revision": protocol["domain_labeling"][
            "classifier_revision"
        ],
        "analysis_scope": "development_only",
        "smoke_or_override": smoke_or_override,
        "selection_status": selection_status,
        "development_gate_ready": development_gate_ready,
        "test_freezable": False,
        "selected_domain": winner_domain if development_gate_ready else None,
        "selected_feature_id": winner_feature_id if development_gate_ready else None,
        "five_matched_random_feature_ids": (
            matched_controls["feature_ids"] if development_gate_ready else []
        ),
        "eligible_activation_threshold": (
            eligible_activation_threshold if development_gate_ready else None
        ),
        "promotion_target": (
            float(np.quantile(positive_values, 0.95))
            if development_gate_ready and len(positive_values)
            else None
        ),
        "selected_pair": diagnostics if development_gate_ready else None,
        "statistical_candidate": diagnostics if statistically_selected else None,
        "best_candidate_diagnostics": diagnostics,
        "selection_checks": passes,
        "matched_random_controls": matched_controls,
        "interpretation_review": review,
        "pre_domain_evidence": timing_evidence,
        "development_gate_plan": {
            "selection": "hash order within above/below development-threshold strata",
            "seed": args.permutation_seed,
            "eligible_activation_threshold": eligible_activation_threshold,
            "prompt_ids": gate_prompt_ids,
            "prompt_count": len(gate_prompt_ids),
            "above_threshold_count": sum(
                prompt_id in set(above_threshold_ids) for prompt_id in gate_prompt_ids
            ),
            "below_threshold_count": sum(
                prompt_id in set(below_threshold_ids) for prompt_id in gate_prompt_ids
            ),
        },
        "domain_label_audit": audit_summary,
        "multiplicity": {
            "method": "prompt-label permutation maximum positive Spearman statistic",
            "permutations": args.permutations,
            "seed": args.permutation_seed,
            "corrected_alpha": args.corrected_alpha,
            "observed_global_max": observed_max,
            "null_max_ci95": [
                float(np.quantile(null_maxima, 0.025)),
                float(np.quantile(null_maxima, 0.975)),
            ],
            "corrected_p_value": corrected_p,
            "features_searched": int(len(eligible_feature_columns)),
            "domains_searched": len(eligible_domains),
            "pair_count": int(len(eligible_feature_columns) * len(eligible_domains)),
        },
        "prompt_level_design": {
            "unit": "source_prompt",
            "prompt_count": len(prompt_ids),
            "generation_label_observation_count": label_observations,
            "samples_per_prompt_histogram": dict(sorted(Counter(sample_counts.values()).items())),
            "activation_rows_per_prompt": 1,
            "outcome": "per-prompt independently labeled target-domain frequency",
        },
        "domain_output_rates": {
            domain: float(domain_rates[index]) for index, domain in enumerate(taxonomy)
        },
        "domain_selection_policy": {
            "descriptive_rate_domains": taxonomy,
            "primary_search_exclusions": primary_domain_exclusions,
            "minimum_development_output_rate": frozen_minimum_domain_rate,
            "other_retained_for_coverage_and_diversity_outcomes": True,
            "source": "protocol_amendment_4.json:domain_selection",
        },
        "eligible_domains": eligible_domains,
        "feature_filter": {
            "width": int(activations.shape[1]),
            "minimum_active_prompts": minimum_active,
            "maximum_active_prompts": maximum_active,
            "activation_epsilon": args.activation_epsilon,
            "eligible_feature_count": int(len(eligible_feature_columns)),
        },
        "top_candidates_exploratory": top_candidates(
            correlations, selected_feature_ids, eligible_domains, args.top_candidates
        ),
        "thresholds": {
            "minimum_domain_rate": args.min_domain_rate,
            "frozen_minimum_domain_rate": frozen_minimum_domain_rate,
            "minimum_cv_correlation_exclusive": args.min_cv_correlation,
            "minimum_bootstrap_positive_fraction": args.min_bootstrap_positive,
        },
    }
    output_hash = atomic_write_json(args.output, result)
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "protocol_amendment_id": amendment["amendment_id"],
        "protocol_amendment_sha256": amendment_hash,
        "domain_labeling_guide_id": audit_binding["domain_labeling_guide_id"],
        "domain_labeling_guide_sha256": audit_binding[
            "domain_labeling_guide_sha256"
        ],
        "primary_search_domain_exclusions": primary_domain_exclusions,
        "minimum_development_output_rate": frozen_minimum_domain_rate,
        "activation_path": str(args.prompt_activations),
        "activation_sha256": sha256_file(args.prompt_activations),
        "label_path": str(args.labels),
        "label_sha256": sha256_file(args.labels),
        "output_sha256": output_hash,
        "selection_status": result["selection_status"],
        "test_data_accessed": False,
    }
    atomic_write_json(args.output.with_name(args.output.name + ".meta.json"), metadata)
    print(
        json.dumps(
            {
                "selection_status": result["selection_status"],
                "best_feature_id": winner_feature_id,
                "best_domain": winner_domain,
                "corrected_p": corrected_p,
                "prompt_count": len(prompt_ids),
                "development_gate_ready": development_gate_ready,
                "test_freezable": False,
            },
            sort_keys=True,
        )
    )
    return 0 if development_gate_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
