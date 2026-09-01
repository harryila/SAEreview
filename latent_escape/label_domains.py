#!/usr/bin/env python3
"""Blind and independently label analogy target domains.

The generated ``target_domain`` field is always removed before classification.
The pinned BART-MNLI zero-shot classifier is the primary automated backend; the
keyword backend exists only for offline plumbing tests and is marked ineligible
for primary analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "latent_escape" / "protocol.json"
CLASSIFIER_REPO = "facebook/bart-large-mnli"
CLASSIFIER_REVISION = "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
HYPOTHESIS_TEMPLATE = "This analogy's target domain is {}."
AUDIT_SEED = "latent-escape-domain-audit-v1"
SELF_LABEL_KEYS = {
    "targetdomain",
    "domainlabel",
    "assigneddomain",
    "predicteddomain",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def jsonl_payload(rows: Iterable[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} has no records")
    return rows


def normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def scrub_self_labels(value: Any) -> Any:
    """Recursively remove fields that directly state the generated domain."""
    if isinstance(value, dict):
        return {
            key: scrub_self_labels(item)
            for key, item in value.items()
            if normalized_key(str(key)) not in SELF_LABEL_KEYS
        }
    if isinstance(value, list):
        return [scrub_self_labels(item) for item in value]
    return value


SELF_LABEL_JSON_RE = re.compile(
    r'''(?ix)
    ( ["']?target[_\s-]*domain["']?\s*:\s* )
    ( "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' | [^,}\]\n]+ )
    '''
)
SELF_LABEL_LINE_RE = re.compile(
    r"(?im)^\s*(?:target[ _-]*domain|domain[ _-]*label)\s*:\s*.*$"
)


def analogy_text(record: dict[str, Any]) -> str:
    """Extract analogy content while excluding the model's own domain claim."""
    parsed = record.get("parsed_output", record.get("parsed_json"))
    if isinstance(parsed, (dict, list)):
        cleaned = scrub_self_labels(parsed)
        return json.dumps(cleaned, ensure_ascii=False, sort_keys=True)

    raw: Any = None
    for key in ("generated_text", "raw_text", "completion", "response_text", "text"):
        if isinstance(record.get(key), str):
            raw = record[key]
            break
    if raw is None:
        raise ValueError(
            "generation record lacks parsed_output/parsed_json and generated_text/raw_text"
        )
    try:
        parsed_raw = json.loads(raw)
    except json.JSONDecodeError:
        redacted = SELF_LABEL_JSON_RE.sub(r'\1"[REDACTED]"', raw)
        redacted = SELF_LABEL_LINE_RE.sub("", redacted)
        return redacted.strip()
    return json.dumps(scrub_self_labels(parsed_raw), ensure_ascii=False, sort_keys=True)


def canonical_seed(record: dict[str, Any]) -> int:
    value = record.get("seed", record.get("paired_seed"))
    if value is None:
        raise ValueError("generation record lacks seed")
    return int(value)


def record_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    required = ("prompt_id", "condition", "sample_index")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"generation record missing keys: {missing}")
    return (
        str(record["prompt_id"]),
        str(record["condition"]),
        int(record["sample_index"]),
        canonical_seed(record),
    )


def blind_records(
    generations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, int, int]]]:
    blinded: list[dict[str, Any]] = []
    mapping: dict[str, tuple[str, str, int, int]] = {}
    seen_keys: set[tuple[str, str, int, int]] = set()
    for record in generations:
        key = record_key(record)
        if key in seen_keys:
            raise ValueError(f"duplicate generation key {key}")
        seen_keys.add(key)
        text = analogy_text(record)
        text_hash = sha256_bytes(text.encode("utf-8"))
        identity = json.dumps(key, separators=(",", ":"), ensure_ascii=False)
        blind_id = sha256_bytes(
            f"latent-escape-blind-v1|{identity}|{text_hash}".encode("utf-8")
        )[:32]
        if blind_id in mapping:
            raise ValueError(f"blind ID collision at {blind_id}")
        mapping[blind_id] = key
        blinded.append(
            {
                "schema_version": 1,
                "record_type": "blinded_analogy",
                "blind_id": blind_id,
                "analogy_text": text,
                "analogy_text_sha256": text_hash,
            }
        )
    blinded.sort(key=lambda row: row["blind_id"])
    return blinded, mapping


HEURISTIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "biology/ecology": ("ecosystem", "species", "predator", "habitat", "organism", "evolution"),
    "medicine/public health": ("patient", "disease", "hospital", "vaccine", "diagnosis", "public health"),
    "physics": ("force", "particle", "energy", "quantum", "gravity", "momentum"),
    "chemistry/materials": ("molecule", "chemical", "catalyst", "polymer", "alloy", "reaction"),
    "engineering/control": ("controller", "feedback loop", "sensor", "actuator", "engineering", "stability"),
    "computer science/software": ("software", "database", "algorithm", "server", "program", "network protocol"),
    "AI/neural networks": ("neural network", "model training", "artificial intelligence", "gradient", "neuron", "machine learning"),
    "economics/markets": ("market", "price", "buyer", "seller", "economy", "supply and demand"),
    "organizations/governance": ("organization", "management", "board", "company", "department", "governance"),
    "sociology/culture": ("community", "social norm", "culture", "society", "social group", "institution"),
    "psychology/cognition": ("memory", "attention", "belief", "cognitive", "emotion", "mind"),
    "education/learning": ("student", "teacher", "classroom", "curriculum", "learning", "school"),
    "law/policy": ("court", "law", "regulation", "legal", "policy", "legislature"),
    "history": ("historical", "empire", "dynasty", "century", "war", "ancient"),
    "arts/literature": ("novel", "music", "painting", "artist", "story", "theater"),
    "sports/games": ("team", "player", "game", "coach", "tournament", "chess"),
    "geography/earth/environment": ("climate", "river", "geological", "landscape", "earth", "watershed"),
    "everyday/household": ("household", "kitchen", "family", "home", "shopping", "cleaning"),
}


def heuristic_labels(
    blinded: list[dict[str, Any]], taxonomy: list[str]
) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    for row in blinded:
        text = row["analogy_text"].casefold()
        scores: dict[str, float] = {}
        for domain in taxonomy:
            patterns = HEURISTIC_PATTERNS.get(domain, ())
            scores[domain] = float(
                sum(1 + (" " in phrase) for phrase in patterns if phrase in text)
            )
        maximum = max(scores.values(), default=0.0)
        if maximum == 0:
            results.append(("other", 0.0))
            continue
        winner = next(domain for domain in taxonomy if scores[domain] == maximum)
        total = sum(scores.values())
        results.append((winner, maximum / total if total else 0.0))
    return results


def hf_zero_shot_labels(
    blinded: list[dict[str, Any]],
    taxonomy: list[str],
    device: str,
    batch_size: int,
    local_files_only: bool,
) -> list[tuple[str, float]]:
    try:
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional run stack
        raise RuntimeError(
            "The primary classifier requires transformers. Sync the latent-run "
            "environment or use --backend heuristic only for an offline smoke test."
        ) from exc

    common = {
        "revision": CLASSIFIER_REVISION,
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_REPO, **common)
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_REPO, **common)
    model.eval()
    classifier = pipeline(
        "zero-shot-classification", model=model, tokenizer=tokenizer, device=device
    )
    outputs = classifier(
        [row["analogy_text"] for row in blinded],
        candidate_labels=taxonomy,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
        batch_size=batch_size,
    )
    if isinstance(outputs, dict):
        outputs = [outputs]
    return [(str(item["labels"][0]), float(item["scores"][0])) for item in outputs]


def load_label_import(path: Path, taxonomy: list[str]) -> dict[str, tuple[str, float | None]]:
    if path.suffix.casefold() == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = read_jsonl(path)
    canonical_domains = {domain.casefold(): domain for domain in taxonomy}
    imported: dict[str, tuple[str, float | None]] = {}
    for row in rows:
        blind_id = str(row.get("blind_id", "")).strip()
        if not blind_id:
            raise ValueError(f"{path}: imported row lacks blind_id")
        raw_domain = next(
            (
                row[key]
                for key in (
                    "manual_domain_label",
                    "domain_label",
                    "assigned_domain",
                    "label",
                )
                if row.get(key) not in (None, "")
            ),
            None,
        )
        if raw_domain is None or str(raw_domain).casefold() not in canonical_domains:
            raise ValueError(f"{path}: invalid domain for {blind_id}: {raw_domain!r}")
        if blind_id in imported:
            raise ValueError(f"{path}: duplicate blind_id {blind_id}")
        confidence = row.get("confidence", row.get("classifier_confidence"))
        imported[blind_id] = (
            canonical_domains[str(raw_domain).casefold()],
            None if confidence in (None, "") else float(confidence),
        )
    return imported


def stratified_audit_ids(
    rows: list[dict[str, Any]], fraction: float, seed: str
) -> set[str]:
    if not 0 <= fraction <= 1:
        raise ValueError("audit fraction must be between zero and one")
    target = int(math.ceil(len(rows) * fraction))
    if target == 0:
        return set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["domain_label"]].append(row)
    allocations = {domain: int(len(group) * fraction) for domain, group in groups.items()}
    remaining = target - sum(allocations.values())
    remainder_order = sorted(
        groups,
        key=lambda domain: (
            -(len(groups[domain]) * fraction - allocations[domain]),
            sha256_bytes(f"{seed}|stratum|{domain}".encode()),
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
        chosen.update(row["blind_id"] for row in ordered[: allocations[domain]])
    return chosen


def default_sibling(output: Path, suffix: str) -> Path:
    stem = output.name[:-6] if output.name.endswith(".jsonl") else output.stem
    return output.with_name(f"{stem}.{suffix}.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("hf-zero-shot", "external", "heuristic"),
        default="hf-zero-shot",
        help="heuristic is smoke-test-only and cannot feed primary analysis",
    )
    parser.add_argument("--external-labels", type=Path)
    parser.add_argument("--manual-overrides", type=Path)
    parser.add_argument("--external-classifier-id", default="external-blinded-classifier")
    parser.add_argument("--blind-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--audit-fraction", type=float, default=0.10)
    parser.add_argument("--audit-seed", default=AUDIT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.backend == "hf-zero-shot" and (
        not math.isclose(args.audit_fraction, 0.10, rel_tol=0.0, abs_tol=1e-12)
        or args.audit_seed != AUDIT_SEED
    ):
        raise ValueError(
            "primary BART labeling requires the frozen 10% audit fraction and seed"
        )

    protocol = json.loads(PROTOCOL_PATH.read_text())
    taxonomy = list(protocol["target_domain_taxonomy"])
    if "other" not in taxonomy:
        raise ValueError("frozen taxonomy must contain 'other'")
    generations = read_jsonl(args.generations)
    protocol_ids = {row.get("protocol_id") for row in generations}
    if protocol_ids != {protocol["protocol_id"]}:
        raise ValueError(f"generation protocol IDs do not match: {protocol_ids}")

    blinded, mapping = blind_records(generations)
    blind_output = args.blind_output or default_sibling(args.output, "blinded")
    blind_payload = jsonl_payload(blinded)
    atomic_write(blind_output, blind_payload)

    classifier_id: str
    primary_eligible: bool
    if args.backend == "hf-zero-shot":
        predictions = hf_zero_shot_labels(
            blinded, taxonomy, args.device, args.batch_size, args.local_files_only
        )
        classifier_id = f"{CLASSIFIER_REPO}@{CLASSIFIER_REVISION}"
        primary_eligible = True
    elif args.backend == "heuristic":
        predictions = heuristic_labels(blinded, taxonomy)
        classifier_id = "deterministic-keyword-smoke-v1"
        primary_eligible = False
    else:
        if args.external_labels is None:
            parser.error("--backend external requires --external-labels")
        imported = load_label_import(args.external_labels, taxonomy)
        missing = sorted(set(mapping) - set(imported))
        unknown = sorted(set(imported) - set(mapping))
        if missing or unknown:
            raise ValueError(
                f"external label join mismatch: {len(missing)} missing, {len(unknown)} unknown"
            )
        predictions = [imported[row["blind_id"]] for row in blinded]
        classifier_id = args.external_classifier_id
        primary_eligible = False

    prediction_map = {
        row["blind_id"]: prediction for row, prediction in zip(blinded, predictions)
    }
    overrides: dict[str, tuple[str, float | None]] = {}
    if args.manual_overrides:
        overrides = load_label_import(args.manual_overrides, taxonomy)
        unknown = sorted(set(overrides) - set(mapping))
        if unknown:
            raise ValueError(f"manual overrides contain {len(unknown)} unknown blind IDs")

    generation_by_key = {record_key(row): row for row in generations}
    labels: list[dict[str, Any]] = []
    for blind in blinded:
        blind_id = blind["blind_id"]
        key = mapping[blind_id]
        source = generation_by_key[key]
        classifier_domain, classifier_confidence = prediction_map[blind_id]
        domain, confidence = overrides.get(
            blind_id, (classifier_domain, classifier_confidence)
        )
        labels.append(
            {
                "schema_version": 1,
                "record_type": "independent_domain_label",
                "protocol_id": protocol["protocol_id"],
                "protocol_revision": protocol.get("protocol_revision"),
                "run_id": source.get("run_id"),
                "prompt_id": key[0],
                "split": source.get("split"),
                "condition": key[1],
                "sample_index": key[2],
                "seed": key[3],
                "blind_id": blind_id,
                "analogy_text_sha256": blind["analogy_text_sha256"],
                "domain_label": domain,
                "classifier_domain_label": classifier_domain,
                "classifier_id": classifier_id,
                "classifier_confidence": classifier_confidence,
                "manual_audited": blind_id in overrides,
                "manual_override": blind_id in overrides and domain != classifier_domain,
                "manual_label_confidence": confidence if blind_id in overrides else None,
                "primary_eligible": primary_eligible,
            }
        )
    labels.sort(key=lambda row: (row["prompt_id"], row["condition"], row["sample_index"], row["seed"]))

    # Audit membership is frozen from classifier predictions, before any manual
    # adjudication. Re-stratifying on final labels could change the queue.
    classifier_rows = [
        {**row, "domain_label": row["classifier_domain_label"]} for row in labels
    ]
    audit_ids = stratified_audit_ids(
        classifier_rows, args.audit_fraction, args.audit_seed
    )
    if overrides and set(overrides) != audit_ids:
        missing = audit_ids - set(overrides)
        extra = set(overrides) - audit_ids
        raise ValueError(
            "manual audit import must contain every frozen audit item and no others: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    for row in labels:
        row["audit_selected"] = row["blind_id"] in audit_ids
        row["audit_fraction"] = args.audit_fraction
        row["audit_seed"] = args.audit_seed
    blind_by_id = {row["blind_id"]: row for row in blinded}
    audit_rows = [
        {
            "schema_version": 1,
            "record_type": "blinded_domain_audit",
            "blind_id": row["blind_id"],
            "analogy_text": blind_by_id[row["blind_id"]]["analogy_text"],
            "analogy_text_sha256": row["analogy_text_sha256"],
            # Deliberately omit the automated label. The double-check must be
            # independent, not an agreement-confirmation task.
            "manual_domain_label": row["domain_label"]
            if row["blind_id"] in overrides
            else None,
        }
        for row in labels
        if row["blind_id"] in audit_ids
    ]
    audit_rows.sort(key=lambda row: row["blind_id"])
    audit_output = args.audit_output or default_sibling(args.output, "audit")

    label_payload = jsonl_payload(labels)
    audit_payload = jsonl_payload(audit_rows)
    atomic_write(args.output, label_payload)
    atomic_write(audit_output, audit_payload)
    metadata = {
        "schema_version": 1,
        "artifact": "independent_domain_labels",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "protocol_revision": protocol.get("protocol_revision"),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "generation_path": str(args.generations),
        "generation_sha256": sha256_file(args.generations),
        "label_sha256": sha256_bytes(label_payload),
        "blind_output_path": str(blind_output),
        "blind_output_sha256": sha256_bytes(blind_payload),
        "audit_output_path": str(audit_output),
        "audit_output_sha256": sha256_bytes(audit_payload),
        "classifier_backend": args.backend,
        "classifier_id": classifier_id,
        "classifier_revision": CLASSIFIER_REVISION if args.backend == "hf-zero-shot" else None,
        "hypothesis_template": HYPOTHESIS_TEMPLATE if args.backend == "hf-zero-shot" else None,
        "primary_eligible": primary_eligible,
        "self_reported_target_domain_used": False,
        "manual_override_count": len(overrides),
        "label_count": len(labels),
        "domain_counts": dict(sorted(Counter(row["domain_label"] for row in labels).items())),
        "audit_fraction": args.audit_fraction,
        "audit_count": len(audit_rows),
        "audit_seed": args.audit_seed,
    }
    meta_path = args.output.with_name(args.output.name + ".meta.json")
    atomic_write(
        meta_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"wrote {args.output} ({len(labels)} independent labels)")
    print(f"wrote {audit_output} ({len(audit_rows)} blinded audit items)")
    if not primary_eligible:
        print("WARNING: heuristic labels are smoke-test-only and primary_eligible=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
