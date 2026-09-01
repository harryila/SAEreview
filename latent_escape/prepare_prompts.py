#!/usr/bin/env python3
"""Build the deterministic Latent Escape development/test prompt manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "scar_system_analogy_en.jsonl"
OUTPUT = ROOT / "latent_escape" / "artifacts" / "prompt_manifest.jsonl"
METADATA = ROOT / "latent_escape" / "artifacts" / "prompt_manifest.meta.json"
DATA_SHA256 = "12883db11de17454b3a4ae30a109f4b64861125b1e94846e17b8edc3f8a12369"
SELECTION_SEED = "latent-escape-prompts-v1"
PROMPT_COUNT = 160
DEVELOPMENT_COUNT = 80


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def stable_score(namespace: str, value: str) -> str:
    return sha256_bytes(f"{SELECTION_SEED}|{namespace}|{value}".encode("utf-8"))


def prompt_text(name: str, domain: str, description: str) -> str:
    return f"""You are given a source mechanism. Generate one structurally faithful analogy in a clearly different domain. Preserve causal roles, relations, and boundary conditions rather than merely matching vocabulary.

Return only valid JSON with these keys:
- target_domain: string
- target_system: string
- mappings: an array of at least three objects with source_role, target_role, and shared_relation
- explanation: string
- limitations: an array of at least two strings

Source system: {name}
Source domain: {domain}
Source description: {description}"""


def load_candidates() -> list[dict[str, str]]:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Run scripts/prepare_retrieval_assets.py --data-only first."
        )
    actual_hash = sha256_file(DATA)
    if actual_hash != DATA_SHA256:
        raise ValueError(f"SCAR SHA-256 is {actual_hash}; expected {DATA_SHA256}")

    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    if len(rows) != 400:
        raise ValueError(f"Expected 400 SCAR rows, found {len(rows)}")

    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        for side in ("a", "b"):
            name = str(row[f"system_{side}"]).strip()
            domain = str(row[f"system_{side}_domain"]).strip()
            description = str(row[f"system_{side}_background"]).strip()
            source_id = f"scar-{int(row['id']):03d}-{side}"
            candidate = {
                "source_id": source_id,
                "source_group_id": f"scar-pair-{int(row['id']):03d}",
                "source_name": name,
                "source_domain": domain,
                "source_description": description,
            }
            key = normalized_name(name)
            current = by_name.get(key)
            if current is None or stable_score("dedupe", source_id) < stable_score(
                "dedupe", current["source_id"]
            ):
                by_name[key] = candidate
    by_pair: dict[str, dict[str, str]] = {}
    for candidate in by_name.values():
        pair_id = candidate["source_group_id"]
        current = by_pair.get(pair_id)
        candidate_score = stable_score("pair-side", candidate["source_id"])
        current_score = (
            stable_score("pair-side", current["source_id"])
            if current is not None
            else None
        )
        if current_score is None or candidate_score < current_score:
            by_pair[pair_id] = candidate
    return list(by_pair.values())


def balanced_selection(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate["source_domain"]].append(candidate)
    for domain, bucket in buckets.items():
        bucket.sort(key=lambda row: stable_score(f"select:{domain}", row["source_id"]))

    selected: list[dict[str, str]] = []
    offsets = {domain: 0 for domain in buckets}
    domains = sorted(buckets)
    while len(selected) < PROMPT_COUNT:
        added = False
        for domain in domains:
            offset = offsets[domain]
            if offset < len(buckets[domain]) and len(selected) < PROMPT_COUNT:
                selected.append(buckets[domain][offset])
                offsets[domain] += 1
                added = True
        if not added:
            raise ValueError(f"Only {len(selected)} unique source systems are available")
    return selected


def build_records(selected: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in selected:
        by_domain[source["source_domain"]].append(source)
    development_per_domain = {
        domain: len(bucket) // 2 for domain, bucket in by_domain.items()
    }
    remaining = DEVELOPMENT_COUNT - sum(development_per_domain.values())
    odd_domains = sorted(
        (domain for domain, bucket in by_domain.items() if len(bucket) % 2),
        key=lambda domain: stable_score("split-domain", domain),
    )
    for domain in odd_domains[:remaining]:
        development_per_domain[domain] += 1

    development_ids: set[str] = set()
    for domain, bucket in by_domain.items():
        split_order = sorted(
            bucket, key=lambda row: stable_score("split", row["source_id"])
        )
        development_ids.update(
            row["source_id"] for row in split_order[: development_per_domain[domain]]
        )
    ordered = sorted(selected, key=lambda row: stable_score("manifest", row["source_id"]))
    records: list[dict[str, Any]] = []
    for index, source in enumerate(ordered, start=1):
        split = "development" if source["source_id"] in development_ids else "test"
        records.append(
            {
                "prompt_id": f"le-{index:03d}",
                "split": split,
                **source,
                "prompt_text": prompt_text(
                    source["source_name"],
                    source["source_domain"],
                    source["source_description"],
                ),
            }
        )
    return records


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, path)


def main() -> int:
    records = build_records(balanced_selection(load_candidates()))
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    manifest_payload = ("\n".join(lines) + "\n").encode("utf-8")
    manifest_hash = sha256_bytes(manifest_payload)
    atomic_write(OUTPUT, manifest_payload)

    metadata = {
        "protocol_id": "latent-escape-mvp-v1",
        "source_sha256": DATA_SHA256,
        "selection_seed": SELECTION_SEED,
        "manifest_sha256": manifest_hash,
        "prompt_count": len(records),
        "split_counts": dict(sorted(Counter(row["split"] for row in records).items())),
        "domain_counts": dict(
            sorted(Counter(row["source_domain"] for row in records).items())
        ),
    }
    atomic_write(
        METADATA,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({manifest_hash})")
    print(json.dumps(metadata["split_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
