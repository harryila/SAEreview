#!/usr/bin/env python3
"""Create the raw OpenAI embedding cache required by the retrieval study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "scar_system_analogy_en.jsonl"
CACHE = ROOT / ".cache" / "openai_text_embedding_3_small_scar.npz"
MODEL = "text-embedding-3-small"
DATA_SHA256 = "12883db11de17454b3a4ae30a109f4b64861125b1e94846e17b8edc3f8a12369"
CANONICAL_CACHE_SHA256 = (
    "ba182c6447de5fd4495e24d04df3e5bbd163737eaf09732401042fe6a76f7446"
)
INPUT_SHA256 = "467f3103e0b5598c20aca4dab6610c5ccb861e36175177f1aabf1b6fb1588030"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows() -> list[dict[str, object]]:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Run scripts/prepare_retrieval_assets.py first."
        )
    actual_hash = sha256_file(DATA)
    if actual_hash != DATA_SHA256:
        raise ValueError(f"SCAR SHA-256 is {actual_hash}; expected {DATA_SHA256}")
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    if len(rows) != 400:
        raise ValueError(f"Expected 400 SCAR rows, found {len(rows)}")
    return rows


def valid_cache(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    if sha256_file(path) == CANONICAL_CACHE_SHA256:
        return True
    with np.load(path) as cached:
        embeddings = np.asarray(cached["embeddings"])
        model = str(cached["model"].item()) if "model" in cached else ""
        input_sha256 = (
            str(cached["input_sha256"].item()) if "input_sha256" in cached else ""
        )
    return (
        embeddings.shape == (expected_rows, 1536)
        and np.isfinite(embeddings).all()
        and model == MODEL
        and input_sha256 == INPUT_SHA256
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    rows = load_rows()
    texts = [str(row["system_a_background"]) for row in rows] + [
        str(row["system_b_background"]) for row in rows
    ]
    serialized = json.dumps(
        texts, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    actual_input_hash = hashlib.sha256(serialized).hexdigest()
    if actual_input_hash != INPUT_SHA256:
        raise ValueError(
            f"Ordered embedding input SHA-256 is {actual_input_hash}; "
            f"expected {INPUT_SHA256}"
        )
    if not args.force and valid_cache(CACHE, len(texts)):
        print(f"verified {CACHE.relative_to(ROOT)}")
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required only when the compatible cache is absent"
        )
    print(
        "WARNING: creating a replication cache with a live hosted model alias; "
        "byte-exact historical embeddings are not guaranteed"
    )

    from openai import OpenAI

    client = OpenAI()
    collected: list[np.ndarray] = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        response = client.embeddings.create(model=MODEL, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        collected.extend(
            np.asarray(item.embedding, dtype=np.float32) for item in ordered
        )
        print(f"embedded {min(start + len(batch), len(texts))}/{len(texts)}")

    embeddings = np.stack(collected)
    if embeddings.shape != (800, 1536) or not np.isfinite(embeddings).all():
        raise ValueError(f"Unexpected embedding output: {embeddings.shape}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        embeddings=embeddings,
        model=np.asarray(MODEL),
        input_sha256=np.asarray(INPUT_SHA256),
    )
    print(f"wrote {CACHE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
