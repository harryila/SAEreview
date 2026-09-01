#!/usr/bin/env python3
"""Download the exact unbundled SCAR data and corrected SAE checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Asset:
    path: Path
    url: str
    sha256: str


ASSETS = (
    Asset(
        path=ROOT / "data" / "scar_system_analogy_en.jsonl",
        url=(
            "https://raw.githubusercontent.com/siyuyuan/scar/"
            "3dfc897cf6cc685531edc80ab64f35660403fc6c/"
            "release/system_analogy_en.json"
        ),
        sha256="12883db11de17454b3a4ae30a109f4b64861125b1e94846e17b8edc3f8a12369",
    ),
    Asset(
        path=ROOT / "weights" / "csLG_64_9216.pth",
        url=(
            "https://huggingface.co/datasets/charlieoneill/saerchModels/resolve/"
            "b2cbb184b58880b77a546511e11d8fd214c40556/"
            "csLG_64_9216.pth?download=true"
        ),
        sha256="29073be46ce5ddceee53f7e9ebf46449e239c1bc29f57dfebced041833698752",
    ),
    Asset(
        path=ROOT / "weights" / "astroPH_64_9216.pth",
        url=(
            "https://huggingface.co/datasets/charlieoneill/saerchModels/resolve/"
            "b2cbb184b58880b77a546511e11d8fd214c40556/"
            "astroPH_64_9216.pth?download=true"
        ),
        sha256="112e8a006ff0cc8e3b4439e1ef28df816564c5d9054974a763eaa69804cf02ed",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset: Asset, *, force: bool) -> None:
    if asset.path.exists() and not force:
        actual = sha256_file(asset.path)
        if actual == asset.sha256:
            print(f"verified {asset.path.relative_to(ROOT)}")
            return
        raise ValueError(
            f"Existing {asset.path} has SHA-256 {actual}; expected {asset.sha256}. "
            "Use --force to replace it."
        )

    asset.path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "SAEreview asset fetcher"},
    )
    with tempfile.NamedTemporaryFile(
        dir=asset.path.parent, prefix=f".{asset.path.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, temporary)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    actual = sha256_file(temporary_path)
    if actual != asset.sha256:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded {asset.path.name} has SHA-256 {actual}; expected {asset.sha256}"
        )
    os.replace(temporary_path, asset.path)
    print(f"downloaded {asset.path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="fetch only SCAR, without the two large SAE checkpoints",
    )
    args = parser.parse_args()
    selected = ASSETS[:1] if args.data_only else ASSETS
    for asset in selected:
        download(asset, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
