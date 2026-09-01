#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

uv sync --frozen
uv run python scripts/prepare_retrieval_assets.py
uv run python scripts/prepare_openai_embeddings.py
mkdir -p .cache
uv run jupyter execute sparse_bridges_quick_test.ipynb \
  --output .cache/reproduced_sparse_bridges_quick_test.ipynb \
  --timeout=300
