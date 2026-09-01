.PHONY: reproduce-retrieval check latent-prompts latent-protocol latent-test latent-dry-run latent-dev-baseline

reproduce-retrieval:
	./scripts/reproduce_retrieval.sh

check:
	uv run python -m py_compile *.py scripts/*.py latent_escape/*.py
	uv run python latent_escape/validate_protocol.py
	uv run python -m pytest -q latent_escape/tests

latent-protocol:
	uv run python latent_escape/validate_protocol.py --show-summary

latent-prompts:
	uv run python scripts/prepare_retrieval_assets.py --data-only
	uv run python latent_escape/prepare_prompts.py
	uv run python latent_escape/validate_protocol.py --require-manifest --show-summary

latent-test:
	uv run python -m pytest -q latent_escape/tests

latent-dry-run: latent-protocol
	uv run python -m latent_escape.generate --dry-run --limit-prompts 2 --overwrite
	uv run python -m latent_escape.capture_activations --dry-run --limit-prompts 2 --overwrite --generations latent_escape/outputs/generations/development_baseline.dry-run.jsonl
	uv run python -m latent_escape.label_domains --generations latent_escape/outputs/generations/development_baseline.dry-run.jsonl --output latent_escape/outputs/labels/development_baseline.dry-run.jsonl --backend heuristic
	uv run python -m latent_escape.evaluate prepare-quality --generations latent_escape/outputs/generations/development_baseline.dry-run.jsonl --output latent_escape/outputs/quality/development_baseline.dry-run.jsonl --split development

latent-dev-baseline: latent-protocol
	uv run python -m latent_escape.generate --split development --condition baseline
	uv run python -m latent_escape.capture_activations --split development --condition baseline
	uv run python -m latent_escape.label_domains --generations latent_escape/outputs/generations/development_baseline.jsonl --output latent_escape/outputs/labels/development_baseline.jsonl --backend hf-zero-shot
