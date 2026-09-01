.PHONY: reproduce-retrieval check latent-prompts latent-protocol

reproduce-retrieval:
	./scripts/reproduce_retrieval.sh

check:
	uv run python -m py_compile *.py scripts/*.py latent_escape/*.py
	uv run python latent_escape/validate_protocol.py

latent-protocol:
	uv run python latent_escape/validate_protocol.py --show-summary

latent-prompts:
	uv run python scripts/prepare_retrieval_assets.py --data-only
	uv run python latent_escape/prepare_prompts.py
	uv run python latent_escape/validate_protocol.py --require-manifest --show-summary
