# Retrieval data

The SCAR dataset is intentionally not committed on `main` because its upstream
repository does not contain an explicit redistribution license at the pinned
revision. Download the exact research artifact and verify its hash with:

```bash
uv run python scripts/prepare_retrieval_assets.py
```

This creates `data/scar_system_analogy_en.jsonl` byte-for-byte from:

- repository: <https://github.com/siyuyuan/scar>
- commit: `3dfc897cf6cc685531edc80ab64f35660403fc6c`
- upstream path: `release/system_analogy_en.json`
- SHA-256: `12883db11de17454b3a4ae30a109f4b64861125b1e94846e17b8edc3f8a12369`

See [`CITATIONS.md`](../CITATIONS.md) and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
