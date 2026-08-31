# Sparse Bridges, Dense Recall — quick test

This workspace runs the pre-specified follow-up from the review note on the 400
English SCAR system-analogy pairs. The evaluation uses the 283 cross-domain pairs
in both directions (566 queries) against 400 candidates per direction. Repeated
candidates with the same normalized target-system name count as relevant.

The corrected test uses raw cached 1,536-D `text-embedding-3-small` vectors and the
exact released scientific Top-K SAE checkpoints:

- `csLG_64_9216.pth` (`k=64`, 9,216 latents)
- `astroPH_64_9216.pth` (`k=64`, 9,216 latents)

Checkpoint hashes are verified before inference. Dense cosine uses L2-normalized
copies; the SAE receives the raw API vectors, matching the released application.

## Results

The complementarity gate **passes**:

- dense top-10: 146/566 (25.8%)
- SAE-only rescues: 54 directional queries
- dense ∪ either-SAE oracle: 200/566 (35.3%), a +9.54-point ceiling
- dense–SAE gold-rank Spearman correlation: 0.710 (cs.LG), 0.662 (astro.PH)

The held-out hybrid does **not** convert that oracle headroom into an SAE-specific
gain. With dense top-100 candidates and five grouped folds, the full IDF-weighted
two-SAE reranker is 146/566—exactly tied with dense—with 30 rescues and 30 losses.
Its canonical-pair-clustered 95% interval is −2.99 to +2.84 points. The seeded
random sparse-projection control is slightly higher at 150/566 (+0.71 points), so
this quick test does not support a `Sparse Bridges` claim.

## Run

```bash
.venv/bin/python complementarity_gate.py
.venv/bin/python hybrid_reranker.py
```

No API call or key is needed while the compatible embedding cache exists. The key
is never written to the workspace.

Generated embedding caches, virtual environments, and the 113 MB public checkpoint
binaries are intentionally excluded from Git; their pinned source is listed below.

Reader-facing, fully executed analysis: `sparse_bridges_quick_test.ipynb`.

Machine-readable outputs:

- `results/complementarity_gate.json`
- `results/complementarity_queries.jsonl`
- `results/hybrid_reranker.json`
- `results/hybrid_queries.jsonl`

The older `sae_retrieval_smoke_test.ipynb` and `results/results.json` are retained
only as the initial k=128/n=3,072 smoke test; they are not the corrected decision
artifacts.

Sources:

- SCAR: <https://github.com/siyuyuan/scar>
- pinned checkpoints: <https://huggingface.co/datasets/charlieoneill/saerchModels/tree/b2cbb184b58880b77a546511e11d8fd214c40556>
- reference SAE inference: <https://github.com/Christine8888/saerch/blob/56137345004821f189011d3f7521dc574815696d/saerch/topk_sae.py>

Scope: SCAR contains system descriptions, not scientific-paper abstracts. This is
a fast proxy and a stopping test, not a tuned final reranker.
