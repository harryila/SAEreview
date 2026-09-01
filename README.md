# SAEreview

This repository contains two deliberately separated studies of sparse
autoencoders (SAEs) and cross-domain analogy:

1. **Retrieval pilot — frozen.** The completed SCAR study is preserved at
   [`v0.1-retrieval-study`](https://github.com/harryila/SAEreview/releases/tag/v0.1-retrieval-study).
2. **Latent Escape — current direction.** A pre-specified causal-generation
   experiment now lives in [`latent_escape/`](latent_escape/README.md).

The retrieval result motivates the pivot but is **not** evidence for the new
generation hypothesis.

## Frozen retrieval result

The corrected study evaluates 283 cross-domain SCAR pairs in both directions
(566 queries, 400 candidates per direction) using raw cached 1,536-dimensional
`text-embedding-3-small` vectors and the exact released cs.LG and astro.PH
Top-K SAE checkpoints (`k=64`, 9,216 latents).

- Dense Recall@10: **146/566 (25.8%)**.
- Dense ∪ either-SAE oracle: **200/566 (35.3%)**, a +9.54-point ceiling.
- Held-out IDF-weighted hybrid: **146/566**, exactly tied with dense.
- The hybrid made 30 rescues and 30 losses; clustered 95% CI: **−2.99 to
  +2.84 points**.
- A seeded random sparse control reached **150/566**, slightly above the SAE
  hybrid.

Conclusion: SAE neighborhoods contain complementary candidates, but this study
does not show that the complementarity is identifiable well enough to improve
retrieval. Retrieval optimization on this SCAR split is stopped.

The low-lexical-overlap subgroup is explicitly **post-hoc and exploratory**. Its
+2.43-point hybrid difference has no standalone confidence interval and is not a
confirmatory claim.

## Reproduce the retrieval study

Install [`uv`](https://docs.astral.sh/uv/), then run:

```bash
make reproduce-retrieval
```

The command uses the committed lockfile, downloads and hash-checks the exact SCAR
artifact and SAE checkpoints, verifies or creates the embedding cache, and runs
the primary notebook into an ignored output file. With the canonical local cache,
this reproduces the frozen run. If that cache is absent, set `OPENAI_API_KEY`; the
command creates a replication cache using the live hosted model alias, whose
outputs may drift. The key is read only from the environment and is never written
to the repository.

The exact historical environment is Python 3.10.18. The two model downloads are
about 226 MB total. SCAR is fetched at runtime because the pinned upstream repo
does not provide an explicit redistribution license.

Primary artifacts:

- [`sparse_bridges_quick_test.ipynb`](sparse_bridges_quick_test.ipynb)
- [`results/complementarity_gate.json`](results/complementarity_gate.json)
- [`results/complementarity_queries.jsonl`](results/complementarity_queries.jsonl)
- [`results/hybrid_reranker.json`](results/hybrid_reranker.json)
- [`results/hybrid_queries.jsonl`](results/hybrid_queries.jsonl)

The older `sae_retrieval_smoke_test.ipynb` and `results/results.json` are retained
only as the initial k=128/n=3,072 smoke test; they are not decision artifacts.

## Latent Escape

The next question is causal: can an interpretable SAE feature associated with an
overused analogy domain be suppressed to change the model's target-domain
distribution without materially reducing structural quality?

The in-repo MVP fixes Gemma 2 9B IT, the layer-20 16k Gemma Scope residual SAE,
160 source prompts split 80 development / 80 untouched test, matched random
features, activation noise, diversity prompting, and temperature controls. See
the [protocol and current run status](latent_escape/README.md).

## License and attribution

Original code and documentation are MIT licensed. Third-party data and weights
retain their own terms and are not bundled on `main`; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Citation metadata and upstream
references are in [`CITATION.cff`](CITATION.cff) and
[`CITATIONS.md`](CITATIONS.md).
