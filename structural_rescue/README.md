# Feature-Grounded Structural Rescue

This is a new, explicitly exploratory development study. It does not modify the
frozen retrieval, Latent Escape, or Latent Choice protocols.

## Status

Candidate preparation is complete and reproducible. The preflight reconstructs
all 566 SCAR query-directions from the pinned embedding cache and checkpoints:

| Candidate pool | Gold present |
| --- | ---: |
| Dense top-10 | 146/566 |
| Dense top-30 | 262/566 |
| Dense top-10 + two SAE top-10s | 200/566 |
| Dense top-10 + two random-sparse top-10s | 193/566 |

The SAE union contains the 54 previously identified dense misses, including 19
whose gold candidate is below dense rank 30. Its realized pool has 11–29 unique
candidates (mean 19.11). The two-projection random control has 13–30 (mean
22.10). These are candidate-oracle facts, not verifier results.

The offline two-query fixture run passes end to end. A real external-API smoke
test passed mechanism extraction (73/73) but exposed a deterministic missing-ID
failure in one eight-feature description batch after 208/256 descriptions. Before
any verifier scores existed, protocol revision 3 froze uniform four-feature
description batches; that compatibility rerun is in progress. No structural-
verifier result exists yet.

## Fixed comparisons

The implementation avoids two confounds: a verifier over exactly ten dense
candidates would be vacuous at top-10 success, and changing both the pool and feature
evidence would not isolate why a result changed.

| Arm | Purpose |
| --- | --- |
| `dense_ranking` | Existing dense top-10 baseline |
| `dense30_structure` | Structural verifier on dense top-30 |
| `sae_union_structure` | Candidate-source value, using the identical structure-only verifier |
| `random_union_structure` | SAE-specificity control with two fixed 9,216-dimensional top-64 random projections |
| `sae_union_feature_grounded` | Effect of the full feature-evidence bundle—opaque aliases, activation percentiles, and frozen descriptions—on the identical SAE pool |

The primary candidate-source comparison is `sae_union_structure` versus
`dense30_structure`. Feature-evidence-bundle value is
`sae_union_feature_grounded` versus `sae_union_structure`; it is not folded into
the candidate-source claim.

## Pipeline

1. `prepare` reconstructs the four candidate pools, qrels sidecar, shared-feature
   evidence, and 256-feature description catalog.
2. `extract` independently converts each system into entities/roles, causal
   relations, dynamics, constraints, and boundary conditions.
3. `describe` interprets each internal namespaced SAE feature once from its
   top-activating development examples. The model sees only opaque,
   representation-local aliases.
4. `verify` scores the outcome-stratified 108-query screen using identical
   deterministic full-superpool batches with and without feature evidence. Only
   coherent descriptions are eligible, and an entire feature is omitted when
   either pair member's normalized content appears anywhere in that feature's
   exact batched description-request context.
   It never receives
   retrieval source, rank, qrels, mappings, explanations, or rescue status.
5. `evaluate` opens the separate qrels sidecar only after predictions exist and
   reports descriptive screen top-10 success, known-rescue conversion, dense-hit
   retention, and separate canonical-pair clustered intervals within the rescue
   and retention strata. These are not population Recall estimates.

The real backend uses the OpenAI Responses API with strict JSON-schema outputs,
`store=false`, reasoning effort `none`, temperature 0, and the snapshot
`gpt-5.4-mini-2026-03-17`. It reads
`OPENAI_API_KEY` from the environment and never writes it. Responses are cached
by model, prompt, schema, and input hashes so interrupted runs can resume.
Every downstream command first verifies all six prepared artifacts and the SCAR
source against the committed source-text-free preparation report.
The `extract` and `describe` stages transmit SCAR system names/backgrounds to
that API; `verify` transmits the derived graphs and eligible feature descriptions.
Run them only when that external processing is authorized.

## Run

Install the locked environment and prepare the ignored third-party assets as in
the root README, then:

```bash
make structural-rescue-prepare
make structural-rescue-dry-run
```

The dry run uses deterministic fixture judgments and is explicitly
non-evidentiary. To run the real development pipeline:

```bash
export OPENAI_API_KEY=...
uv run python -m structural_rescue.run extract
uv run python -m structural_rescue.run describe
uv run python -m structural_rescue.run verify
uv run python -m structural_rescue.run evaluate
```

For the first live smoke test, use a separate ignored output directory and prepare
it before limiting `extract` and `verify` to two screen queries:

```bash
uv run python -m structural_rescue.run prepare --output-dir structural_rescue/outputs/smoke --overwrite
uv run python -m structural_rescue.run extract --output-dir structural_rescue/outputs/smoke --limit-queries 2
uv run python -m structural_rescue.run describe --output-dir structural_rescue/outputs/smoke
uv run python -m structural_rescue.run verify --output-dir structural_rescue/outputs/smoke --limit-queries 2
uv run python -m structural_rescue.run evaluate --output-dir structural_rescue/outputs/smoke
```

All generated SCAR-derived text, qrels, feature evidence, and API responses stay
under ignored `structural_rescue/outputs/`.

## Claim boundary

SCAR is fully inspected development data, and the 108 rows are selected using
known outcomes. Even a positive screen would only justify freezing a method for
an outcome-independent external benchmark. Dense-versus-random arm ordering here
cannot establish SAE specificity. The feature-grounded contrast only measures
the effect of adding this particular evidence bundle. It would not test
an SAE intervention, establish serendipity, or demonstrate scientific
discovery. Confirmation requires freezing the complete method and evaluating a
genuinely untouched external analogy benchmark. The unused 120 Latent Choice
prompts remain excluded.

See [`protocol.json`](protocol.json) and the source-text-free
[`prepare_report.json`](prepare_report.json).
