# Feature-Grounded Structural Rescue

This is a separate, explicitly exploratory development study. It does not modify
the frozen retrieval, Latent Escape, or Latent Choice protocols.

## Status

Revision 5 candidate preparation is complete and reproducible. The preflight
reconstructs all 566 SCAR query-directions from the pinned embedding cache and
checkpoints:

| Candidate pool | Gold present |
| --- | ---: |
| Dense top-10 | 146/566 |
| Dense top-30 | 262/566 |
| Unpadded dense + two-SAE source union | 200/566 |
| 64 unpadded random-sparse source unions | 179–216/566 (median 195) |

The SAE union contains the 54 previously identified dense misses, including 19
whose gold candidate is below dense rank 30. Across 64 prespecified random seed
pairs, 18 random source unions matched or exceeded the SAE oracle; the higher
95th percentile was 204 and the plus-one tail probability was 19/65 = 0.292.
The frozen SAE-specific candidate-source gate has therefore already failed.
This is a development finding, not a population or confirmatory claim, and the
seeds and gate will not be retuned. Random source unions are also larger on
average (22.20 candidates versus 19.11 for SAE), so this rejects the frozen
total-hit gate—not a size-matched efficiency claim or the broader SAE hypothesis.

Every verifier pool is now padded to exactly 30 candidates with the next unused
dense results. Unpadded unions are retained only for source-oracle reporting and
SAE feature selection. The still-open development question is narrower: whether
correctly aligned feature descriptions improve a fixed verifier over the same
SAE-padded pool with structure alone, activation percentiles alone, or
frequency-matched shuffled descriptions.

The revision-4 fixture and real two-query external-API smoke passed end to end
(73 mechanisms, 256 descriptions, and 166 judgments). That historical smoke
exposed two model-compliance edge cases, both documented before any 108-query
evaluation. Its two-query readout remains strictly non-evidentiary; see
[`smoke_report.json`](smoke_report.json).

## Fixed comparisons

The implementation fixes candidate budget at 30 and separates candidate-source
value from feature-description value.

| Arm | Purpose |
| --- | --- |
| `dense_ranking` | Existing dense top-10 baseline |
| `dense30_structure` | Structural verifier on dense top-30 |
| `sae_union_padded30_structure` | Padded SAE source union with the identical structure-only verifier |
| `random_union_padded30_structure_1..3` | Three prespecified padded random-projection controls |
| `sae_union_padded30_activation_only` | Same SAE pool and usable features, with aliases and percentiles but no descriptions |
| `sae_union_padded30_aligned_description` | Same rows with correctly aligned frozen descriptions |
| `sae_union_padded30_shuffled_description` | Same rows with within-representation, frequency-bin-matched donor descriptions |

Aligned, activation-only, and shuffled modes receive identical feature aliases,
ordering, activation percentiles, candidate positions, and batches. Only
description presence or text changes. Features are included only when both the
aligned and donor descriptions pass the same coherence and direct-example
exclusion rules.

## Pipeline

1. `prepare` reconstructs the source pools, exact-30 verifier pools, qrels
   sidecar, 64-pair random oracle distribution, shared-feature evidence,
   deterministic description shuffle, and 256-feature catalog.
2. `extract` independently converts each system into entities/roles, causal
   relations, dynamics, constraints, and boundary conditions.
3. `describe` interprets each internal namespaced SAE feature once from its
   top-activating development examples. The model sees only opaque,
   representation-local aliases.
4. `audit-coverage` applies the frozen feature-eligibility rules to the entire
   108-query screen without reading qrels and must pass before real verification.
5. A qrels-free capacity smoke processes the selected 85-candidate superpool,
   including one full 64-candidate request batch.
6. `verify` scores the 108-query screen in four exactly paired evidence modes.
   Only coherent descriptions are eligible, and a feature is omitted when either
   pair member appears anywhere in the aligned or shuffled donor's exact
   description-request context. It never receives retrieval source, rank, qrels,
   mappings, explanations, or rescue status.
7. `evaluate` opens the separate qrels sidecar only after predictions exist and
   reports descriptive screen top-10 success, known-rescue conversion, dense-hit
   retention, separate canonical-pair clustered intervals, and the frozen
   10,000-resample paired-bootstrap gate. These are not population Recall
   estimates.

The real backend uses the OpenAI Responses API with strict JSON-schema outputs,
`store=false`, reasoning effort `none`, temperature 0, and the snapshot
`gpt-5.4-mini-2026-03-17`. It reads `OPENAI_API_KEY` from the environment and
never writes it. Responses are cached by model, prompt, schema, and input hashes
so interrupted runs can resume.

Every downstream command first verifies all eight prepared artifacts and the
SCAR source against the committed source-text-free preparation report. The
`extract` and `describe` stages transmit SCAR system names/backgrounds to that
API; `verify` transmits the derived graphs and eligible feature descriptions.
Run them only when that external processing is authorized.

## Run

Install the locked environment and prepare the ignored third-party assets as in
the root README, then:

```bash
make structural-rescue-prepare
make structural-rescue-dry-run
```

The dry run uses deterministic fixture judgments and is explicitly
non-evidentiary. For the real development pipeline, generate descriptions and
perform the qrels-free checks first:

```bash
export OPENAI_API_KEY=...
uv run python -m structural_rescue.run describe
uv run python -m structural_rescue.run audit-coverage
uv run python -m structural_rescue.run extract --selection-path structural_rescue/outputs/development/capacity_smoke_selection.json
uv run python -m structural_rescue.run verify --selection-path structural_rescue/outputs/development/capacity_smoke_selection.json
uv run python -m structural_rescue.run extract
uv run python -m structural_rescue.run verify
uv run python -m structural_rescue.run evaluate
```

All generated SCAR-derived text, qrels, feature evidence, and API responses stay
under ignored `structural_rescue/outputs/`.

The frozen development gate requires at least 33/54 known rescues, at most four
losses among the 54 dense-retention controls, and net utility of at least 29.
Required pairwise margins are five net successes with at least 0.90 positive
bootstrap fraction. The aligned-description arm must also retain usable evidence
on at least 41 queries per stratum, recover no fewer rescues than each feature
control, and add at most one dense loss. Exact full-, narrow-, and no-go policies
are in [`protocol.json`](protocol.json).

## Claim boundary

SCAR is fully inspected development data, and the 108 rows are selected using
known outcomes. The random-oracle result already rejects SAE candidate-source
specificity under the frozen gate. A positive aligned-description screen could
only support the narrower decision to freeze that verifier method for an
outcome-independent external benchmark. It would not test an SAE intervention,
establish serendipity, or demonstrate scientific discovery. Confirmation
requires freezing the complete method and evaluating a genuinely untouched
external analogy benchmark. The unused 120 Latent Choice prompts remain
excluded. Twenty-four screen queries span two request batches, whose fixed
integer rubric scores are assumed comparable across fixed batches; any apparent
description benefit is therefore a batch-level verifier-method effect, not proof
that an individual feature description locally explains a candidate.

See [`protocol.json`](protocol.json) and the source-text-free
[`prepare_report.json`](prepare_report.json).
