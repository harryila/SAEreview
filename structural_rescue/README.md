# Feature-Grounded Structural Rescue

This is a separate, explicitly exploratory development study. It does not modify
the frozen retrieval, Latent Escape, or Latent Choice protocols.

## Status

The frozen 108-query development screen is complete and returned
`no_go_stop_structural_rescue_on_scar`. Evidence coverage was adequate, but
neither candidate-source specificity nor aligned-description utility passed.
Structural Rescue is therefore stopped on SCAR without retuning.

The preflight reconstructs all 566 SCAR query-directions from the pinned
embedding cache and checkpoints:

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
The frozen SAE-specific candidate-source gate therefore failed.
This is a development finding, not a population or confirmatory claim, and the
seeds and gate will not be retuned. Random source unions are also larger on
average (22.20 candidates versus 19.11 for SAE), so this rejects the frozen
total-hit gate—not a size-matched efficiency claim or the broader SAE hypothesis.

Every verifier pool was padded to exactly 30 candidates with the next unused
dense results. Unpadded unions are retained only for source-oracle reporting and
SAE feature selection.

The one-time screen completed 24,708 judgments across 108 outcome-stratified
queries. These are development-screen diagnostics, not population Recall
estimates:

| Arm | Rescues / 54 | Dense losses / 54 | Net utility |
| --- | ---: | ---: | ---: |
| Dense-30 + structure | 13 | 11 | 2 |
| SAE-padded + structure | 16 | 11 | 5 |
| Random-padded + structure 1 | 11 | 9 | 2 |
| Random-padded + structure 2 | 11 | 11 | 0 |
| Random-padded + structure 3 | 12 | 10 | 2 |
| SAE-padded + activations only | 19 | 21 | -2 |
| SAE-padded + aligned descriptions | 15 | 13 | 2 |
| SAE-padded + shuffled descriptions | 24 | 18 | 6 |

The frozen absolute gate required at least 33 rescues, at most four dense
losses, and net utility of at least 29. SAE-padded structure cleared none of
those thresholds. Aligned descriptions also trailed structure-only by three net
successes and shuffled descriptions by four; they cleared none of the frozen
feature-grounding comparisons. Usable evidence covered 2,819/3,240 SAE-pool
pairs (87.0%), all 108 queries, and all 54 queries in each stratum. The no-go is
therefore performance-based, not an operational or coverage failure. The exact
aggregate output is in
[`evaluation_report.json`](outputs/development/evaluation_report.json).

The revision-4 fixture and real two-query external-API smoke passed end to end
(73 mechanisms, 256 descriptions, and 166 judgments). That historical smoke
exposed two model-compliance edge cases, both documented before any 108-query
evaluation. Its two-query readout remains strictly non-evidentiary; see
[`smoke_report.json`](smoke_report.json).

Revision 6 records one further pre-coverage API-compliance repair. After 200/256
real descriptions completed, a response repeatedly marked a feature incoherent
but did not use the exact fallback wording. Incoherent features were already
ineligible for every evidence arm. The loader now preserves that raw text,
records the normalization, and replaces only the unusable description with the
frozen fallback. No real coverage audit, verifier score, or qrels-based
evaluation preceded the change.

Revision 7 changes only request scheduling after the qrels-free gates passed and
892/24,708 judgments had been cached sequentially. The four already-frozen
evidence-mode requests for each candidate batch now run concurrently and are
validated and written in the original deterministic order. Request bytes,
hashes, prompts, schemas, retries, scoring, endpoints, and thresholds are
unchanged. No rubric values, rankings, qrels, or evaluation outcome were
inspected before this throughput-only amendment.

Revision 8 fixes a concurrency edge case found before the parallel scheduler
was used on the real screen. Four zero-evidence batches have identical request
hashes across all evidence modes; those requests are now coalesced by exact hash
and their validated response is copied to the matching modes before the usual
deterministic write. This avoids duplicate charges and cache-file races without
changing any request or scientific choice. The normalization diagnostic is also
computed from the final rows, making resumed and fresh reports identical. No
additional verifier values, rankings, qrels, or evaluation outcome were
inspected before the repair.

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

All SCAR-derived text, qrels, feature evidence, per-query rows, predictions, and
API responses stay ignored under `structural_rescue/outputs/`. Only the
source-text-free aggregate evaluation report is tracked.

The frozen development gate requires at least 33/54 known rescues, at most four
losses among the 54 dense-retention controls, and net utility of at least 29.
Required pairwise margins are five net successes with at least 0.90 positive
bootstrap fraction. The aligned-description arm must also retain usable evidence
on at least 41 queries per stratum, recover no fewer rescues than each feature
control, and add at most one dense loss. Exact full-, narrow-, and no-go policies
are in [`protocol.json`](protocol.json).

## Claim boundary

SCAR is fully inspected development data, and the 108 rows were selected using
known outcomes. The frozen result closes Structural Rescue on SCAR: neither the
candidate-source gate nor the feature-grounding gate passed, so the method will
not advance to an external benchmark or be retuned on this screen. This does not
test an SAE intervention, disprove the broader SAE hypothesis, establish
serendipity, or demonstrate scientific discovery. The unused 120 Latent Choice
prompts remain excluded. Twenty-four screen queries span two request batches,
whose fixed integer rubric scores are assumed comparable across fixed batches;
the description comparison is a batch-level verifier-method diagnostic, not a
test of local feature faithfulness.

See [`protocol.json`](protocol.json), the source-text-free
[`prepare_report.json`](prepare_report.json), and the source-text-free
[`evaluation_report.json`](outputs/development/evaluation_report.json).
