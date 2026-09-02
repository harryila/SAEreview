# Latent Escape: Causal Control of Analogy Target-Domain Selection with Sparse Autoencoder Features

**Status: the 640-output development baseline, prompt-activation capture, and
Amendment 4 blinded label audit are complete. The frozen audit gate failed:
exact agreement was 18/64 (28.125%), including 0/38 for the pinned classifier's
`other` class, in the accepted neutral-path rating attempt. The protocol therefore
stops before feature discovery. No feature has been selected, no causal
intervention or confirmatory test has been run, and no empirical causal result is
claimed. Protocol Amendment 4 was adopted after reviewing only the aggregate
baseline report and before the audit or feature search.**

Research question: do sparse, interpretable domain-attractor features causally
contribute to Gemma 2 repeatedly choosing familiar analogy domains, and can
suppressing one such feature change the output-domain distribution without
materially reducing structural quality?

The retrieval pilot in the repository is closed. Its negative held-out result is
not reused as evidence for this hypothesis.

## Frozen MVP

- Model: `google/gemma-2-9b-it` at revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`, BF16, no quantization.
  The confirmatory runtime is one CUDA device (`cuda:0`) with no offload.
- SAE: `google/gemma-scope-9b-it-res` at revision
  `e86af97a5b6fbbccca28ab654f2fda1b0768f770`, layer 20, width 16k,
  `average_l0_91`.
- Stimuli: 200 source systems, deterministically balanced across SCAR domains;
  80 development and 120 untouched test, with at most one side of any SCAR pair.
  Known SCAR target systems and mappings are never included in the generation
  manifest or evaluation.
- Feature discovery: use development generations to freeze exactly one primary
  overrepresented-domain/SAE-feature pair. `other` remains a reported taxonomy
  class but is ineligible as the selected primary domain; every other domain
  still must account for at least 10% of development outputs. The feature must
  predict that domain before its first domain-revealing output token and pass the
  stability checks in [`protocol.json`](protocol.json) and
  [`protocol_amendment_4.json`](protocol_amendment_4.json).
- Confirmatory controls: baseline, targeted suppression, five matched-random SAE
  features, L2-matched activation noise, a diversity instruction, and a
  higher-temperature baseline. Feature promotion is secondary.
- Primary causal endpoint: prompt-clustered change in the selected target-domain
  rate under targeted suppression, versus baseline and matched-random suppression.
- Guardrail: blinded structural-quality non-inferiority with a 0.25-point margin
  on a five-point rubric, rated only for baseline versus full-strength targeted
  suppression. Random controls still receive independent domain labels.
- Secondary distance metric: normalized cosine distance from source text to the
  generated target system/roles using `sentence-transformers/all-MiniLM-L6-v2`
  at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

“Suppression” is operationally a decoder-direction ablation: the hook subtracts
the selected activation times that feature's decoder vector from the untouched
residual stream. It does not assume that re-encoding the edited residual makes
that latent exactly zero or leaves every other encoder coordinate unchanged.

The immutable revision-3 base specification is [`protocol.json`](protocol.json),
whose SHA-256 remains
`a9bdeb15de798bc56f888715fbe7bef47b69f1dd06f06dc7322013567ed9297a`.
The effective development protocol is that base plus the disclosed
[`protocol_amendment_4.json`](protocol_amendment_4.json). Choices learned on
development—feature ID, matched controls, strength, exact judge, and prompt
hash—must be written to `test_frozen.json` before any test generation.
[`test_config.template.json`](test_config.template.json) binds the base protocol,
amendment, and labeling-guide hashes in addition to the required learned fields.

## Real GPU smoke test

The isolated two-prompt development smoke passed on an NVIDIA A100-SXM4-40GB at
commit `c28f65a`. It loaded the exact pinned Gemma and Gemma Scope revisions,
hash-checked the SAE, reproduced identical logits with a zero-strength hook,
captured real prompt activations, applied a nonzero full-strength decoder-direction
ablation with paired seeds, matched hook calls to generated-token counts, produced
schema-valid JSON in both arms, and independently labeled the targeted outputs
with the pinned BART classifier. The untouched test split was not accessed.

See [`gpu_smoke_report.json`](gpu_smoke_report.json) for artifact hashes and exact
runtime details. Feature `8161` was chosen only because it had the largest positive
activation in these two prompts; it is a plumbing check and is not the development
feature-discovery result or evidence for the hypothesis.

## Development baseline

The frozen 80-prompt development split was run at eight samples per prompt on
the real stack, producing 640 baseline generations, 80 prompt-level SAE vectors
of width 16,384, 640 pinned BART domain labels, and a blinded 64-item manual-audit
queue. The untouched test split was not accessed. Strict JSON validation passed
for 584/640 generations. Of the 56 strict-invalid outputs, 55 reached the
384-token limit (17 after a complete schema-valid object and 38 while incomplete),
and one ended early with malformed quoting. All rows are preserved; none were
selectively regenerated, salvaged, or dropped.

A capture-only replay at commit `8ab8781` corrected token alignment after the
causal domain boundary and resolved 640/640 pre-domain boundaries, above the
frozen 90% minimum. Prompt activation values and decoder norms remained
bitwise-identical; generations, independent labels, and the audit queue remained
byte-identical.

See [`development_baseline_report.json`](development_baseline_report.json) for
the frozen commit, artifact hashes, aggregate label counts, and audit status.
That report remains frozen at its pre-audit state. The completed audit and stop
result are recorded separately below; these labels are not eligible for feature
discovery.

Amendment 4 leaves [`protocol.json`](protocol.json) byte-for-byte unchanged and
records its predecessor hash explicitly. At adoption time, the full unaudited
BART count vector was known, including `other` at 383/640, along with the
584/640 strict-schema count and 640/640 boundary resolution. The audit was 0/64
complete; no prompt-level label/activation joins, feature correlations,
candidate rankings, selection statistics, interventions, or test outputs had
been examined. Because `other` is a heterogeneous residual rather than a single
interpretable intervention target, it is excluded only from primary domain
selection. It remains in labeling coverage, audit summaries, domain rates,
entropy, and distinct-domain counts. The pre-existing 10% selection threshold is
unchanged. This is a post-baseline, outcome-informed development amendment, not
a claim that the exclusion was preregistered.

This is a **transparent post-baseline, pre-analysis amendment**. Here,
“pre-analysis” means before the manual audit, prompt-level label/activation
associations, or feature analysis; the aggregate baseline report and counts had
already been reviewed. It is not a fully preregistered study.

## Amendment 4 audit result

A first AI-rating attempt was excluded after a final provenance review found that
its absolute input path contained `development_baseline`, disclosing split and
condition despite row-level field blinding. Its outcome was known before
remediation but was not used for the authoritative gate. A fresh rater was
committed in advance as authoritative regardless of its result and received no
context or result from the excluded attempt.

The accepted rater was an isolated-context AI rater—not a human rater. It was
instructed to use only a neutral `items.jsonl` containing blind ID, analogy text,
text hash, and empty label plus the frozen
[`domain_labeling_guide.md`](domain_labeling_guide.md), and reported using only
those artifacts. The packet contained no split, condition, seed, feature,
intervention, classifier-label, or protocol fields. This was procedural
blinding: shared-workspace filesystem access was not technically restricted. All
64 labels were completed in the frozen taxonomy, then mechanically reattached to
the original provenance rows without changing any non-label field.

The repository's strict evaluator rejected the audit: overall exact agreement
was 18/64 (28.125%), below the frozen 80% threshold. Among classifier classes
with at least five audited examples, `biology/ecology` passed at 6/7,
`computer science/software` passed at 6/6, and `other` failed at 0/38. The
accepted attempt had no incomplete items or evaluator-detected size/seed errors;
an independent recomputation matched the frozen selection. The classifier/rater
labels did not satisfy the frozen validation criterion, but that disagreement
alone does not identify which measurement component is substantively responsible.
These labels are ineligible for feature discovery. No downstream bypass was used,
no label/activation association was computed, and the untouched test split was
not accessed.

See [`domain_audit_report.json`](domain_audit_report.json) for thresholds,
class-level results, rater disclosure, adjudication details, and SHA-256 bindings
for the ignored local artifacts used in this audit and adjudication. It also
preserves the excluded first attempt and an aborted malformed-packet execution
without using either result.

## Prepare and verify prompts

```bash
make latent-prompts
```

This downloads only the pinned SCAR file, builds ignored local prompt artifacts,
and verifies their split, provenance, and hash. To validate the protocol without
downloading anything:

```bash
make latent-protocol
```

This also verifies Amendment 4, the exact taxonomy and aggregate knowledge
snapshot, the domain-labeling guide hash, the quality-sampling arithmetic, and
the amendment hashes prefilled in `test_config.template.json`.

## Run the development pipeline

A model-free two-prompt contract check downloads no model weights (and prepares
the small pinned prompt source automatically on a fresh clone):

```bash
make latent-dry-run
```

On a single CUDA GPU, after setting `HF_TOKEN` for an account with Gemma access,
run only the 80-prompt development baseline:

```bash
make latent-dev-baseline
```

That command generates eight paired-seed outputs per prompt, captures exactly
one pre-generation SAE vector per prompt (plus nested pre-domain diagnostics),
labels targets with the pinned independent classifier, and writes a blinded 10%
manual-audit queue. The already-completed revision-3 baseline can be bound to
Amendment 4 without rerunning BART:

```bash
python -m latent_escape.label_domains \
  --generations latent_escape/outputs/development-8ab8781/generations/development_baseline.jsonl \
  --source-classifier-labels latent_escape/outputs/development-8ab8781/labels/development_baseline.jsonl \
  --output latent_escape/outputs/development-8ab8781/labels/development_baseline.amendment4.jsonl \
  --audit-output latent_escape/outputs/development-8ab8781/labels/development_baseline.amendment4.audit.jsonl
```

Do not give that repository path to the rater: it reveals `development_baseline`.
First export a neutrally named `items.jsonl` containing only `blind_id`,
`analogy_text`, `analogy_text_sha256`, and an empty `manual_domain_label`, and
copy [`domain_labeling_guide.md`](domain_labeling_guide.md) to a neutral
`guide.md` beside it outside the repository. Give only those two neutral paths to
the independent rater. After all 64 labels are filled, verify the blind IDs,
texts, hashes, order, taxonomy, and non-label fields against the frozen queue;
then mechanically reattach only `manual_domain_label` to the original
Amendment-4 queue. Archive the neutral packet and rater record only after rating.
The resulting full-provenance artifact may then be used to create a separate
adjudicated artifact—never overwrite the frozen BART source:

```bash
python -m latent_escape.label_domains \
  --generations latent_escape/outputs/development-8ab8781/generations/development_baseline.jsonl \
  --source-classifier-labels latent_escape/outputs/development-8ab8781/labels/development_baseline.amendment4.jsonl \
  --manual-overrides latent_escape/outputs/development-8ab8781/labels/development_baseline.amendment4.audit.neutral.completed.jsonl \
  --output latent_escape/outputs/development-8ab8781/labels/development_baseline.neutral.adjudicated.jsonl \
  --audit-output latent_escape/outputs/development-8ab8781/labels/development_baseline.neutral.adjudicated.audit.jsonl
```

The import rejects guide/hash drift, exposed classifier fields, changed blinded
text, incomplete queues, and any non-frozen audit membership. Discovery blocks
unless the audit gate, prompt-level cross-validation, 1,000-permutation global
maximum-statistic correction, prompt bootstrap, matched controls, and the 4/5
semantic review all pass. The discovery artifact supplies the frozen activation
threshold and 24 hash-selected gate prompt IDs; use
`semantic_review.template.json` for the required review record and pass the
captured `development_baseline_prompt.pre_domain.jsonl` through
`--pre-domain-activations` for the quantitative timing gate.

After this gate failure, no feature-discovery, intervention, or test step below
was run.

The remaining CLIs are intentionally separate at the manual checkpoints:

- `generate.py` runs baseline, dose, targeted, matched-random, activation-noise,
  diversity-prompt, temperature, and promotion conditions with paired seeds.
- `evaluate.py prepare-quality` exports a condition-blinded 1–5 quality queue
  containing only baseline and full-strength targeted outputs; partial doses and
  random/other controls are excluded automatically.
- `evaluate.py distance` computes the pinned secondary semantic-distance metric.
- `evaluate.py run` reports clustered bootstrap results for the full population
  first and the development-frozen eligible-prompt population second.

Structural quality uses one hash-selected paired sample index per prompt, shared
between baseline and full-strength targeted suppression. The pair-selection seed
is `latent-escape-quality-pair-v1`. Only the primary rater score enters the
quality endpoint. Prompts selected by
`latent-escape-quality-reliability-v1` receive an additional independent rating
for both arms solely to estimate reliability: `ceil(10% * prompts)` gives three
development-gate prompts and 12 test prompts. The resulting workloads are 48
unique plus six duplicate ratings (54 tasks) for the 24-prompt development gate,
and 240 unique plus 24 duplicate ratings (264 tasks) for the 120-prompt test.

For the intervention gate, pass the completed discovery artifact to every
generation as `--development-plan`; this automatically selects the exact 24
prompts, four samples, feature, and five controls. Generate suppression doses
`0.25`, `0.5`, and `1.0`, then evaluate with the same
`--development-plan`. The evaluator refuses any prompt/sample mismatch and emits
`development_intervention_gate.status=pass|stop`.

The selected-domain contrasts can support causal influence on target-domain
selection. Entropy and distinct-domain gains with non-inferior quality can
additionally support reduced domain homogeneity. This study does not evaluate
serendipity, usefulness, scientific discovery, or overall analogy improvement.

Copy `test_config.template.json` to ignored `test_frozen.json` only after the
development gate passes. Test commands require that completed file, the exact
clean code commit and lockfile hashes, every test prompt, and an explicit
`--confirm-test`. The frozen config must point to the passing gate report and
its hash, and it must retain the prefilled base-protocol, amendment, and labeling
guide hashes. An ignored append-only arm ledger prevents rerunning a confirmatory
arm under a second output path, and test outputs cannot be overwritten through
the CLI.

SCAR is used only as a quick source-mechanism stimulus bank. It is not a fresh
benchmark and may have appeared in model pretraining; a positive MVP should later
be replicated on newly written source mechanisms.

## Access needed for the actual run

Prompt preparation needs no credential. Generation requires a Hugging Face read
token for an account that has accepted the gated Gemma 2 terms, plus suitable
compute. The 9B BF16 model is about 18.5 GB before runtime overhead; 24 GB GPU
memory is borderline and 40 GB is comfortable. The confirmatory run must not be
quantized because quantization would add an intervention confound.

No model or SAE weights are committed here. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
