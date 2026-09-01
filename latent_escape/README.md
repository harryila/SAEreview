# Latent Escape: Causal Control of Analogy Target-Domain Selection with Sparse Autoencoder Features

**Status: the real-stack two-prompt GPU smoke test passed; the development
experiment has not been run and no empirical result is claimed.**

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
  overrepresented-domain/SAE-feature pair. The feature must predict that domain
  before its first domain-revealing output token and pass the stability checks in
  [`protocol.json`](protocol.json).
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

The complete machine-readable specification is [`protocol.json`](protocol.json).
Choices learned on development—feature ID, matched controls, strength, exact
judge, and prompt hash—must be written to `test_frozen.json` before any test
generation. [`test_config.template.json`](test_config.template.json) lists the
required fields.

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

## Run the development pipeline

An offline two-prompt contract check downloads no model weights:

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
manual-audit queue. Fill `manual_domain_label` in the emitted
`development_baseline.audit.jsonl`, then rerun the labeler with
`--manual-overrides` before invoking `discover_feature.py`. Discovery blocks
unless the audit gate, prompt-level cross-validation, 1,000-permutation global
maximum-statistic correction, prompt bootstrap, matched controls, and the 4/5
semantic review all pass. The discovery artifact supplies the frozen activation
threshold and 24 hash-selected gate prompt IDs; use
`semantic_review.template.json` for the required review record and pass the
captured `development_baseline_prompt.pre_domain.jsonl` through
`--pre-domain-activations` for the quantitative timing gate.

The remaining CLIs are intentionally separate at the manual checkpoints:

- `generate.py` runs baseline, dose, targeted, matched-random, activation-noise,
  diversity-prompt, temperature, and promotion conditions with paired seeds.
- `evaluate.py prepare-quality` exports a condition-blinded 1–5 quality queue
  containing only baseline and full-strength targeted outputs; partial doses and
  random/other controls are excluded automatically.
- `evaluate.py distance` computes the pinned secondary semantic-distance metric.
- `evaluate.py run` reports clustered bootstrap results for the full population
  first and the development-frozen eligible-prompt population second.

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
its hash. An ignored append-only arm ledger prevents rerunning a confirmatory
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
