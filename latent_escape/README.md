# Latent Escape

**Status: protocol prepared; no generation result has been run or claimed.**

Research question: do sparse, interpretable domain-attractor features causally
contribute to Gemma 2 repeatedly choosing familiar analogy domains, and can
suppressing one such feature change the output-domain distribution without
materially reducing structural quality?

The retrieval pilot in the repository is closed. Its negative held-out result is
not reused as evidence for this hypothesis.

## Frozen MVP

- Model: `google/gemma-2-9b-it` at revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`, BF16, no quantization.
- SAE: `google/gemma-scope-9b-it-res` at revision
  `e86af97a5b6fbbccca28ab654f2fda1b0768f770`, layer 20, width 16k,
  `average_l0_91`.
- Stimuli: 160 source systems, deterministically balanced across SCAR domains;
  80 development and 80 untouched test, with at most one side of any SCAR pair.
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
  on a five-point rubric.

The complete machine-readable specification is [`protocol.json`](protocol.json).
Choices learned on development—feature ID, matched controls, strength, exact
judge, and prompt hash—must be written to `test_frozen.json` before any test
generation. [`test_config.template.json`](test_config.template.json) lists the
required fields.

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
