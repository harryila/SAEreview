# Latent Choice v1

**Status: Latent Choice v1 remains frozen at its failed compliance gate. A
separate 24-prompt exploratory endpoint calibration fixed the token-boundary
problem but failed its pre-run mapping-invariance rubric. Coded choice is now
stopped rather than retuned. No feature association, study intervention, human
rating, or test output exists; the 120-prompt test split remains untouched.**

Latent Choice is a distinct successor to the stopped Latent Escape study. It
asks whether a sparse-autoencoder feature causally changes Gemma 2's explicit
choice among 18 named analogy target domains. It does not reuse the predecessor's
BART labels, AI-rater labels, feature candidates, or prompt activations.

The predecessor is preserved at `v0.2-measurement-gate-failure`. Its audit found
that BART and an AI rater disagreed; it did not test or disprove an SAE causal
hypothesis.

## Direct endpoint

Each source prompt presents all 18 substantive domains behind the one-token
completions ` A` through ` R`, following the exact assistant prefill `CHOICE:`.
The catch-all class `other` is **excluded by design**; it
is neither an action, analysis outcome, nor feature-search target.
The instruction tells the model that `CHOICE:` is already fixed and that it must
complete that prefix with exactly one listed code.

The code-to-domain mapping is a deterministic balanced rotation that varies by
prompt. Prompt IDs are hash-ordered with the frozen seed, and domain `i` maps to
code `(i + prompt_rank) mod 18`. Across the 80 development prompts, every code represents every domain
four or five times. This prevents one domain from being confounded with one
letter token while leaving all within-prompt condition comparisons identical.

At the exact next-token choice position, the runner will record:

- the 18 raw code logits;
- candidate-restricted probabilities `q(d)` in canonical domain order;
- full-vocabulary probability mass on the code set as a diagnostic;
- one layer-20 SAE activation vector in the untreated development baseline;
- deterministic paired realized choices drawn from `q`.

Before feature discovery, at least 90% of all 80 development prompts must assign
at least 50% of their unmasked next-token probability mass to the 18 code tokens.
All rows remain reported; low-mass prompts are never removed. If this compliance
gate fails, the study stops without changing the prefix, codes, or threshold.

## Development baseline result

The immutable real-model baseline ran at commit `44ed128b` on one NVIDIA
A100-SXM4-40GB. It loaded the pinned Gemma 2 and Gemma Scope revisions, validated
all 80 exact code-token boundaries before the first forward, performed one clean
choice forward per development prompt, and captured an `80 x 16,384` untreated
SAE activation matrix. The worktree was clean and the test split was not
generated.

The compliance result was:

- 13/80 prompts at or above 0.5 candidate-code mass: **16.25%**;
- required: 72/80 prompts, or **90%**;
- median candidate-code mass: **0.3413**;
- minimum / maximum: **0.1328 / 0.6988**;
- gate status: **stop**.

Under the frozen protocol, feature discovery is now forbidden. No activation-
domain association was computed, no feature was selected, no intervention was
performed, and the threshold or prompt prefix will not be revised inside v1.
The result means this exact forced letter-code endpoint did not satisfy its
predeclared compliance criterion; it does not test or disprove the SAE causal
hypothesis.

Evidence:

- [`development_baseline_report.json`](development_baseline_report.json)
- [`results/development_choice_baseline.jsonl`](results/development_choice_baseline.jsonl)
  (80 text-free row-level logit/probability records)
- [`development_baseline_verification.json`](development_baseline_verification.json)
  (independent row/activation/hash checks)

The source-text-free activation matrix remains outside the public repository;
its SHA-256 is bound in the report and verification artifact.

## Exploratory endpoint calibration result

The v1 stop remains final. A separate, explicitly exploratory calibration
tested the narrower diagnosis suggested by the public rows: token formatting
and letter identity might obscure an otherwise stable constrained choice. It
hash-selected 24 development prompts, applied six within-prompt code rotations,
recorded the top 20 unmasked next tokens, and compared the original flat
endpoint with a tokenizer-valid newline boundary and a 6-by-3 hierarchy. It
performed no SAE feature analysis or intervention and did not select, generate,
or score any test prompt. See
[`exploratory_calibration_plan.json`](exploratory_calibration_plan.json).

The literal trailing-space variant was rejected before model scoring because
the pinned tokenizer retokenizes the completed `CHOICE: A` sequence instead of
treating `A` as a prefix-preserving next token. A newline is the nearest valid
committed-whitespace boundary.

The real A100 calibration completed all 1,296 planned decision positions at
clean commit `33d7804`, with 648 fixed two-prompt batch forwards. The direct
formatting diagnosis was confirmed:

- original-prefix mean candidate mass: **40.5%**;
- newline-prefix mean candidate mass: **99.4%**;
- original-prefix top token: raw no-space letter in **104/144** trials and an
  allowed leading-space letter in **40/144** trials;
- mean raw-letter mass visible within the stored top 20 was at least **59.0%**.

Formatting was not the decisive validation criterion. Under six within-prompt
code rotations:

| Endpoint | Median Spearman | >=4/6 modal prompts | Stable LOO prompts | Result |
| --- | ---: | ---: | ---: | --- |
| Flat, original prefix | 0.675 | 22/24 | 23/24 | Stop |
| Flat, newline prefix | 0.648 | 24/24 | 23/24 | Stop |
| 6-by-3 hierarchy, newline | 0.757 | 22/24 | 21/24 | Stop |

The descriptive rubric required median Spearman at least 0.70, at least 20/24
prompts with a 4-of-6 modal domain, and at least 22/24 leave-one-rotation-out
stable prompts. No arm passed all three. The two flat rotation-averaged
endpoints were very similar (23/24 top-domain agreement; median Spearman 0.979),
but the hierarchy selected the same top domain as newline-flat for only 12/24
prompts. This is evidence that the coded endpoint remains mapping-sensitive,
not evidence for or against any SAE causal effect.

Per the plan written before these calibration logits were inspected, no Latent
Choice v2 is created and no fourth coded endpoint is tried on these prompts.
The complete source-text-free evidence is in
[`results/exploratory_calibration/`](results/exploratory_calibration/):

- [`calibration_report.json`](results/exploratory_calibration/calibration_report.json)
- [`calibration_rotation_averages.jsonl`](results/exploratory_calibration/calibration_rotation_averages.jsonl)
- [`calibration_forwards.jsonl`](results/exploratory_calibration/calibration_forwards.jsonl)
- [`calibration_verification.json`](results/exploratory_calibration/calibration_verification.json)

The primary endpoint is the prompt-mean change in the selected domain's `q`
under full targeted suppression versus baseline. Specificity contrasts compare
that change with five activation/norm-matched random-feature suppressions and
with L2-matched activation noise. Realized choice rate is secondary. No
post-hoc domain classifier or generated `target_domain` field is used.

## Causal scope

The existing real-GPU-tested model, SAE loader, and residual intervention are
reused from `latent_escape.model_sae` and `latent_escape.intervene`. For a choice
condition, the hook is installed for exactly one forward pass and edits only the
final prompt position whose state produces the code logits.

After a code is sampled, the hook is removed and every intervened cache is
discarded. Analogy generation restarts from a clean prompt that includes the
textual declared domain. Full analogies are needed only for baseline and
full-strength targeted suppression; matched-feature and noise controls are
evaluated directly at the choice endpoint. If two paired arms choose the same
domain, their clean prompt, seed, and generated bytes must be identical.

This supports only a claim about explicit constrained-menu domain choice. It
does not measure spontaneous domain choice, serendipity, creativity, discovery,
or general analogy improvement.

## Frozen sequence

1. Validate the protocol and the 18 one-token action codes before any model run.
2. On the 80 development prompts only, measure untreated `q` and one choice-position
   SAE vector per prompt.
3. Apply the frozen candidate-mass compliance gate before computing any
   feature/domain association; stop if it fails.
4. Select at most one domain/feature pair with the frozen prompt-row maximum-statistic,
   cross-validation, bootstrap, timing, semantic, and matched-control gates.
5. Run the 24-prompt development gate: target doses `0.25`, `0.5`, and `1.0`,
   five matched controls, and L2-matched noise.
6. Have human raters—not AI systems—evaluate the frozen, condition-blind baseline
   and full-targeted subset for declared-domain consistency and structural quality.
7. Stop if any development gate fails. Do not substitute another feature or
   alter thresholds.
8. Only after a passing gate, fill and hash `test_frozen.json`, bind the clean
   code commit and environment, and materialize the test code mapping.
9. Run all eight confirmatory choice arms once on all 120 test prompts.

The human queue shows the source, textual declared domain, and analogy, but hides
the letter code, condition, feature, intervention, logits, and probabilities.
One primary human rating enters each endpoint; a different human duplicates the
frozen 10% reliability subset. `unclear` counts as inconsistent. No AI rating
may supply or adjudicate these outcomes.

## Validation

Structural validation downloads nothing and does not read or materialize test
content:

```bash
uv run python -m latent_choice.validate_protocol --show-summary
```

To reproduce the preflight, verify the pinned Gemma tokenizer contract. This
requires access to the pinned tokenizer and writes no experiment data:

```bash
HF_TOKEN=... uv run python -m latent_choice.validate_protocol --check-tokenizer --show-summary
```

The command must establish that, after the complete chat template and `CHOICE:`
prefill, appending each frozen leading-space code completion adds exactly one
unique token without retokenizing the prompt prefix. The verified IDs and exact
tokenizer revision are committed in
[`code_token_manifest.json`](code_token_manifest.json); the protocol, runner,
and `test_frozen.json` bind that file by SHA-256.

Run the model-free two-prompt contract check with:

```bash
make latent-choice-dry-run
```

On a fresh clone with no existing output, reproduce the immutable 80-prompt
development baseline on one CUDA GPU with:

```bash
HF_TOKEN=... uv run python -m latent_choice.run baseline
```

The real runner refuses a dirty worktree, altered protocol, token drift,
partial prompt list, existing output, or test split. It validates all 80 exact
choice boundaries before the first model forward.

[`protocol.json`](protocol.json) is the complete frozen design.
[`test_config.template.json`](test_config.template.json) is deliberately
incomplete and cannot authorize test access.
