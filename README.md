# SAEreview

This repository contains four deliberately separated studies of sparse
autoencoders (SAEs) and cross-domain analogy:

1. **Retrieval pilot — frozen.** The completed SCAR study is preserved at
   [`v0.1-retrieval-study`](https://github.com/harryila/SAEreview/releases/tag/v0.1-retrieval-study).
2. **Latent Escape — frozen at its measurement gate.** The development baseline
   and independent AI-rater audit are preserved at
   [`v0.2-measurement-gate-failure`](https://github.com/harryila/SAEreview/releases/tag/v0.2-measurement-gate-failure).
   BART and the accepted AI rater agreed on only 18/64 audit items, so the study
   stopped before feature discovery or any hypothesis-testing intervention.
3. **Latent Choice — coded endpoint stopped.** Its frozen v1 baseline failed the
   choice-compliance gate. A subsequent explicitly exploratory calibration
   showed that newline alignment fixes code mass, but none of the three tested
   endpoints met the pre-run mapping-invariance rubric. It therefore remains
   stopped before feature discovery or intervention; see
   [`latent_choice/`](latent_choice/README.md).
4. **Feature-Grounded Structural Rescue — stopped at its frozen development
   gate.** The complete 108-query screen found adequate feature-evidence
   coverage, but neither candidate-source specificity nor aligned-description
   utility passed. See
   [`structural_rescue/`](structural_rescue/README.md).

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

Latent Escape asked whether an interpretable SAE feature associated with an
overused analogy domain could be suppressed to change the model's target-domain
distribution without materially reducing structural quality. Its 640-output
development baseline completed, but the frozen measurement gate failed: BART
and the accepted AI rater agreed on 18/64 labels (28.125%), including 0/38 for
BART's `other` class.

That disagreement does not establish which instrument was correct. The study
therefore makes no claim about the causal hypothesis: no feature was selected,
no hypothesis-testing intervention was run, and the confirmatory split remains
untouched. A separate two-prompt nonzero intervention was only a plumbing smoke
test. See the [frozen protocol and audit report](latent_escape/README.md).

The distinct Latent Choice successor then tested an explicit domain-choice
measurement, without reusing the failed audit. Its pre-feature compliance gate
also stopped the study: the exact letter-code action did not receive enough
unmasked next-token probability on enough development prompts.

A 24-prompt, development-only endpoint calibration then located the formatting
failure precisely: in all 144 original-prefix rotation trials, the top token was
either a raw letter (104) or the allowed leading-space letter (40). Moving the
choice to the next line raised mean allowed-code mass from 40.5% to 99.4%.
However, the original and newline flat endpoints missed the 0.70 median
within-prompt rank-correlation rubric (0.675 and 0.648), while the hierarchy
passed rank correlation (0.757) but missed leave-one-rotation-out stability by
one prompt (21/24 versus 22/24). Per the pre-run rule, coded-domain choice is
stopped rather than retuned. This remains a measurement result, not a test of
the SAE causal hypothesis; no feature-domain association or study intervention
was run, and the confirmatory split remains untouched.

## Feature-Grounded Structural Rescue

The next study returns to the retrieval pilot's candidate oracle. It separates
candidate-source value from verifier value and feature-description value across
nine fixed arms, with every verifier pool padded to exactly 30 candidates. The
real-data preparation pass reproduces dense top-10 **146/566**, dense top-30
**262/566**, and the unpadded dense-plus-two-SAE union **200/566**.

The expanded cheap control is already informative: across 64 prespecified
two-projection random-sparse unions, oracle scores range from **179 to 216/566**
with median **195**; **18/64** match or exceed the SAE score. The higher 95th
percentile is 204 and the plus-one tail probability is **19/65 = 0.292**.
Consequently, the frozen SAE-specific candidate-source gate failed and was not
retuned. The one-time outcome-stratified 108-query screen then completed all
24,708 verifier judgments. The SAE-padded structure arm recovered **16/54**
known rescues while losing **11/54** dense hits (net utility **5**), far below
the frozen requirements of 33 rescues, at most four losses, and net utility 29.
Aligned feature descriptions recovered **15/54** and lost **13/54** (net
utility **2**), underperforming both structure-only and the shuffled-description
control (24 rescues, 18 losses, net utility 6).

Usable feature evidence covered **2,819/3,240 pairs (87.0%)** and all 108
queries, so this was a performance-based no-go rather than a coverage failure.
Structural Rescue is stopped on SCAR under the frozen protocol; no external
benchmark will be opened for this method. This does not test an SAE causal
intervention or disprove the broader SAE hypothesis. See the
[source-free aggregate report](structural_rescue/outputs/development/evaluation_report.json).

## License and attribution

Original code and documentation are MIT licensed. Third-party data and weights
retain their own terms and are not bundled on `main`; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Citation metadata and upstream
references are in [`CITATION.cff`](CITATION.cff) and
[`CITATIONS.md`](CITATIONS.md).
