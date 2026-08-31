#!/usr/bin/env python3
"""Build the reader-facing experiment notebook with nbformat."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "sae_retrieval_smoke_test.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (SAE smoke test)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.10"},
}

notebook["cells"] = [
    markdown(
        """
# SAE cross-domain analogy retrieval: quick go/no-go test

## tl;dr

**No-go for pure SAE-latent retrieval on this proxy.** On 566 cross-domain retrieval
queries, raw `text-embedding-3-small` reached **25.8% Recall@10**, while the two-SAE
ensemble reached **18.2%** (−7.6 percentage points; paired bootstrap 95% CI
−10.8 to −4.6 pp). Mean reciprocal rank fell from **0.118 to 0.088**.

On the deliberately harder low-lexical-overlap half, dense Recall@10 was **10.4%**
versus **8.0%** for the ensemble; its MRR also fell from **0.051 to 0.041**. The
astro.PH SAE was the strongest sparse variant, but it still did not beat its own raw
dense input.

This is a stop signal for the broad claim that shared SAE latents, by themselves,
improve far-domain analogy retrieval. It is not a definitive test of stable feature
subspaces or a hybrid SAE-candidate-generator plus relational verifier.
        """
    ),
    markdown(
        """
## Context & Methods

The decision is deliberately narrow: should we spend time on a larger SAE analogy
engine? The cheap gate is whether public scientific SAEs improve retrieval of known
cross-domain analogues over the exact dense embeddings they transform.

Methods:

- SCAR's 400 English system-analogy pairs; the 283 cross-domain pairs are queried in
  both directions, producing 566 queries against 400 candidates per direction.
- Relevance is the paired system. Candidates with the same normalized system name
  also count as relevant so repeated concepts are not obvious false negatives.
- Metrics are MRR and Recall@1/5/10. The low-overlap scope is the bottom half of
  cross-domain pairs by stopword-filtered token Jaccard similarity.
- Baselines: seeded random, Okapi BM25, local MiniLM, and OpenAI
  `text-embedding-3-small`.
- SAE variants: the public O'Neill et al. Top-K checkpoints trained on cs.LG and
  astro.PH embeddings (`k=128`, 3,072 latents), plus a mean-score ensemble.
- Uncertainty: 2,000 paired bootstrap resamples at the analogy-pair level.

### Key Assumptions

SCAR descriptions are a fast structural-analogy proxy, not scientific-paper
abstracts. The candidate pool supplies natural distractors but not manually designed
same-topic/different-mechanism hard negatives. The two checkpoints test domain
sensitivity, not stability across random SAE seeds.

Sources: [SCAR](https://github.com/siyuyuan/scar),
[scientific embedding SAEs](https://huggingface.co/charlieoneill/embedding-saes), and
[reference Top-K inference code](https://huggingface.co/spaces/charlieoneill/saerch.ai/blob/main/topk_sae.py).
        """
    ),
    markdown("## Data"),
    code(
        """
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path.cwd()
command = [sys.executable, "sae_smoke_test.py"]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
if completed.returncode != 0:
    raise RuntimeError(completed.stderr or completed.stdout)
print(completed.stdout.strip().splitlines()[-1])

results_path = ROOT / "results" / "results.json"
results = json.loads(results_path.read_text(encoding="utf-8"))
benchmark = results["benchmark"]
print(
    f"{benchmark['total_pairs']} total pairs; "
    f"{benchmark['cross_domain_pairs']} cross-domain pairs; "
    f"{benchmark['queries_cross_domain_bidirectional']} bidirectional queries; "
    f"{benchmark['candidate_count_per_direction']} candidates per query."
)
        """
    ),
    markdown(
        """
The raw input is the released English SCAR JSONL file. Dataset and checkpoint SHA-256
hashes are recorded in `results/results.json`; the OpenAI key is never stored. Only
the returned 1,536-dimensional embeddings are cached for reproducible reruns.
        """
    ),
    markdown("## Results"),
    code(
        """
from IPython.display import Markdown, display

method_order = [
    "random_seeded",
    "bm25",
    "minilm_dense",
    "openai_dense",
    "sae_cslg_k128_n3072",
    "sae_astroph_k128_n3072",
    "sae_dual_domain_ensemble",
]
labels = {
    "random_seeded": "Random",
    "bm25": "BM25",
    "minilm_dense": "MiniLM dense",
    "openai_dense": "OpenAI dense",
    "sae_cslg_k128_n3072": "cs.LG SAE",
    "sae_astroph_k128_n3072": "astro.PH SAE",
    "sae_dual_domain_ensemble": "Two-SAE ensemble",
}

lines = [
    "| Method | Cross-domain MRR | Cross-domain R@10 | Low-overlap MRR | Low-overlap R@10 |",
    "|---|---:|---:|---:|---:|",
]
for method in method_order:
    metric = results["metrics"][method]
    lines.append(
        f"| {labels[method]} | {metric['cross_domain']['mrr']:.3f} | "
        f"{metric['cross_domain']['recall_at_10']:.1%} | "
        f"{metric['low_lexical_overlap']['mrr']:.3f} | "
        f"{metric['low_lexical_overlap']['recall_at_10']:.1%} |"
    )
display(Markdown("\\n".join(lines)))
        """
    ),
    markdown(
        """
Raw OpenAI dense retrieval is the strongest tested method. Sparsification loses
substantial ranking signal in the full cross-domain set. On low-overlap pairs, the
gap is smaller but still does not reverse; BM25 collapses as expected, which confirms
that this slice is less driven by shared words.
        """
    ),
    code(
        """
comparisons = results["paired_bootstrap_comparisons"]
comparison_order = [
    "sae_cslg_k128_n3072_vs_openai_dense",
    "sae_astroph_k128_n3072_vs_openai_dense",
    "sae_dual_domain_ensemble_vs_openai_dense",
]
comparison_labels = {
    comparison_order[0]: "cs.LG SAE − dense",
    comparison_order[1]: "astro.PH SAE − dense",
    comparison_order[2]: "Two-SAE ensemble − dense",
}

lines = [
    "| Paired comparison | Scope | Δ MRR (95% CI) | Δ R@10 (95% CI) |",
    "|---|---|---:|---:|",
]
for name in comparison_order:
    for scope, scope_label in [
        ("cross_domain", "Cross-domain"),
        ("low_lexical_overlap", "Low overlap"),
    ]:
        mrr = comparisons[name][scope]["mrr"]
        r10 = comparisons[name][scope]["recall_at_10"]
        lines.append(
            f"| {comparison_labels[name]} | {scope_label} | "
            f"{mrr['delta']:+.3f} [{mrr['ci_95_low']:+.3f}, {mrr['ci_95_high']:+.3f}] | "
            f"{r10['delta']:+.1%} [{r10['ci_95_low']:+.1%}, {r10['ci_95_high']:+.1%}] |"
        )
display(Markdown("\\n".join(lines)))
        """
    ),
    markdown("### Retrieval spot check"),
    code(
        """
worst = results["examples"]["worst"]
lines = [
    "| SAE-ensemble query | Gold analogue | Top retrieved | Gold rank |",
    "|---|---|---|---:|",
]
for item in worst:
    lines.append(
        f"| {item['query']} | {item['gold']} | {item['retrieved_top1']} | "
        f"{item['gold_rank']:.0f} |"
    )
display(Markdown("\\n".join(lines)))
        """
    ),
    markdown("### Checks"),
    code(
        """
assert benchmark["total_pairs"] == 400
assert benchmark["cross_domain_pairs"] == 283
assert benchmark["queries_cross_domain_bidirectional"] == 566
assert benchmark["candidate_count_per_direction"] == 400
assert benchmark["low_lexical_overlap_pairs"] == 144
assert results["skipped"] == {}

dense = results["metrics"]["openai_dense"]
for sae_method in [
    "sae_cslg_k128_n3072",
    "sae_astroph_k128_n3072",
    "sae_dual_domain_ensemble",
]:
    sae = results["metrics"][sae_method]
    assert dense["cross_domain"]["mrr"] > sae["cross_domain"]["mrr"]
    assert dense["cross_domain"]["recall_at_10"] > sae["cross_domain"]["recall_at_10"]

ensemble_delta = comparisons[
    "sae_dual_domain_ensemble_vs_openai_dense"
]["cross_domain"]["recall_at_10"]["delta"]
recomputed_delta = (
    results["metrics"]["sae_dual_domain_ensemble"]["cross_domain"]["recall_at_10"]
    - dense["cross_domain"]["recall_at_10"]
)
assert abs(ensemble_delta - recomputed_delta) < 1e-12

serialized = results_path.read_text(encoding="utf-8")
assert "OPENAI_API_KEY" not in serialized and "sk-proj" not in serialized
print("All targeted integrity checks passed.")
        """
    ),
    markdown(
        """
## Takeaways

1. **Do not proceed with a pure SAE-nearest-neighbor engine.** Both domain-specific
   dictionaries and their ensemble underperform the exact dense representation they
   sparsify.
2. **The failure is consistent with the review's central concern.** Top-K features can
   preserve salient concepts while discarding relational detail needed to match an
   analogue.
3. **Only continue under a narrower, different hypothesis:** use SAE features for
   interpretable candidate proposals, then verify explicit entity-role and causal
   mappings. A real second stage would need paper abstracts, hard negatives,
   purpose–mechanism and LLM-schema baselines, and multiple SAE seeds or aligned
   feature subspaces.

Validation assessment: **share with caveats**. The arithmetic, pairing, duplicate-label
handling, hashes, and cache-only rerun are checked. The conclusion is valid for this
proxy and these public checkpoints, not a universal statement about every SAE or
hybrid architecture.
        """
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
