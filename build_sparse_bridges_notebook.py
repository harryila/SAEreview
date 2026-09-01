#!/usr/bin/env python3
"""Build the concise, executable Sparse Bridges quick-test notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "sparse_bridges_quick_test.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.10"},
}
notebook["cells"] = [
    markdown(
        """
# Sparse Bridges, Dense Recall — quick test

This notebook executes the corrected complementarity gate and the small held-out
dense-top-100 reranker requested in the review. It makes no API calls: it uses the
cached raw `text-embedding-3-small` vectors and verifies the exact released
`k=64, n=9,216` checkpoint hashes before inference.

Primary grain: 566 directional queries from 283 cross-domain SCAR pairs. Both
directions of a pair remain in the same one of five folds.
"""
    ),
    code(
        """
from pathlib import Path
from html import escape
import json
import subprocess
import sys

from IPython.display import HTML, display

ROOT = Path.cwd()
assert (ROOT / "complementarity_gate.py").exists(), ROOT
PYTHON = Path(sys.executable)
RUN_DIR = ROOT / ".cache" / "notebook_run"
RUN_DIR.mkdir(parents=True, exist_ok=True)
GATE_SUMMARY = RUN_DIR / "complementarity_gate.json"
GATE_QUERIES = RUN_DIR / "complementarity_queries.jsonl"
HYBRID_SUMMARY = RUN_DIR / "hybrid_reranker.json"
HYBRID_QUERIES = RUN_DIR / "hybrid_queries.jsonl"

def show_table(rows, formatters=None):
    formatters = formatters or {}
    headers = list(rows[0])
    body = []
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header)
            if header in formatters:
                value = formatters[header](value)
            elif value is None:
                value = "—"
            cells.append(f"<td>{escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    display(HTML(
        "<table><thead><tr>" + header_html + "</tr></thead><tbody>" +
        "".join(body) + "</tbody></table>"
    ))

jobs = (
    ("complementarity_gate.py", GATE_SUMMARY, GATE_QUERIES),
    ("hybrid_reranker.py", HYBRID_SUMMARY, HYBRID_QUERIES),
)
for script, summary_path, queries_path in jobs:
    completed = subprocess.run(
        [
            str(PYTHON),
            script,
            "--summary",
            str(summary_path),
            "--queries",
            str(queries_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"{script}: OK")
"""
    ),
    markdown(
        """
## 1. Complementarity gate

The gate was fixed before inspecting the corrected checkpoint run: proceed only
with at least 20 unique SAE rescues and at least +4.0 percentage points in the
dense-plus-SAE oracle ceiling. The either-SAE result below is explicitly an oracle
diagnostic, not a deployable score fusion.
"""
    ),
    code(
        """
gate = json.loads(GATE_SUMMARY.read_text())
gate_queries = [
    json.loads(line)
    for line in GATE_QUERIES.read_text().splitlines()
    if line.strip()
]
overall = gate["overall"]

assert gate["gate"]["passes"] is True
assert overall["queries"] == len(gate_queries) == 566
assert len({row["query_id"] for row in gate_queries}) == 566
assert (
    overall["shared_dense_and_either_sae"]
    + overall["sae_unique_rescues"]
    + overall["dense_unique_rescues"]
    + overall["neither_succeeds"]
    == 566
)
assert overall["oracle_union_successes"] == (
    overall["dense_top10_successes"] + overall["sae_unique_rescues"]
)
assert sum(row["sae_rescue"] for row in gate_queries) == 54
assert all(
    row["source_domain"] != row["target_domain"] for row in gate_queries
)

gate_table = [
        {
            "measure": "Dense top-10",
            "queries": overall["dense_top10_successes"],
            "Recall@10": overall["dense_recall_at_10"],
        },
        {
            "measure": "Either SAE top-10 (oracle checkpoint choice)",
            "queries": overall["either_sae_top10_successes"],
            "Recall@10": overall["either_sae_oracle_recall_at_10"],
        },
        {
            "measure": "Dense ∪ either SAE oracle",
            "queries": overall["oracle_union_successes"],
            "Recall@10": overall["oracle_union_recall_at_10"],
        },
]
show_table(gate_table, {"Recall@10": lambda value: f"{value:.1%}"})
"""
    ),
    code(
        """
show_table(
    [
        {"gate statistic": "SAE-only rescues", "value": overall["sae_unique_rescues"]},
        {"gate statistic": "Dense-only rescues", "value": overall["dense_unique_rescues"]},
        {
            "gate statistic": "Oracle gain over dense",
            "value": f"{overall['oracle_improvement_percentage_points']:.2f} pp",
        },
        {
            "gate statistic": "Dense–cs.LG rank correlation",
            "value": f"{gate['rank_correlations']['dense_vs_cslg_sae']['spearman_rho']:.3f}",
        },
        {
            "gate statistic": "Dense–astro.PH rank correlation",
            "value": f"{gate['rank_correlations']['dense_vs_astroph_sae']['spearman_rho']:.3f}",
        },
        {
            "gate statistic": "Top-10 cutoff ties",
            "value": sum(
                sum(value.values()) for value in gate["top10_cutoff_ties"].values()
            ),
        },
    ]
)
"""
    ),
    markdown(
        """
**Gate decision:** pass. The 54 unique directional rescues imply a +9.54-point
oracle ceiling, comfortably above both pre-specified thresholds. This justifies
one proper shortlist reranker test, but is not itself evidence that a deployable
hybrid can choose those rescues.
"""
    ),
    markdown(
        """
## 2. Pair-grouped held-out reranker

Dense retrieval supplies exactly 100 candidates. A fixed linear pairwise ranker
uses out-of-fold predictions only. The full model includes dense score/rank,
BM25, lexical Jaccard, separate cs.LG and astro.PH cosine, the requested
IDF-weighted minimum-activation bridge score, shared-feature rarity/count, and
reconstruction cosine. Controls use the same folds and learner.
"""
    ),
    code(
        """
hybrid = json.loads(HYBRID_SUMMARY.read_text())
hybrid_queries = [
    json.loads(line)
    for line in HYBRID_QUERIES.read_text().splitlines()
    if line.strip()
]
metrics = hybrid["metrics"]

assert len(hybrid_queries) == 566
assert len({row["query_id"] for row in hybrid_queries}) == 566
assert hybrid["protocol"]["pair_groups"] == 283
assert hybrid["protocol"]["canonical_fold_groups"] == 277
assert hybrid["protocol"]["folds"] == 5
assert hybrid["protocol"]["shortlist_retrievable_queries"] == 432
assert metrics["dense_only"]["top10_successes"] == 146
assert metrics["dense_idf_sae"]["top10_successes"] == 146

full_wins = sum(
    (not row["dense_only_top10"]) and row["dense_idf_sae_top10"]
    for row in hybrid_queries
)
full_losses = sum(
    row["dense_only_top10"] and (not row["dense_idf_sae_top10"])
    for row in hybrid_queries
)
assert (full_wins, full_losses) == (30, 30)

labels = {
    "dense_only": "Dense only",
    "dense_bm25": "Dense + BM25",
    "dense_random_sparse": "Dense + random sparse projection",
    "dense_unweighted_sae": "Dense + unweighted SAE cosine",
    "dense_idf_sae": "Dense + IDF-weighted SAE features (primary)",
}
control_rows = []
for method, label in labels.items():
    values = metrics[method]
    comparison = values.get("vs_dense", {})
    control_rows.append(
        {
            "method": label,
            "hits / 566": values["top10_successes"],
            "Recall@10": values["recall_at_10"],
            "Δ vs dense (pp)": comparison.get("delta_percentage_points", 0.0),
            "95% CI low (pp)": comparison.get("ci_95_low_percentage_points"),
            "95% CI high (pp)": comparison.get("ci_95_high_percentage_points"),
            "rescues": comparison.get("rescues"),
            "losses": comparison.get("losses"),
            "MRR": values["mrr_with_shortlist_misses_as_zero"],
        }
    )

show_table(
    control_rows,
    {
        "Recall@10": lambda value: f"{value:.1%}",
        "Δ vs dense (pp)": lambda value: f"{value:+.2f}",
        "95% CI low (pp)": lambda value: "—" if value is None else f"{value:+.2f}",
        "95% CI high (pp)": lambda value: "—" if value is None else f"{value:+.2f}",
        "MRR": lambda value: f"{value:.3f}",
    },
)
"""
    ),
    code(
        """
primary = hybrid["primary_result"]
print(f"Primary verdict: {primary['verdict']}")
print(
    "Full SAE hybrid: "
    f"{metrics['dense_idf_sae']['top10_successes']}/566, "
    f"Δ={primary['delta_percentage_points']:+.2f} pp, "
    f"clustered 95% CI "
    f"[{100*primary['canonical_pair_cluster_bootstrap_ci_95'][0]:+.2f}, "
    f"{100*primary['canonical_pair_cluster_bootstrap_ci_95'][1]:+.2f}] pp."
)
print(
    "Random sparse control: "
    f"{metrics['dense_random_sparse']['top10_successes']}/566 "
    f"({metrics['dense_random_sparse']['vs_dense']['delta_percentage_points']:+.2f} pp)."
)
"""
    ),
    markdown(
        """
## Decision

The complementarity hypothesis passes, but this first deployable bridge model
does not convert the oracle headroom into held-out Recall@10 gain. It ties dense
exactly (30 rescues, 30 losses), its uncertainty interval crosses zero, and the
random sparse control is slightly better. Therefore this run does **not** support
an SAE-specific `Sparse Bridges, Dense Recall` claim.

Per the review’s stop rule, do not tune this same SCAR reranker until it turns
positive. The useful finding is narrower: SAE rankings contain real complementary
signal, but this fixed linear bridge scorer cannot identify it out of sample.
"""
    ),
]

nbf.write(notebook, OUTPUT)
print(OUTPUT)
