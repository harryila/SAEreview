"""Focused regressions for the frozen domain-audit and search policy."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from latent_escape.discover_feature import (
    deterministic_audit_ids,
    load_prompt_domain_frequencies,
    primary_domain_columns,
)
from latent_escape.label_domains import load_manual_audit_import


PROVENANCE = {
    "protocol_amendment_id": "amendment-4",
    "protocol_amendment_sha256": "a" * 64,
    "domain_labeling_guide_id": "guide-v1",
    "domain_labeling_guide_sha256": "b" * 64,
}


def test_manual_audit_import_requires_exact_guide_provenance(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    analogy_text = "A blinded analogy"
    analogy_hash = hashlib.sha256(analogy_text.encode("utf-8")).hexdigest()
    row = {
        "blind_id": "blind-1",
        "analogy_text": analogy_text,
        "analogy_text_sha256": analogy_hash,
        "manual_domain_label": "other",
        **PROVENANCE,
    }
    path.write_text(json.dumps(row) + "\n")
    assert load_manual_audit_import(
        path, ["other"], PROVENANCE, {"blind-1": analogy_hash}
    ) == {
        "blind-1": ("other", None)
    }

    row["domain_labeling_guide_sha256"] = "wrong"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="labeling guide/protocol amendment"):
        load_manual_audit_import(path, ["other"], PROVENANCE)

    row["domain_labeling_guide_sha256"] = PROVENANCE[
        "domain_labeling_guide_sha256"
    ]
    row["classifier_domain_label"] = "other"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="exposes forbidden fields"):
        load_manual_audit_import(path, ["other"], PROVENANCE)


def test_feature_discovery_requires_completed_frozen_audit(tmp_path) -> None:
    rows = [
        {
            "blind_id": f"blind-{index}",
            "prompt_id": "prompt-1",
            "condition": "baseline",
            "sample_index": index,
            "seed": index,
            "split": "development",
            "domain_label": "biology/ecology",
            "classifier_domain_label": "biology/ecology",
            "classifier_id": "pinned-classifier",
            "primary_eligible": True,
            "audit_fraction": 0.10,
            "audit_seed": "latent-escape-domain-audit-v1",
            "audit_selected": False,
            "manual_audited": False,
            **PROVENANCE,
        }
        for index in range(10)
    ]
    selected = deterministic_audit_ids(
        rows, 0.10, "latent-escape-domain-audit-v1"
    )
    for row in rows:
        if row["blind_id"] in selected:
            row["audit_selected"] = True
            row["manual_audited"] = True
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    frequencies, counts, observations, audit = load_prompt_domain_frequencies(
        path,
        ["biology/ecology", "other"],
        "baseline",
        10,
        False,
        False,
        False,
        "pinned-classifier",
        PROVENANCE,
        0.10,
    )
    assert counts == {"prompt-1": 10}
    assert observations == 10
    assert frequencies["prompt-1"].tolist() == [1.0, 0.0]
    assert audit["gate_pass"] is True
    assert audit["domain_labeling_guide_id"] == "guide-v1"

    rows[0]["protocol_amendment_sha256"] = "wrong"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="not bound to the frozen"):
        load_prompt_domain_frequencies(
            path,
            ["biology/ecology", "other"],
            "baseline",
            10,
            False,
            False,
            False,
            "pinned-classifier",
            PROVENANCE,
            0.10,
        )


def test_other_rate_is_reported_but_excluded_from_primary_search() -> None:
    taxonomy = ["biology/ecology", "physics", "other"]
    prompt_frequencies = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rates, columns = primary_domain_columns(
        prompt_frequencies, taxonomy, 0.10, ["other"]
    )

    assert rates.tolist() == [0.5, 0.0, 0.5]
    assert [taxonomy[index] for index in columns] == ["biology/ecology"]
