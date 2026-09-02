#!/usr/bin/env python3
"""Load and validate the post-baseline Latent Escape protocol amendment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_PROTOCOL_PATH = ROOT / "latent_escape" / "protocol.json"
AMENDMENT_PATH = ROOT / "latent_escape" / "protocol_amendment_4.json"
GUIDE_PATH = ROOT / "latent_escape" / "domain_labeling_guide.md"
DEVELOPMENT_REPORT_PATH = ROOT / "latent_escape" / "development_baseline_report.json"
TEST_CONFIG_TEMPLATE_PATH = ROOT / "latent_escape" / "test_config.template.json"

BASE_PROTOCOL_SHA256 = "a9bdeb15de798bc56f888715fbe7bef47b69f1dd06f06dc7322013567ed9297a"
PRE_AMENDMENT_COMMIT = "ea4368347a79b6ed11bc9f71df2c8debd4529d93"
DEVELOPMENT_REPORT_SHA256 = (
    "60c862e95cf1b5003e84cdda063e030d7151304761b08bf69d7e47bca0b53258"
)
GUIDE_ID = "latent-escape-domain-labeling-v1"
AMENDMENT_ID = "latent-escape-protocol-amendment-4"
TEST_QUALITY_SAMPLING_PLAN_SHA256 = (
    "4da0e27a2cb67f3942710799bf1db8e295229163a509b2c6c9c7a7c0d9882c99"
)
TEST_QUALITY_RELIABILITY_SUBSET_SHA256 = (
    "0f96b4038f81e40487acf9fd8b83e7cb9ca6e8a6678c9504b67b88dca549b677"
)

EXACT_TAXONOMY = (
    "biology/ecology",
    "medicine/public health",
    "physics",
    "chemistry/materials",
    "engineering/control",
    "computer science/software",
    "AI/neural networks",
    "economics/markets",
    "organizations/governance",
    "sociology/culture",
    "psychology/cognition",
    "education/learning",
    "law/policy",
    "history",
    "arts/literature",
    "sports/games",
    "geography/earth/environment",
    "everyday/household",
    "other",
)

EXPECTED_CLASSIFIER_COUNTS = {
    "biology/ecology": 70,
    "medicine/public health": 3,
    "physics": 2,
    "chemistry/materials": 21,
    "engineering/control": 5,
    "computer science/software": 60,
    "AI/neural networks": 0,
    "economics/markets": 11,
    "organizations/governance": 5,
    "sociology/culture": 1,
    "psychology/cognition": 1,
    "education/learning": 17,
    "law/policy": 7,
    "history": 18,
    "arts/literature": 15,
    "sports/games": 7,
    "geography/earth/environment": 11,
    "everyday/household": 3,
    "other": 383,
}

EXPECTED_QUALITY_KEYS = {
    "pair_selection_seed",
    "reliability_selection_seed",
    "samples_per_prompt",
    "reliability_prompt_fraction",
    "required_arms",
    "primary_rater_estimand",
    "duplicate_rater_use",
    "expected_workload",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def amendment_sha256(path: Path = AMENDMENT_PATH) -> str:
    return sha256_file(path)


def _validate_workload(name: str, workload: dict[str, Any]) -> None:
    expected_prompts = 24 if name == "development_gate" else 120
    expected_reliability = int(math.ceil(0.10 * expected_prompts))
    require(workload.get("prompt_count") == expected_prompts, f"{name} prompt count drift")
    require(
        workload.get("unique_generation_ratings") == 2 * expected_prompts,
        f"{name} unique quality workload drift",
    )
    require(
        workload.get("reliability_prompt_count") == expected_reliability,
        f"{name} reliability-prompt count drift",
    )
    require(
        workload.get("duplicate_ratings") == 2 * expected_reliability,
        f"{name} duplicate-rating count drift",
    )
    require(
        workload.get("total_rating_tasks")
        == workload["unique_generation_ratings"] + workload["duplicate_ratings"],
        f"{name} total quality workload does not add up",
    )


def validate_protocol_amendment(
    protocol: dict[str, Any], amendment: dict[str, Any]
) -> None:
    """Fail closed if the base protocol, amendment, or knowledge snapshot drifts."""

    require(sha256_file(BASE_PROTOCOL_PATH) == BASE_PROTOCOL_SHA256, "base protocol bytes drifted")
    require(protocol.get("schema_version") == 1, "unsupported base protocol schema")
    require(protocol.get("protocol_id") == "latent-escape-mvp-v1", "wrong protocol ID")
    require(protocol.get("protocol_revision") == 3, "base protocol must remain revision 3")
    require(tuple(protocol.get("target_domain_taxonomy", ())) == EXACT_TAXONOMY, "taxonomy drift")

    require(amendment.get("schema_version") == 1, "unsupported amendment schema")
    require(amendment.get("amendment_id") == AMENDMENT_ID, "wrong amendment ID")
    require(amendment.get("protocol_id") == protocol["protocol_id"], "amendment protocol mismatch")
    require(amendment.get("effective_protocol_revision") == 4, "wrong effective revision")
    require(
        amendment.get("status") == "adopted_before_manual_audit_and_feature_discovery",
        "unexpected amendment status",
    )

    base = amendment.get("base_protocol", {})
    require(base.get("revision") == 3, "wrong amendment base revision")
    require(base.get("sha256") == BASE_PROTOCOL_SHA256, "wrong amendment base hash")
    require(
        base.get("pre_amendment_commit") == PRE_AMENDMENT_COMMIT,
        "wrong pre-amendment commit",
    )

    guide = amendment.get("domain_labeling_guide", {})
    require(guide.get("id") == GUIDE_ID, "wrong domain-labeling guide ID")
    require(
        guide.get("path") == "latent_escape/domain_labeling_guide.md",
        "wrong domain-labeling guide path",
    )
    require(guide.get("sha256") == sha256_file(GUIDE_PATH), "domain-labeling guide hash drift")
    require(
        f"Guide ID: `{GUIDE_ID}`" in GUIDE_PATH.read_text(),
        "domain-labeling guide does not declare its ID",
    )

    selection = amendment.get("domain_selection", {})
    require(
        selection.get("minimum_development_output_rate") == 0.10,
        "minimum domain rate must remain 10 percent",
    )
    require(
        selection.get("primary_selected_domain_exclusions") == ["other"],
        "primary domain exclusions must be exactly ['other']",
    )
    require("other" in EXACT_TAXONOMY, "other must remain in taxonomy coverage")
    policy = str(selection.get("coverage_and_diversity_policy", "")).casefold()
    require(
        all(term in policy for term in ("including other", "entropy", "distinct-domain")),
        "amendment must retain other in coverage and diversity outcomes",
    )

    snapshot = amendment.get("knowledge_snapshot", {})
    source_report = snapshot.get("source_report", {})
    require(
        source_report.get("path") == "latent_escape/development_baseline_report.json",
        "wrong knowledge-snapshot report path",
    )
    require(
        source_report.get("sha256") == DEVELOPMENT_REPORT_SHA256,
        "wrong knowledge-snapshot report hash",
    )
    require(
        sha256_file(DEVELOPMENT_REPORT_PATH) == DEVELOPMENT_REPORT_SHA256,
        "development report bytes drifted",
    )
    report = read_json_object(DEVELOPMENT_REPORT_PATH)
    known = snapshot.get("aggregate_information_known", {})
    require(known.get("prompt_count") == report.get("prompt_count") == 80, "prompt count drift")
    require(known.get("samples_per_prompt") == 8, "sample count drift")
    require(report.get("samples_per_prompt") == [8], "report sample count drift")
    require(known.get("generation_count") == report.get("generation_count") == 640, "generation count drift")
    require(known.get("schema_valid_count") == report.get("schema_valid_count") == 584, "schema-valid count drift")
    require(sum(EXPECTED_CLASSIFIER_COUNTS.values()) == 640, "classifier counts do not total 640")
    require(
        known.get("classifier_domain_counts_before_manual_audit")
        == EXPECTED_CLASSIFIER_COUNTS,
        "known classifier-domain counts drifted",
    )
    report_counts = {
        domain: int(report.get("classifier_domain_counts", {}).get(domain, 0))
        for domain in EXACT_TAXONOMY
    }
    require(report_counts == EXPECTED_CLASSIFIER_COUNTS, "report classifier-domain counts drifted")
    report_classifier_ids = report.get("classifier_id")
    require(
        report_classifier_ids == [known.get("classifier_id")],
        "known classifier ID differs from report",
    )
    require(known.get("manual_audit") == report.get("manual_audit"), "rater-audit status drift")
    require(
        known.get("pre_domain_boundary", {}).get("resolved_count") == 640
        and known.get("pre_domain_boundary", {}).get("total_count") == 640
        and known.get("pre_domain_boundary", {}).get("resolution_rate") == 1.0,
        "known pre-domain boundary status drifted",
    )
    require(
        known.get("test_split_touched") is False
        and report.get("test_split_touched") is False,
        "knowledge snapshot must keep the test untouched",
    )
    not_examined = snapshot.get("not_examined_or_computed_before_adoption", [])
    require(
        any("feature-domain correlations" in str(item) for item in not_examined)
        and any("test generations" in str(item) for item in not_examined),
        "knowledge boundary is incomplete",
    )

    quality = amendment.get("quality_guardrail_sampling", {})
    require(set(quality) == EXPECTED_QUALITY_KEYS, "quality sampling fields drifted")
    require(
        quality.get("pair_selection_seed") == "latent-escape-quality-pair-v1",
        "quality pair-selection seed drifted",
    )
    require(
        quality.get("reliability_selection_seed")
        == "latent-escape-quality-reliability-v1",
        "quality reliability-selection seed drifted",
    )
    require(quality.get("samples_per_prompt") == 1, "quality samples per prompt drifted")
    require(
        quality.get("reliability_prompt_fraction") == 0.10,
        "quality reliability fraction drifted",
    )
    require(
        quality.get("required_arms")
        == ["baseline", "targeted_feature_suppression"],
        "quality required arms drifted",
    )
    require(
        "Only the primary rater's score" in str(quality.get("primary_rater_estimand")),
        "quality primary estimand is not explicit",
    )
    require(
        "only for inter-rater reliability" in str(quality.get("duplicate_rater_use")),
        "duplicate-rating use is not reliability-only",
    )
    workloads = quality.get("expected_workload", {})
    require(set(workloads) == {"development_gate", "test"}, "quality workload scopes drifted")
    _validate_workload("development_gate", workloads["development_gate"])
    _validate_workload("test", workloads["test"])


def load_protocol_amendment(
    protocol: dict[str, Any] | None = None,
    path: Path = AMENDMENT_PATH,
) -> dict[str, Any]:
    if path.resolve() != AMENDMENT_PATH.resolve():
        raise ValueError("primary analysis requires the repository amendment file")
    base = protocol if protocol is not None else read_json_object(BASE_PROTOCOL_PATH)
    amendment = read_json_object(path)
    validate_protocol_amendment(base, amendment)
    return amendment


def validate_test_config_amendment_bindings(
    config: dict[str, Any], amendment: dict[str, Any] | None = None
) -> None:
    effective = amendment if amendment is not None else load_protocol_amendment()
    expected = {
        "protocol_id": "latent-escape-mvp-v1",
        "effective_protocol_revision": 4,
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
        "protocol_amendment_id": AMENDMENT_ID,
        "protocol_amendment_sha256": amendment_sha256(),
        "domain_labeling_guide_id": GUIDE_ID,
        "domain_labeling_guide_sha256": effective["domain_labeling_guide"]["sha256"],
        "quality_sampling_plan_sha256": TEST_QUALITY_SAMPLING_PLAN_SHA256,
        "quality_reliability_subset_sha256": (
            TEST_QUALITY_RELIABILITY_SUBSET_SHA256
        ),
    }
    drift = {
        key: {"observed": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    require(not drift, f"test-config amendment bindings drifted: {drift}")


def main() -> int:
    amendment = load_protocol_amendment()
    print(
        json.dumps(
            {
                "amendment_id": amendment["amendment_id"],
                "amendment_sha256": amendment_sha256(),
                "base_protocol_sha256": BASE_PROTOCOL_SHA256,
                "domain_labeling_guide_sha256": amendment["domain_labeling_guide"][
                    "sha256"
                ],
                "primary_selected_domain_exclusions": amendment["domain_selection"][
                    "primary_selected_domain_exclusions"
                ],
                "minimum_development_output_rate": amendment["domain_selection"][
                    "minimum_development_output_rate"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
