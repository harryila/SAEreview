"""CLI for the exploratory Feature-Grounded Structural Rescue pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from sae_smoke_test import load_jsonl

from .core import (
    DEFAULT_DATA,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROTOCOL,
    FEATURE_DESCRIPTION_BATCH_SIZE,
    ROOT,
    VERIFIER_BATCH_SIZE,
    canonical_json_sha256,
    prepare_development,
    sha256_file,
    system_payload,
    write_json,
    write_jsonl,
)
from .evaluate import evaluate_rows
from .llm import (
    FEATURE_DESCRIPTION_BATCH_SCHEMA,
    FEATURE_INSTRUCTIONS,
    MECHANISM_BATCH_SCHEMA,
    MECHANISM_INSTRUCTIONS,
    MODEL,
    PROMPT_VERSION,
    VERDICT_BATCH_SCHEMA,
    VERIFIER_INSTRUCTIONS,
    OpenAIStructuredBackend,
    feature_description_payload,
    mechanism_payload,
    opaque_feature_alias,
    pair_payload_hash,
    prompt_hash,
    schema_hash,
    structured_request_hash,
    validate_exact_ids,
    verdict_score,
    verifier_payload,
)


MECHANISM_BATCH_SIZE = 8
PREPARED_FILENAMES = {
    "candidate_manifest": "candidate_manifest.jsonl",
    "qrels_sidecar": "qrels_sidecar.jsonl",
    "feature_catalog": "feature_catalog.jsonl",
    "pair_feature_evidence": "pair_feature_evidence.jsonl",
    "screen_selection": "screen_selection.json",
    "verifier_batch_plan": "verifier_batch_plan.json",
}


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_rows_by_id(data_path: Path) -> dict[int, dict[str, Any]]:
    rows = load_jsonl(data_path)
    by_id = {int(row["id"]): row for row in rows}
    if len(rows) != 400 or len(by_id) != 400:
        raise ValueError("Expected 400 unique SCAR rows")
    return by_id


def _read_existing(path: Path, key: str, *, overwrite: bool) -> dict[str, dict[str, Any]]:
    if overwrite or not path.exists():
        return {}
    rows = load_jsonl(path)
    indexed = {str(row[key]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {key} rows in {path}")
    return indexed


def _save_index(path: Path, values: Mapping[str, dict[str, Any]]) -> None:
    write_jsonl(path, [values[key] for key in sorted(values)], overwrite=path.exists())


def normalized_system_content_sha256(name: str, background: str) -> str:
    normalize = lambda value: " ".join(str(value).casefold().split())
    return canonical_json_sha256(
        {"name": normalize(name), "background": normalize(background)}
    )


def git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    return commit, dirty


def validate_prepared_bundle(
    output_dir: Path,
    *,
    canonical_report_path: Path | None = None,
    data_path: Path | None = None,
) -> dict[str, str]:
    """Bind every downstream stage to the committed SCAR preparation manifest."""

    canonical_path = canonical_report_path or DEFAULT_PROTOCOL.with_name(
        "prepare_report.json"
    )
    observed_path = output_dir / "prepare_report.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    observed = json.loads(observed_path.read_text(encoding="utf-8"))
    stable_keys = (
        "protocol_sha256",
        "source_sha256",
        "preflight",
        "feature_catalog_rows",
        "pair_feature_evidence_rows",
        "verifier_screen",
        "verifier_batch_plan",
    )
    for key in stable_keys:
        if observed.get(key) != canonical.get(key):
            raise ValueError(f"Prepared bundle differs from committed report: {key}")
    if observed["protocol_sha256"] != sha256_file(DEFAULT_PROTOCOL):
        raise ValueError("Prepared bundle uses a stale protocol")
    if data_path is not None and sha256_file(data_path) != canonical["source_sha256"][
        "scar"
    ]:
        raise ValueError("SCAR source hash differs from the prepared bundle")

    hashes: dict[str, str] = {}
    for key, filename in PREPARED_FILENAMES.items():
        path = output_dir / filename
        actual = sha256_file(path)
        observed_hash = observed["artifacts"][key]["sha256"]
        canonical_hash = canonical["artifacts"][key]["sha256"]
        if actual != observed_hash or actual != canonical_hash:
            raise ValueError(f"Prepared artifact hash mismatch: {key}")
        hashes[key] = actual
    return hashes


def require_clean_real_backend(backend, *, dirty: bool) -> None:
    if backend.model == MODEL and dirty:
        raise RuntimeError("Real API stages require a clean committed worktree")


def validate_verdict_batch(
    output: Mapping[str, Any],
    *,
    aliases: Mapping[str, str],
    empty_evidence_aliases: set[str],
) -> None:
    validate_exact_ids(
        output,
        collection_key="candidates",
        id_key="candidate_alias",
        expected=aliases,
    )


def normalize_empty_evidence_verdicts(
    output: Mapping[str, Any], *, empty_evidence_aliases: set[str]
) -> int:
    """Apply the frozen evidence-only invariant while preserving raw model fields."""

    overrides = 0
    for verdict in output["candidates"]:
        raw_support = int(verdict["feature_support"])
        raw_overlap = bool(verdict["accidental_feature_overlap"])
        is_empty = str(verdict["candidate_alias"]) in empty_evidence_aliases
        verdict["raw_feature_support"] = raw_support
        verdict["raw_accidental_feature_overlap"] = raw_overlap
        verdict["empty_evidence_normalized"] = is_empty and (
            raw_support != 0 or raw_overlap
        )
        if is_empty:
            verdict["feature_support"] = 0
            verdict["accidental_feature_overlap"] = False
            overrides += int(bool(verdict["empty_evidence_normalized"]))
    return overrides


def validate_feature_description_batch(
    output: Mapping[str, Any], *, expected_aliases: Sequence[str]
) -> None:
    validate_exact_ids(
        output,
        collection_key="features",
        id_key="feature_key",
        expected=expected_aliases,
    )
    fallback = "no coherent mechanistic interpretation"
    for row in output["features"]:
        description = str(row["description"]).strip().casefold().rstrip(".")
        if not bool(row["coherent"]) and description != fallback:
            raise ValueError("Incoherent feature description must use the frozen fallback")


def _pair_feature_context(
    shared_features: Sequence[Mapping[str, Any]],
    descriptions: Mapping[str, Mapping[str, Any]],
    *,
    query_content_sha256: str,
    candidate_content_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Exclude incoherent or directly self-describing evidence for one pair."""

    usable: list[dict[str, Any]] = []
    counts = {"raw": len(shared_features), "incoherent": 0, "direct_example": 0}
    for feature in shared_features:
        feature_key = str(feature["feature_key"])
        if feature_key not in descriptions:
            raise ValueError(f"Missing feature description for {feature_key}")
        description = descriptions[feature_key]
        if not bool(description.get("coherent")):
            counts["incoherent"] += 1
            continue
        example_hashes = set(
            map(str, description.get("request_example_content_sha256", []))
        )
        if not example_hashes:
            raise ValueError(f"Feature description lacks example provenance: {feature_key}")
        if (
            query_content_sha256 in example_hashes
            or candidate_content_sha256 in example_hashes
        ):
            counts["direct_example"] += 1
            continue
        usable.append(dict(feature))
    counts["usable"] = len(usable)
    return usable, counts


class FixtureBackend:
    """Deterministic plumbing backend. Its outputs are never evidentiary."""

    model = "fixture-no-model"

    def complete(
        self,
        *,
        schema_name: str,
        schema: Mapping[str, Any],
        instructions: str,
        payload: Mapping[str, Any],
        max_output_tokens: int,
        collection_key: str | None = None,
        id_key: str | None = None,
        expected_ids: Sequence[str] | None = None,
        output_validator=None,
    ) -> dict[str, Any]:
        del schema, instructions, max_output_tokens, collection_key, id_key, expected_ids
        if schema_name == "mechanism_batch":
            result = {
                "systems": [
                    {
                        "system_id": row["system_id"],
                        "summary": f"Fixture mechanism for {row['name']}",
                        "entities_and_roles": [
                            {"entity": row["name"], "role": "fixture role"}
                        ],
                        "causal_relations": [],
                        "dynamics": ["fixture dynamics"],
                        "constraints": [],
                        "boundary_conditions": [],
                    }
                    for row in payload["systems"]
                ]
            }
        elif schema_name == "feature_description_batch":
            result = {
                "features": [
                    {
                        "feature_key": row["feature_key"],
                        "description": "fixture feature pattern",
                        "coherent": True,
                    }
                    for row in payload["features"]
                ]
            }
        elif schema_name == "verdict_batch":
            output = []
            for row in payload["pairs"]:
                digest = int(canonical_json_sha256(row)[:8], 16)
                has_features = bool(row["shared_feature_evidence"])
                output.append(
                    {
                        "candidate_alias": row["candidate_alias"],
                        "role_alignment": digest % 5,
                        "causal_alignment": (digest // 5) % 5,
                        "dynamics_alignment": (digest // 25) % 5,
                        "constraint_alignment": (digest // 125) % 5,
                        "feature_support": 2 if has_features else 0,
                        "lexical_only": False,
                        "same_domain_only": False,
                        "accidental_feature_overlap": False,
                        "break_severity": (digest // 625) % 5,
                        "mechanism_mappings": [],
                        "analogy_breakpoints": [],
                    }
                )
            result = {"candidates": output}
        else:
            raise ValueError(f"Unknown fixture schema {schema_name}")
        if output_validator is not None:
            output_validator(result)
        return result


def make_backend(kind: str, output_dir: Path):
    if kind == "fixture":
        return FixtureBackend()
    if kind == "openai":
        return OpenAIStructuredBackend(
            model=MODEL, cache_dir=output_dir / "api_cache"
        )
    raise ValueError(f"Unknown backend {kind}")


def selected_candidates(
    candidate_path: Path,
    *,
    selection_path: Path,
    limit_queries: int | None,
) -> list[dict[str, Any]]:
    rows = load_jsonl(candidate_path)
    by_id = {str(row["query_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate candidate query IDs")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_ids = list(map(str, selection["query_ids"]))
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate query IDs in verifier screen")
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise ValueError(f"Verifier screen contains unknown query IDs: {missing[:3]}")
    if limit_queries is not None:
        if limit_queries <= 0:
            raise ValueError("--limit-queries must be positive")
        selected_ids = selected_ids[:limit_queries]
    return [by_id[query_id] for query_id in selected_ids]


def _stage_request_hash(
    *,
    backend,
    schema_name: str,
    schema: Mapping[str, Any],
    instructions: str,
    payload: Mapping[str, Any],
    max_output_tokens: int,
) -> str:
    return structured_request_hash(
        model=backend.model,
        schema_name=schema_name,
        schema=schema,
        instructions=instructions,
        payload=payload,
        max_output_tokens=max_output_tokens,
    )


def _validate_fixed_batch(
    existing_rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    request_sha256: str,
    backend_model: str,
    current_git_commit: str,
    label: str,
) -> bool:
    """Return True for a complete compatible batch; reject partial/stale batches."""

    if not existing_rows:
        return False
    if len(existing_rows) != expected_count:
        raise ValueError(
            f"Partial {label} batch found ({len(existing_rows)}/{expected_count}); "
            "rerun this stage with --overwrite"
        )
    for row in existing_rows:
        if row.get("request_sha256") != request_sha256:
            raise ValueError(f"Stale {label} result; rerun this stage with --overwrite")
        if row.get("model") != backend_model:
            raise ValueError(f"Mixed-model {label} result; rerun with --overwrite")
        if backend_model == MODEL and (
            row.get("generation_git_commit") != current_git_commit
            or bool(row.get("generation_git_worktree_dirty"))
        ):
            raise ValueError(f"Stale code provenance in {label}; rerun with --overwrite")
    return True


def extract_mechanisms(
    *,
    data_path: Path,
    candidate_path: Path,
    selection_path: Path,
    output_path: Path,
    backend,
    limit_queries: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    candidates = selected_candidates(
        candidate_path,
        selection_path=selection_path,
        limit_queries=limit_queries,
    )
    required_ids = set()
    for query in candidates:
        required_ids.add(str(query["query_system_id"]))
        required_ids.update(map(str, query["superpool"]))
    rows_by_id = load_rows_by_id(data_path)
    generation_commit, generation_dirty = git_state()
    require_clean_real_backend(backend, dirty=generation_dirty)
    existing = _read_existing(output_path, "system_id", overwrite=overwrite)
    for batch in batched(sorted(required_ids), MECHANISM_BATCH_SIZE):
        systems = [system_payload(system, rows_by_id) for system in batch]
        payload = mechanism_payload(systems)
        request_sha256 = _stage_request_hash(
            backend=backend,
            schema_name="mechanism_batch",
            schema=MECHANISM_BATCH_SCHEMA,
            instructions=MECHANISM_INSTRUCTIONS,
            payload=payload,
            max_output_tokens=12000,
        )
        batch_existing = [existing[system_id] for system_id in batch if system_id in existing]
        if _validate_fixed_batch(
            batch_existing,
            expected_count=len(batch),
            request_sha256=request_sha256,
            backend_model=backend.model,
            current_git_commit=generation_commit,
            label="mechanism",
        ):
            continue
        result = backend.complete(
            schema_name="mechanism_batch",
            schema=MECHANISM_BATCH_SCHEMA,
            instructions=MECHANISM_INSTRUCTIONS,
            payload=payload,
            max_output_tokens=12000,
            collection_key="systems",
            id_key="system_id",
            expected_ids=list(batch),
        )
        for graph in result["systems"]:
            graph = dict(graph)
            source = next(
                row for row in systems if row["system_id"] == graph["system_id"]
            )
            graph["input_sha256"] = canonical_json_sha256(source)
            graph["source_content_sha256"] = normalized_system_content_sha256(
                source["name"], source["background"]
            )
            graph["model"] = backend.model
            graph["prompt_version"] = PROMPT_VERSION
            graph["instructions_sha256"] = prompt_hash(MECHANISM_INSTRUCTIONS)
            graph["schema_sha256"] = schema_hash(MECHANISM_BATCH_SCHEMA)
            graph["request_sha256"] = request_sha256
            graph["generation_git_commit"] = generation_commit
            graph["generation_git_worktree_dirty"] = generation_dirty
            existing[str(graph["system_id"])] = graph
        _save_index(output_path, existing)
        print(f"mechanisms {len(existing)}/{len(required_ids)}", flush=True)
    return {
        "required_systems": len(required_ids),
        "completed_systems": len(required_ids.intersection(existing)),
        "backend": backend.model,
        "output_sha256": sha256_file(output_path),
    }


def describe_features(
    *,
    data_path: Path,
    feature_catalog_path: Path,
    output_path: Path,
    backend,
    overwrite: bool,
) -> dict[str, Any]:
    features = load_jsonl(feature_catalog_path)
    rows_by_id = load_rows_by_id(data_path)
    generation_commit, generation_dirty = git_state()
    require_clean_real_backend(backend, dirty=generation_dirty)
    examples_by_id: dict[str, dict[str, str]] = {}
    for feature in features:
        for example in feature["top_examples"]:
            identifier = str(example["system_id"])
            examples_by_id[identifier] = system_payload(identifier, rows_by_id)
    existing = _read_existing(output_path, "feature_key", overwrite=overwrite)
    for batch in batched(features, FEATURE_DESCRIPTION_BATCH_SIZE):
        payload = feature_description_payload(batch, examples_by_id)
        internal_keys = [str(row["feature_key"]) for row in batch]
        alias_to_internal = {
            opaque_feature_alias(feature_key): feature_key
            for feature_key in internal_keys
        }
        expected = list(alias_to_internal)
        request_sha256 = _stage_request_hash(
            backend=backend,
            schema_name="feature_description_batch",
            schema=FEATURE_DESCRIPTION_BATCH_SCHEMA,
            instructions=FEATURE_INSTRUCTIONS,
            payload=payload,
            max_output_tokens=4000,
        )
        request_example_hashes = sorted(
            {
                normalized_system_content_sha256(
                    examples_by_id[str(example["system_id"])]["name"],
                    examples_by_id[str(example["system_id"])]["background"],
                )
                for feature in batch
                for example in feature["top_examples"]
            }
        )
        batch_existing = [existing[key] for key in internal_keys if key in existing]
        if _validate_fixed_batch(
            batch_existing,
            expected_count=len(batch),
            request_sha256=request_sha256,
            backend_model=backend.model,
            current_git_commit=generation_commit,
            label="feature-description",
        ):
            continue
        result = backend.complete(
            schema_name="feature_description_batch",
            schema=FEATURE_DESCRIPTION_BATCH_SCHEMA,
            instructions=FEATURE_INSTRUCTIONS,
            payload=payload,
            max_output_tokens=4000,
            collection_key="features",
            id_key="feature_key",
            expected_ids=expected,
            output_validator=lambda output, expected=expected: validate_feature_description_batch(
                output, expected_aliases=expected
            ),
        )
        for description in result["features"]:
            row = dict(description)
            alias = str(row.pop("feature_key"))
            internal_key = alias_to_internal[alias]
            feature = next(
                feature for feature in batch if str(feature["feature_key"]) == internal_key
            )
            row["feature_key"] = internal_key
            row["model_feature_alias"] = alias
            row["example_system_ids"] = [
                str(example["system_id"]) for example in feature["top_examples"]
            ]
            row["example_content_sha256"] = [
                normalized_system_content_sha256(
                    examples_by_id[str(example["system_id"])]["name"],
                    examples_by_id[str(example["system_id"])]["background"],
                )
                for example in feature["top_examples"]
            ]
            row["request_example_content_sha256"] = request_example_hashes
            row["input_sha256"] = canonical_json_sha256(
                feature_description_payload([feature], examples_by_id)
            )
            row["model"] = backend.model
            row["prompt_version"] = PROMPT_VERSION
            row["instructions_sha256"] = prompt_hash(FEATURE_INSTRUCTIONS)
            row["schema_sha256"] = schema_hash(FEATURE_DESCRIPTION_BATCH_SCHEMA)
            row["request_sha256"] = request_sha256
            row["generation_git_commit"] = generation_commit
            row["generation_git_worktree_dirty"] = generation_dirty
            existing[internal_key] = row
        _save_index(output_path, existing)
        print(f"feature descriptions {len(existing)}/{len(features)}", flush=True)
    return {
        "required_features": len(features),
        "completed_features": len(features),
        "backend": backend.model,
        "output_sha256": sha256_file(output_path),
    }


def verify_pairs(
    *,
    candidate_path: Path,
    selection_path: Path,
    batch_plan_path: Path,
    mechanisms_path: Path,
    pair_evidence_path: Path,
    feature_descriptions_path: Path,
    output_path: Path,
    backend,
    limit_queries: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    candidates = selected_candidates(
        candidate_path,
        selection_path=selection_path,
        limit_queries=limit_queries,
    )
    generation_commit, generation_dirty = git_state()
    require_clean_real_backend(backend, dirty=generation_dirty)
    batch_plan = json.loads(batch_plan_path.read_text(encoding="utf-8"))
    if int(batch_plan["batch_size"]) != VERIFIER_BATCH_SIZE:
        raise ValueError("Verifier batch-plan size differs from implementation")
    batch_plan_sha256 = sha256_file(batch_plan_path)
    plan_by_query = {
        str(row["query_id"]): row["batches"] for row in batch_plan["queries"]
    }
    if len(plan_by_query) != len(batch_plan["queries"]):
        raise ValueError("Duplicate query IDs in verifier batch plan")
    mechanism_rows = load_jsonl(mechanisms_path)
    mechanisms = {str(row["system_id"]): row for row in mechanism_rows}
    if len(mechanisms) != len(mechanism_rows):
        raise ValueError("Duplicate mechanism graphs")
    description_rows = load_jsonl(feature_descriptions_path)
    descriptions = {str(row["feature_key"]): row for row in description_rows}
    if len(descriptions) != len(description_rows):
        raise ValueError("Duplicate feature descriptions")
    evidence_rows = load_jsonl(pair_evidence_path)
    pair_evidence = {
        (str(row["query_id"]), str(row["candidate_id"])): row["shared_features"]
        for row in evidence_rows
    }
    if len(pair_evidence) != len(evidence_rows):
        raise ValueError("Duplicate query-candidate feature evidence")

    required_system_ids: set[str] = set()
    required_evidence_pairs: set[tuple[str, str]] = set()
    for query in candidates:
        required_system_ids.add(str(query["query_system_id"]))
        required_system_ids.update(map(str, query["superpool"]))
        required_evidence_pairs.update(
            (str(query["query_id"]), str(candidate_id))
            for candidate_id in query["pools"]["sae_union_structure"]
        )
    missing_graphs = sorted(required_system_ids - set(mechanisms))
    if missing_graphs:
        raise ValueError(f"Missing mechanism graphs: {missing_graphs[:3]}")
    for system_id in required_system_ids:
        graph = mechanisms[system_id]
        if graph.get("model") != backend.model:
            raise ValueError("Mechanism model differs from verifier model")
        if graph.get("instructions_sha256") != prompt_hash(MECHANISM_INSTRUCTIONS):
            raise ValueError("Stale mechanism prompt provenance")
        if graph.get("schema_sha256") != schema_hash(MECHANISM_BATCH_SCHEMA):
            raise ValueError("Stale mechanism schema provenance")
        if backend.model == MODEL and (
            graph.get("generation_git_commit") != generation_commit
            or bool(graph.get("generation_git_worktree_dirty"))
        ):
            raise ValueError("Mechanism code provenance differs from verifier run")
    missing_pairs = sorted(required_evidence_pairs - set(pair_evidence))
    if missing_pairs:
        raise ValueError(f"Missing pair feature evidence: {missing_pairs[:3]}")
    for description in descriptions.values():
        if description.get("model") != backend.model:
            raise ValueError("Feature-description model differs from verifier model")
        if description.get("instructions_sha256") != prompt_hash(FEATURE_INSTRUCTIONS):
            raise ValueError("Stale feature-description prompt provenance")
        if description.get("schema_sha256") != schema_hash(
            FEATURE_DESCRIPTION_BATCH_SCHEMA
        ):
            raise ValueError("Stale feature-description schema provenance")
        if backend.model == MODEL and (
            description.get("generation_git_commit") != generation_commit
            or bool(description.get("generation_git_worktree_dirty"))
        ):
            raise ValueError("Feature-description code provenance differs from verifier run")

    existing_rows = [] if overwrite or not output_path.exists() else load_jsonl(output_path)
    existing = {
        (str(row["query_id"]), str(row["candidate_id"]), str(row["mode"])): row
        for row in existing_rows
    }
    if len(existing) != len(existing_rows):
        raise ValueError("Duplicate verifier predictions")

    required = sum(2 * len(query["superpool"]) for query in candidates)
    evidence_counts = {
        "pairs_scored": 0,
        "pairs_with_raw_evidence": 0,
        "pairs_with_usable_evidence": 0,
        "raw": 0,
        "incoherent": 0,
        "direct_example": 0,
        "usable": 0,
    }
    empty_evidence_normalizations = 0
    for query in candidates:
        query_id = str(query["query_id"])
        query_system_id = str(query["query_system_id"])
        query_graph = mechanisms[query_system_id]
        if query_id not in plan_by_query:
            raise ValueError(f"Missing verifier batch plan for {query_id}")
        planned_batches = plan_by_query[query_id]
        candidate_ids = [
            str(candidate_id)
            for batch in planned_batches
            for candidate_id in batch["candidate_ids"]
        ]
        if len(candidate_ids) != len(set(candidate_ids)) or set(candidate_ids) != set(
            map(str, query["superpool"])
        ):
            raise ValueError(f"Invalid verifier batch plan candidates for {query_id}")
        sae_candidates = set(map(str, query["pools"]["sae_union_structure"]))
        for batch_index, planned_batch in enumerate(planned_batches, start=1):
            batch_ids = list(map(str, planned_batch["candidate_ids"]))
            candidate_graphs = [
                (candidate, mechanisms[candidate]) for candidate in batch_ids
            ]
            structure_payload, aliases = verifier_payload(query_graph, candidate_graphs)
            if canonical_json_sha256(aliases) != planned_batch["alias_map_sha256"]:
                raise ValueError(f"Verifier alias plan mismatch for {query_id}")
            batch_id = canonical_json_sha256(
                {"query_id": query_id, "candidate_ids": batch_ids}
            )
            if batch_id != planned_batch["batch_id"]:
                raise ValueError(f"Verifier batch ID mismatch for {query_id}")
            feature_evidence: dict[str, list[dict[str, Any]]] = {}
            for candidate_id in batch_ids:
                shared = (
                    pair_evidence[(query_id, candidate_id)]
                    if candidate_id in sae_candidates
                    else []
                )
                usable, counts = _pair_feature_context(
                    shared,
                    descriptions,
                    query_content_sha256=str(query_graph["source_content_sha256"]),
                    candidate_content_sha256=str(
                        mechanisms[candidate_id]["source_content_sha256"]
                    ),
                )
                feature_evidence[candidate_id] = usable
                evidence_counts["pairs_scored"] += 1
                evidence_counts["pairs_with_raw_evidence"] += int(bool(shared))
                evidence_counts["pairs_with_usable_evidence"] += int(bool(usable))
                for key, value in counts.items():
                    evidence_counts[key] += value
            description_text = {
                key: str(row["description"])
                for key, row in descriptions.items()
                if bool(row["coherent"])
            }
            grounded_payload, grounded_aliases = verifier_payload(
                query_graph,
                candidate_graphs,
                feature_evidence=feature_evidence,
                feature_descriptions=description_text,
            )
            if aliases != grounded_aliases:
                raise AssertionError("Verifier aliases differ between paired modes")
            stripped_grounded = {
                "pairs": [
                    {**row, "shared_feature_evidence": []}
                    for row in grounded_payload["pairs"]
                ]
            }
            if stripped_grounded != structure_payload:
                raise AssertionError("Verifier batch context differs between paired modes")
            batch_context_sha256 = canonical_json_sha256(structure_payload)

            for mode, payload in (
                ("structure", structure_payload),
                ("feature_grounded", grounded_payload),
            ):
                empty_aliases = {
                    str(row["candidate_alias"])
                    for row in payload["pairs"]
                    if not row["shared_feature_evidence"]
                }
                request_sha256 = _stage_request_hash(
                    backend=backend,
                    schema_name="verdict_batch",
                    schema=VERDICT_BATCH_SCHEMA,
                    instructions=VERIFIER_INSTRUCTIONS,
                    payload=payload,
                    max_output_tokens=12000,
                )
                batch_existing = [
                    existing[(query_id, candidate_id, mode)]
                    for candidate_id in batch_ids
                    if (query_id, candidate_id, mode) in existing
                ]
                if _validate_fixed_batch(
                    batch_existing,
                    expected_count=len(batch_ids),
                    request_sha256=request_sha256,
                    backend_model=backend.model,
                    current_git_commit=generation_commit,
                    label=f"verifier {query_id}/{batch_index}/{mode}",
                ):
                    continue
                result = backend.complete(
                    schema_name="verdict_batch",
                    schema=VERDICT_BATCH_SCHEMA,
                    instructions=VERIFIER_INSTRUCTIONS,
                    payload=payload,
                    max_output_tokens=12000,
                    collection_key="candidates",
                    id_key="candidate_alias",
                    expected_ids=list(aliases),
                    output_validator=lambda output, aliases=aliases, empty_aliases=empty_aliases: validate_verdict_batch(
                        output,
                        aliases=aliases,
                        empty_evidence_aliases=empty_aliases,
                    ),
                )
                empty_evidence_normalizations += normalize_empty_evidence_verdicts(
                    result, empty_evidence_aliases=empty_aliases
                )
                graph_by_candidate = dict(candidate_graphs)
                payload_by_alias = {
                    str(row["candidate_alias"]): row for row in payload["pairs"]
                }
                for verdict in result["candidates"]:
                    candidate_id = aliases[str(verdict["candidate_alias"])]
                    pair_input = payload_by_alias[str(verdict["candidate_alias"])]
                    row = {
                        "query_id": query_id,
                        "candidate_id": candidate_id,
                        "mode": mode,
                        "pair_payload_sha256": pair_payload_hash(
                            query_graph, graph_by_candidate[candidate_id]
                        ),
                        "verifier_pair_input_sha256": canonical_json_sha256(pair_input),
                        "feature_evidence_count": len(
                            pair_input["shared_feature_evidence"]
                        ),
                        "batch_context_sha256": batch_context_sha256,
                        "batch_plan_sha256": batch_plan_sha256,
                        "batch_id": batch_id,
                        "alias_map_sha256": str(planned_batch["alias_map_sha256"]),
                        "batch_index": batch_index,
                        "model": backend.model,
                        "prompt_version": PROMPT_VERSION,
                        "instructions_sha256": prompt_hash(VERIFIER_INSTRUCTIONS),
                        "schema_sha256": schema_hash(VERDICT_BATCH_SCHEMA),
                        "request_sha256": request_sha256,
                        "generation_git_commit": generation_commit,
                        "generation_git_worktree_dirty": generation_dirty,
                        **dict(verdict),
                    }
                    row.pop("candidate_alias")
                    row["score"] = verdict_score(row)
                    existing[(query_id, candidate_id, mode)] = row
                ordered = [existing[key] for key in sorted(existing)]
                write_jsonl(output_path, ordered, overwrite=output_path.exists())
                print(f"verifier predictions {len(existing)}/{required}+", flush=True)

    selected_ids = {str(row["query_id"]) for row in candidates}
    completed = sum(key[0] in selected_ids for key in existing)
    if completed != required:
        raise ValueError(f"Expected {required} selected predictions, found {completed}")
    return {
        "queries": len(candidates),
        "required_predictions": required,
        "completed_predictions": completed,
        "backend": backend.model,
        "paired_superpool_batches": True,
        "feature_evidence_filter_counts": evidence_counts,
        "empty_evidence_field_normalizations": empty_evidence_normalizations,
        "output_sha256": sha256_file(output_path),
    }


def evaluate_command(
    *,
    candidate_path: Path,
    qrels_path: Path,
    selection_path: Path,
    predictions_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    predictions = load_jsonl(predictions_path)
    selected_ids = {str(row["query_id"]) for row in predictions}
    screen = json.loads(selection_path.read_text(encoding="utf-8"))
    screen_ids = set(map(str, screen["query_ids"]))
    if not selected_ids or not selected_ids.issubset(screen_ids):
        raise ValueError("Predictions are not a subset of the frozen verifier screen")
    candidates = [
        row for row in load_jsonl(candidate_path) if str(row["query_id"]) in selected_ids
    ]
    qrels = [row for row in load_jsonl(qrels_path) if str(row["query_id"]) in selected_ids]
    report, per_query = evaluate_rows(
        candidates,
        qrels,
        predictions,
    )
    artifact_paths = {
        "candidate_manifest": candidate_path,
        "screen_selection": selection_path,
        "verifier_batch_plan": output_dir / "verifier_batch_plan.json",
        "qrels_sidecar": qrels_path,
        "mechanisms": output_dir / "mechanisms.jsonl",
        "feature_catalog": output_dir / "feature_catalog.jsonl",
        "feature_descriptions": output_dir / "feature_descriptions.jsonl",
        "pair_feature_evidence": output_dir / "pair_feature_evidence.jsonl",
        "predictions": predictions_path,
        "protocol": DEFAULT_PROTOCOL,
        "lockfile": ROOT / "uv.lock",
    }
    missing_artifacts = [name for name, path in artifact_paths.items() if not path.exists()]
    if missing_artifacts:
        raise ValueError(f"Missing evaluation artifacts: {missing_artifacts}")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    prediction_commits = {
        str(row.get("generation_git_commit", "")) for row in predictions
    }
    prediction_dirty = any(
        bool(row.get("generation_git_worktree_dirty", True)) for row in predictions
    )
    report.update(
        {
            "artifact_sha256": {
                name: sha256_file(path) for name, path in artifact_paths.items()
            },
            "git_commit": git_commit,
            "git_worktree_dirty": git_dirty,
            "prediction_generation_git_commits": sorted(prediction_commits),
            "prediction_generation_worktree_dirty": prediction_dirty,
            "complete_frozen_screen": selected_ids == screen_ids,
        }
    )
    if report["evidentiary"] and (
        git_dirty or prediction_dirty or prediction_commits != {git_commit}
    ):
        raise ValueError("Real prediction code provenance is dirty, mixed, or stale")
    if report["evidentiary"] and not report["complete_frozen_screen"]:
        report["status"] = "live_smoke_non_evidentiary"
        report["evidentiary"] = False
    write_json(output_dir / "evaluation_report.json", report, overwrite=overwrite)
    write_jsonl(
        output_dir / "evaluation_per_query.jsonl", per_query, overwrite=overwrite
    )
    return report


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "candidates": output_dir / "candidate_manifest.jsonl",
        "qrels": output_dir / "qrels_sidecar.jsonl",
        "features": output_dir / "feature_catalog.jsonl",
        "evidence": output_dir / "pair_feature_evidence.jsonl",
        "screen": output_dir / "screen_selection.json",
        "batch_plan": output_dir / "verifier_batch_plan.json",
        "mechanisms": output_dir / "mechanisms.jsonl",
        "descriptions": output_dir / "feature_descriptions.jsonl",
        "predictions": output_dir / "verifier_predictions.jsonl",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "extract", "describe", "verify", "evaluate", "dry-run")
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=("openai", "fixture"), default="openai")
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    lowered_paths = f"{args.data} {args.output_dir}".lower()
    if "latent_choice" in lowered_paths or "test_frozen" in lowered_paths:
        raise ValueError("Structural Rescue must not read Latent Choice/test manifests")

    paths = _paths(args.output_dir)
    if args.command != "prepare":
        validate_prepared_bundle(args.output_dir, data_path=args.data)
    if args.command == "prepare":
        report = prepare_development(
            data_path=args.data, output_dir=args.output_dir, overwrite=args.overwrite
        )
    elif args.command == "evaluate":
        report = evaluate_command(
            candidate_path=paths["candidates"],
            qrels_path=paths["qrels"],
            selection_path=paths["screen"],
            predictions_path=paths["predictions"],
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    else:
        backend = make_backend(args.backend, args.output_dir)
        if args.command == "extract":
            report = extract_mechanisms(
                data_path=args.data,
                candidate_path=paths["candidates"],
                selection_path=paths["screen"],
                output_path=paths["mechanisms"],
                backend=backend,
                limit_queries=args.limit_queries,
                overwrite=args.overwrite,
            )
        elif args.command == "describe":
            report = describe_features(
                data_path=args.data,
                feature_catalog_path=paths["features"],
                output_path=paths["descriptions"],
                backend=backend,
                overwrite=args.overwrite,
            )
        elif args.command == "verify":
            report = verify_pairs(
                candidate_path=paths["candidates"],
                selection_path=paths["screen"],
                batch_plan_path=paths["batch_plan"],
                mechanisms_path=paths["mechanisms"],
                pair_evidence_path=paths["evidence"],
                feature_descriptions_path=paths["descriptions"],
                output_path=paths["predictions"],
                backend=backend,
                limit_queries=args.limit_queries,
                overwrite=args.overwrite,
            )
        else:
            if args.backend != "fixture":
                raise ValueError("dry-run requires --backend fixture")
            extract_report = extract_mechanisms(
                data_path=args.data,
                candidate_path=paths["candidates"],
                selection_path=paths["screen"],
                output_path=paths["mechanisms"],
                backend=backend,
                limit_queries=args.limit_queries or 2,
                overwrite=args.overwrite,
            )
            write_json(
                args.output_dir / "extract_report.json",
                extract_report,
                overwrite=(args.output_dir / "extract_report.json").exists(),
            )
            description_report = describe_features(
                data_path=args.data,
                feature_catalog_path=paths["features"],
                output_path=paths["descriptions"],
                backend=backend,
                overwrite=args.overwrite,
            )
            write_json(
                args.output_dir / "description_report.json",
                description_report,
                overwrite=(args.output_dir / "description_report.json").exists(),
            )
            verify_report = verify_pairs(
                candidate_path=paths["candidates"],
                selection_path=paths["screen"],
                batch_plan_path=paths["batch_plan"],
                mechanisms_path=paths["mechanisms"],
                pair_evidence_path=paths["evidence"],
                feature_descriptions_path=paths["descriptions"],
                output_path=paths["predictions"],
                backend=backend,
                limit_queries=args.limit_queries or 2,
                overwrite=args.overwrite,
            )
            write_json(
                args.output_dir / "verify_report.json",
                verify_report,
                overwrite=(args.output_dir / "verify_report.json").exists(),
            )
            report = evaluate_command(
                candidate_path=paths["candidates"],
                qrels_path=paths["qrels"],
                selection_path=paths["screen"],
                predictions_path=paths["predictions"],
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
    if args.command in {"extract", "describe", "verify"}:
        report_path = args.output_dir / f"{args.command}_report.json"
        write_json(report_path, report, overwrite=report_path.exists())
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
