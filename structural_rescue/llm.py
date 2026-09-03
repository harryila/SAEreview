"""Strict structured-output prompts and the resumable OpenAI backend."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .core import canonical_json_sha256, write_json


MODEL = "gpt-5.4-mini-2026-03-17"
PROMPT_VERSION = "structural-rescue-v3"
REASONING_EFFORT = "none"
TEMPERATURE = 0.0
MAX_ATTEMPTS = 3
VERIFIER_EVIDENCE_MODES = (
    "structure",
    "activation_only",
    "aligned_description",
    "shuffled_description",
)

MECHANISM_INSTRUCTIONS = """You extract a compact causal mechanism graph for each
system independently. Use only the supplied name and background. Do not infer a
comparison, analogy partner, retrieval outcome, or domain label. Prefer functional
roles and causal/dynamical relations over topic words. Keep each list short and set
unknown details to an empty list rather than guessing. Return the strict schema."""

FEATURE_INSTRUCTIONS = """You describe sparse-feature semantics from independent
top-activating system examples. Produce one short, literal description of the common
mechanistic pattern. Do not mention analogy pairs, relevance, retrieval, ranking, or
whether a feature is useful. If the examples lack a coherent pattern, say
'no coherent mechanistic interpretation'. Return the strict schema."""

VERIFIER_INSTRUCTIONS = """You are a blinded relational verifier. Score every
query-candidate pair independently for structural analogy, not topical relatedness.
Reward matching functional roles, causal organization, dynamics, and constraints.
Flag lexical resemblance, same-domain resemblance without the same mechanism, and
accidental feature overlap. Feature evidence is supporting context only and must not
override contradictory mechanism graphs. Identify concise mappings and where the
analogy breaks. You do not know retrieval source, rank, gold labels, or study arm.
When shared_feature_evidence is empty, feature_support must be 0 and
accidental_feature_overlap must be false.
Return the strict schema."""


def _strict_object(properties: dict[str, Any], required: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


STRING_ARRAY = {"type": "array", "items": {"type": "string"}, "maxItems": 6}

MECHANISM_ITEM_SCHEMA = _strict_object(
    {
        "system_id": {"type": "string"},
        "summary": {"type": "string"},
        "entities_and_roles": {
            "type": "array",
            "maxItems": 8,
            "items": _strict_object(
                {"entity": {"type": "string"}, "role": {"type": "string"}},
                ["entity", "role"],
            ),
        },
        "causal_relations": {
            "type": "array",
            "maxItems": 8,
            "items": _strict_object(
                {
                    "source": {"type": "string"},
                    "relation": {"type": "string"},
                    "target": {"type": "string"},
                },
                ["source", "relation", "target"],
            ),
        },
        "dynamics": STRING_ARRAY,
        "constraints": STRING_ARRAY,
        "boundary_conditions": STRING_ARRAY,
    },
    [
        "system_id",
        "summary",
        "entities_and_roles",
        "causal_relations",
        "dynamics",
        "constraints",
        "boundary_conditions",
    ],
)

MECHANISM_BATCH_SCHEMA = _strict_object(
    {"systems": {"type": "array", "items": MECHANISM_ITEM_SCHEMA}}, ["systems"]
)

FEATURE_DESCRIPTION_ITEM_SCHEMA = _strict_object(
    {
        "feature_key": {"type": "string"},
        "description": {"type": "string"},
        "coherent": {"type": "boolean"},
    },
    ["feature_key", "description", "coherent"],
)

FEATURE_DESCRIPTION_BATCH_SCHEMA = _strict_object(
    {
        "features": {
            "type": "array",
            "items": FEATURE_DESCRIPTION_ITEM_SCHEMA,
        }
    },
    ["features"],
)

VERDICT_ITEM_SCHEMA = _strict_object(
    {
        "candidate_alias": {"type": "string"},
        "role_alignment": {"type": "integer", "minimum": 0, "maximum": 4},
        "causal_alignment": {"type": "integer", "minimum": 0, "maximum": 4},
        "dynamics_alignment": {"type": "integer", "minimum": 0, "maximum": 4},
        "constraint_alignment": {"type": "integer", "minimum": 0, "maximum": 4},
        "feature_support": {"type": "integer", "minimum": 0, "maximum": 4},
        "lexical_only": {"type": "boolean"},
        "same_domain_only": {"type": "boolean"},
        "accidental_feature_overlap": {"type": "boolean"},
        "break_severity": {"type": "integer", "minimum": 0, "maximum": 4},
        "mechanism_mappings": STRING_ARRAY,
        "analogy_breakpoints": STRING_ARRAY,
    },
    [
        "candidate_alias",
        "role_alignment",
        "causal_alignment",
        "dynamics_alignment",
        "constraint_alignment",
        "feature_support",
        "lexical_only",
        "same_domain_only",
        "accidental_feature_overlap",
        "break_severity",
        "mechanism_mappings",
        "analogy_breakpoints",
    ],
)

VERDICT_BATCH_SCHEMA = _strict_object(
    {"candidates": {"type": "array", "items": VERDICT_ITEM_SCHEMA}}, ["candidates"]
)


def schema_hash(schema: Mapping[str, Any]) -> str:
    return canonical_json_sha256(schema)


def prompt_hash(instructions: str) -> str:
    return hashlib.sha256(
        f"{PROMPT_VERSION}\n{instructions}".encode("utf-8")
    ).hexdigest()


def structured_request_hash(
    *,
    model: str,
    schema_name: str,
    schema: Mapping[str, Any],
    instructions: str,
    payload: Mapping[str, Any],
    max_output_tokens: int,
) -> str:
    """Bind a result to every model-visible input and decoding setting."""

    return canonical_json_sha256(
        {
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "instructions_sha256": prompt_hash(instructions),
            "schema_name": schema_name,
            "schema_sha256": schema_hash(schema),
            "payload": payload,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": REASONING_EFFORT,
            "temperature": TEMPERATURE,
        }
    )


def response_usage_payload(response: Any) -> dict[str, Any] | None:
    """Return JSON-safe token accounting when the SDK supplies it."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        value = dict(usage)
    else:
        model_dump = getattr(usage, "model_dump", None)
        if not callable(model_dump):
            return None
        value = model_dump(mode="json")
    if not isinstance(value, dict):
        return None
    # Usage accounting must never turn a valid paid response into a failed run.
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None
    return value


def opaque_feature_alias(feature_key: str) -> str:
    try:
        namespace, raw_id = feature_key.split(":", 1)
        feature_id = int(raw_id)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid internal feature key {feature_key}") from exc
    representation = {"cslg": "R1", "astroph": "R2"}.get(namespace)
    if representation is None or feature_id < 0:
        raise ValueError(f"Invalid internal feature namespace {feature_key}")
    return f"{representation}:F{feature_id:04d}"


def validate_exact_ids(
    output: Mapping[str, Any], *, collection_key: str, id_key: str, expected: Iterable[str]
) -> None:
    expected_ids = list(expected)
    rows = output.get(collection_key)
    if not isinstance(rows, list):
        raise ValueError(f"Structured output is missing {collection_key}")
    observed = [row.get(id_key) for row in rows if isinstance(row, dict)]
    if len(observed) != len(rows) or sorted(observed) != sorted(expected_ids):
        raise ValueError(
            f"Structured output IDs differ: expected {sorted(expected_ids)}, "
            f"observed {sorted(str(value) for value in observed)}"
        )


def mechanism_payload(systems: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    allowed = {"system_id", "name", "background"}
    clean: list[dict[str, str]] = []
    for system in systems:
        if set(system) != allowed:
            raise ValueError(
                f"Mechanism inputs must contain exactly {sorted(allowed)}; "
                f"found {sorted(system)}"
            )
        clean.append({key: str(system[key]) for key in ("system_id", "name", "background")})
    return {"systems": clean}


def feature_description_payload(
    features: Sequence[Mapping[str, Any]],
    examples_by_system_id: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    clean: list[dict[str, Any]] = []
    for feature in features:
        feature_key = opaque_feature_alias(str(feature["feature_key"]))
        examples: list[dict[str, str]] = []
        for rank, example in enumerate(feature["top_examples"], start=1):
            system = examples_by_system_id[str(example["system_id"])]
            if set(system) != {"system_id", "name", "background"}:
                raise ValueError("Feature examples must use independent system payloads")
            examples.append(
                {
                    "example_alias": f"E{rank:02d}",
                    "name": str(system["name"]),
                    "background": str(system["background"]),
                }
            )
        clean.append({"feature_key": feature_key, "top_activating_examples": examples})
    return {"features": clean}


def public_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "summary",
        "entities_and_roles",
        "causal_relations",
        "dynamics",
        "constraints",
        "boundary_conditions",
    }
    missing = allowed - set(graph)
    if missing:
        raise ValueError(f"Mechanism graph missing fields: {sorted(missing)}")
    return {key: graph[key] for key in sorted(allowed)}


def pair_payload_hash(query_graph: Mapping[str, Any], candidate_graph: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {"query": public_graph(query_graph), "candidate": public_graph(candidate_graph)}
    )


def verifier_payload(
    query_graph: Mapping[str, Any],
    candidates: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    evidence_mode: str = "structure",
    feature_evidence: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    feature_descriptions: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build a blinded batch of independent pair judgments.

    Candidate IDs are replaced with local aliases. Retrieval arm, rank, qrels,
    and rescue status are deliberately not accepted by this API.
    """

    if evidence_mode not in VERIFIER_EVIDENCE_MODES:
        raise ValueError(f"Unknown verifier evidence mode: {evidence_mode}")
    if evidence_mode == "structure":
        if feature_evidence is not None or feature_descriptions is not None:
            raise ValueError("Structure mode cannot receive feature evidence")
    elif feature_evidence is None:
        raise ValueError(f"{evidence_mode} mode requires feature evidence")
    if evidence_mode == "activation_only" and feature_descriptions is not None:
        raise ValueError("Activation-only mode must omit feature descriptions")
    if evidence_mode in {"aligned_description", "shuffled_description"} and (
        feature_descriptions is None
    ):
        raise ValueError(f"{evidence_mode} mode requires feature descriptions")

    payload_rows: list[dict[str, Any]] = []
    alias_to_candidate: dict[str, str] = {}
    for index, (candidate_id, candidate_graph) in enumerate(candidates, start=1):
        alias = f"C{index:03d}"
        alias_to_candidate[alias] = candidate_id
        evidence_rows: list[dict[str, Any]] = []
        if evidence_mode != "structure":
            assert feature_evidence is not None
            for evidence in feature_evidence.get(candidate_id, []):
                internal_feature_key = str(evidence["feature_key"])
                feature_key = opaque_feature_alias(internal_feature_key)
                evidence_row = {
                    "feature_key": feature_key,
                    "query_activation_percentile": float(
                        evidence["query_activation_percentile"]
                    ),
                    "candidate_activation_percentile": float(
                        evidence["candidate_activation_percentile"]
                    ),
                }
                if evidence_mode in {
                    "aligned_description",
                    "shuffled_description",
                }:
                    assert feature_descriptions is not None
                    if internal_feature_key not in feature_descriptions:
                        raise ValueError(
                            f"Missing feature description for {internal_feature_key}"
                        )
                    description = str(feature_descriptions[internal_feature_key])
                    if not description.strip():
                        raise ValueError(
                            f"Empty feature description for {internal_feature_key}"
                        )
                    evidence_row["description"] = description
                evidence_rows.append(evidence_row)
        payload_rows.append(
            {
                "candidate_alias": alias,
                "query_mechanism": public_graph(query_graph),
                "candidate_mechanism": public_graph(candidate_graph),
                "shared_feature_evidence": evidence_rows,
            }
        )
    return {"pairs": payload_rows}, alias_to_candidate


def strip_verifier_feature_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with every evidence list emptied for paired-mode assertions."""

    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("Verifier payload is missing pairs")
    stripped: list[dict[str, Any]] = []
    for row in pairs:
        if not isinstance(row, Mapping) or "shared_feature_evidence" not in row:
            raise ValueError("Verifier payload pair is missing feature evidence")
        stripped.append({**dict(row), "shared_feature_evidence": []})
    return {"pairs": stripped}


def strip_verifier_feature_descriptions(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy with only description text removed from feature evidence."""

    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("Verifier payload is missing pairs")
    stripped: list[dict[str, Any]] = []
    for row in pairs:
        if not isinstance(row, Mapping):
            raise ValueError("Verifier payload pairs must be objects")
        evidence = row.get("shared_feature_evidence")
        if not isinstance(evidence, list):
            raise ValueError("Verifier payload pair is missing feature evidence")
        stripped.append(
            {
                **dict(row),
                "shared_feature_evidence": [
                    {
                        key: value
                        for key, value in dict(item).items()
                        if key != "description"
                    }
                    for item in evidence
                ],
            }
        )
    return {"pairs": stripped}


def verdict_score(row: Mapping[str, Any]) -> int:
    positive = (
        2 * int(row["role_alignment"])
        + 3 * int(row["causal_alignment"])
        + 3 * int(row["dynamics_alignment"])
        + 2 * int(row["constraint_alignment"])
        + int(row["feature_support"])
    )
    penalties = (
        4 * int(bool(row["lexical_only"]))
        + 4 * int(bool(row["same_domain_only"]))
        + 4 * int(bool(row["accidental_feature_overlap"]))
        + 2 * int(row["break_severity"])
    )
    return positive - penalties


class OpenAIStructuredBackend:
    """Small Responses API wrapper with strict schemas and request caching."""

    def __init__(
        self,
        *,
        model: str = MODEL,
        cache_dir: Path,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the real backend")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.cache_dir = cache_dir
        self.max_attempts = max_attempts

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
        output_validator: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        request = {
            "request_sha256": structured_request_hash(
                model=self.model,
                schema_name=schema_name,
                schema=schema,
                instructions=instructions,
                payload=payload,
                max_output_tokens=max_output_tokens,
            )
        }
        request_hash = request["request_sha256"]
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("request_sha256") != request_hash:
                raise ValueError(f"Cache-key mismatch in {cache_path}")
            output = dict(cached["output"])
            if expected_ids is not None:
                if collection_key is None or id_key is None:
                    raise ValueError("ID validation keys are required")
                validate_exact_ids(
                    output,
                    collection_key=collection_key,
                    id_key=id_key,
                    expected=expected_ids,
                )
            if output_validator is not None:
                output_validator(output)
            return output

        error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "strict": True,
                            "schema": dict(schema),
                        }
                    },
                    max_output_tokens=max_output_tokens,
                    reasoning={"effort": REASONING_EFFORT},
                    temperature=TEMPERATURE,
                    store=False,
                )
                output = json.loads(response.output_text)
                if expected_ids is not None:
                    if collection_key is None or id_key is None:
                        raise ValueError("ID validation keys are required")
                    validate_exact_ids(
                        output,
                        collection_key=collection_key,
                        id_key=id_key,
                        expected=expected_ids,
                    )
                if output_validator is not None:
                    output_validator(output)
                receipt = {
                    "request_sha256": request_hash,
                    "model": self.model,
                    "response_id": response.id,
                    "schema_sha256": schema_hash(schema),
                    "instructions_sha256": prompt_hash(instructions),
                    "output": output,
                }
                usage = response_usage_payload(response)
                if usage is not None:
                    receipt["usage"] = usage
                write_json(cache_path, receipt)
                return output
            except Exception as exc:  # API errors differ by SDK/version.
                error = exc
                if attempt < self.max_attempts:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(
            f"Structured response failed after {self.max_attempts} attempts"
        ) from error
