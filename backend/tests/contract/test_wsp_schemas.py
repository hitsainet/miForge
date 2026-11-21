"""miForge's three owned WSP contracts (feature 016, FTASKS 1.1; FTDD §3.1-3.3).

``test_gate_contracts.py`` already checks every schema in the directory is a valid draft 2020-12
schema and closes every object with declared properties. This file pins what is specific to WSP:

- the three files exist and are draft 2020-12 documents with the expected ``kind`` constants;
- closure holds here too, **including** objects reached only through ``anyOf`` (nullable) and
  ``$defs``, so a nullable sub-object cannot be the one open door;
- ``HeldoutSet`` is exactly the five fields miWorkbench sends to its scorer, so the bundle has no
  field that can carry held-out rows, text or activations (FR-016.1);
- the run schema's ``request`` (``FrozenRequest``) names held-out sets but holds no held-out field;
- real-shaped documents validate, and an invented key, a held-out row smuggled into ``HeldoutSet``
  and a ``request_sha256`` that is not hex64 are each refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"

WSP_SCHEMAS = {
    "miforge-wsp-bundle-v1.json": "miforge.wsp-bundle/v1",
    "miforge-wsp-patch-v1.json": "miforge.wsp-patch/v1",
    "miforge-wsp-run-v1.json": "miforge.wsp-run/v1",
}

#: FTDD §3.1: exactly the five fields 044 §5.7.1 sends. ``positives``, ``truncated_rows`` and
#: ``min_rows`` were removed on purpose (R1B-X6); a row, text or activation field never belongs.
HELDOUT_SET_FIELDS = {"name", "view_id", "role", "rows", "row_ids_sha256"}

#: Words that would mean a held-out field reached the frozen request.
HELDOUT_WORDS = ("heldout", "held_out", "eval_rows", "labels", "activations")

HEX = "a" * 64


def _load(filename: str) -> dict[str, Any]:
    path = SCHEMA_DIR / filename
    assert path.is_file(), f"{path} is missing"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"{path} is empty"
    loaded: dict[str, Any] = json.loads(text)
    return loaded


def _validator(filename: str) -> Draft202012Validator:
    return Draft202012Validator(_load(filename))


def _walk(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        if "properties" in node:
            found.append((path, node))
        for key, value in node.items():
            found.extend(_walk(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk(value, f"{path}[{index}]"))
    return found


@pytest.mark.parametrize("filename", sorted(WSP_SCHEMAS))
def test_schema_is_draft_2020_12_with_its_kind(filename: str) -> None:
    schema = _load(filename)
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(schema)
    assert schema["properties"]["kind"]["const"] == WSP_SCHEMAS[filename]


@pytest.mark.parametrize("filename", sorted(WSP_SCHEMAS))
def test_every_object_is_closed_and_requires_every_field(filename: str) -> None:
    """Every object closes, and every field is required (nullable where a writer may say none).

    README property 2: an absent key is an omission, ``null`` is a statement. FTDD §3's outlines
    mark every field ``req``; none is ``opt``.
    """
    objects = _walk(_load(filename))
    assert objects, f"{filename}: no objects found, the walk proves nothing"
    open_objects = [p for p, n in objects if n.get("additionalProperties") is not False]
    assert not open_objects, f"{filename}: open objects {open_objects}"
    allowed = OPTIONAL_FIELDS.get(filename, {})
    partial = [
        (p, sorted(set(n["properties"]) - set(n.get("required", []))))
        for p, n in objects
        if set(n["properties"]) - set(n.get("required", [])) != allowed.get(p, set())
    ]
    assert not partial, f"{filename}: objects with an optional field {partial}"


#: The THREE optional fields in the WSP contracts, with their reason (2026-11-21, reviews
#: ``018_thinking_2026-11-21.md`` and ``018_paging_2026-11-21.md``): ``thinking_tokens`` exists
#: only with ``thinking: on``, ``output_cap`` and ``paging`` only when off, and as optional keys
#: every request frozen before them, and every request at the defaults, stays valid under both
#: this copy and a vendored copy from before them. Anything else optional is red.
OPTIONAL_FIELDS: dict[str, dict[str, set[str]]] = {
    "miforge-wsp-run-v1.json": {
        "$.$defs.FrozenRequest": {"thinking_tokens", "output_cap", "paging"}
    },
}


def test_thinking_tokens_is_bounded_and_thinking_admits_on() -> None:
    request = _load("miforge-wsp-run-v1.json")["$defs"]["FrozenRequest"]
    assert request["properties"]["thinking_tokens"]["type"] == "integer"
    assert request["properties"]["thinking_tokens"]["minimum"] == 1
    assert "on" in request["properties"]["thinking"]["anyOf"][0]["enum"]
    assert "thinking_tokens" not in request["required"]


@pytest.mark.parametrize("filename", sorted(WSP_SCHEMAS))
def test_an_object_without_declared_fields_is_a_typed_map(filename: str) -> None:
    """F-2: a map (``truncated_fired``, label -> count) declares no properties, so the closure
    walk above never sees it. It must still type its values: ``additionalProperties`` a schema,
    never ``true`` or absent, which would be an open door."""

    def walk(node: Any, path: str = "$") -> list[tuple[str, Any]]:
        found: list[tuple[str, Any]] = []
        if isinstance(node, dict):
            kind = node.get("type")
            if (kind == "object" or (isinstance(kind, list) and "object" in kind)) and (
                "properties" not in node
            ):
                found.append((path, node.get("additionalProperties")))
            for key, value in node.items():
                found.extend(walk(value, f"{path}.{key}"))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                found.extend(walk(value, f"{path}[{index}]"))
        return found

    maps = walk(_load(filename))
    untyped = [p for p, extra in maps if not (isinstance(extra, dict) and extra.get("type"))]
    assert not untyped, f"{filename}: objects with untyped values {untyped}"
    if filename == "miforge-wsp-run-v1.json":
        assert [p for p, _ in maps] == ["$.$defs.SetVerdict.properties.truncated_fired"]


def test_heldout_set_is_exactly_the_five_scorer_fields() -> None:
    heldout = _load("miforge-wsp-bundle-v1.json")["$defs"]["HeldoutSet"]
    assert set(heldout["properties"]) == HELDOUT_SET_FIELDS


def test_frozen_request_holds_no_heldout_field() -> None:
    run = _load("miforge-wsp-run-v1.json")
    request = run["$defs"]["FrozenRequest"]
    assert run["properties"]["request"] == {"$ref": "#/$defs/FrozenRequest"}
    # Every property name reachable from FrozenRequest, following $refs transitively (through
    # anyOf too, so a nullable sub-object is not a blind spot).
    names: list[str] = []
    seen: set[str] = set()
    pending: list[Any] = [request]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            names.extend(node.get("properties", {}) if "properties" in node else [])
            target = node.get("$ref")
            if isinstance(target, str) and target not in seen:
                seen.add(target)
                pending.append(run["$defs"][target.rsplit("/", 1)[1]])
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    assert "budgets" in names and "max_changed_entries" in names, "the walk missed $refs"
    leaked = [n for n in names if any(w in n for w in HELDOUT_WORDS)]
    assert not leaked, f"FrozenRequest carries held-out fields: {leaked}"
    # It names the primary sets, and nothing more about them.
    assert "primary_sets" in request["properties"]


def test_caller_ref_is_outside_the_digested_request() -> None:
    """R4A-8: caller_ref moves no digest, so it sits beside request, never inside it."""
    run = _load("miforge-wsp-run-v1.json")
    assert "caller_ref" in run["properties"]
    assert "caller_ref" not in run["$defs"]["FrozenRequest"]["properties"]


# --------------------------------------------------------------------------------------
# Real-shaped documents
# --------------------------------------------------------------------------------------


def bundle_document() -> dict[str, Any]:
    return {
        "kind": "miforge.wsp-bundle/v1",
        "bundle_kind": "linear_probe",
        "bundle_id": "wsb_abc123",
        "created_at": "2026-11-16T05:00:00Z",
        "identity": {
            "owner": "miworkbench",
            "artifact_kind": "probe",
            "artifact_id": "probe-1",
            "detector_digest": None,
            "model": "google/gemma-2-2b",
            "layer": 12,
            "run_id": "ptr-1",
        },
        "files": [
            {"name": "heads.safetensors", "role": "heads", "sha256": HEX, "bytes": 1024},
            {"name": "practice.npy", "role": "practice_activations", "sha256": HEX, "bytes": 4096},
        ],
        "tensors": [
            {"name": "weight", "shape": [8], "dtype": "float32", "sha256": HEX, "editable": True},
            {"name": "bias", "shape": [1], "dtype": "float64", "sha256": HEX, "editable": True},
        ],
        "function": {
            "kind": "linear_probe/v1",
            "description": "mean-pooled linear probe",
            "linear_probe": {
                "rule": "mean",
                "params": {"tau": None, "window": None},
                "layer": 12,
                "scope": "all",
                "skip_template_prefix": False,
            },
            "sae_fn": None,
            "parity": {"rows": list(range(8)), "scores": [0.5] * 8, "tolerance": 1e-5},
        },
        "practice": {
            "format": "token_activations",
            "dtype": "float32",
            "rows": 40,
            "tokens": 400,
            "d_model": 8,
            "view_id": "view-1",
            "row_ids_sha256": HEX,
            "label_counts": {"positive": 20, "negative": 20},
            "validation_rows": 8,
            "truncated_rows": 0,
        },
        "heldout_ref": {
            "owner": "miworkbench",
            "sets": [
                {
                    "name": "eval",
                    "view_id": "v2",
                    "role": "eval",
                    "rows": 100,
                    "row_ids_sha256": HEX,
                }
            ],
        },
        "baselines_available": ["lda"],
        "operating_point": {
            "target_fpr": 0.01,
            "threshold": 0.3,
            "threshold_source": "calibration",
        },
        "sensitivity": {
            "value": "not_sensitive",
            "basis": [{"source": "operator_mark", "ref": None, "value": "not_sensitive"}],
        },
        "sae": None,
        "bundle_sha256": HEX,
    }


def run_document() -> dict[str, Any]:
    return {
        "kind": "miforge.wsp-run/v1",
        "run_id": "wsp_0123456789ab",
        "created_at": "2026-11-16T05:00:00Z",
        "request": {
            "kind": "repair",
            "goal": None,
            "mode": "session",
            "task": "fix the false alarms",
            "bundle": {"owner": "miworkbench", "bundle_id": "wsb_abc123", "bundle_sha256": HEX},
            "sensitivity_declared": "not_sensitive",
            "parent": {"kind": "bundle", "ref": HEX},
            "chain": False,
            "keep_weights": False,
            "provider": {"id": "weightspace", "version": "0.3.0"},
            "endpoint": {"id": "ep-1", "locality": "local", "model": "qwen"},
            "thinking": None,
            "budgets": {
                "forward_passes": 0,
                "tokens": 1000,
                "cost_usd": None,
                "time_s": 600,
                "max_steps": 10,
            },
            "change_limits": {
                "max_changed_entries": 4,
                "allowed_tensors": None,
                "frozen_tensors": ["mean", "std"],
            },
            "observation_tools": [],
            "primary_sets": ["eval"],
            "program_sha256": None,
            "support_set": None,
            "of_run_id": None,
        },
        "request_sha256": HEX,
        "sensitivity": {
            "value": "not_sensitive",
            "effective": "not_sensitive",
            "source": "operator_mark",
        },
        "approval": None,
        "status": "queued",
        "stop_reason": None,
        "outcome": None,
        "failure": None,
        "usage": {
            "metered_by": "wsp_gateway",
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0,
            "self_reported": None,
        },
        "forward_passes_used": None,
        "transcript_sha256": None,
        "result_sha256": None,
        "patch_sha256": None,
        "content_sha256": None,
        "boot_id": None,
        "candidate_id": None,
        "program_sha256": None,
        "program_reruns": [],
        "caller_ref": "exp-7",
        "replay": None,
        "verdict": None,
        "run_sha256": None,
        "lineage": {
            "parent": {"kind": "bundle", "ref": HEX},
            "bundle_sha256": HEX,
            "chain": [],
            "provider": "weightspace",
            "provider_version": "0.3.0",
        },
    }


def patch_document() -> dict[str, Any]:
    return {
        "kind": "miforge.wsp-patch/v1",
        "format_version": 1,
        "bundle_base_sha256": HEX,
        "parent": {"kind": "bundle", "ref": HEX, "patch_sha256": None},
        "parent_patch_sha256": None,
        "arrays_file": "edit.npz",
        "arrays_sha256": HEX,
        "tensors": [
            {
                "index": 0,
                "name": "weight",
                "dtype": "float32",
                "shape": [8],
                "edited_shape": [8],
                "form": "sparse",
                "rank": None,
                "append": None,
                "base_sha256": HEX,
                "edited_sha256": HEX,
                "changed_entries": 1,
                "l2_delta": 0.5,
                "max_abs_delta": 0.5,
            }
        ],
        "changed_entries_total": 1,
        "created_by": {
            "provider": "weightspace",
            "provider_version": "0.3.0",
            "run_id": "wsp_0123456789ab",
        },
        "created_at": "2026-11-16T05:00:00Z",
        "content_sha256": HEX,
        "patch_sha256": HEX,
    }


DOCUMENTS = {
    "miforge-wsp-bundle-v1.json": bundle_document,
    "miforge-wsp-patch-v1.json": patch_document,
    "miforge-wsp-run-v1.json": run_document,
}


@pytest.mark.parametrize("filename", sorted(DOCUMENTS))
def test_a_real_shaped_document_validates(filename: str) -> None:
    errors = list(_validator(filename).iter_errors(DOCUMENTS[filename]()))
    assert not errors, [e.message for e in errors]


@pytest.mark.parametrize("filename", sorted(DOCUMENTS))
def test_an_invented_key_is_refused(filename: str) -> None:
    doc = DOCUMENTS[filename]()
    doc["invented"] = 1
    assert not _validator(filename).is_valid(doc)


def test_a_heldout_row_smuggled_into_a_set_is_refused() -> None:
    doc = bundle_document()
    doc["heldout_ref"]["sets"][0]["rows_text"] = ["a private sentence"]
    assert not _validator("miforge-wsp-bundle-v1.json").is_valid(doc)


def test_a_nested_nullable_object_is_closed_too() -> None:
    doc = bundle_document()
    assert doc["operating_point"] is not None
    doc["operating_point"]["extra"] = True
    assert not _validator("miforge-wsp-bundle-v1.json").is_valid(doc)


def test_a_patch_that_changes_nothing_is_refused() -> None:
    doc = patch_document()
    doc["changed_entries_total"] = 0
    assert not _validator("miforge-wsp-patch-v1.json").is_valid(doc)


@pytest.mark.parametrize("bad", ["A" * 64, "a" * 63, "g" * 64])
def test_digests_are_lowercase_hex64(bad: str) -> None:
    doc = run_document()
    doc["request_sha256"] = bad
    assert not _validator("miforge-wsp-run-v1.json").is_valid(doc)


# --------------------------------------------------------------------------------------
# F-2 (2026-11-19): SetVerdict.truncated_fired is a map, label -> count
# --------------------------------------------------------------------------------------


def _set_verdict_validator() -> Draft202012Validator:
    run = _load("miforge-wsp-run-v1.json")
    return Draft202012Validator({"$defs": run["$defs"], "$ref": "#/$defs/SetVerdict"})


def set_verdict(truncated_fired: Any) -> dict[str, Any]:
    return {
        "name": "eval",
        "role": "eval",
        "primary": True,
        "rows": 409,
        "positives": 200,
        "negatives": 200,
        "excluded_truncated": 9,
        "truncated_fired": truncated_fired,
        "heldout_uses": 1,
        "heads": [],
        "deltas": [],
        "value": "improved",
    }


EVERY_HEAD = {"parent": 1, "candidate": 2, "control": 3, "baseline:lda": 0}


def test_truncated_fired_takes_every_heads_count() -> None:
    """The keys are the head labels the scorer answered (parent, candidate, control,
    baseline:<kind>); each value is that head's count."""
    errors = [e.message for e in _set_verdict_validator().iter_errors(set_verdict(EVERY_HEAD))]
    assert not errors, errors
    doc = run_document()
    doc["verdict"] = {
        "verdict_id": "wspv_0123456789ab",
        "value": "improved",
        "reasons": [],
        "request_sha256": HEX,
        "statistic": "paired_bootstrap",
        "B": 1000,
        "seed": 7,
        "ci": 0.95,
        "control_seed": 7,
        "control_figures": {
            "candidate": {"changed_entries": 1, "l2_delta": 0.25},
            "control_post_cast": {"changed_entries": 1, "l2_delta": 0.25},
        },
        "primary_metric": "auroc",
        "scorer": {
            "owner": "miworkbench",
            "contract": "miworkbench-wsp-scoring-v1",
            "call_id": "wsc_x",
            "metrics_agree": True,
        },
        "guarded_comparisons": 0,
        "completeness": {"ok": True, "missing": []},
        "per_set": [set_verdict(EVERY_HEAD)],
        "evaluated_at": "2026-11-19T05:00:00Z",
    }
    errors = [e.message for e in _validator("miforge-wsp-run-v1.json").iter_errors(doc)]
    assert not errors, errors


def test_the_pre_f2_shape_still_validates() -> None:
    """The change only loosens: the old one-literal-key object is a map with one key."""
    assert _set_verdict_validator().is_valid(set_verdict({"label": 4}))
    assert _set_verdict_validator().is_valid(set_verdict({}))


@pytest.mark.parametrize("count", [-1, 1.5, "2", True, None])
def test_a_count_that_is_not_a_non_negative_integer_is_refused(count: Any) -> None:
    assert not _set_verdict_validator().is_valid(set_verdict({**EVERY_HEAD, "control": count}))


def test_truncated_fired_stays_required_and_an_object() -> None:
    doc = set_verdict(EVERY_HEAD)
    del doc["truncated_fired"]
    assert not _set_verdict_validator().is_valid(doc)
    assert not _set_verdict_validator().is_valid(set_verdict([["candidate", 2]]))
    assert not _set_verdict_validator().is_valid(set_verdict(3))
