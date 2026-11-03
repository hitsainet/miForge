"""The contract gate's four schemas, exercised against real documents  (task 3.0).

These are the shared interfaces of PPRD section 12. The argument for settling them before
any parallel feature work is not stylistic: fan agents out first and **each one invents
its own batch record**, which is the drift that has already cost this project two sessions
and produced several cross-document claims their own cited sources did not support.

So this file does three things a reading pass cannot:

1. **Checks each schema is a valid JSON Schema**, not merely parseable JSON.
2. **Asserts every object closes** (``additionalProperties: false``). R-02.35 says a
   manifest's fields are *exactly* the schema's; before 2026-10-13 neither vendored schema
   closed its objects, so an invented key validated clean and that rule was enforced by
   nothing. Five features confirmed it independently by running it.
3. **Validates real documents through the cross-file ``$ref``s**, so "one definition,
   referenced" is a fact rather than an intention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"

#: The four schemas the contract gate added. The three manifests are covered by the
#: sync test; these are the ones whose *content* this file exercises.
GATE_SCHEMAS = (
    "batch-record-v1.json",
    "step-record-v1.json",
    "selector-config-v1.json",
    "pack-definition-v1.json",
)


def _registry() -> Registry:
    """Every schema in the directory, so relative ``$ref``s across files resolve."""
    resources = []
    for path in SCHEMA_DIR.glob("*.json"):
        contents = json.loads(path.read_text(encoding="utf-8"))
        resources.append(
            (path.name, Resource.from_contents(contents, default_specification=DRAFT202012))
        )
    return Registry().with_resources(resources)


def _validator(filename: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / filename).read_text(encoding="utf-8"))
    return Draft202012Validator(schema, registry=_registry())


# --------------------------------------------------------------------------------------
# Structural properties of every schema in the directory
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("filename", sorted(p.name for p in SCHEMA_DIR.glob("*.json")))
def test_schema_is_itself_valid(filename: str) -> None:
    """Parseable JSON is not the same as a valid schema; a bad keyword silently never fires."""
    Draft202012Validator.check_schema(json.loads((SCHEMA_DIR / filename).read_text()))


def _objects(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    """Every subschema that describes an object with declared properties."""
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            found.append((path, node))
        for key, value in node.items():
            found.extend(_objects(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_objects(value, f"{path}[{index}]"))
    return found


#: Vendored contracts whose openness is their OWNER's design, not miForge's to close.
OPEN_BY_DESIGN = {
    "millm-weights-api-v1.json": (
        "miLLM's response contract, GENERATED from its pydantic models; it is open exactly "
        "where those models are (the envelope, StorageHandle/ParameterHandle extra=allow). "
        "miForge validates its fake surface against it rather than authoring it."
    ),
    "millm-scoring-api-v1.json": (
        "miLLM's /v1/completions contract, GENERATED from its pydantic models (FR-25.15, D61); "
        "open where they are (the request is extra=allow, as OpenAI's is). miForge validates "
        "its fake scoring endpoint against it rather than authoring it."
    ),
}


@pytest.mark.parametrize("filename", sorted(p.name for p in SCHEMA_DIR.glob("*.json")))
def test_every_object_is_closed_to_unknown_keys(filename: str) -> None:
    """``additionalProperties: false`` on every object with properties.

    This is what makes R-02.35's "the manifest's fields are exactly the schema's" true
    rather than asserted. An open object accepts an invented or **misspelled** key, and a
    misspelled key is the dangerous one: it validates, it writes, and the field the writer
    meant to set is simply absent.
    """
    if filename in OPEN_BY_DESIGN:
        from tests.contract.test_schema_sync import OWNERSHIP, Vendored

        # Only a SIBLING-owned contract may be exempt: closure is the owner's choice.
        assert isinstance(OWNERSHIP.get(filename), Vendored), filename
        pytest.skip(f"{filename}: {OPEN_BY_DESIGN[filename]}")
    schema = json.loads((SCHEMA_DIR / filename).read_text(encoding="utf-8"))
    open_objects = [
        path for path, node in _objects(schema) if node.get("additionalProperties") is not False
    ]
    assert not open_objects, (
        f"{filename} has objects open to unknown keys: {open_objects}. An invented or "
        f"misspelled key would validate, and nothing downstream would notice."
    )


# --------------------------------------------------------------------------------------
# step-record-v1: the five blocks
# --------------------------------------------------------------------------------------

ENVELOPE = {"run_id": "run-abc", "step": 100, "ts": "2026-10-14T05:00:00Z"}

STEP_BLOCK = {
    **ENVELOPE,
    "block": "step",
    "loss": 1.8125,
    "objective": "cross_entropy",
    "learning_rate": 0.0001,
    "wall_time_ms": 812.5,
}

EVALUATION_BLOCK = {
    **ENVELOPE,
    "block": "evaluation",
    "scores": [{"set": "legal-test", "score": 0.82, "n": 200, "floor": None, "provider": "x"}],
    "provider": "in_process:lfm2-1.2b",
    "aggregation": "mean",
    "decode_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
    "subsample_seed": 7,
    "split": ["held_out"],
    "retention_floor_breached": False,
}

BOUNDARY_BLOCK = {
    **ENVELOPE,
    "block": "boundary",
    "round": 3,
    "stop_conditions": [
        {"condition": "max_steps", "triggered": False, "observed_at_step": 100},
    ],
    "phase_timings_ms": {"generate": 4000.0, "step": 812.5},
}

DEFINITION_BLOCK = {
    **ENVELOPE,
    "step": None,  # written before step 0 exists
    "block": "definition",
    "submitted": {"gpu": "auto", "pack": "legal@3"},
    "resolved": {"gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},
    "definition_sha256": "a" * 64,
}

BATCH_RECORD = {
    "source": "self-generation",
    "prompt_ids": ["p-1", "p-2"],
    "parts": [
        {
            "kind": "prompt_set",
            "contract_reference": {"type": "prompt-set", "name": "legal", "version": "3"},
            "external": None,
        }
    ],
    "verdicts": {"accept": 2, "reject": 1, "undecided": 1},
    "selection_health": {
        "candidates_generated": 4,
        "candidates_admitted": 2,
        "acceptance_rate": 0.5,
        "exhausted": False,
    },
    "selector_config_sha256": None,
    "grader_config_sha256": None,
    "batch_sha256": "b" * 64,
    "resolved_model": "in_process:lfm2-1.2b",
}
BATCH_BLOCK = {**ENVELOPE, "block": "batch", "batch": BATCH_RECORD}

ALL_BLOCKS = {
    "step": STEP_BLOCK,
    "evaluation": EVALUATION_BLOCK,
    "boundary": BOUNDARY_BLOCK,
    "definition": DEFINITION_BLOCK,
    "batch": BATCH_BLOCK,
}


@pytest.mark.parametrize("name", sorted(ALL_BLOCKS))
def test_each_run_log_block_validates(name: str) -> None:
    """All five blocks, one schema, one file. Three of the five had nowhere to go before."""
    errors = sorted(_validator("step-record-v1.json").iter_errors(ALL_BLOCKS[name]), key=str)
    assert not errors, f"{name} block rejected: {[e.message for e in errors]}"


def test_the_run_log_has_exactly_five_blocks() -> None:
    """A count assertion, because this project's commonest defect is a count that
    disagrees with its own body - the gate task's own heading said FOUR while its body
    listed five."""
    schema = json.loads((SCHEMA_DIR / "step-record-v1.json").read_text())
    assert len(schema["$defs"]["StepRecordV1"]["oneOf"]) == 5
    assert set(ALL_BLOCKS) == {"step", "evaluation", "boundary", "definition", "batch"}


def test_batch_block_resolves_the_cross_file_reference() -> None:
    """The batch payload is a ``$ref`` into ``batch-record-v1.json`` - ONE definition.

    If the reference silently failed to resolve, the payload would be unconstrained and
    every malformed batch record would pass. So the negative case is the real assertion.
    """
    validator = _validator("step-record-v1.json")
    broken = json.loads(json.dumps(BATCH_BLOCK))
    del broken["batch"]["verdicts"]
    assert not validator.is_valid(broken), (
        "a batch record missing `verdicts` was accepted, which means the cross-file $ref "
        "did not resolve and the payload is unconstrained"
    )


def test_an_unknown_block_discriminator_is_rejected() -> None:
    assert not _validator("step-record-v1.json").is_valid({**STEP_BLOCK, "block": "nonsense"})


def test_prompt_text_cannot_be_smuggled_into_the_run_log() -> None:
    """ADR-020: the run log carries identifiers and hashes, **never prompt or completion text**.

    Not a size optimisation. The run log leaves this machine in exports and support
    bundles, so user text in it is a privacy breach, and a sibling project shipped exactly
    that by flipping one ``include_context`` argument - with its suite staying green. Here
    the closed objects make it a *validation* failure rather than a matter of discipline.
    """
    validator = _validator("step-record-v1.json")
    for block in (STEP_BLOCK, BATCH_BLOCK):
        smuggled = json.loads(json.dumps(block))
        smuggled["prompt_text"] = "what is the penalty for late filing"
        assert not validator.is_valid(
            smuggled
        ), f"prompt text accepted into a {block['block']} block"
    inner = json.loads(json.dumps(BATCH_BLOCK))
    inner["batch"]["prompts"] = ["the actual prompt text"]
    assert not validator.is_valid(inner), "prompt text accepted inside the batch record"


@pytest.mark.parametrize("field", ["quiesce_hold_ms", "commit_duration_ms"])
def test_the_two_protocol_timings_are_optional(field: str) -> None:
    """They measure the optional hot-update mode's protocol and **cannot exist standalone**.

    R-02.14 requires them on "every step record", which was written when every step held a
    barrier. The standalone reframe rewrote the requirement that *does* something with a
    barrier and left alone the one that merely *measures* it - so as written, the core
    product could not satisfy its own requirement. They are nullable, and the mode fills
    them.
    """
    validator = _validator("step-record-v1.json")
    assert validator.is_valid(STEP_BLOCK), "a standalone step record must validate without them"
    assert validator.is_valid({**STEP_BLOCK, field: 1850.0}), "the mode must be able to fill it"


# --------------------------------------------------------------------------------------
# batch-record-v1: the part shape both features deferred to the other
# --------------------------------------------------------------------------------------


def test_a_part_may_be_a_contract_reference_or_external() -> None:
    validator = _validator("batch-record-v1.json")
    assert validator.is_valid(BATCH_RECORD)
    external = json.loads(json.dumps(BATCH_RECORD))
    external["parts"][0] = {
        "kind": "corpus",
        "contract_reference": None,
        "external": {
            "source": "https://example.org/corpus",
            "content_sha256": "c" * 64,
            "retrieved_at": "2026-10-01T00:00:00Z",
            "licence": "CC-BY-4.0",
        },
    }
    assert validator.is_valid(external)


def test_a_part_must_state_both_keys_even_when_one_is_null() -> None:
    """Both keys are **required**, each nullable - the nullable-but-required pattern.

    ``null`` is an assertion ("this part has no contract reference"); an **absent key** is
    an omission ("nobody said"). Collapsing them is the defect the nullable-``domain``
    decision exists to prevent, and it recurs wherever a writer is allowed to stay silent.
    """
    validator = _validator("batch-record-v1.json")
    silent = json.loads(json.dumps(BATCH_RECORD))
    del silent["parts"][0]["external"]
    assert not validator.is_valid(silent)


def test_the_verdict_tally_has_three_counts() -> None:
    """Two counts would force every pending candidate to be miscounted as accept or reject."""
    validator = _validator("batch-record-v1.json")
    two_valued = json.loads(json.dumps(BATCH_RECORD))
    del two_valued["verdicts"]["undecided"]
    assert not validator.is_valid(two_valued)


# --------------------------------------------------------------------------------------
# selector-config-v1: five types, and the two capabilities deliberately absent
# --------------------------------------------------------------------------------------

SELECTORS: dict[str, dict[str, Any]] = {
    "rule": {"kind": "rule", "name": "no-refusals", "config_sha256": None, "expression": "x"},
    "judge_model": {
        "kind": "judge_model",
        "name": "judge",
        "config_sha256": "d" * 64,
        "provider": "openai_compatible_cpu",
        "threshold": 0.7,
    },
    "reference_match": {
        "kind": "reference_match",
        "name": "refmatch",
        "config_sha256": None,
        "mode": "fuzzy",
        "threshold": 0.9,
    },
    "human_review": {
        "kind": "human_review",
        "name": "review",
        "config_sha256": None,
        "queue": "legal-review",
        "timeout_seconds": None,
    },
    "composite": {
        "kind": "composite",
        "name": "both",
        "config_sha256": None,
        "children": ["no-refusals", "refmatch"],
        "combine": "all",
    },
}


@pytest.mark.parametrize("kind", sorted(SELECTORS))
def test_each_selector_type_validates(kind: str) -> None:
    errors = sorted(_validator("selector-config-v1.json").iter_errors(SELECTORS[kind]), key=str)
    assert not errors, f"{kind} rejected: {[e.message for e in errors]}"


def test_there_are_exactly_five_selector_types() -> None:
    schema = json.loads((SCHEMA_DIR / "selector-config-v1.json").read_text())
    assert len(schema["$defs"]["SelectorConfigV1"]["oneOf"]) == 5
    assert set(SELECTORS) == {
        "rule",
        "judge_model",
        "reference_match",
        "human_review",
        "composite",
    }


def test_embedding_similarity_is_refused_not_silently_accepted() -> None:
    """R-02.9's amendment offered ``exact``, ``fuzzy`` **or embedding similarity**.

    Embedding similarity has **no mechanism**: the provider abstraction is scoped to
    generation and the word "embedding" occurs exactly once in the whole BRD - in that
    amendment. Same shape as the missing load call: a capability named in prose with no
    source. So the enum refuses it rather than a selector silently matching fuzzily when
    asked for embeddings, which would be a fabricated result (ADR-018).
    """
    asked = {**SELECTORS["reference_match"], "mode": "embedding"}
    assert not _validator("selector-config-v1.json").is_valid(asked)


# --------------------------------------------------------------------------------------
# pack-definition-v1
# --------------------------------------------------------------------------------------

PACK: dict[str, Any] = {
    "schema_version": "1",
    "name": "legal-drafting",
    "version": "3",
    "domain": "legal",
    "created_at": "2026-10-14T05:00:00Z",
    "definition_sha256": "e" * 64,
    "slots": {
        "prompt_set": {
            "contract_reference": {"type": "prompt-set", "name": "legal", "version": "3"},
            "external": None,
            "floor": None,
        },
        "corpus": None,
        "preference_pairs": None,
        "selector_config": None,
        "grader_config": None,
        "test_set": None,
        "retention_set": None,
    },
}


def test_a_pack_validates_and_requires_all_seven_slot_keys() -> None:
    validator = _validator("pack-definition-v1.json")
    assert validator.is_valid(PACK)
    silent = json.loads(json.dumps(PACK))
    del silent["slots"]["retention_set"]
    assert not validator.is_valid(silent), (
        "a pack that simply omits a slot was accepted; every slot must be stated, null "
        "included, so 'no retention set' is an assertion rather than a silence"
    )


def test_a_pack_domain_is_nullable_but_required() -> None:
    validator = _validator("pack-definition-v1.json")
    assert validator.is_valid({**PACK, "domain": None}), "null is a legitimate assertion"
    absent = json.loads(json.dumps(PACK))
    del absent["domain"]
    assert not validator.is_valid(absent), "an absent key is an omission, not an assertion"
