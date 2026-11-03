"""The two ScoreRecord payloads F5 writes into a manifest  (FR-004.29, 30, 31; E-22; M-4).

## Read the key set from the schema; do NOT retype it

:data:`DECLARED_KEYS` is read out of ``docs/schemas/checkpoint-manifest-v1.json`` at import.
This module was specified when ``ScoreRecord`` had **four** properties and the schema gained
a fifth the same day; a hardcoded tuple is exactly how that becomes a silent contract breach,
and reading the file is the only reason the specification survived the change underneath it.
The same discipline F10 applies to the ``miforge_set_kind`` enum and F2 to ``signal_source``.

## The direction that matters is OMISSION, and a validator cannot see it

Since the 2026-10-13 re-vendor every object in the manifest sets
``additionalProperties: false``, so an **extra** key is now refused by ``jsonschema`` itself.
What a validator still cannot see is a **declared, non-required** key being left out - and
``provider`` is exactly that: declared ``anyOf [string, null]``, defaulted ``null``, and
**not in ``required``**. A payload of the old four keys therefore **validates clean** while
leaving R-02.16's obligation unmet in the one artifact that travels:

    "together with which provider produced it - a score is only comparable against another
    taken the same way."

So :func:`score_record_payload` writes all five, and
``backend/tests/contract/test_manifest_score_payload.py`` asserts **key-set equality in both
directions** against the schema file. That assertion is mutation control **M-4**, re-pointed
from *adding* the key to *omitting* it - and omission is the mutation a well-meaning
implementer actually produces, by copying the old four-key builder.

**RE-RUN M-4 AFTER ANY EDIT TO THIS FILE.**

## Null is present, absent is an omission

A save at a step where no evaluation ran writes ``test: null`` and ``retention: null`` -
**present and null** (E-22). Both properties are nullable with a null default, and null is an
assertion that nothing ran where an absent key is an omission. The same distinction the
manifest's own ``domain`` field is documented with.

## Both kinds, although the requirement names one

R-02.18's text says *"into the checkpoint manifest under ``test``"* and names ``retention``
nowhere. The schema **has** a ``retention`` property, nullable, annotated *"Written by the
trainer. The out-of-domain score at this step (BRD-02 R-02.17), carrying the floor the run
stops at"*, and acceptance criteria 3 and 19 both need a durable retention series. F4 writes
both, on the schema's own writer annotation. **The requirement's text is narrower than its
own normative schema; that is reported (DEF-9), not amended here.**

## No prompt text, no completion text, no credential

A manifest travels with the artifact and is published beside a checkpoint. Prompt and
completion text live in the run log and ``mf_score_items`` only; a credential exists nowhere
in a record at all, and :class:`providers.base.ProviderIdentity` has no field to hold one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.errors import refuse

#: Where the vendored contracts are. The same env override and the same walk-up as
#: :mod:`services.reindex` and F3's resolver, for the same reason: a hard-coded absolute path
#: works on one workstation.
_SCHEMA_DIR_ENV = "MIFORGE_SCHEMA_DIR"
_DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[4] / "docs" / "schemas"
SCHEMA_FILENAME = "checkpoint-manifest-v1.json"

#: The two manifest properties F4 fills. Named here so the key set and the property names are
#: not two separate pieces of knowledge at the call site.
MANIFEST_TEST_KEY = "test"
MANIFEST_RETENTION_KEY = "retention"


def schema_dir() -> Path:
    return Path(os.environ.get(_SCHEMA_DIR_ENV, str(_DEFAULT_SCHEMA_DIR)))


def declared_keys() -> frozenset[str]:
    """``$defs.ScoreRecord.properties``, **read from the vendored file**.

    A function rather than a module constant so a test that points ``MIFORGE_SCHEMA_DIR`` at
    a copy sees that copy. The contract test calls it and the builder calls it, so the
    builder and its guard cannot disagree about what the schema says.
    """
    path = schema_dir() / SCHEMA_FILENAME
    if not path.is_file():
        raise refuse(
            "schema_not_vendored",
            f"{SCHEMA_FILENAME} is not present in {schema_dir()}; set {_SCHEMA_DIR_ENV} to "
            f"the directory holding the vendored contracts. The ScoreRecord key set is read "
            f"from this file rather than retyped, so without it a payload cannot be built "
            f"at all - which is the correct failure: a guessed key set is how a cross-repo "
            f"contract gets widened by accident.",
            schema=SCHEMA_FILENAME,
            searched=str(schema_dir()),
        )
    document = json.loads(path.read_text("utf-8"))
    return frozenset(document["$defs"]["ScoreRecord"]["properties"])


def required_keys() -> frozenset[str]:
    """``$defs.ScoreRecord.required``. Read for the same reason, and used by the guard.

    The difference between this and :func:`declared_keys` is the whole point of M-4:
    ``provider`` is declared and **not** required, so the validator accepts its absence.
    """
    path = schema_dir() / SCHEMA_FILENAME
    document = json.loads(path.read_text("utf-8"))
    return frozenset(document["$defs"]["ScoreRecord"].get("required") or ())


def score_record_payload(score: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """One ``ScoreRecord``, or ``None`` for a kind that did not run at this step.

    Args:
        score: A mapping carrying ``set``, ``score``, ``n``, ``floor`` and ``provider`` - in
            practice a :class:`services.scoring.record.ScoreRecordData`. ``None`` means no
            evaluation of this kind ran, which writes ``null`` **present** (E-22).

    Raises:
        Refusal: ``manifest_score_payload_invalid`` when a declared key has no value to
            write. Refused rather than defaulted: the one key that could plausibly be
            defaulted is ``provider``, and defaulting it to ``null`` is mutation M-4 by
            another route - the record would validate and R-02.16's obligation would be
            unmet in the artifact that travels.
    """
    if score is None:
        return None

    declared = declared_keys()
    payload: dict[str, Any] = {}
    for key in sorted(declared):
        if key not in score:
            raise refuse(
                "manifest_score_payload_invalid",
                f"The schema declares ScoreRecord.{key} and this score carries no value "
                f"for it. Refusing rather than writing a default: a defaulted 'provider' "
                f"validates clean - it is declared and not required - and leaves the score's "
                f"instrument unnamed in the one artifact that travels without its run log.",
                key=key,
                declared=sorted(declared),
                carried=sorted(str(k) for k in score),
            )
        payload[key] = score[key]
    return payload


def manifest_scores(
    *, test: Mapping[str, Any] | None, retention: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The two manifest properties, **both always present** (E-22, FR-004.31).

    Returns a mapping with exactly ``test`` and ``retention``, each a ScoreRecord payload or
    ``null``. F5's manifest writer merges this; it never has to decide whether to include a
    key, which is how one of them would eventually go missing at a non-evaluation save.
    """
    return {
        MANIFEST_TEST_KEY: score_record_payload(test),
        MANIFEST_RETENTION_KEY: score_record_payload(retention),
    }
