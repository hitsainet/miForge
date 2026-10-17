"""The batch record: assemble, validate, log, index, emit  (FR-002.28 - FR-002.32).

**In that order, and the order is the requirement.**

    assemble -> validate (jsonschema, against the gate contract)
             -> hash     (core/canonical.py - the ONE serialiser, ADR-005)
             -> append to the run log      <-- AUTHORITATIVE (ADR-004)
             -> upsert the index row       <-- REBUILDABLE
             -> emit a summary on miforge/run/<id>   <-- A PROJECTION
             -> touch the heartbeat        <-- ADR-007
             -> return the content hash    --> F1's step record (R-02.14)

A crash between the log and the index costs a ``reindex``. A crash the other way round
would cost **the record**, and the index would then be the only evidence of a batch the
authoritative log denies. **If validation fails, nothing is written and the batch is not
returned** (E-17) - byte-for-byte F1's rule for the step record, kept identical on purpose
so the two records fail the same way.

## The hash covers the record without its own hash field

``batch_sha256`` is a required field of the record *and* is the record's content hash, so
the digest is taken over the canonical JSON of the record **with that key removed** and then
written in. A reader verifies the same way - :func:`verify` - and there is one function for
both, because two canonicalisers are two answers (ADR-005).

## IDENTIFIERS AND HASHES, NEVER TEXT

The contract's own words: *"it carries identifiers and hashes and never prompt or completion
text (ADR-020). That is not a size optimisation: the run log leaves this machine in exports
and support bundles, and user prompt text in it would be a privacy breach that no schema
check downstream could detect. A sibling project shipped exactly that bug by flipping one
``include_context`` argument, and its suite stayed green."* Every object in the schema sets
``additionalProperties: false``, so smuggled text is a **validation failure** rather than a
silent one - and :func:`summary_of` reaches no content either, because the socket payload is
the other thing that leaves this process.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.canonical import canonical_bytes, sha256_of
from core.errors import refuse
from training import run_log
from training.signals import batch_index
from training.signals.base import BatchCounts

log = logging.getLogger(__name__)

#: The event name on the run's **existing** room. No new room and no new event stream: the
#: run is the unit an operator watches, and a second channel is a second truth.
BATCH_EVENT = "batch"

_SCHEMA_NAME = "batch-record-v1.json"
_schema: dict[str, Any] | None = None
_validator: Any | None = None


def schema_path() -> Path:
    """``docs/schemas/batch-record-v1.json``, found relative to this file.

    Relative rather than configured so the schema travels with the repository, exactly as
    F1's run log finds the step record's. ``docs/schemas/`` is the one ``docs/`` path
    published to the public mirror, because these are public contracts.
    """
    default = Path(__file__).resolve().parents[4] / "docs" / "schemas"
    # MIFORGE_SCHEMA_DIR, as the other seven modules resolve it. See run_log.py.
    return Path(os.environ.get("MIFORGE_SCHEMA_DIR", str(default))) / _SCHEMA_NAME


def schema() -> dict[str, Any]:
    """The gate contract, parsed once. **Parsed, never described.**"""
    global _schema
    if _schema is None:
        path = schema_path()
        if not path.is_file():
            raise refuse(
                "schema_not_vendored",
                f"{_SCHEMA_NAME} is not present at {path}. It is foundation task 3.2's gate "
                f"contract and F2 is its principal writer; without it a batch record cannot "
                f"be validated, and an unvalidated provenance record is worse than a gap.",
                schema=_SCHEMA_NAME,
                searched=str(path),
            )
        _schema = json.loads(path.read_text("utf-8"))
    return _schema


def validator() -> Any:
    """The compiled validator, built once at first use (ADR-024: per batch, not per example)."""
    global _validator
    if _validator is None:
        from jsonschema import Draft202012Validator

        _validator = Draft202012Validator(schema())
    return _validator


def source_enum() -> frozenset[str]:
    """``source``'s enum, **read from the contract** rather than retyped.

    The same discipline F10 applies to its ``kind_map``: a value added or removed upstream
    then produces an explicit failure instead of a silent disagreement.
    """
    node = schema()["$defs"]["BatchRecordV1"]["properties"]["source"]
    return frozenset(str(value) for value in node["enum"])


def record_source(name: str) -> str:
    """A registry name as the contract's ``source`` value. **Now an identity check.**

    The two vocabularies are **one vocabulary**: ``batch-record-v1.json``'s ``source`` and
    ``checkpoint-manifest-v1.json``'s ``signal_source`` carry the same four values with the
    same spellings, so a registry name needs no translation to become either.

    *This function used to translate - hyphens to underscores - and it was right to, because
    the two enums genuinely disagreed: they shared exactly ONE value out of four. F2 found
    that and pinned it with a test asserting the two are not interchangeable. The better fix,
    made 2026-10-14, was to remove the disagreement: the enums are now identical, ``reward``
    was added (so a reward batch can be recorded at all), and ``mixed`` was removed (no
    requirement describes a mixed signal source - R-02.27 makes a curriculum CHAINED runs,
    each with one source, and "mixed" appears in BRD-02 only about pack parts). Both enums
    were mine, and the underscore spelling was mine.*

    **The refusal is kept and still matters.** A registry name with no value in the enum is
    refused at pre-flight, before a run starts, rather than at step 1 - because writing a
    different source instead would be a false claim about what trained the checkpoint, and
    the record is the one thing that can answer that question six months later. What has
    changed is that no *currently registered* source hits it.
    """
    enum = source_enum()
    if name not in enum:
        raise refuse(
            "batch_source_not_in_contract",
            f"Signal source {name!r} has no value in batch-record-v1.json's `source` enum "
            f"({sorted(enum)}), so a batch it filled could not be recorded. Its objects are "
            f"closed, and writing a different source instead would be a false claim about "
            f"what trained the checkpoint. Refusing before the run starts rather than at "
            f"step 1.",
            source=name,
            enum=sorted(enum),
            schema=_SCHEMA_NAME,
        )
    return name


def assemble(
    *,
    source: str,
    prompt_ids: Sequence[str],
    parts: Sequence[Mapping[str, Any]],
    counts: BatchCounts,
    resolved_model: str | None = None,
    selector_config_sha256: str | None = None,
    grader_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Assemble a record from the values F2 contributes, and hash it.

    Every key here is the contract's and none is this module's invention - the objects are
    closed, so an invented one would fail validation rather than being quietly accepted,
    which is the gate working as designed.

    ``acceptance_rate`` is ``admitted / generated`` and **0.0 when nothing was generated**,
    which is the corpus case: a corpus batch runs no selection, so its three verdict counts
    and both candidate counts are zero and the rate is zero with them. The alternative -
    reporting ``1.0`` over the examples read - would assert that *a selector accepted
    everything*, which is one of the three named degenerate signals, about a batch where no
    selector ran at all.
    """
    generated = counts.generated
    record: dict[str, Any] = {
        "source": record_source(source),
        "prompt_ids": list(prompt_ids),
        "parts": [dict(part) for part in parts],
        "verdicts": {
            "accept": counts.admitted,
            "reject": counts.rejected,
            # THREE counts. `undecided` is carried as itself and never folded into
            # `reject`: a pending candidate miscounted as a rejection makes the health
            # record lie about an unattended queue, and the run looks like it is selecting.
            "undecided": counts.pending,
        },
        "selection_health": {
            "candidates_generated": generated,
            "candidates_admitted": counts.admitted if generated else 0,
            "acceptance_rate": (counts.admitted / generated) if generated else 0.0,
            "exhausted": counts.exhausted,
        },
        "selector_config_sha256": selector_config_sha256,
        "grader_config_sha256": grader_config_sha256,
        "resolved_model": resolved_model,
    }
    record["batch_sha256"] = content_hash(record)
    return record


def content_hash(record: Mapping[str, Any]) -> str:
    """The record's content hash: canonical JSON **without** ``batch_sha256``, SHA-256.

    Through ``core.canonical`` (ADR-005). **No local canonicaliser, not even a two-line
    one**: the bytes that are hashed and the bytes that are written must come from one code
    path, or a recorded hash stops matching the record it describes and nothing says so.
    """
    payload = {key: value for key, value in record.items() if key != "batch_sha256"}
    return sha256_of(payload)


def verify(record: Mapping[str, Any]) -> bool:
    """Whether a record's ``batch_sha256`` matches its content. One function, both sides."""
    return record.get("batch_sha256") == content_hash(record)


def validate(record: Mapping[str, Any]) -> None:
    """Validate against the gate contract, **naming the failing field** (E-17).

    Raises:
        Refusal: ``batch_record_invalid``. Nothing is written and the batch is not fed to a
            step - a record a reader cannot trust is worse than a gap, because the gap is
            visible and the bad record is not.
    """
    errors = sorted(validator().iter_errors(dict(record)), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise refuse(
            "batch_record_invalid",
            f"This batch record does not validate against {_SCHEMA_NAME}: {first.message} "
            f"(at {where}). Nothing was written and the batch is not handed to a step.",
            pointer=where,
            errors=[error.message for error in errors[:5]],
        )
    if not verify(record):
        raise refuse(
            "batch_record_hash_mismatch",
            "This batch record's batch_sha256 does not cover its own content. The hash is "
            "the batch's identity and the only join between a step and the material it "
            "consumed (R-02.14), so a record whose hash was computed over different content "
            "would make that join point at the wrong batch.",
            recorded=record.get("batch_sha256"),
            computed=content_hash(record),
        )


def block(
    record: Mapping[str, Any], *, run_id: str, step: int | None, ts: datetime | None = None
) -> dict[str, Any]:
    """The run log's ``batch`` block around a record (``step-record-v1.json``).

    **The fifth block of one log.** The discriminator is in the line, not implied by the
    file, because ``reindex`` rebuilds from the log alone and has to classify a line it is
    holding. The payload is a ``$ref`` into ``batch-record-v1.json`` - one definition,
    referenced - so this wraps the record rather than copying its fields.
    """
    return {
        "block": "batch",
        "run_id": run_id,
        "step": step,
        "ts": (ts or datetime.now(UTC)).isoformat(),
        "batch": dict(record),
    }


def summary_of(record: Mapping[str, Any], *, counts: BatchCounts, objective: str) -> dict[str, Any]:
    """The socket payload. **Reaches no content, by construction.**

    Source, counts, the objective, the hash and the signal names - nothing that could carry
    a prompt or a completion, and nothing that could carry one by accident: it is built from
    the record's scalar fields and the counts, never from the batch's material. A summary
    with a sample completion in it is the content leak this feature's one data-minimisation
    guarantee exists to prevent, and it is a named mutation control.
    """
    return {
        "batch_sha256": record["batch_sha256"],
        "source": record["source"],
        "objective": objective,
        "resolved_model": record.get("resolved_model"),
        "admitted": counts.admitted,
        "rejected": counts.rejected,
        "pending": counts.pending,
        "dropped": counts.dropped,
        "ties": counts.ties,
        # ADR-012: every number names its scale and sample. The rate never travels without
        # the count it was taken over.
        "candidates_generated": counts.generated,
        "exhausted": counts.exhausted,
    }


def write(
    record: Mapping[str, Any],
    *,
    run_id: str,
    step: int | None,
    counts: BatchCounts,
    objective: str,
    log_root: Path | None = None,
    session: Any | None = None,
    emit: Any | None = None,
    heartbeat: Any | None = None,
    signals: Sequence[str] = (),
    effects: list[str] | None = None,
) -> str:
    """Record one batch and return its content hash. **Durable first.**

    Args:
        effects: Appended to in order, so the ordering test asserts the **sequence** rather
            than the outcome. An inversion changes nothing observable until the machine
            dies, which is exactly why it is asserted directly.
        signals: F3's named degenerate signals, logged beside the batch. They have **no
            field in the record** - its ``SelectionHealth`` is four fields - so they travel
            on the in-process batch for F6 and in this log line for a human.

    Raises:
        Refusal: ``batch_record_invalid`` or ``batch_record_hash_mismatch``, before any
            side effect.
    """
    validate(record)
    effects = effects if effects is not None else []
    line = block(record, run_id=run_id, step=step)

    if log_root is not None:
        # 1. DURABLE, AND FIRST. `append` validates the whole block against
        #    step-record-v1.json too - including the cross-file `$ref` into the batch
        #    contract - so an invalid line leaves no trace rather than an index row with no
        #    record behind it.
        run_log.append(run_log.run_log_path(Path(log_root), run_id), line)
        effects.append("log")

    if session is not None:
        batch_index.upsert(session, line)
        effects.append("index")

    if emit is not None:
        # 3. A projection, after the record exists. Fire-and-forget: a worker with no Redis
        #    reachable still completes a batch.
        emit(run_id, BATCH_EVENT, summary_of(record, counts=counts, objective=objective))
        effects.append("socket")

    if heartbeat is not None and session is not None:
        # 4. Liveness. The Heartbeat owns the 60 s throttle and the terminal-row guard; this
        #    does not re-implement either. A long self-generation assembly is exactly the
        #    unit that would otherwise look like a lost worker - progress reported only over
        #    the socket is invisible to the janitor.
        heartbeat.touch(session)
        effects.append("heartbeat")

    log.info(
        "batch recorded",
        extra={
            "code": "batch_recorded",
            "run_id": run_id,
            "step": step,
            "source": record["source"],
            "objective": objective,
            "batch_sha256": record["batch_sha256"],
            "admitted": counts.admitted,
            "rejected": counts.rejected,
            "pending": counts.pending,
            # The two counts the contract has no field for. Recorded here so they are not
            # lost entirely, and reported as owed on the gate contract.
            "dropped": counts.dropped,
            "ties": counts.ties,
            "signals": list(signals),
        },
    )
    for signal in signals:
        # One line per named signal (ADR-021), and none per verdict: a per-verdict flood
        # buries the signals beside it.
        log.warning(
            "selection signal",
            extra={"code": signal, "run_id": run_id, "batch_sha256": record["batch_sha256"]},
        )
    return str(record["batch_sha256"])


def canonical_record_bytes(record: Mapping[str, Any]) -> bytes:
    """The record's canonical bytes, for a caller that needs to compare two records."""
    return canonical_bytes(dict(record))
