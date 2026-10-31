"""The run log: the **authoritative** record of a run  (ADR-004; FR-001.24, FR-001.27).

JSON Lines, append-only, one ``step-record-v1.json`` block per line. ADR-004 makes the
filesystem the source of truth and PostgreSQL *a rebuildable index*, so this file is where a
step record actually lives and ``mf_steps`` is a query surface over it. The ordering rule
follows from that and is not negotiable: **durable write, then index, then socket.** A crash
between them loses a projection, never a record.

## Five blocks, one file, one discriminator

``step-record-v1.json`` is a discriminated union on ``block``: ``step`` (F1), ``evaluation``
(F4), ``boundary`` and ``definition`` (F6), ``batch`` (F2). Four features write five things
to one log, interleaved, and before that was settled three of them had nowhere defined to
put theirs. The discriminator is **in the line**, not implied by which file it came from,
because ``reindex`` rebuilds from the log alone and a reader must be able to classify a line
it is holding.

**F1 writes the ``step`` block and validates every line against the schema before writing
it.** A record a reader cannot trust is worse than a gap, so a validation failure fails the
step and writes nothing.

## The two section 5.10 fields

``quiesce_hold_ms`` and ``commit_duration_ms`` are **present and unset** on every step F1
writes, never dropped. They measure a protocol with a serving process that does not exist on
the default path; F11 populates them and defines neither. The precedent is the checkpoint
manifest's ``domain`` field, which says it plainly: *null is a statement, an absent key is
an omission.*

``commit_duration_ms`` is **not local telemetry**: miLLM Feature 31's acceptance gate is set
from its p99.9, so preserving it and keeping it queryable **as a distribution** is a
cross-repo product obligation (FR-001.26). :func:`commit_duration_percentile` is that query,
and it reads the log rather than the index - which is the right source anyway, since the log
is the truth and survives a dropped index.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.canonical import canonical_bytes
from core.errors import refuse
from core.paths import contained_child

log = logging.getLogger(__name__)

#: The log's file name inside a run's directory.
RUN_LOG_NAME = "run.jsonl"

#: The five ``block`` discriminators ``step-record-v1.json`` defines, and who writes each.
#: Enumerated so an unknown block is refused at the writer rather than discovered by a
#: reader that silently skips it.
BLOCK_WRITERS: Mapping[str, str] = {
    "step": "F1",
    "evaluation": "F4",
    "boundary": "F6",
    "definition": "F6",
    "batch": "F2",
}

#: Each block's ``$defs`` entry, so a refusal can be raised against the block's own
#: subschema and therefore name the field that failed. Derived from the discriminator
#: rather than guessed: the mapping is one line per block and the test asserts every
#: discriminator has one.
BLOCK_DEFINITIONS: Mapping[str, str] = {
    "step": "StepBlock",
    "evaluation": "EvaluationBlock",
    "boundary": "BoundaryBlock",
    "definition": "DefinitionBlock",
    "batch": "BatchBlock",
}

#: Blocks whose ``step`` may be null. Only ``definition``, which is written before step 0
#: exists - the schema says so, and restating it here would be a second rule, so this is
#: derived from the schema at validation time and used only for the index rebuild.
NULL_STEP_BLOCKS = frozenset({"definition"})

_VALIDATORS: dict[str, Any] = {}


def schema_path() -> Path:
    """``docs/schemas/step-record-v1.json``, found relative to this file.

    Relative rather than configured so the schema travels with the repository, the same way
    the contract tests find it. ``docs/schemas/`` is the one ``docs/`` path published to the
    public mirror, because it is a public contract.
    """
    default = Path(__file__).resolve().parents[3] / "docs" / "schemas"
    # MIFORGE_SCHEMA_DIR: the convention five other modules already follow, and what
    # lets the backend image find its contracts - `parents[3]` is the repository root
    # from source and `/usr/local/lib` once pip-installed.
    return Path(os.environ.get("MIFORGE_SCHEMA_DIR", str(default))) / "step-record-v1.json"


def validator(block: str) -> Any:
    """The compiled validator for one block, built **once** per block at first use.

    Compiled once because validating per step is cheap and recompiling is not, and F1's own
    overhead budget is under 5% of step wall time (ADR-024). Built lazily rather than at
    import so a machine without the schema still imports this module.

    **Per block rather than against the union**, and that is not an optimisation. The schema
    is a ``oneOf`` discriminated on ``block``, and ``jsonschema`` reports a failure against a
    ``oneOf`` as *"is not valid under any of the given schemas"* at the root, with the whole
    instance quoted and **no mention of the field that was wrong**. Validating against the
    block's own ``$defs`` entry - selected by the discriminator, which is checked first -
    produces a message that names the field, which is what ADR-013 means by a refusal saying
    what to do.
    """
    if block not in _VALIDATORS:
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        directory = schema_path().parent
        document = json.loads(schema_path().read_text(encoding="utf-8"))
        resources = [
            (
                path.name,
                Resource.from_contents(
                    json.loads(path.read_text(encoding="utf-8")),
                    default_specification=DRAFT202012,
                ),
            )
            for path in directory.glob("*.json")
        ]
        # A registry over the whole directory: the step record's `batch` block is a `$ref`
        # into `batch-record-v1.json`, and "one definition, referenced" only works if the
        # validator can follow it.
        registry = Registry().with_resources(resources)
        subschema = document["$defs"][BLOCK_DEFINITIONS[block]]
        # The `$defs` are carried along so the subschema's own internal `$ref`s resolve.
        _VALIDATORS[block] = Draft202012Validator(
            {**subschema, "$defs": document["$defs"]}, registry=registry
        )
    return _VALIDATORS[block]


def validate_block(record: Mapping[str, Any]) -> None:
    """Validate one line against ``step-record-v1.json``, refusing by name on failure.

    Raises:
        Refusal: ``step_record_invalid``, carrying the validator's own message. **The step
            fails and nothing is written** - a record a reader cannot trust is worse than a
            gap, because the gap is visible and the bad record is not.
    """
    block = record.get("block")
    if not isinstance(block, str) or block not in BLOCK_WRITERS:
        raise refuse(
            "step_record_invalid",
            f"{block!r} is not one of step-record-v1.json's blocks "
            f"({sorted(BLOCK_WRITERS)}). The discriminator must be in the line: reindex "
            f"rebuilds the index from this log alone and classifies each line it reads.",
            block=block,
        )
    _assert_finite_numbers(record, block)
    errors = sorted(validator(block).iter_errors(dict(record)), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise refuse(
            "step_record_invalid",
            f"A {block!r} block does not validate against step-record-v1.json: "
            f"{first.message} (at {where}). Refusing to write it: the log is the "
            f"authoritative record and a line a reader cannot trust is worse than a missing "
            f"one.",
            block=block,
            errors=[e.message for e in errors[:5]],
        )


def _assert_finite_numbers(record: Mapping[str, Any], block: str) -> None:
    """Refuse a block carrying NaN or an infinity, **naming the field**.

    The schema cannot catch this: JSON Schema's ``number`` admits a Python float that is NaN,
    so a diverged loss validates clean. ``canonical_bytes`` then refuses it - correctly, with
    ``allow_nan=False``, because ``json`` would otherwise emit the literal ``NaN``, which no
    parser is required to accept, so the log would validate going out and fail coming back.

    But that refusal arrives as a bare ``ValueError`` from the serialiser, which reaches a
    client as a 500 with a traceback rather than as a named refusal. **A diverged loss should
    refuse loudly at the point it is recorded** - ``core/canonical.py`` says so in terms - so
    it is named here, before any write, with the field that diverged.
    """
    for key, value in record.items():
        if isinstance(value, bool) or not isinstance(value, float):
            continue
        if value != value or value in (float("inf"), float("-inf")):
            raise refuse(
                "step_record_not_finite",
                f"A {block!r} block carries a non-finite {key} ({value}). The value is a "
                f"real measurement and is reported rather than silently dropped, but it "
                f"cannot be written as JSON - no parser is required to accept the literal "
                f"NaN - so the run stops here. Lower the learning rate or check the batch.",
                block=block,
                field=key,
                value=str(value),
            )


#: A run's log is ONE append-only record written by SEVERAL processes - the API writes the
#: definition block at submit, the train worker appends every step - so it is GROUP-writable
#: by construction (group = the app's own, inherited through the setgid directory) and does
#: not depend on any creator's umask. Until 2026-10-17 the API created the directory 0755
#: and the file 0644 under umask 0022, so a train worker running as a different uid (the F11
#: hot-update session runs it as miLLM's uid, gid unchanged) could not append, and the run
#: failed before its first step (found live, F11 window 2).
RUN_LOG_DIR_MODE = 0o2775
RUN_LOG_FILE_MODE = 0o664


def _ensure_shared_dir(directory: Path) -> None:
    """Create a run's log directory group-writable and setgid; leave an existing one alone."""
    if directory.is_dir():
        return
    directory.mkdir(parents=True, exist_ok=True)
    # Created concurrently by another process, which owns it and set the mode itself.
    with contextlib.suppress(PermissionError):
        os.chmod(directory, RUN_LOG_DIR_MODE)


def run_log_path(root: Path, run_id: str) -> Path:
    """``<root>/<run_id>/run.jsonl``.

    One directory per run, so deleting a run's row never touches its log (ADR-004) and a
    support bundle is a directory copy.
    """
    if not run_id:
        raise refuse("run_log_run_id_required", "A run log must belong to a named run.")
    # The identifier arrives from URL paths and submitted definitions: contained, or refused.
    return contained_child(root, run_id, code="run_log_run_id_invalid", what="a run log") / (
        RUN_LOG_NAME
    )


def append(path: Path, record: Mapping[str, Any]) -> None:
    """Validate, then append one block, **durably**.

    The write is ``O_APPEND`` plus ``fsync``: a step record that is in the page cache when
    the machine loses power is a step that happened and cannot be proved, and this file is
    the thing other records are reconciled against.

    Serialised through ``core.canonical.canonical_bytes`` - the codebase's **only**
    serialiser (ADR-005). That matters here beyond tidiness: it sorts keys, forbids NaN, and
    a NaN loss would otherwise be written as the literal ``NaN``, which no JSON parser is
    required to accept, so the log would validate on the way out and fail on the way back in.
    """
    validate_block(record)
    _ensure_shared_dir(path.parent)
    payload = canonical_bytes(dict(record)) + b"\n"
    created = not path.exists()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, RUN_LOG_FILE_MODE)
    try:
        if created:
            # Explicit, not the creator's umask: see RUN_LOG_FILE_MODE. A concurrent creator
            # that won the race owns the file, and its own call set the mode.
            with contextlib.suppress(PermissionError):
                os.fchmod(fd, RUN_LOG_FILE_MODE)
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_blocks(path: Path, *, block: str | None = None) -> Iterator[dict[str, Any]]:
    """Every block in the log, in written order, optionally filtered by discriminator.

    A malformed line is **refused, not skipped**. Skipping would make a truncated log read
    as a shorter run, and ``reindex`` would then rebuild an index that silently disagrees
    with the record it was built from - which is the exact divergence ADR-004's equivalence
    test exists to detect.
    """
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except ValueError as exc:
                raise refuse(
                    "run_log_corrupt",
                    f"{path} line {number} is not valid JSON: {exc}. Refusing rather than "
                    f"skipping it - a skipped line makes a truncated log read as a shorter "
                    f"run, and the index rebuilt from it would disagree silently.",
                    path=str(path),
                    line=number,
                ) from exc
            if block is None or parsed.get("block") == block:
                yield parsed


def step_block(
    *,
    run_id: str,
    step: int,
    loss: float,
    objective: str,
    learning_rate: float,
    wall_time_ms: float,
    grad_norm: float | None = None,
    batch_sha256: str | None = None,
    ts: datetime | None = None,
    quiesce_hold_ms: float | None = None,
    commit_duration_ms: float | None = None,
) -> dict[str, Any]:
    """Assemble F1's ``step`` block (R-02.14).

    **Every field here is the schema's and none is F1's invention.** The two section 5.10
    timings are written **present and null**: dropping them would take away the input to
    another repository's acceptance gate, and an absent key is an omission where a null is a
    statement that this run had no barrier to measure.
    """
    return {
        "block": "step",
        "run_id": run_id,
        "step": step,
        "ts": (ts or datetime.now(UTC)).isoformat(),
        "loss": loss,
        "objective": objective,
        "learning_rate": learning_rate,
        "wall_time_ms": wall_time_ms,
        "gradient_norm": grad_norm,
        "batch_sha256": batch_sha256,
        # §5.10 ONLY. Present and unset on the default path - F11's stepper populates them
        # through `StepOutcome` and defines neither (its FR-011.57). See the module docstring.
        "quiesce_hold_ms": quiesce_hold_ms,
        "commit_duration_ms": commit_duration_ms,
    }


# --------------------------------------------------------------------------------------
# The index rebuild, and the distribution FR-001.26 owes another repository
# --------------------------------------------------------------------------------------


def step_rows(path: Path) -> list[dict[str, Any]]:
    """Every step block as an ``mf_steps`` row, for an index rebuild.

    **This is the half of ``reindex`` the log can serve.** ``services/reindex.py`` rebuilds
    ``mf_checkpoints`` and ``mf_adapters`` from on-disk *manifests*; a step has no manifest,
    its authoritative copy is this log, and so the rebuild for ``mf_steps`` reads from here.
    The run row itself is **not** rebuildable - a run's status and the gap between its
    heartbeats appear in no artifact - and that distinction is why this returns step rows
    only.
    """
    # ONE ROW PER STEP, THE LAST RECORD WINNING. A resumed run (D52, F11 FTASKS 18.0) writes
    # steps after its checkpoint a second time - the earlier attempt's records of them are
    # superseded, not deleted (the log is append-only and the discontinuity is recorded) - and
    # `mf_steps` is unique on (run_id, step). Keeping every record made a failed resumed run's
    # index rebuild violate that index and leave the steps unrebuilt. Log order is time order,
    # so the last record is the current attempt's; the live writer's upsert agrees.
    rows: dict[int, dict[str, Any]] = {}
    for record in read_blocks(path, block="step"):
        row = step_rows_from_block(record)
        rows[int(row["step"])] = row
    return [rows[step] for step in sorted(rows)]


def step_rows_from_block(record: Mapping[str, Any]) -> dict[str, Any]:
    """One ``step`` block as one ``mf_steps`` row.

    **The single projection from the record to the index.** The live writer and the rebuild
    both call it, which is what makes ADR-016 #4's equivalence test meaningful: two
    projections would agree until one of them was changed, and the test compares API
    responses, so the disagreement would surface as a reindex that quietly returns different
    numbers.

    ~~``objective``, ``quiesce_hold_ms`` and ``commit_duration_ms`` are **deliberately not
    here**: ``mf_steps`` has no column for them today.~~ **ALL THREE ARE PROJECTED SINCE
    2026-10-14, and the gap between the columns arriving and this line was a real defect.**

    Migration ``0011`` added all three columns and a partial index
    ``ix_mf_steps_commit_duration``, and **nothing widened this projection** — so for a day
    the columns existed and were null on every row, and the index was over a column nothing
    wrote. That is *declaring a mechanism without wiring it*, committed in the same change
    whose own docstring states the rule. **Found by F11** while reading for its timings work,
    not by anyone reading the migration.

    ``objective`` is the costly one. Its own column comment says it is *"the load-bearing one
    of the three"* because **F9 draws loss curves from this table**, and a curve that silently
    splices two objectives together looks exactly like a regression — which is the failure the
    column was added to prevent, sitting empty.

    ``quiesce_hold_ms`` and ``commit_duration_ms`` are ``None`` on every standalone step and
    **that is an assertion, not an omission**: there is no quiesce and no commit bracket
    standalone. F11 fills them in §5.10 mode, and ``commit_duration_ms`` is not local
    telemetry — miLLM Feature 31's acceptance gate is set from its p99.9, which is why it has
    an index rather than a file scan.
    """
    return {
        "run_id": record["run_id"],
        "step": record["step"],
        "loss": record.get("loss"),
        "grad_norm": record.get("gradient_norm"),
        "learning_rate": record.get("learning_rate"),
        # The column is an integer of milliseconds; the schema's field is a number. Rounded
        # at the index boundary and nowhere else, so the authoritative value keeps its
        # precision.
        "wall_time_ms": (
            None if record.get("wall_time_ms") is None else round(float(record["wall_time_ms"]))
        ),
        "batch_source_hashes": _batch_hashes(record),
        # Projected since 2026-10-14. `record.get` rather than `record[...]`: a step record
        # written before migration 0011 has no such key, and a rebuild must read old logs.
        "objective": record.get("objective"),
        # Section 5.10 only, and None here is the ASSERTION that no bracket was held. Floats,
        # not rounded ints: the gate miLLM Feature 31 sets from `commit_duration_ms`'s p99.9
        # reads tens of milliseconds, where rounding to whole ms would coarsen the tail that
        # the gate is derived from.
        "quiesce_hold_ms": record.get("quiesce_hold_ms"),
        "commit_duration_ms": record.get("commit_duration_ms"),
    }


def _batch_hashes(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The batch provenance an ``mf_steps`` row carries (R-02.14's batch source hashes).

    A mapping rather than a bare string, because ``mf_steps.batch_source_hashes`` is plural
    and F2's batch block carries selector and grader config hashes beside the content hash.
    ``None`` when the step recorded none, which a fixture batch legitimately does.
    """
    value = record.get("batch_sha256")
    return None if value is None else {"batch_sha256": value}


def commit_duration_percentile(path: Path, percentile: float) -> float | None:
    """A percentile of ``commit_duration_ms`` over a run's step blocks (FR-001.26).

    **Why this exists at all, and why it reads the log.** miLLM Feature 31's commit watchdog
    is set to ten times the measured **p99.9** of this value, so the field is a cross-repo
    product obligation rather than telemetry - F1's duty is that the writer and the storage
    *preserve* it and keep it queryable as a distribution, not merely as a last value.

    It reads the log because the log is the authoritative record (ADR-004): the answer
    survives a dropped index, and ``mf_steps`` carries no column for this field today (see
    this feature's report).

    Returns ``None`` when no step carries a value, which is the **normal** standalone case -
    the default path has no commit bracket to time. ``None`` is not zero: zero would be a
    measured instantaneous commit, and reporting one would hand miLLM a watchdog of nothing.
    """
    if not 0.0 < percentile <= 100.0:
        raise refuse(
            "percentile_out_of_range",
            f"A percentile must be in (0, 100]; got {percentile}.",
            percentile=percentile,
        )
    values = sorted(
        float(record["commit_duration_ms"])
        for record in read_blocks(path, block="step")
        if record.get("commit_duration_ms") is not None
    )
    if not values:
        return None
    # Nearest-rank: ``ceil(p/100 * N)``, the definition that never interpolates between two
    # observed commits. For a watchdog threshold an interpolated value is a duration that was
    # never measured, and the gate is ten times this number. ``ceil`` rather than ``round``
    # so that p100 is the maximum and a p50 over an even count takes the upper of the two
    # middle values - erring toward the longer commit, which is the safe direction for a
    # threshold that protects availability.
    rank = max(1, math.ceil(percentile / 100.0 * len(values)))
    return values[min(rank, len(values)) - 1]
