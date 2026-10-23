"""The score record: run log, then index, then socket  (FR-004.9, 18, 28, 30; ADR-004, ADR-007).

**That order, not another.** The on-disk run log is the authoritative record (ADR-004), the
``mf_scores`` row is the index over it, and the Socket.IO emission is a convenience. A crash
between them loses a projection, never a record. The reason is recorded rather than assumed:
*"progress reported only over the socket is invisible to the janitor"* - which is how miStudio
reaped a live 5.8-hour job as "worker lost", and a generation-bound scoring pass over a large
test set is exactly that shape of job.

A breach is published the same way: record, index, then emit. **Never emit-only** - a
detected breach nobody is told about is the same failure as not detecting it (M-2).

## Every field here exists because a number without it is not a measurement

R-02.16: *"a score is only comparable against another taken the same way."* The provider, the
model identity, the parameter-set digest, the decode parameters, the aggregation method, the
subsample seed and the split markers are each as load-bearing as the number. ``ScoreRecord``
carries the first of those on the artifact since 2026-10-13; the rest live here and in the
run log's evaluation block, and DEF-2 keeps their manifest home an open question.

## A re-score appends

No ``UPDATE`` on a score row exists anywhere in ``backend/src``, and a test asserts it. A
mutated history is worse than a duplicated one: the duplicate is visible.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from providers.base import GenerationParams, ProviderIdentity
from services.scoring import evaluation_block as blocks
from services.scoring import manifest_payload
from services.scoring.expectations import PromptOutcome
from services.scoring.retention import BREACH_EVENT, BreachSignal
from services.scoring.tables import FLOOR_LAYER_NONE, KIND_RETENTION, SCORE_ITEMS, SCORES
from training import run_log

log = logging.getLogger("miforge.scoring")

#: The socket event one evaluation publishes into ``miforge/run/<id>``.
SCORE_EVENT = "score"

#: How much of a completion is kept for a drill-down. Truncated because a run view's
#: drill-down is not a transcript, and because ``mf_score_items`` grows with
#: ``n x evaluations`` (FPRD OQ-6, which has no answer and is reported).
COMPLETION_EXCERPT_CHARS = 500


@dataclass(frozen=True, slots=True)
class ScoreRecordData:
    """One evaluation's durable record. **The shape F4 writes, in one place.**

    ``kind`` is F4's own axis and is deliberately absent from both contracts: the manifest
    reaches a ScoreRecord through its ``test`` and ``retention`` properties, and the run
    log's ``ScoreEntry`` distinguishes them by whether ``floor`` is set.
    """

    run_id: str
    step: int
    kind: str
    set_rendered: str
    set_ref: Mapping[str, Any] | None
    score: float
    n: int
    floor: float | None
    floor_layer: str
    breached: bool
    provider: ProviderIdentity
    parameter_set_digest: str | None
    aggregation: str
    decode: GenerationParams
    subsample_seed: int | None
    split_markers: tuple[str, ...] | None
    selector_hash: str | None = None
    grader_hash: str | None = None
    reproducible: bool = True
    not_reproducible_reason: str | None = None
    out_of_cadence: bool = False
    provider_discontinuity: bool = False
    rescore_reason: str | None = None
    part_id: str | None = None
    outcomes: tuple[PromptOutcome, ...] = ()
    completions: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # ---------------------------------------------------------------- projections
    def score_record(self) -> dict[str, Any]:
        """The five ``ScoreRecord`` fields, keyed as the schema declares them.

        ``provider`` is the **rendered string**, because ``ScoreRecord.provider`` is typed
        ``anyOf [string, null]`` - the schema decides the shape, not this module. The full
        identity goes to the evaluation block.
        """
        return {
            "set": self.set_rendered,
            "score": self.score,
            "n": self.n,
            "floor": self.floor,
            "provider": self.provider.record(),
        }

    def score_entry(self) -> dict[str, Any]:
        """One run-log ``ScoreEntry``."""
        return blocks.score_entry(
            set_ref=self.set_rendered,
            score=self.score,
            n=self.n,
            floor=self.floor,
            provider=self.provider.record(),
        )

    def row(self) -> dict[str, Any]:
        """The ``mf_scores`` row. **No credential and no prompt text** (ADR-020, E-23)."""
        return {
            "run_id": self.run_id,
            "step": self.step,
            "kind": self.kind,
            "set_ref": dict(self.set_ref) if self.set_ref else None,
            "set_rendered": self.set_rendered,
            "part_id": self.part_id,
            "score": self.score,
            "n": self.n,
            "floor": self.floor,
            "floor_layer": self.floor_layer,
            "breached": self.breached,
            "provider": self.provider.record(),
            "provider_kind": self.provider.kind,
            "provider_model": self.provider.model,
            "provider_base_url": self.provider.base_url,
            "parameter_set_digest": self.parameter_set_digest,
            "aggregation": self.aggregation,
            "decode": blocks.decode_params_payload(self.decode),
            "subsample_seed": self.subsample_seed,
            "split_markers": None if self.split_markers is None else list(self.split_markers),
            "selector_hash": self.selector_hash,
            "grader_hash": self.grader_hash,
            "reproducible": self.reproducible,
            "not_reproducible_reason": self.not_reproducible_reason,
            "out_of_cadence": self.out_of_cadence,
            "provider_discontinuity": self.provider_discontinuity,
            "rescore_reason": self.rescore_reason,
            "created_at": self.created_at,
        }

    def api_payload(self) -> dict[str, Any]:
        """One score as an API response element.

        **Every number names its scale and sample** (ADR-012): ``n`` and the provider are
        on every element, so F9 cannot render a bare ``0.84`` even by accident. That is
        enforced by the payload rather than by F9's diligence, which is the point of
        FR-004.9 living here.
        """
        return {
            "run_id": self.run_id,
            "step": self.step,
            "kind": self.kind,
            "set": self.set_rendered,
            "score": self.score,
            "n": self.n,
            "floor": self.floor,
            "floor_layer": self.floor_layer,
            "breached": self.breached,
            "provider": self.provider.record(),
            "provider_kind": self.provider.kind,
            "provider_model": self.provider.model,
            "aggregation": self.aggregation,
            "subsample_seed": self.subsample_seed,
            "split_markers": None if self.split_markers is None else list(self.split_markers),
            "reproducible": self.reproducible,
            "not_reproducible_reason": self.not_reproducible_reason,
            "out_of_cadence": self.out_of_cadence,
            "provider_discontinuity": self.provider_discontinuity,
        }


def write_pass(
    records: Sequence[ScoreRecordData],
    *,
    session: Any,
    log_path: Path,
    breach: BreachSignal | None = None,
    emit: Callable[[str, str, dict[str, Any]], Any] | None = None,
    heartbeat: Any = None,
) -> dict[str, Any]:
    """Write one evaluation pass **durably, then index it, then announce it**.

    Args:
        records: The pass's score records - one per kind, so a paired measurement is two.
        session: An open session. **The caller commits**, so a pass whose transaction is
            rolled back leaves no row for a score it did not keep.
        log_path: The run's ``run.jsonl``.
        breach: The signal, when the retention score fell below its floor.
        emit: ``api.sockets.emit_run_event``, injected so a worker with no Redis reachable
            still completes a pass - the socket is a convenience, never the record.
        heartbeat: The foundation's throttled database heartbeat. A long generation pass is
            exactly the job shape a janitor reaps when it reports only over the socket.

    Returns:
        The ``evaluation`` block that was written, so a caller can assert it field by field.

    Raises:
        Refusal: ``step_record_invalid`` from the log writer when the block does not conform
            - **the pass fails and nothing is written**, because a record a reader cannot
            trust is worse than a gap.
    """
    if not records:
        from core.errors import refuse

        raise refuse(
            "evaluation_block_invalid",
            "A pass with no score records has nothing to write. A refusal is recorded by "
            "the caller; an empty pass is not.",
        )

    first = records[0]
    block = blocks.evaluation_block(
        run_id=first.run_id,
        step=first.step,
        scores=[record.score_entry() for record in records],
        provider=first.provider.record(),
        aggregation=first.aggregation,
        decode_params=blocks.decode_params_payload(first.decode),
        subsample_seed=first.subsample_seed,
        split=_merged_split(records),
        retention_floor_breached=breach is not None,
        ts=first.created_at,
    )

    # 1. THE AUTHORITATIVE WRITE. O_APPEND + fsync, validated against the schema first.
    run_log.append(log_path, block)

    # 2. The index over it.
    for record in records:
        result = session.execute(sa.insert(SCORES).values(**record.row()))
        score_id = _inserted_id(session, result, record)
        if record.outcomes:
            session.execute(
                sa.insert(SCORE_ITEMS),
                [
                    {
                        "score_id": score_id,
                        "prompt_index": outcome.index,
                        "prompt_id": None,
                        "verdict": outcome.verdict,
                        "scalar": outcome.scalar,
                        "reason_code": outcome.reason_code,
                        "completion_excerpt": _excerpt(record.completions, outcome.index),
                    }
                    for outcome in record.outcomes
                ],
            )

    if heartbeat is not None:
        # A pass over a large set legitimately exceeds the 60 s write interval, which is a
        # MINIMUM between writes and not a maximum gap. Advancing here is what keeps a live
        # pass from looking like a dead worker.
        heartbeat.touch(session)

    # 3. The convenience, and only after the record exists.
    if emit is not None:
        for record in records:
            emit(record.run_id, SCORE_EVENT, record.api_payload())
        if breach is not None:
            emit(breach.run_id, BREACH_EVENT, breach.payload())

    if breach is not None:
        log.warning(
            breach.message(),
            extra={
                "code": BREACH_EVENT,
                "run_id": breach.run_id,
                "step": breach.step,
                "score": breach.score,
                "floor": breach.floor,
                "n": breach.n,
                "provider": breach.provider,
            },
        )
    return block


def _merged_split(records: Sequence[ScoreRecordData]) -> list[str]:
    """The pass's split markers: every marker true on any set in the pass.

    One array for the pass because that is the block's shape. A test set and a retention set
    are both evaluation material, so in practice they agree; the union is the honest read
    when they do not, because the stronger claim - that some of this pass's data is held out
    - is the one a reader must not lose.
    """
    markers: set[str] = set()
    for record in records:
        markers.update(record.split_markers or ())
    return sorted(markers)


def _inserted_id(session: Any, result: Any, record: ScoreRecordData) -> int:
    """The new row's primary key, by whichever route the driver offers.

    ``inserted_primary_key`` is present on SQLite and PostgreSQL alike for a single-row
    insert; the fallback query exists for a driver that does not populate it, and it selects
    by the row's own unique key rather than by ``max(id)`` - which would pick up a
    concurrent run's row.
    """
    key = getattr(result, "inserted_primary_key", None)
    if key:
        return int(key[0])
    return int(
        session.execute(
            sa.select(SCORES.c.id).where(
                SCORES.c.run_id == record.run_id,
                SCORES.c.step == record.step,
                SCORES.c.kind == record.kind,
                SCORES.c.created_at == record.created_at,
            )
        ).scalar_one()
    )


def _excerpt(completions: Sequence[str], index: int) -> str | None:
    """A truncated completion for a drill-down, or ``None``.

    Truncated and **index-only**: it never enters a manifest, which travels with the artifact
    and is published beside a checkpoint (ADR-020).
    """
    if index >= len(completions):
        return None
    return completions[index][:COMPLETION_EXCERPT_CHARS]


# --------------------------------------------------------------------------------------
# The publication to F5's manifest writer  (FR-004.29, FR-004.30; M-4)
# --------------------------------------------------------------------------------------


def manifest_payload_for_step(session: Any, *, run_id: str, step: int) -> dict[str, Any]:
    """The two ScoreRecord payloads for one step - **both keys, always** (E-22).

    Reads the index rather than being handed the pass, so a save at a step evaluated earlier
    in the loop still carries that step's scores, and a save at a step where nothing ran
    writes ``test: null`` and ``retention: null`` **present**. ``null`` is an assertion that
    nothing ran; an absent key is an omission.

    The latest row per kind wins, which is the re-score rule: history appends, and the later
    line is the later fact.
    """
    rows = session.execute(
        sa.select(SCORES)
        .where(SCORES.c.run_id == run_id, SCORES.c.step == step, SCORES.c.out_of_cadence.is_(False))
        .order_by(SCORES.c.created_at)
    ).mappings()
    latest: dict[str, Any] = {}
    for row in rows:
        latest[str(row["kind"])] = {
            "set": row["set_rendered"],
            "score": row["score"],
            "n": row["n"],
            "floor": row["floor"],
            "provider": row["provider"],
        }
    return manifest_payload.manifest_scores(
        test=latest.get("test"), retention=latest.get(KIND_RETENTION)
    )


def publish_to_manifest(
    session: Any,
    *,
    run_id: str,
    step: int,
    publish: Callable[[dict[str, Any] | None, dict[str, Any] | None], Any],
) -> dict[str, Any]:
    """Hand F5 the two payloads. **The seam ADR-016 #1 requires a red for.**

    ``publish`` takes ``(test, retention)`` positionally and is called **exactly once per
    save**, with both arguments present and ``None`` where nothing ran. Removing this call
    must make ``backend/tests/unit/test_scoring_manifest_publication.py`` red, and the test
    asserts the **payload field by field and the call count** - *"was called" passes against
    a call sending the wrong arguments*, which is exactly how a score could reach the
    manifest with the wrong step's number.
    """
    scores = manifest_payload_for_step(session, run_id=run_id, step=step)
    publish(
        scores[manifest_payload.MANIFEST_TEST_KEY], scores[manifest_payload.MANIFEST_RETENTION_KEY]
    )
    return scores


def floor_layer_of(record: Mapping[str, Any] | None) -> str:
    """The floor layer a rebuilt row reports. ``none`` where the log cannot say.

    ``EvaluationBlock`` has no field for the layer and closes itself to unknown keys, so a
    reindexed row genuinely does not know which layer supplied the floor. Reported as
    ``none`` and **named as a gap** rather than guessed at ``pack_slot``, which would be a
    fabricated provenance for the one value FPRD OQ-2 says is unsettled.
    """
    del record
    return FLOOR_LAYER_NONE
