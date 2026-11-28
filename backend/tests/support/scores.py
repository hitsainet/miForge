"""F4's fixtures - **round-tripped, never hand-built**  (ADR-016; FPRD §12).

ADR-016's rule, and this is the feature where it matters most: *"fixtures whose fields agree
by construction are how a suite stays green over a real bug."* The concrete bug here, since
the 2026-10-13 re-vendor, is a ``ScoreRecord`` written **without** its declared ``provider``:
it validates, because the field is declared and **not required**, and a hand-built fixture
would simply omit it too - so the test and the defect would agree with each other and the
suite would stay green over exactly the thing F4 exists to make catchable.

So the manifest fixtures here go through **F5's real writer and reader**, and the score
fixtures go through F4's real builder.

``mf_scores`` and ``mf_score_items`` live on F4's own ``MetaData`` and are **not** in the
Alembic migration yet - the revision graph is shared state and the coordinator owns it - so
:func:`create_score_tables` emits them onto the migrated engine. When the migration lands,
this function becomes a no-op and the tables come from the artefact production applies, which
is the arrangement ``support/db.py`` explains at length.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from providers.base import GenerationParams, ProviderIdentity
from services.scoring import record as record_module
from services.scoring.tables import KIND_TEST
from services.scoring.tables import metadata as score_metadata

#: A provider identity with every field an in-process score needs. Built rather than
#: hand-written so ``record()`` - the one rendering of a provider - is what the tests assert.
IN_PROCESS = ProviderIdentity(
    kind="in_process",
    model="fake-model",
    base_url=None,
    engine_version="fake-parameter-set@step0",
)

HTTP = ProviderIdentity(
    kind="openai_compatible", model="judge-7b", base_url="http://judge.test:8000"
)

GREEDY = GenerationParams(max_tokens=32, temperature=0.0, top_p=1.0)


def create_score_tables(engine: Any) -> None:
    """Emit F4's tables onto a migrated engine. See the module docstring."""
    score_metadata.create_all(engine)


def score_record(
    *,
    run_id: str = "run-1",
    step: int = 0,
    kind: str = KIND_TEST,
    set_rendered: str = "test_set/general@1",
    score: float = 0.84,
    n: int = 200,
    floor: float | None = None,
    floor_layer: str = "none",
    breached: bool = False,
    provider: ProviderIdentity = IN_PROCESS,
    aggregation: str = "mean",
    decode: GenerationParams = GREEDY,
    subsample_seed: int | None = None,
    split_markers: tuple[str, ...] | None = ("evaluation_only",),
    created_at: datetime | None = None,
    **extra: Any,
) -> record_module.ScoreRecordData:
    """One :class:`~services.scoring.record.ScoreRecordData`, through the real type.

    Through the real type rather than a dict so that a field added to the record appears in
    every fixture at once - and so that the projections (``row``, ``score_record``,
    ``score_entry``, ``api_payload``) are the ones under test rather than a second
    hand-written copy of them.
    """
    return record_module.ScoreRecordData(
        run_id=run_id,
        step=step,
        kind=kind,
        set_rendered=set_rendered,
        set_ref={"type": "test_set", "name": "general", "version": "1"},
        score=score,
        n=n,
        floor=floor,
        floor_layer=floor_layer,
        breached=breached,
        provider=provider,
        parameter_set_digest=provider.engine_version,
        aggregation=aggregation,
        decode=decode,
        subsample_seed=subsample_seed,
        split_markers=split_markers,
        created_at=created_at or datetime.now(UTC),
        **extra,
    )


def written_and_read_manifest(
    root: Path,
    *,
    run_id: str = "run-1",
    step: int = 100,
    test: dict[str, Any] | None,
    retention: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write a manifest through **F5's real builder**, read it back off the disk, return it.

    A hand-built dict agrees with the test's expectation **by construction** - which is
    exactly how this suite would stay green over a ScoreRecord written without its declared
    ``provider``. Going through the real builder means the assertion is about what F5 would
    actually write.
    """
    import json

    from services.checkpoints.manifest import build

    manifest = build(
        run_id=run_id,
        step=step,
        domain=None,
        signal_source="corpus",
        base_model="LFM2-1.2B",
        parent=None,
        optimizer={"name": "adamw-8bit", "state": "fresh"},
        weights_sha256="0" * 64,
        test=test,
        retention=retention,
    )
    directory = root / f"{run_id}-{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return dict(json.loads(path.read_text(encoding="utf-8")))


class RecordingEmitter:
    """A socket emitter that records **what** it was called with and **how often**.

    Both, because ADR-016 requires the payload and the call count: *"was called" passes
    against a call sending the wrong arguments* - which is how a breach could be published
    for the wrong step.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(self, run_id: str, event: str, payload: dict[str, Any]) -> None:
        self.calls.append((run_id, event, payload))

    def events(self, name: str) -> list[dict[str, Any]]:
        return [payload for _run, event, payload in self.calls if event == name]

    @property
    def count(self) -> int:
        return len(self.calls)


class RecordingPublisher:
    """F5's manifest-write seam, recorded. One call per save, both arguments present."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []

    def __call__(self, test: dict[str, Any] | None, retention: dict[str, Any] | None) -> None:
        self.calls.append((test, retention))

    @property
    def count(self) -> int:
        return len(self.calls)


class StandInStopCondition:
    """F6's stop condition, stood in for, so the breach signal is proved **consumable**.

    F4 publishes and F6 stops (R-02.25). A test that only asserted the publication would
    prove the signal was emitted, not that anything downstream could read it - so this reads
    the published payload the way a stop condition would and records what it decided.
    """

    def __init__(self) -> None:
        self.observed: list[dict[str, Any]] = []

    def consume(self, payloads: Sequence[dict[str, Any]]) -> bool:
        self.observed = list(payloads)
        return any(
            payload["score"] < payload["floor"] for payload in payloads if payload.get("floor")
        )
