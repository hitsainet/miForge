"""The longest heartbeat gap across a **complete** run.  ``gpu``-marked.  (F6 task 11.6)

OQ-5, and the input to foundation task 6.5's janitor threshold.

## What this can and cannot settle, stated before the number

F6 task 11.6 asks for *"the longest heartbeat gap across a full run of each type"* so the
janitor threshold can be set from a measurement rather than a guess.

**It is achievable for BOTH ``change`` and ``create`` as of 2026-10-16.** Each finishes in
the pod in well under a minute, and this file drives both to termination with a real
:class:`workers.heartbeat.Heartbeat`, a real scoring pass and a real checkpoint save inside
them.

*This paragraph said ``create`` was "NOT achievable, and the blocker is a defect rather than
a time limit" until 2026-10-16, and it was true when written.* The blocker was that
``acquisition/scratch.py`` returned ``(module, None)`` for the tokenizer while nothing in
``backend/src/`` supplied one, so a create run trained and then died at its first save point
on ``tokenizer.save_pretrained`` - R-02.41 producing no artifact at all. ``AcquisitionSpec``
now carries ``tokenizer_path`` / ``tokenizer_repo``, resolved **at submit** by
``ScratchAcquirer.preflight``, so naming none is a named refusal before the lease rather than
a traceback three phases later. The tripwire that recorded the defect fired and has been
replaced by the measurement it was holding the place for, plus a case asserting the refusal
from the other side.

**But the figure it produces is a LOWER BOUND, and setting the janitor threshold from it
directly would reap healthy runs.** That is the single most important sentence here, so the
reasoning is laid out rather than left implicit:

* The gaps inside a boundary are bounded by its longest sub-operation, and
  ``training/boundary.py`` advances the heartbeat after the step, after the test, after the
  save and after the refresh. So the longest gap in any run is the longest of those four.
* The step on this fixture is ~130 ms. **The test pass is ~4.8 s** at ``n`` = 16 against an
  86 M-parameter model, measured in ``test_scoring_pass_gpu.py``, and it is **99.9 %
  generation** - so it scales with ``n`` and with model size, neither of which this fixture
  represents. At BRD-02's own worked example (1.2 B parameters) and a realistic evaluation
  set, one test pass is **minutes**.
* And ``test_signal_source_throughput_gpu.py`` measures that generation under time-slicing
  at **2.1-2.4x** wall time. A gap measured on an idle card understates a shared one by that
  factor again.

So the measured maximum below is *"at least this"*, and the quantity the janitor threshold
actually needs is **the longest test pass and the longest save at production scale**. This is
the same hazard in the opposite direction from this project's recorded F9 defect, where a
liveness threshold *"matched to foundation 6.4's 60 s heartbeat"* would have shown every
healthy run as stalled during exactly these sub-operations: **60 s is a minimum interval
between writes, not a maximum gap**, and a threshold derived from a toy fixture's gap would
make the same mistake with a smaller number.

## Two types, and why not three

``change`` and ``create`` are both driven to completion here. **``adapter`` is still
deliberately absent**, and the reason is unchanged and was re-checked rather than carried
forward: ``training/runtypes/{adapter,base,change}.py``, ``training/tasks.py`` and
``training/run_driver.py`` were all **being edited by another agent** when this was written
(checked 2026-10-16), so a second file driving an adapter run would be a second owner of a
path in flight, and a figure taken from a half-written module is not a measurement.

The gap it would report is in any case bounded by the same two sub-operations - R-02.43's
adapter step is *cheaper* than a full-parameter one (no full-parameter gradients, no FP32
master copy of the base, no optimizer state over it) and holds no barrier, so its **step**
gap is strictly smaller, while its **test pass** is identical because scoring generates from
the same resident model. Since the longest gap is dominated by the test pass in both types
measured below, the adapter figure is bounded above by the ``change`` figure on the same
fixture. **Named as not-covered rather than silently omitted**, and the reason the figure is
unaffected is given rather than assumed.

**One caveat on that bound, found while re-scaling the adapter benchmark the same day:**
``base_parameter_digest`` refuses ``base_digest_unreadable`` on a **bfloat16** base, which is
the dtype ``write_bench_model`` and the checkpoint writer both emit - so an adapter run
against a real checkpoint currently refuses at ``prepare`` and has no gap series at all.
Pinned by ``tests/unit/test_base_digest_refuses_bfloat16.py``; owner is
``training/runtypes/adapter.py``.

## Sample and topology (ADR-012, ADR-024)

One **RTX 3090** on ``mcs-lnxhost02``, PCIe **x4 negotiated**, slot ``width.max`` 16. The
figures are read from the **run log's own** ``heartbeat_gap_ms``, which F6 records per
boundary as a float, and cross-checked against ``mf_runs.longest_heartbeat_gap_seconds``.

**Those two are not the same measurement and the difference matters here.** The column is an
``Integer`` of **seconds**, so it cannot distinguish a 0.9 s gap from a 0.1 s one; the run
log carries milliseconds. For a janitor threshold in minutes the column is adequate, but the
*input to choosing* that threshold is the run log, and this file reports both so a reader
does not take the coarser one for the measurement.

Run with::

    pytest -m gpu backend/tests/performance/test_heartbeat_gap_full_run_gpu.py -s
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.bench_model import BENCH_BATCH, BENCH_LLAMA, BENCH_SEQ, bench_prompts

pytestmark = pytest.mark.gpu

MAX_STEPS = 3
SAVE_EVERY = 3
TEST_EVERY = 2

#: The evaluation set size. Small so the run completes as a gate; **named** because the gap
#: this file measures is dominated by the test pass and therefore by this number.
EVAL_N = 8
EVAL_TOKENS = 16


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def device():
    from core.errors import Refusal
    from core.gpu import resolve_device

    try:
        return resolve_device("auto")
    except Refusal as exc:
        pytest.skip(f"no CUDA device resolved ({exc.code}); this case needs the GPU node")


@pytest.fixture
def roots(tmp_path: Path):
    import core.config as config
    from core.config import Settings, reset_settings

    checkpoints = tmp_path / "checkpoints"
    parts = tmp_path / "parts"
    states = tmp_path / "trainer-state"
    for directory in (checkpoints, parts, states):
        directory.mkdir(parents=True, exist_ok=True)
    original = config._settings
    config._settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        checkpoint_root=checkpoints,
        parts_root=parts,
        trainer_state_root=states,
    )
    yield checkpoints, states
    config._settings = original
    reset_settings()


@pytest.fixture
def session(tmp_path: Path):
    from tests.support import scores as fixtures
    from tests.support.db import migrated_engine, session_for

    engine = migrated_engine(tmp_path / "db")
    fixtures.create_score_tables(engine)
    with session_for(engine) as open_session:
        yield open_session
    engine.dispose()


# --------------------------------------------------------------------------- the run's parts


class _Batch:
    """A real tokenised batch on the device, with a real provenance record.

    F2 owns the real sources; this is the minimum ``training/step.py`` reads, built here
    because a *source* would bring part resolution and selection into a measurement of
    heartbeat gaps.
    """

    objective = "sft"

    def __init__(self, input_ids: Any) -> None:
        self.input_ids = input_ids
        self.record = {"batch_sha256": "a" * 64}


class _Source:
    def __init__(self) -> None:
        import torch

        self.batch = _Batch(
            torch.randint(0, BENCH_LLAMA["vocab_size"], (BENCH_BATCH, BENCH_SEQ), device="cuda")
        )

    def next(self) -> _Batch:
        return self.batch


class _CrossEntropy:
    """**A fixture objective, in a test file** - F1 FR-001.17 forbids one in ``src``.

    *"Write no cross-entropy there, not even as a placeholder: it would be a second
    definition of F2's contract and it is the one that would survive."*
    """

    name = "sft"

    def loss(self, model: Any, batch: _Batch) -> Any:
        return model(input_ids=batch.input_ids, labels=batch.input_ids).loss


class _Objectives:
    def __init__(self) -> None:
        self.objective = _CrossEntropy()

    def get(self, name: str) -> _CrossEntropy:
        return self.objective


def _evaluator(handle: Any, session: Any, log_path: Path):
    """A **real** scoring pass as the run's ``evaluate`` callable.

    This is the sub-operation that produces the longest gap, so stubbing it would make the
    whole measurement meaningless - the figure would be the step's gap and would miss the
    thing the janitor threshold has to survive.
    """
    from providers.base import GenerationParams
    from providers.in_process import InProcessProvider
    from services.scoring import expectations, retention, service, sets

    expectation = expectations.resolve_expectation(
        {"kind": "rule", "name": "len", "config_sha256": None, "expression": '{"min_length": 1}'},
        source="pack",
    )

    def resolved(kind: str):
        reference = sets.SetReference(
            rendered=f"{kind}_set/general@1",
            structured={"type": f"{kind}_set", "name": "general", "version": "1"},
        )
        return sets.resolve(
            kind,
            reference,
            [{"prompt": text, "group": "g0"} for text in bench_prompts(EVAL_N)],
            markers=("held_out", "evaluation_only"),
        )

    def evaluate(step: int) -> Any:
        spec = service.EvaluationSpec(
            run_id=log_path.parent.name,
            provider_name="in_process",
            aggregation="mean",
            decode=GenerationParams(max_tokens=EVAL_TOKENS, temperature=0.0),
            test_set=resolved("test"),
            retention_set=resolved("retention"),
            test_expectation=expectation,
            retention_expectation=expectation,
            floor=retention.ResolvedFloor(value=0.0, layer="pack_slot"),
            batch_bound=EVAL_N,
        )
        outcome = service.evaluate(
            spec,
            step,
            session=session,
            log_path=log_path,
            provider=InProcessProvider(handle, batch_bound=EVAL_N),
        )
        session.commit()
        return outcome

    return evaluate


# --------------------------------------------------------------------------- reporting


def _gaps_from_log(log_path: Path) -> list[dict[str, Any]]:
    """Every boundary block's ``heartbeat_gap_ms``, read off disk.

    From the **run log**, which ADR-004 makes authoritative, rather than from the outcome
    objects the run returned - so the figure is the one a later reader of the run would get,
    not one that depends on this process still being alive.
    """
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text().splitlines():
        block = json.loads(line)
        if block.get("block") == "boundary":
            rows.append(
                {
                    "step": block["step"],
                    "gap_ms": block.get("heartbeat_gap_ms"),
                    "phases": block.get("phase_timings_ms", {}),
                }
            )
    return rows


def _report(run_type: str, rows: list[dict[str, Any]], persisted: Any) -> float:
    gaps = [row["gap_ms"] for row in rows if row["gap_ms"] is not None]
    longest = max(gaps) if gaps else 0.0
    print(f"\n── F6 11.6: longest heartbeat gap, run_type={run_type} ──")
    print("   RTX 3090, PCIe x4 negotiated (slot x16)")
    print(
        f"   run: {MAX_STEPS} steps, test_every={TEST_EVERY} (n={EVAL_N} per set x 2 sets, "
        f"<= {EVAL_TOKENS} tokens), save_every={SAVE_EVERY}; "
        f"model {BENCH_LLAMA['num_hidden_layers']}L/{BENCH_LLAMA['hidden_size']}h "
        f"batch {BENCH_BATCH} x seq {BENCH_SEQ} bf16"
    )
    for row in rows:
        phases = ", ".join(f"{name}={value:.0f}ms" for name, value in sorted(row["phases"].items()))
        print(f"   step {row['step']}: gap {row['gap_ms']} ms   [{phases}]")
    print(f"   LONGEST GAP (run log, ms): {longest}")
    print(f"   mf_runs.longest_heartbeat_gap_seconds: {persisted}  (Integer SECONDS - coarser)")
    print(
        "   ** LOWER BOUND. The gap is dominated by the test pass, which is 99.9 % "
        "generation and scales with n and model size; see this module's docstring before "
        "setting a janitor threshold from it. **"
    )
    return longest


# --------------------------------------------------------------------------- the runs


def _drive(
    *, run_type: str, spec: Any, roots: tuple[Path, Path], session: Any, tmp_path: Path, device
) -> tuple[list[dict[str, Any]], Any]:
    """Drive one run to completion with a real heartbeat, a real test pass and a real save.

    ``signal_source`` and ``domain`` are supplied because without them ``tasks.py`` refuses
    the weights save by name (``signal_source_unavailable``) and the run completes with an
    empty checkpoint list - *which still produces a gap series, and that is the trap*: the
    reported figure would silently exclude the save phase while reading as a full run. It
    did, until 2026-10-16.
    """
    import sqlalchemy as sa

    from providers.model_handle import AcquiredModelHandle
    from tests.support.jobs import insert_run
    from training import run_log, tasks
    from training.run_config import RunConfig
    from workers.heartbeat import Heartbeat
    from workers.run_record import RUNS

    _, states = roots
    run_id = f"hbgap-{run_type}"
    insert_run(session, run_id, status="running", run_type=run_type)
    session.commit()

    acquired = tasks.acquire_for_run(spec)
    log_root = tmp_path / "runs"
    log_path = run_log.run_log_path(tasks.run_log_root(log_root), run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # min_interval_seconds=0 so EVERY advance writes. Production's 60 s throttle would make
    # a 3-step run write once and report a single gap spanning the whole run - which is a
    # measurement of the throttle, not of the sub-operations.
    heartbeat = Heartbeat(run_id, min_interval_seconds=0)
    heartbeat.start(session)

    result = tasks.drive_run(
        run_id=run_id,
        spec=spec,
        cfg=RunConfig(
            learning_rate=1e-4, warmup_steps=0, batch_size=BENCH_BATCH, gradient_accumulation=1
        ),
        batch_source=_Source(),
        objectives=_Objectives(),
        session=session,
        heartbeat=heartbeat,
        max_steps=MAX_STEPS,
        save_every=SAVE_EVERY,
        test_every=TEST_EVERY,
        retention_every=TEST_EVERY,
        log_root=log_root,
        trainer_state_root=states,
        stop_conditions={"step_count": {"total": MAX_STEPS}},
        evaluate=_evaluator(AcquiredModelHandle(acquired), session, log_path),
        signal_source="corpus",
        domain="bench",
    )
    session.commit()
    assert result.steps_completed == MAX_STEPS, (
        f"the run did not complete ({result.steps_completed}/{MAX_STEPS} steps), so the "
        f"figure is not 'across a full run' and F6 11.6 is not discharged by it"
    )
    # THE POSITIVE CHECK THAT THE SAVE RAN. The gap this file reports is the longest
    # sub-operation, and the save is one of the four candidates; a run whose save was
    # refused produces a complete-looking gap series with that phase missing. Asserted
    # rather than assumed because that is exactly what happened before this line existed.
    saved = [row for row in _gaps_from_log(log_path) if "save" in row["phases"]]
    assert saved, (
        "no boundary block recorded a `save` phase, so the save was skipped or refused and "
        "the longest gap below is not 'across a full run' - it is across a run with one of "
        "its four candidate sub-operations absent"
    )
    persisted = session.execute(
        sa.select(RUNS.c.longest_heartbeat_gap_seconds).where(RUNS.c.id == run_id)
    ).scalar()
    return _gaps_from_log(log_path), persisted


class TestTheLongestGapAcrossAFullRun:
    """F6 11.6 for the two run types this file owns."""

    def test_change_run(self, roots, session: Any, tmp_path: Path, device) -> None:
        """A complete ``change`` run: a real base model, trained, tested and saved."""
        from tests.support.bench_model import write_bench_model
        from training.acquisition.base import AcquisitionSpec

        checkpoints, _ = roots
        written = write_bench_model(checkpoints / "base")
        rows, persisted = _drive(
            run_type="change",
            spec=AcquisitionSpec(path=written, gpu_request=device.uuid),
            roots=roots,
            session=session,
            tmp_path=tmp_path,
            device=device,
        )
        longest = _report("change", rows, persisted)
        assert rows, "no boundary blocks were written, so there is no gap series to report"
        assert longest > 0.0, (
            "every heartbeat gap was zero, which is not a measurement: the heartbeat was "
            "not advanced inside the boundary, or the run did no work between advances"
        )

    def test_create_run(self, roots, session: Any, tmp_path: Path, device) -> None:
        """A complete ``create`` run: a random-init model, trained, tested and saved.

        **This case was a tripwire asserting a defect until 2026-10-16**, and the tripwire
        fired. The defect was that ``acquisition/scratch.py`` returned ``(module, None)`` for
        the tokenizer and **nothing in** ``backend/src/`` **supplied one**, so a create run
        trained and then died at its first save point with a bare ``AttributeError`` on
        ``tokenizer.save_pretrained`` - R-02.41 produced no artifact at all, and the run
        could not be scored either (``TypeError`` in ``providers/model_handle.py``).

        **It is fixed, and the fix is a supply mechanism rather than a ``None`` check at
        either crash site** - which is what the tripwire's docstring argued for.
        ``AcquisitionSpec`` carries ``tokenizer_path`` / ``tokenizer_repo``,
        ``ScratchAcquirer.preflight`` resolves it **at submit** (so
        ``tokenizer_required``, the three source refusals and ``tokenizer_vocab_mismatch``
        all land before the lease and before a tensor is allocated), and ``acquire`` loads
        it onto the acquisition. Naming no tokenizer is now a **named refusal at submit**,
        not a traceback three phases later - verified by running it, not by reading it.

        So F6 task 11.6 is dischargeable for the second run type, and the figure is below.
        **It is a LOWER BOUND for the same reasons the ``change`` figure is** - see this
        module's docstring before setting a janitor threshold from it.
        """
        from tests.support.bench_model import BENCH_LLAMA
        from tests.support.tiny_hf_model import write_tiny_tokenizer
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec

        checkpoints, _ = roots
        # Inside the configured checkpoint root: `local_dir.resolve_within_roots` confines a
        # submitted tokenizer path to a configured model root, and `tmp_path` is outside it.
        # Sized from the config, because BENCH_LLAMA's vocab_size is 512 and a real 32k
        # tokenizer would hand the model an id outside its embedding table.
        tokenizer_dir = checkpoints / "given-tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        write_tiny_tokenizer(tokenizer_dir, vocab_size=BENCH_LLAMA["vocab_size"])

        rows, persisted = _drive(
            run_type="create",
            spec=AcquisitionSpec(
                mode=AcquisitionMode.CREATE,
                config=dict(BENCH_LLAMA),
                seed=1234,
                gpu_request=device.uuid,
                tokenizer_path=tokenizer_dir,
            ),
            roots=roots,
            session=session,
            tmp_path=tmp_path,
            device=device,
        )
        longest = _report("create", rows, persisted)
        assert rows, "no boundary blocks were written, so there is no gap series to report"
        assert longest > 0.0, (
            "every heartbeat gap was zero, which is not a measurement: the heartbeat was "
            "not advanced inside the boundary, or the run did no work between advances"
        )

    def test_a_create_run_naming_NO_tokenizer_is_REFUSED_at_submit(
        self, roots, session: Any, tmp_path: Path, device
    ) -> None:
        """The other half of the fix, and the half a happy-path test cannot show.

        The old defect's signature was that acquisition **succeeded** and the run died three
        phases later. What replaced it must therefore be checked from the failing side too:
        a create run naming no tokenizer is refused **by name, at preflight**, before the
        lease and before a tensor is allocated.

        *Kept as a permanent case rather than deleted with the tripwire*: if a later change
        makes ``tokenizer`` optional again, the happy path above still passes and only this
        one goes red.
        """
        from core.errors import Refusal
        from tests.support.bench_model import BENCH_LLAMA
        from training import tasks
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec

        with pytest.raises(Refusal) as caught:
            tasks.acquire_for_run(
                AcquisitionSpec(
                    mode=AcquisitionMode.CREATE,
                    config=dict(BENCH_LLAMA),
                    seed=1234,
                    gpu_request=device.uuid,
                )
            )
        assert caught.value.code == "tokenizer_required", (
            f"a create run naming no tokenizer was not refused `tokenizer_required` but "
            f"`{caught.value.code}`. If it is no longer refused at all, the run will reach "
            f"its first save point and raise AttributeError there instead - which is the "
            f"defect this case exists to keep closed."
        )
