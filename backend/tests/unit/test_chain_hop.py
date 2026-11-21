"""The hop: the previous stage's last checkpoint, a new run id, a fresh optimizer.

FR-006.40, .41, .42, .44; AC-006.29, AC-006.30; E-16, E-17.

**Checkpoints here are written by the real writer** (``tests.support.checkpoints``), not
planted, so the manifest the hop reads is the manifest a save produces and the weights hash
the branch resolution verifies is the hash of the bytes on disk. A planted manifest with
``weights_sha256`` of ``"b" * 64`` would fail that verification - which is the point: the
resolution F6 calls is not a directory lookup.

## What these tests prove, and what they deliberately do not

F6 chooses **which** checkpoint a hop starts from and routes it through F5. The hop's first
*manifest* is written by F5's writer at the stage's first save, from the plan F5 itself
produced - so AC-006.29's ``parent`` and AC-006.30's ``optimizer.state = "fresh"`` are
asserted here **at F6's boundary**: on the resolution F6 performs, through the real
:func:`services.checkpoints.resume.branch_plan`. Driving a real save would require a
training step and a card (ADR-023), and F5's own suite covers the writer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from services.chains import service as chain_service
from services.chains.tables import CHAINS
from tests.support import checkpoints as ckpt
from tests.support.db import migrated_engine, session_for
from tests.unit.test_chain_service import GPU, _set_terminal, _stage, spec, spies
from workers.run_record import RUNS


@pytest.fixture
def engine(tmp_path: Path):
    built = migrated_engine(tmp_path / "db")
    yield built
    built.dispose()


@pytest.fixture
def logs(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture
def ckpt_root(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints"


def _create(session: Any, logs: Path, **over: Any) -> str:
    return str(chain_service.create(session, spec(**over), log_root=logs, **spies())["chain_id"])


class TestTheHopStartsFromTheLastCheckpoint:
    def test_the_next_stage_loads_the_previous_stages_last_save(
        self, engine, logs, ckpt_root, caplog
    ) -> None:
        """FR-006.40, AC-006.29. **The LAST save, not the first.**

        Three saves are written. Starting the next stage from step 100 would silently
        discard two hundred steps of training and report success.
        """
        with session_for(engine) as session:
            chain_id = _create(session, logs)
            previous = str(_stage(session, chain_id, 0).run_id)
            written = ckpt.write_chain(ckpt_root, run_id=previous, steps=(100, 200, 300))
            _set_terminal(session, previous, stop_reason="step_count")

            with caplog.at_level(logging.INFO, logger="services.chains.service"):
                outcome = chain_service.advance_one(
                    session, chain_id, log_root=logs, checkpoint_root=ckpt_root, **spies()
                )

        assert outcome["action"] == "started"
        assert outcome["stage_index"] == 1
        assert outcome["source"]["mode"] == "load"
        assert outcome["source"]["path"] == str(written[-1].directory)

        resolved = [
            record
            for record in caplog.records
            if getattr(record, "code", None) == "chain_hop_resolved"
        ]
        assert len(resolved) == 1, "one resolution per hop, through F5, and only one"
        # AC-006.29: what the hop's first save will record as `parent` is the previous
        # stage's LAST checkpoint, carried by F5's own plan rather than chosen here.
        assert resolved[0].parent == written[-1].name
        assert resolved[0].step == 300
        # AC-006.30: `fresh`, by construction - trainer state is addressed by `run_id` and a
        # hop takes a new one (FR-006.42, DEF-6).
        assert resolved[0].optimizer_state == "fresh"

    def test_the_hop_is_a_branch_with_a_new_run_id(self, engine, logs, ckpt_root) -> None:
        """FR-006.41: a run records exactly one ``signal_source``, so one ``run_id`` cannot
        carry two stages."""
        with session_for(engine) as session:
            chain_id = _create(session, logs)
            previous = str(_stage(session, chain_id, 0).run_id)
            ckpt.write_chain(ckpt_root, run_id=previous, steps=(100,))
            _set_terminal(session, previous, stop_reason="step_count")
            chain_service.advance_one(
                session, chain_id, log_root=logs, checkpoint_root=ckpt_root, **spies()
            )
        with session_for(engine) as session:
            hop = str(_stage(session, chain_id, 1).run_id)
            assert hop != previous
            run = session.execute(sa.select(RUNS).where(RUNS.c.id == hop)).one()
            assert run.chain_id == chain_id and run.stage_index == 1
            assert run.definition_sha256 is not None
            # The chain moved; the previous stage is closed out.
            chain = session.execute(sa.select(CHAINS)).one()
            assert chain.current_stage == 1 and chain.status == "running"
            assert _stage(session, chain_id, 0).finished_at is not None

    def test_no_trainer_state_is_copied_forward(self, engine, logs, ckpt_root, tmp_path) -> None:
        """FR-006.42, AC-006.30, and **F6 invents no copy-forward.**

        No requirement authorises copying another run's trainer state, so a curriculum
        carries R-02.22's loss discontinuity once per stage - DEF-6, recorded rather than
        fixed. The assertion is on the **filesystem**: a hop leaves no trainer-state
        directory for the new run.
        """
        state_root = tmp_path / "trainer-state"
        state_root.mkdir()
        with session_for(engine) as session:
            chain_id = _create(session, logs)
            previous = str(_stage(session, chain_id, 0).run_id)
            (state_root / previous / "step-000100").mkdir(parents=True)
            ckpt.write_chain(ckpt_root, run_id=previous, steps=(100,))
            _set_terminal(session, previous, stop_reason="step_count")
            chain_service.advance_one(
                session, chain_id, log_root=logs, checkpoint_root=ckpt_root, **spies()
            )
            hop = str(_stage(session, chain_id, 1).run_id)
        assert not (state_root / hop).exists()
        assert (state_root / previous).exists(), "the previous run's state is not moved either"


class TestAStageWithNoSaveHaltsTheChain:
    def test_chain_no_checkpoint(self, engine, logs, ckpt_root) -> None:
        """E-16. R-02.27's mechanism **is** the checkpoint.

        Starting the next stage from the base model would discard the completed stage
        without saying so - a curriculum that silently trained its second stage on the
        model its first stage started from.
        """
        ckpt_root.mkdir()
        with session_for(engine) as session:
            chain_id = _create(session, logs)
            previous = str(_stage(session, chain_id, 0).run_id)
            _set_terminal(session, previous, stop_reason="step_count")
            outcome = chain_service.advance_one(
                session, chain_id, log_root=logs, checkpoint_root=ckpt_root, **spies()
            )
        assert outcome["action"] == "halted"
        assert outcome["reason"] == "chain_no_checkpoint"
        with session_for(engine) as session:
            assert session.execute(sa.select(CHAINS)).one().halt_code == "chain_no_checkpoint"
            assert _stage(session, chain_id, 1).run_id is None

    def test_a_corrupt_checkpoint_is_not_a_halt(self, engine, logs, ckpt_root, caplog) -> None:
        """A hash mismatch is a **disk** problem an operator can fix, and ``halted`` is
        terminal.

        So it is raised rather than halting the chain: discarding a recoverable curriculum
        because one file was damaged is the irreversible answer to a reversible problem.
        Only *"there is no checkpoint at all"* is E-16's halt.
        """
        from core.errors import Refusal

        # Three sessions, because the refusal's rollback must discard only the advance. One
        # session would roll the chain's own creation back with it and the assertion below
        # would pass against a chain that never existed.
        with session_for(engine) as session:
            chain_id = _create(session, logs)
            previous = str(_stage(session, chain_id, 0).run_id)
            written = ckpt.write_chain(ckpt_root, run_id=previous, steps=(100,))
            (written[-1].directory / "model.safetensors").write_bytes(b"corrupted")
            _set_terminal(session, previous, stop_reason="step_count")
        with session_for(engine) as session:
            with pytest.raises(Refusal) as caught:
                chain_service.advance_one(
                    session, chain_id, log_root=logs, checkpoint_root=ckpt_root, **spies()
                )
            session.rollback()
        assert caught.value.code == "checkpoint_hash_mismatch"
        with session_for(engine) as session:
            chain = session.execute(sa.select(CHAINS)).one()
            assert chain.status == "running", "a damaged file must not discard the curriculum"
            assert chain.halt_code is None


class TestTheLeaseIsTheChainsForItsWholeDuration:
    def test_a_hop_renews_rather_than_re_acquiring(self, engine, logs, ckpt_root) -> None:
        """FR-006.44: **one** lease row for the whole curriculum.

        A hop that released and re-acquired could lose the card between the two, and a
        blind re-acquire could land on a different card - which silently changes the device
        every later comparison was taken on.
        """
        from tests.support.jobs import read_leases

        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, spec(), log_root=logs, **spies(gpu_uuid=GPU))[
                    "chain_id"
                ]
            )
            previous = str(_stage(session, chain_id, 0).run_id)
            ckpt.write_chain(ckpt_root, run_id=previous, steps=(100,))
            _set_terminal(session, previous, stop_reason="step_count")
            chain_service.advance_one(
                session,
                chain_id,
                log_root=logs,
                checkpoint_root=ckpt_root,
                **spies(gpu_uuid=GPU),
            )
        with session_for(engine) as session:
            rows = read_leases(session, GPU)
            assert len(rows) == 1, "a hop must not release and re-acquire (FR-006.44)"
            assert rows[0]["holder_run_id"] == f"chain:{chain_id}"
            assert rows[0]["released_at"] is None

    def test_a_lost_lease_halts_and_does_not_re_acquire(self, engine, logs, ckpt_root) -> None:
        """E-17, FR-006.44. The card is named, and **nothing is re-acquired.**"""
        from tests.support.jobs import read_leases
        from workers import lease as lease_module

        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, spec(), log_root=logs, **spies(gpu_uuid=GPU))[
                    "chain_id"
                ]
            )
            previous = str(_stage(session, chain_id, 0).run_id)
            ckpt.write_chain(ckpt_root, run_id=previous, steps=(100,))
            _set_terminal(session, previous, stop_reason="step_count")
            # The chain's lease goes away and someone else takes the card.
            lease_module.release(session, f"chain:{chain_id}", GPU)
            from core.gpu import Device

            lease_module.acquire(
                session, "another-run", "auto", device=Device(uuid=GPU, name="", index=0)
            )
            outcome = chain_service.advance_one(
                session,
                chain_id,
                log_root=logs,
                checkpoint_root=ckpt_root,
                **spies(gpu_uuid=GPU),
            )

        assert outcome["action"] == "halted"
        assert outcome["reason"] == "chain_lease_lost"
        assert outcome["gpu_uuid"] == GPU
        assert outcome["holder_run_id"] == "another-run"
        with session_for(engine) as session:
            assert session.execute(sa.select(CHAINS)).one().halt_code == "chain_lease_lost"
            holders = [row["holder_run_id"] for row in read_leases(session, GPU)]
            assert (
                holders.count(f"chain:{chain_id}") == 1
            ), "the released row stays released; a blind re-acquire would add a second"
            assert _stage(session, chain_id, 1).run_id is None

    def test_a_hop_RENEWS_the_chains_lease(self, engine, logs, ckpt_root) -> None:
        """**MC-9's regression.** A sweep that never renewed would let the lease lapse.

        FR-006.44 holds the card for the *whole duration*, and a lease is not a boolean - it
        has an expiry, precisely so a dead holder does not hold it forever. The consequence
        of never renewing is the one failure mode that looks like nothing for an hour and
        then hands a four-hour curriculum's card to another run: ``reclaim_expired`` is run
        by *any* other acquirer, so a lapsed-but-unreleased lease stops blocking the index.
        """
        from tests.support.jobs import read_leases

        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, spec(), log_root=logs, **spies(gpu_uuid=GPU))[
                    "chain_id"
                ]
            )
            previous = str(_stage(session, chain_id, 0).run_id)
            ckpt.write_chain(ckpt_root, run_id=previous, steps=(100,))
        with session_for(engine) as session:
            before = read_leases(session, GPU)[0]["expires_at"]
            # Wind the expiry back so the renewal is a measurable move rather than a
            # same-second no-op the assertion could not see.
            from datetime import timedelta

            from workers.run_record import LEASES

            session.execute(
                sa.update(LEASES)
                .where(LEASES.c.resource == GPU)
                .values(expires_at=before - timedelta(seconds=300))
            )
            wound_back = read_leases(session, GPU)[0]["expires_at"]
            _set_terminal(session, previous, stop_reason="step_count")

        with session_for(engine) as session:
            chain_service.advance_one(
                session,
                chain_id,
                log_root=logs,
                checkpoint_root=ckpt_root,
                **spies(gpu_uuid=GPU),
            )
        with session_for(engine) as session:
            rows = read_leases(session, GPU)
            assert len(rows) == 1, "renewed in place, not released and re-acquired"
            assert rows[0]["expires_at"] > wound_back, (
                "the hop did not renew the chain's lease; it would lapse mid-curriculum and "
                "the next run to call acquire() would reclaim the card"
            )

    def test_the_start_path_refuses_when_the_card_changed_hands(
        self, engine, logs, ckpt_root
    ) -> None:
        """**MC-6's regression**, on the path where that guard is the ONLY lease check.

        ``advance_one``'s own check runs on the advance decision; the **recovery** path -
        a current stage the rows say has no run - starts the stage without one, so
        :func:`services.chains.service._chain_lease` is what stands there. And it is
        reachable rather than defensive: ``lease.acquire`` reclaims any *expired*
        unreleased row for a card, so a chain whose lease lapsed can genuinely find the
        card held by another run by the time it starts a stage.
        """
        from core.errors import Refusal
        from core.gpu import Device
        from workers import lease as lease_module

        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, spec(), log_root=logs, **spies(gpu_uuid=GPU))[
                    "chain_id"
                ]
            )
            previous = str(_stage(session, chain_id, 0).run_id)
            ckpt.write_chain(ckpt_root, run_id=previous, steps=(100,))
            # The rows say stage 1 is current and unstarted - the recovery shape.
            session.execute(
                sa.update(CHAINS).where(CHAINS.c.id == chain_id).values(current_stage=1)
            )
            lease_module.release(session, f"chain:{chain_id}", GPU)
            lease_module.acquire(
                session, "another-run", "auto", device=Device(uuid=GPU, name="", index=0)
            )
        with session_for(engine) as session:
            with pytest.raises(Refusal) as caught:
                chain_service.advance_one(
                    session,
                    chain_id,
                    log_root=logs,
                    checkpoint_root=ckpt_root,
                    **spies(gpu_uuid=GPU),
                )
            session.rollback()
        assert caught.value.code == "chain_lease_lost"
        assert caught.value.details["gpu_uuid"] == GPU
        assert caught.value.details["holder_run_id"] == "another-run"
        with session_for(engine) as session:
            assert (
                _stage(session, chain_id, 1).run_id is None
            ), "a stage must not start on a card the chain no longer holds"

    def test_the_lease_is_checked_before_the_checkpoint_is_read(
        self, engine, logs, ckpt_root
    ) -> None:
        """The order in the module docstring, asserted: a chain that has lost its card does
        not read a manifest to find out.

        With **no checkpoint on disk at all**, a lost lease must still report
        ``chain_lease_lost`` rather than ``chain_no_checkpoint`` - which is only true if the
        lease check runs first.
        """
        from workers import lease as lease_module

        ckpt_root.mkdir()
        with session_for(engine) as session:
            chain_id = str(
                chain_service.create(session, spec(), log_root=logs, **spies(gpu_uuid=GPU))[
                    "chain_id"
                ]
            )
            previous = str(_stage(session, chain_id, 0).run_id)
            _set_terminal(session, previous, stop_reason="step_count")
            lease_module.release(session, f"chain:{chain_id}", GPU)
            outcome = chain_service.advance_one(
                session,
                chain_id,
                log_root=logs,
                checkpoint_root=ckpt_root,
                **spies(gpu_uuid=GPU),
            )
        assert outcome["reason"] == "chain_lease_lost"
