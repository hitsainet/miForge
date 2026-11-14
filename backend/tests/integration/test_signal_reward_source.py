"""The reward source, and the gate-contract gap that **has since closed**  (R-02.7).

**The gap is gone and this file no longer simulates it.** ``batch-record-v1.json``'s
``source`` enum was ``corpus | self_generation | preference_pairs | mixed`` - sharing
exactly one value with ``checkpoint-manifest-v1.json``'s ``signal_source``, which R-02.7
and FR-002.1 name - so a reward batch could not be recorded at all. On 2026-10-14 the two
enums were made one vocabulary: ``reward`` was added, ``mixed`` removed and the separators
aligned. So the reward source records normally, through the real contract, and every test
below runs against the file on disk.

**What happened to the two tests that used to be first in this file.** They asserted the
pre-flight refusal ``batch_source_not_in_contract`` and each opened with
``if "reward" in batch_record.source_enum(): pytest.skip(...)`` - a tripwire written so
that the day the contract carried ``reward`` the tests would say so rather than silently
become lies. **The tripwire fired**, and from that day the two tests skipped on every
machine, which is a check nobody was making. They are replaced rather than deleted,
because the refusal they covered is **not** dead code (see below) - and two further facts
are worth recording, since both are this project's false-measurement family again:

* a *skipped* test is a test that reports nothing, so its own correctness was never
  re-checked - and these two asserted ``details["contract_value"]``, **a key production
  has never set** (:func:`training.signals.batch_record.record_source` passes ``source``,
  ``enum`` and ``schema``). Had either ever run, it would have been red for a reason that
  has nothing to do with the contract;
* the refusal is reached through :meth:`training.signals.common.BaseSource.preflight`,
  which calls ``record_source(self.NAME)`` **before** it resolves a single part - so the
  class below asserts the ordering too (R-02.48: *a run that cannot proceed says so before
  it starts*).

The grader is a judge over a stub OpenAI-compatible server: the in-process provider cannot
be a judge (R-02.11) and ``rule`` produces no scalar, so a grader needs an endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.errors import Refusal
from services.selectors import provider_port
from services.selectors.settings import reset_selector_settings
from tests.support import db as support_db
from tests.support import signal_fixtures as fixtures
from tests.support.selector_stubs import StubJudgeServer, StubProviderResolver, judge_config
from training.signals import batch_index, batch_record, ensure_registered, registry
from training.signals.base import PartSpec, RunContext, SourceCapability, SourceSpec
from training.signals.common import BaseSource
from training.signals.settings import reset_signal_settings


class ScoringJudge(StubJudgeServer):
    """Scores by content so a group can have spread - or, deliberately, none."""

    def __init__(self, *, flat: bool = False) -> None:
        super().__init__()
        self.flat = flat

    def handler(self, request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content.decode())["messages"][-1]["content"]
        if self.flat:
            self.reply = "SCORE: 0.50\nREASON: identical"
        else:
            self.reply = (
                "SCORE: 0.90\nREASON: good" if "good" in content else "SCORE: 0.10\nREASON: poor"
            )
        return super().handler(request)


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    engine = support_db.migrated_engine(tmp_path / "db")
    return engine


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("SIGNAL_PENDING_SHORTFALL", "short_batch")
    monkeypatch.setenv("SIGNAL_REWARD_GROUP_SIZE", "2")
    monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", "Grade.\nQ: {prompt}\nA: {completion}")
    monkeypatch.setenv("JUDGE_PROVIDER_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("JUDGE_PROVIDER_MODEL", "judge-7b")
    reset_signal_settings()
    reset_selector_settings()
    yield
    provider_port.bind_provider_resolver(None)
    reset_signal_settings()
    reset_selector_settings()


def build(session: Any, tmp_path: Path, *, answers: Any, flat: bool = False, **context: Any) -> Any:
    provider_port.bind_provider_resolver(StubProviderResolver(ScoringJudge(flat=flat).provider()))
    part = fixtures.prompt_part(session, tmp_path, prompts=["q1", "q2"])
    ensure_registered()
    source = registry.get("reward")(
        RunContext(
            run_id="run-1",
            session=session,
            provider=fixtures.in_process_provider(answers),
            current_step=lambda: 5,
            **context,
        )
    )
    source.preflight(
        SourceSpec(
            source="reward",
            parts=(PartSpec(kind="prompt_set", part_id=str(part["id"])),),
            grader=judge_config(threshold=0.5),
            config_source="inline",
            group_size=2,
            groups_per_batch=2,
        )
    )
    return source


class PluginSource(BaseSource):
    """A source registered the way a plugin registers one, named outside the enum.

    *"The plugin interface is open, the recorded vocabulary is not."* This is the shape the
    refusal exists for: ``training.signals.registry.register`` is public and takes any
    name, while ``batch-record-v1.json``'s ``source`` enum is closed and
    ``checkpoint-manifest-v1.json``'s ``signal_source`` is **required**, so a batch filled
    by a fifth source could not be written down and the run that used it could not be
    saved. Nothing is hand-rolled here that production does not do: the class is a
    :class:`BaseSource`, so the pre-flight walked below is the real one.
    """

    NAME = "handwritten-signal"
    CAPABILITY = SourceCapability(
        name="handwritten-signal",
        part_kind="prompt_set",
        needs_selector=False,
        needs_grader=False,
        needs_generation=False,
        objective="sft",
        summary="a fifth source, registered by a plugin, that the contract cannot record",
    )

    def fill(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("pre-flight must refuse before anything fills a batch")


class TestASourceTheContractCannotRecordIsRefusedBeforeTheRunStarts:
    """The refusal is **live code on a live path**, and that is why it was not deleted.

    No *built-in* source can reach it: :func:`training.signals.registry.names` is asserted
    equal to the manifest's enum, so all four built-ins are recordable by construction -
    which is what made the old tests skip. The path that remains is the one the registry's
    own docstring describes as open: a plugin registering a fifth name. The guard is the
    **only** enforcement of that asymmetry, so deleting it as "unreachable" would have
    deleted the enforcement and left the documented promise with nothing behind it.
    """

    @pytest.fixture
    def plugin_source(self) -> Any:
        """Register the plugin source, and **prove the premise** before asserting on it."""
        name = PluginSource.NAME
        assert name not in batch_record.source_enum(), (
            "this test's premise is that the name is outside the contract's enum; if the "
            "enum ever carries it, choose another name rather than skipping"
        )
        assert name not in registry.names()
        registry.register(name, PluginSource, PluginSource.CAPABILITY)
        try:
            yield name
        finally:
            registry._FACTORIES.pop(name, None)
            registry._CAPABILITIES.pop(name, None)
            assert name not in registry.names(), "the registry was left mutated"

    def test_preflight_refuses_naming_the_enum_the_contract_carries(
        self, engine: Any, tmp_path: Path, plugin_source: str
    ) -> None:
        """R-02.48: *a run that cannot proceed says so before it starts, not partway
        through.* A batch that cannot be written down is a run that cannot proceed."""
        with support_db.session_for(engine) as session:
            source = registry.get(plugin_source)(RunContext(run_id="run-1", session=session))
            with pytest.raises(Refusal) as caught:
                source.preflight(
                    SourceSpec(
                        source=plugin_source,
                        parts=(PartSpec(kind="prompt_set", part_id="does-not-matter"),),
                    )
                )
        assert caught.value.code == "batch_source_not_in_contract"
        assert caught.value.details["source"] == plugin_source
        assert sorted(caught.value.details["enum"]) == sorted(batch_record.source_enum())
        assert caught.value.details["schema"] == "batch-record-v1.json"
        # The four recordable names are named in the refusal, so the operator is told what
        # is possible rather than only what failed (ADR-013, ADR-018).
        assert "reward" in caught.value.details["enum"]

    def test_the_refusal_comes_BEFORE_any_part_is_resolved(
        self, engine: Any, tmp_path: Path, plugin_source: str
    ) -> None:
        """The ordering is the requirement, not a detail.

        The part id above does not exist in F10's index, so a pre-flight that resolved
        parts first would refuse ``part_index_unavailable`` or ``part_directory_missing`` -
        a true refusal about the wrong thing, sending the operator to look for a dataset
        when the problem is the source's name. ``record_source`` is therefore called before
        ``resolve_parts``, and this asserts the code that comes back.
        """
        with support_db.session_for(engine) as session:
            source = registry.get(plugin_source)(RunContext(run_id="run-1", session=session))
            with pytest.raises(Refusal) as caught:
                source.preflight(
                    SourceSpec(
                        source=plugin_source,
                        parts=(PartSpec(kind="prompt_set", part_id="99999"),),
                    )
                )
        assert caught.value.code == "batch_source_not_in_contract"

    def test_nothing_is_recorded_under_another_source_name(
        self, engine: Any, tmp_path: Path, plugin_source: str
    ) -> None:
        """The failure mode this refusal exists to prevent: a batch from an unrecordable
        source written as ``corpus`` or ``reward`` would be a false provenance claim that
        nothing downstream could detect. So no index row and no run log are written."""
        with support_db.session_for(engine) as session:
            source = registry.get(plugin_source)(
                RunContext(run_id="run-1", session=session, log_root=tmp_path / "runs")
            )
            with pytest.raises(Refusal) as caught:
                source.preflight(
                    SourceSpec(
                        source=plugin_source,
                        parts=(PartSpec(kind="prompt_set", part_id="1"),),
                    )
                )
            # The CODE, not merely "a refusal": with the guard neutered, pre-flight still
            # refuses - further down, for the part it cannot resolve - so a bare
            # `pytest.raises(Refusal)` here passed against the mutation and proved nothing.
            # Found by running the control rather than by reading this test.
            assert caught.value.code == "batch_source_not_in_contract"
            assert batch_index.list_batches(session, run_id="run-1", limit=5) == []
        assert not (tmp_path / "runs").exists()


class TestEverythingBehindTheGapIsBuilt:
    """The whole path, **against the real vendored contract.**

    This class carried an autouse fixture that deep-copied the parsed schema and appended
    ``reward`` to the ``source`` enum, because the contract did not carry it. The contract
    carries it now, so the simulation was both redundant and dishonest - it would have kept
    passing against a *patched* enum if the real one ever regressed, which is the opposite
    of what a gate contract test is for. Removed, and with it the one remaining reason the
    run log was left out of these tests.
    """

    def test_a_reward_batch_carries_its_groups_and_their_statistics(
        self, engine: Any, tmp_path: Path
    ) -> None:
        """AC-002.10: min, max, mean and variance **and the grader's declared range**."""
        answers = iter(["good one", "poor one", "good two", "poor two"])
        with support_db.session_for(engine) as session:
            # WITH a `log_root`, which the simulated enum could not reach: the run log
            # validates the whole block against `step-record-v1.json`, whose batch payload
            # is a `$ref` into the file on disk. That reference now resolves to an enum
            # carrying `reward`, so the authoritative record is written here rather than
            # asserted about.
            source = build(
                session,
                tmp_path,
                answers=lambda prompt, index: next(answers),
                log_root=tmp_path / "runs",
            )
            batch = source.next()
        assert batch.objective == "grpo"
        assert len(batch.examples) == 2
        group = batch.examples[0]
        assert group.minimum == pytest.approx(0.1)
        assert group.maximum == pytest.approx(0.9)
        assert group.declared_range == (0.0, 1.0)
        assert group.zero_spread is False
        assert batch.record["source"] == "reward"
        assert batch.record["resolved_model"] == "in_process:fake-model@5"
        # The positive check that the log_root above did something: a run log that was
        # never written would make the paragraph about it a claim with no measurement.
        log_lines = [
            json.loads(line)
            for line in (tmp_path / "runs" / "run-1" / "run.jsonl").read_text("utf-8").splitlines()
            if line.strip()
        ]
        batch_blocks = [line for line in log_lines if line.get("block") == "batch"]
        assert len(batch_blocks) == 1, log_lines
        assert batch_blocks[0]["batch"]["source"] == "reward"

    def test_a_zero_spread_group_is_named_and_the_run_is_not_stopped(
        self, engine: Any, tmp_path: Path
    ) -> None:
        """E-15 / AC-002.10. *The step runs in full and applies numerically nothing* - so
        the batch is produced, the signal is on it, and **F2 does not stop the run**:
        stopping is R-02.25's and F6's."""
        with support_db.session_for(engine) as session:
            source = build(session, tmp_path, answers="the same every time", flat=True)
            batch = source.next()
        assert all(group.zero_spread for group in batch.examples)
        assert batch.size == 2  # produced, not skipped

    def test_a_group_below_two_is_refused_at_preflight(self, engine: Any, tmp_path: Path) -> None:
        """AC-002.12 / E-16: a single completion has no relative advantage by
        construction."""
        provider_port.bind_provider_resolver(StubProviderResolver(ScoringJudge().provider()))
        with support_db.session_for(engine) as session:
            part = fixtures.prompt_part(session, tmp_path, prompts=["q1"])
            ensure_registered()
            source = registry.get("reward")(
                RunContext(run_id="run-1", session=session, provider=fixtures.in_process_provider())
            )
            with pytest.raises(Refusal) as caught:
                source.preflight(
                    SourceSpec(
                        source="reward",
                        parts=(PartSpec(kind="prompt_set", part_id=str(part["id"])),),
                        grader=judge_config(),
                        config_source="inline",
                        group_size=1,
                    )
                )
        assert caught.value.code == "reward_group_too_small"
        assert caught.value.details["minimum"] == 2

    def test_a_grader_is_required(self, engine: Any, tmp_path: Path) -> None:
        with support_db.session_for(engine) as session:
            part = fixtures.prompt_part(session, tmp_path, prompts=["q1"])
            ensure_registered()
            source = registry.get("reward")(
                RunContext(run_id="run-1", session=session, provider=fixtures.in_process_provider())
            )
            with pytest.raises(Refusal) as caught:
                source.preflight(
                    SourceSpec(
                        source="reward",
                        parts=(PartSpec(kind="prompt_set", part_id=str(part["id"])),),
                        group_size=2,
                    )
                )
        assert caught.value.code == "grader_not_configured"
