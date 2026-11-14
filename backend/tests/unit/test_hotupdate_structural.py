"""F11's two remaining structural obligations, both **negative**  (ADR-016 #3 and #4).

F11 defines no contract and writes no index row, so its obligations under two of the four
structural tests are things that must **stay** true rather than things it adds:

* **Schema sync.** ``step-record-v1.json`` is unchanged by this feature and stays in
  foundation 1.4's parametrised set. A feature that quietly added a field to a contract it
  does not own would break the byte-identity of every vendor of it.
* **Reindex equivalence.** The two §5.10 fields survive a drop-and-reindex **from the
  authoritative on-disk run log** (ADR-004: the filesystem is the source of truth, the
  database is a rebuildable index).

The second one found something, and it is F1's rather than F11's - see
``TestTheIndexDoesNotCarryThemYet``.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMAS = Path(__file__).resolve().parents[3] / "docs" / "schemas"

#: The op R-02.34 forbids miForge ever to call. See the class at the foot of this file.
FORBIDDEN_OP = "clear-stale"
STEP_RECORD = SCHEMAS / "step-record-v1.json"


class TestSchemaSyncIsUntouchedByThisFeature:
    def test_the_step_record_contract_is_in_the_parametrised_set(self) -> None:
        """A contract outside that set is a contract nothing checks."""
        from tests.contract.test_schema_sync import OWNERSHIP

        assert "step-record-v1.json" in OWNERSHIP

    def test_the_two_section_5_10_fields_are_the_contracts_and_not_F11s(self) -> None:
        """FR-011.57: **F11 populates them and defines neither.** Their descriptions say
        ``SECTION 5.10 ONLY`` in the schema, which is where that fact belongs."""
        block = json.loads(STEP_RECORD.read_text("utf-8"))["$defs"]["StepBlock"]
        for field in ("quiesce_hold_ms", "commit_duration_ms"):
            assert field in block["properties"]
            assert "5.10" in block["properties"][field]["description"]
            assert field not in block["required"], (
                "both are nullable and NOT required - which is why dropping them validates "
                "cleanly and why a test, not the schema, is what catches MUT-15"
            )

    def test_the_block_is_closed_so_F11_cannot_add_a_field_by_accident(self) -> None:
        block = json.loads(STEP_RECORD.read_text("utf-8"))["$defs"]["StepBlock"]
        assert block["additionalProperties"] is False

    def test_no_module_in_this_feature_writes_to_docs_schemas(self) -> None:
        """F11 adds no vendored contract, and *"do not edit ``docs/schemas/``"* is one of
        this feature's standing instructions. Asserted rather than remembered."""
        package = Path(__file__).resolve().parents[2] / "src" / "training" / "hotupdate"
        for path in package.glob("*.py"):
            text = path.read_text("utf-8")
            assert "docs/schemas" not in text or "step-record-v1.json" in text


class TestReindexEquivalenceForTheTwoFields:
    def _write_log(self, root: Path, durations: list[float]) -> Path:
        from training import run_log
        from training.hotupdate.timings import StepTimings, fill

        path = run_log.run_log_path(root, "hot-run")
        path.parent.mkdir(parents=True, exist_ok=True)
        timings = StepTimings(watchdog_ms=2000.0, creep_fraction=0.9, run_id="hot-run")
        for step, duration in enumerate(durations):
            timings.record_commit(duration)
            timings.record_hold(duration + 5.0, {"held_requests": 1})
            run_log.append(
                path,
                # **Through the real writer and the real filler**, so the assertion is about
                # the code rather than about a hand-built dict whose fields agree with it by
                # construction (ADR-016).
                fill(
                    run_log.step_block(
                        run_id="hot-run",
                        step=step,
                        loss=1.0 - step / 100,
                        objective="causal_lm",
                        learning_rate=1e-5,
                        wall_time_ms=120.0,
                    ),
                    timings,
                ),
            )
        return path

    def test_the_figure_another_repositorys_gate_reads_survives_a_dropped_index(
        self, tmp_path: Path
    ) -> None:
        """**The equivalence that matters here.** ADR-016 #4 compares API responses; the
        §5.10 fields' consumer is miLLM Feature 31's watchdog gate, which reads a p99.9 - so
        the question is whether that figure is identical after the index is thrown away. It
        is, because it is computed from the log, which is authoritative.
        """
        from training import run_log
        from training.hotupdate import timings as timings_module

        self._write_log(tmp_path, [40.0 + (index % 5) for index in range(200)])
        path = run_log.run_log_path(tmp_path, "hot-run")
        before = timings_module.report(path, conditions="fixture")
        # "Drop the index": nothing is read from the database at all on this path, which is
        # the property - a figure that needed the index would be lost with it.
        after = timings_module.report(path, conditions="fixture")
        assert before == after
        assert before["value_ms"] == run_log.commit_duration_percentile(path, 99.9)
        assert before["sample_size"] == 200

    def test_a_rebuild_from_the_log_restores_every_step_row(self, tmp_path: Path) -> None:
        """F1's own rebuild, run over a log F11 wrote: the §5.10 fields must not make a
        block unreindexable."""
        from tests.support.jobs import insert_run, memory_engine, session_for
        from training import step_record

        self._write_log(tmp_path, [41.0, 42.0, 43.0])
        engine = memory_engine()
        try:
            with session_for(engine) as session:
                insert_run(session, "hot-run", status="running")
                rebuilt = step_record.rebuild_step_index(
                    session, log_root=tmp_path, run_id="hot-run"
                )
                assert rebuilt == 3

                # STRENGTHENED 2026-10-14, on the flipped tripwire's own instruction:
                # *"compare the two §5.10 columns rather than only the row count."*
                #
                # A row count proves the blocks were readable, not that their values
                # arrived. While the projection silently dropped these three, this test
                # passed at `rebuilt == 3` with every column null - so the row count was
                # exactly the assertion that could not see the defect.
                import sqlalchemy as sa

                from models.run import Step

                rows = session.execute(
                    sa.select(Step.step, Step.commit_duration_ms, Step.objective).order_by(
                        Step.step
                    )
                ).all()
                assert [row[1] for row in rows] == [41.0, 42.0, 43.0], (
                    "the rebuild restored the rows but not commit_duration_ms. miLLM Feature "
                    "31's gate reads this column's p99.9, so a reindex that loses it loses a "
                    "cross-repo measurement - and ADR-004 says the log is authoritative, "
                    "which is only true if a rebuild from it is complete."
                )
                assert {row[2] for row in rows} == {
                    "causal_lm"
                }, "the rebuild lost `objective`, which F9's loss curves read"
        finally:
            engine.dispose()


class TestTheIndexAndTheProjectionAgree:
    """**A finding, and it is F1's rather than F11's.** Reported, pinned, not fixed here.

    ``mf_steps`` **has** ``quiesce_hold_ms``, ``commit_duration_ms`` and ``objective``
    columns, and a partial index ``ix_mf_steps_commit_duration`` whose stated purpose is
    *"supports FR-001.26's commit-duration distribution"*. But ``run_log.step_rows_from_block``
    - the single projection from the record to the index, which the live writer and the
    rebuild both use - **does not project any of the three**, and its docstring says they
    are *"deliberately not here: ``mf_steps`` has no column for them today."*

    That sentence was true when it was written and is now false: the columns exist. So today
    the partial index is over a column nothing writes, and ``objective`` - *"the
    load-bearing one of the three"* by its own comment, because F9 draws loss curves from
    that table and *"a curve that silently splices two objectives together looks like a
    regression"* - is null on every row.

    **It is not F11's to change**: that projection is F1's, three features' reindex-
    equivalence assertions compare API responses built from it, and widening it is a
    behaviour change to the index rather than an addition to §5.10. Pinned here so the
    finding cannot go stale quietly: when the projection is widened, this test goes red and
    says what to do.
    """

    def test_the_columns_exist_on_mf_steps(self) -> None:
        from models.run import Step

        for column in ("quiesce_hold_ms", "commit_duration_ms", "objective"):
            assert column in Step.__table__.c, f"{column} is gone from mf_steps"

    def test_the_partial_index_on_commit_duration_exists(self) -> None:
        from models.run import Step

        names = {index.name for index in Step.__table__.indexes}
        assert "ix_mf_steps_commit_duration" in names

    def test_the_projection_NOW_FILLS_THEM_and_this_is_the_flipped_tripwire(self) -> None:
        """**This asserted the opposite, and it fired on 2026-10-14 as designed.**

        It read ``assert "commit_duration_ms" not in row`` with the instruction *"when this
        goes red, F1's projection has been widened: delete this class, and strengthen the
        reindex assertion above."* The projection was widened and the instruction followed.

        **The finding it recorded was a real defect in the coordinator's own work.** Migration
        ``0011`` added all three columns *and* a partial index
        ``ix_mf_steps_commit_duration``, and nothing widened the projection — so the columns
        were null on every row and the index was over a column nothing wrote. ``objective``'s
        own column comment calls it *"the load-bearing one of the three"* because **F9 draws
        loss curves from this table**, and a curve splicing two objectives together looks like
        a regression: the exact failure the column was added to prevent, sitting empty for a
        day. *Declaring a mechanism without wiring it, committed in the same change whose
        docstring states that rule.*

        Found by F11 reading for its timings work, not by anyone reading the migration.
        """
        from training import run_log
        from training.hotupdate.timings import StepTimings, fill

        timings = StepTimings(watchdog_ms=2000.0, creep_fraction=0.9)
        timings.record_commit(41.0)
        timings.record_hold(46.0, {"held_requests": 2})
        block = fill(
            run_log.step_block(
                run_id="hot-run",
                step=0,
                loss=1.0,
                objective="causal_lm",
                learning_rate=1e-5,
                wall_time_ms=120.0,
            ),
            timings,
        )
        assert block["commit_duration_ms"] == 41.0, "the log carries it"

        row = run_log.step_rows_from_block(block)
        assert row["commit_duration_ms"] == 41.0, (
            "the projection dropped commit_duration_ms. miLLM Feature 31's acceptance gate is "
            "set from this column's p99.9, so an empty column is a cross-repo obligation "
            "silently unmet - and the partial index on it would be over nothing."
        )
        assert row["quiesce_hold_ms"] == 46.0
        assert row["objective"] == "causal_lm", (
            "the projection dropped `objective`, which is the costly one: F9 draws loss "
            "curves from mf_steps and a curve that splices two objectives together reads as a "
            "regression"
        )

    def test_a_standalone_step_asserts_null_rather_than_omitting(self) -> None:
        """The other direction, and the distinction is the project's own rule.

        A standalone step holds no bracket, so both §5.10 timings are ``None`` — **present and
        null, which is an assertion**, not an absent key, which would be an omission. A
        projection that dropped the keys entirely would make a standalone step and a §5.10
        step with unmeasured timings indistinguishable in the index.
        """
        from training import run_log

        row = run_log.step_rows_from_block(
            run_log.step_block(
                run_id="cold-run",
                step=7,
                loss=0.5,
                objective="causal_lm",
                learning_rate=1e-5,
                wall_time_ms=90.0,
            )
        )
        assert row["quiesce_hold_ms"] is None
        assert row["commit_duration_ms"] is None
        assert "quiesce_hold_ms" in row and "commit_duration_ms" in row
        assert row["objective"] == "causal_lm", "objective is NOT section 5.10-only"


class TestTheClearStaleCallIsAbsentWithNoMODULEAllowlist:
    """**MUT-7's regression test, written because MUT-7 SURVIVED its designated catcher.**

    TDD §10.4 names MUT-7 as *"add a ``clear-stale`` call to ``client.py``"* and names the
    absence test as what must fail. ``test_no_clear_stale_call.py`` scans by AST for a call
    carrying the forbidden op - correctly - but it **excludes three whole modules by name**,
    and ``training/hotupdate/client.py`` is the first of them. So the mutation placed in the
    module the plan names is the one mutation that scan cannot see.

    Run on 2026-10-14: adding ``def clear_stale(self): return self.call("clear-stale")`` to
    ``WeightsClient`` left the F11 suite at **430 passed**. Nothing failed. The method is
    never called by a test, so the ``FORBIDDEN_OP`` guard inside ``call()`` never runs
    either - *the runtime guard protects the calls that happen and says nothing about the
    code that exists*, which is the distinction ADR-016 #1 is about.

    **The allowlist is not needed for the call scan at all.** Verified by running the same
    AST scan over every module under ``backend/src/`` with no exclusions: zero hits. The
    three allowlisted modules *name* the op - in a dict key, in a module constant, in a
    refusal message - and none of those is an ``ast.Call`` with the op as a constant
    argument. So the exclusion bought nothing and cost the one mutation it existed for.

    This test is therefore the same scan **with no module allowlist**. The path-text scan in
    the sibling file still needs its allowlist, because a text match cannot tell a route
    table from a request; an AST call scan can.
    """

    FORBIDDEN = FORBIDDEN_OP

    @staticmethod
    def _call_sites(tree: object) -> list[int]:
        import ast

        found: list[int] = []
        for node in ast.walk(tree):  # type: ignore[arg-type]
            if not isinstance(node, ast.Call):
                continue
            for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                if isinstance(argument, ast.Constant) and argument.value == FORBIDDEN_OP:
                    found.append(node.lineno)
        return found

    def test_no_module_anywhere_calls_anything_with_the_op_not_even_the_client(self) -> None:
        """R-02.34, FR-011.34. **No module is exempt** - that is the whole point of this
        test existing beside the allowlisted one."""
        import ast

        src = Path(__file__).resolve().parents[2] / "src"
        offenders = {
            str(path.relative_to(src)): lines
            for path in sorted(src.rglob("*.py"))
            if (lines := self._call_sites(ast.parse(path.read_text("utf-8"))))
        }
        assert offenders == {}, (
            f"{offenders} calls something with the {self.FORBIDDEN!r} op. R-02.34: miForge "
            f"never clears another process's attachment on its own initiative - that is how "
            f"two trainers end up writing the same tensors, and the holder may be mid-commit. "
            f"A call inside training/hotupdate/client.py is NOT exempt: the sibling scan in "
            f"test_no_clear_stale_call.py allowlists that module and MUT-7 survived it."
        )

    def test_the_scanner_sees_the_exact_form_MUT_7_SURVIVED_as(self) -> None:
        """The self-test, written as the surviving mutation's own source.

        *A guard whose self-test uses a different spelling than the defect is a guard whose
        self-test proves the wrong thing.* This is the literal body that went green.
        """
        import ast
        import textwrap

        survived = textwrap.dedent("""
            class WeightsClient:
                def clear_stale(self) -> dict:
                    return self.call("clear-stale")
            """)
        assert self._call_sites(ast.parse(survived)) != [], "the scanner is blind to MUT-7"

    def test_a_mere_mention_is_still_not_a_call(self) -> None:
        """So dropping the allowlist does not turn the three legitimate mentions into
        findings - which is the reason the exclusion was written in the first place, and
        the reason it belonged on the *text* scan rather than on this one."""
        import ast

        assert self._call_sites(ast.parse('OPS = {"clear-stale": ("POST", "/x", "R")}')) == []
        assert self._call_sites(ast.parse('FORBIDDEN_OP = "clear-stale"')) == []
        assert self._call_sites(ast.parse('"""clear-stale is operator-only."""')) == []
