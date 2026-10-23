"""The base schema: the seven tables, the prefix, and the join key  (tasks 4.2, 4.2b).

**The boundary rule is asserted, not merely written down.** The base schema holds the
tables **more than one feature reads**. A table only one feature touches ships with that
feature's own migration - so this test pins the set to exactly seven and fails when an
eighth appears here. *Why a test:* "file your foundation gaps here" without a boundary
turns task 4.2 into every feature's migration, and then no feature owns its own schema.
F2's task list instructs someone to file its batch index into this schema; on this rule
that filing is **declined**, and this test is what makes the decline durable rather than a
remark in a report nobody reads again.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from db.base import TABLE_PREFIX, Base
from models import Adapter, Checkpoint, Lease, Pack, PackPart, Run, Step
from models.domain_state import DomainState
from tests.support import db as support

#: The base schema, exactly. An addition here is a decision, which is the point.
BASE_SCHEMA_TABLES = frozenset(
    {
        "mf_runs",
        "mf_steps",
        "mf_packs",
        "mf_pack_parts",
        "mf_checkpoints",
        "mf_adapters",
        "mf_leases",
    }
)


def test_the_base_schema_is_exactly_these_seven_tables() -> None:
    assert set(Base.metadata.tables) == set(BASE_SCHEMA_TABLES), (
        "the base schema changed. A table only one feature reads belongs in that "
        "feature's own migration (foundation task 4.2's boundary rule)."
    )


def test_every_table_carries_the_mf_prefix() -> None:
    """ADR-003. A ratchet, because an unprefixed table collides in a shared database."""
    offenders = [name for name in Base.metadata.tables if not name.startswith(TABLE_PREFIX)]
    assert offenders == [], offenders


def test_every_constraint_and_index_is_named() -> None:
    """Unnamed constraints cannot be dropped by a later migration.

    SQLite in particular emits anonymous CHECKs, so a migration that is reversible on
    PostgreSQL silently is not on the engine the unit suite runs against.
    """
    unnamed: list[str] = []
    for table in Base.metadata.tables.values():
        unnamed += [f"{table.name}: index without a name" for ix in table.indexes if not ix.name]
        unnamed += [
            f"{table.name}: {type(constraint).__name__} without a name"
            for constraint in table.constraints
            if constraint.name is None
        ]
    assert unnamed == [], unnamed


# ------------------------------------------------------------------ task 4.2b


@pytest.fixture()
def engine(tmp_path: Path):
    return support.migrated_engine(tmp_path / "db")


@pytest.mark.parametrize("table", ["mf_checkpoints", "mf_adapters"])
def test_name_is_indexed_as_the_join_key(engine, table: str) -> None:
    """**Task 4.2b.** ``name`` is the key R-02.29's tree joins on, so it must be indexed.

    The primary key satisfies this and is asserted as such rather than assumed: a later
    change to a surrogate id would move ``name`` off the index and the tree walk would
    degrade to a scan with nothing going red.
    """
    inspector = inspect(engine)
    pk = inspector.get_pk_constraint(table)["constrained_columns"]
    indexed = {tuple(ix["column_names"]) for ix in inspector.get_indexes(table)}
    assert pk == ["name"] or ("name",) in indexed, (pk, indexed)


def test_the_tree_walks_parent_and_base_name_and_both_are_indexed(engine) -> None:
    """The two different edges of R-02.29's tree, each with an index behind it.

    A checkpoint's edge is ``parent``; an adapter's is the base **its own manifest names**.
    Both are traversed child-to-parent, which is a lookup by the *other* column.
    """
    inspector = inspect(engine)
    checkpoint_indexed = {
        col for ix in inspector.get_indexes("mf_checkpoints") for col in ix["column_names"]
    }
    adapter_indexed = {
        col for ix in inspector.get_indexes("mf_adapters") for col in ix["column_names"]
    }
    assert "parent" in checkpoint_indexed
    assert "base_name" in adapter_indexed


# ------------------------------------------------------------------ the record tables


def test_a_terminal_run_must_carry_a_finish_time(engine) -> None:
    """A terminal row with no ``finished_at`` is how a janitor cannot tell done from lost."""
    with pytest.raises(IntegrityError), support.session_for(engine) as session:
        session.add(Run(id="r1", status="completed", run_type="change"))
        session.flush()


def test_a_lease_must_expire(engine) -> None:
    """ADR-008: a lease with no expiry is the bare boolean, and a crash costs the card."""
    with pytest.raises((IntegrityError, TypeError)), support.session_for(engine) as session:
        session.add(Lease(resource="GPU-abc", holder_run_id="r1"))
        session.flush()


def test_one_unreleased_lease_per_resource(engine) -> None:
    """Exclusivity in the database, not in a service's good intentions.

    Two workers racing for the same card both pass an application-level check; only one
    passes the partial unique index.
    """
    now = support.utc_now()
    later = now.replace(year=now.year + 1)
    with support.session_for(engine) as session:
        session.add(Lease(resource="GPU-abc", holder_run_id="r1", expires_at=later))
        session.flush()
    with pytest.raises(IntegrityError), support.session_for(engine) as session:
        session.add(Lease(resource="GPU-abc", holder_run_id="r2", expires_at=later))
        session.flush()


def test_a_released_lease_frees_the_resource(engine) -> None:
    """Partial on purpose: history stays, exclusivity applies only to the live row."""
    now = support.utc_now()
    later = now.replace(year=now.year + 1)
    with support.session_for(engine) as session:
        session.add(
            Lease(resource="GPU-abc", holder_run_id="r1", expires_at=later, released_at=now)
        )
        session.flush()
    with support.session_for(engine) as session:
        session.add(Lease(resource="GPU-abc", holder_run_id="r2", expires_at=later))
        session.flush()
        assert session.query(Lease).count() == 2


def test_an_invalid_pack_names_a_reason(engine) -> None:
    """ADR-013: refuse with a reason. ``valid=False`` and no reason is a refusal without one."""
    with pytest.raises(IntegrityError), support.session_for(engine) as session:
        session.add(Pack(id="p1", name="support", version="1", valid=False))
        session.flush()


def test_a_pack_part_must_be_pinned(engine) -> None:
    """An unpinned part makes a pack **version** meaningless, which is the whole artifact."""
    with support.session_for(engine) as session:
        session.add(Pack(id="p1", name="support", version="1", valid=True))
        session.flush()
    with pytest.raises(IntegrityError), support.session_for(engine) as session:
        session.add(PackPart(pack_id="p1", part_kind="prompt-set", source="midataworks"))
        session.flush()


def test_a_steps_row_is_unique_per_run_and_step(engine) -> None:
    with support.session_for(engine) as session:
        session.add(Run(id="r1", status="running", run_type="change"))
        session.add(Step(run_id="r1", step=1, loss=2.0))
        session.flush()
    with pytest.raises(IntegrityError), support.session_for(engine) as session:
        session.add(Step(run_id="r1", step=1, loss=1.0))
        session.flush()


def test_a_checkpoint_indexes_a_run_it_has_never_heard_of(engine) -> None:
    """**``run_id`` is deliberately not a foreign key** (ADR-004).

    A checkpoint copied from another machine keeps its meaning and arrives with a
    ``run_id`` that has no local run row. A FK would refuse to index it and the artifact
    would be invisible - the index silently overruling the filesystem.
    """
    with support.session_for(engine) as session:
        session.add(
            Checkpoint(
                name="foreign:run-elsewhere:step-1",
                run_id="run-elsewhere",
                step=1,
                domain=None,
                domain_state=DomainState.ASSERTED_NULL.value,
                signal_source="corpus",
                parent=None,
                base_model="foreign",
                weights_sha256="b" * 64,
                manifest_sha256="e" * 64,
                path="elsewhere/step-000001",
                created_at=support.utc_now(),
            )
        )
        session.flush()
        assert session.query(Checkpoint).count() == 1
        assert session.query(Run).count() == 0


def test_an_adapter_defaults_to_not_in_kind(engine) -> None:
    """The default is the only legal value, so a writer cannot forget it into a wrong state."""
    with support.session_for(engine) as session:
        session.add(
            Adapter(
                name="base:run-2:adapter-1",
                run_id="run-2",
                step=1,
                base_name="base:run-1:step-1",
                weights_sha256="d" * 64,
                manifest_sha256="f" * 64,
                path="run-2/adapter-000001",
                created_at=support.utc_now(),
            )
        )
        session.flush()
        row = session.query(Adapter).one()
        assert row.domain_state == DomainState.NOT_IN_KIND.value
        assert row.domain is None
