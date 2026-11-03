"""Where index and manifest disagree, the manifest wins **and it is logged**  (task 4.5).

ADR-004. The manifest beside the weights is the record; this database is a cache. So a
disagreement has exactly one resolution, and the resolution is not the interesting half -
**the log line is.** A divergence means something wrote the index without going through
the manifest, or edited a manifest in place. Both are bugs, and a silent self-heal is how
they stay unfound for months.

So every test here asserts **both** halves, as the task requires: the outcome *and* the
log line. A test that asserted only the outcome would pass against a correct resolution
that said nothing, which is the going-quietly-dark failure ADR-013 names.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.errors import Refusal
from models.checkpoint import Checkpoint
from models.domain_state import DomainState
from services.reindex import DIVERGENCE_EVENT, reindex, resolve_checkpoint
from tests.support import db as support

NAME = "LFM2-1.2B:run-1:step-100"


@pytest.fixture()
def engine(tmp_path: Path):
    return support.migrated_engine(tmp_path / "db")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    path = tmp_path / "checkpoints"
    path.mkdir()
    return path


def _divergence_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.message == DIVERGENCE_EVENT]


def test_no_divergence_means_no_log_line(engine, root: Path, caplog) -> None:
    """The negative case first: an agreeing index must be silent.

    Without this, a function that logged the event unconditionally would satisfy every
    other test in this file.
    """
    support.write_checkpoint(root, support.checkpoint_manifest(step=100, domain="support"))
    with support.session_for(engine) as session:
        reindex(session, root=root)

    with (
        caplog.at_level(logging.WARNING, logger="miforge.reindex"),
        support.session_for(engine) as session,
    ):
        payload = resolve_checkpoint(session, NAME, root=root)

    assert payload["domain"] == "support"
    assert _divergence_records(caplog) == []


def test_the_manifest_wins_and_the_disagreement_is_logged(engine, root: Path, caplog) -> None:
    """A planted divergence on a plain field. Outcome **and** log line."""
    support.write_checkpoint(root, support.checkpoint_manifest(step=100, domain="support"))
    with support.session_for(engine) as session:
        reindex(session, root=root)
        # Plant the divergence by corrupting the INDEX, which is the direction that
        # actually happens: a writer that updated a row without rewriting the manifest.
        session.get(Checkpoint, NAME).weights_sha256 = "9" * 64  # type: ignore[union-attr]
        session.flush()

    with (
        caplog.at_level(logging.WARNING, logger="miforge.reindex"),
        support.session_for(engine) as session,
    ):
        payload = resolve_checkpoint(session, NAME, root=root)

    assert payload["weights_sha256"] == "b" * 64, "the index overruled the manifest"

    records = _divergence_records(caplog)
    assert len(records) == 1, records
    record = records[0]
    assert record.artifact == NAME  # type: ignore[attr-defined]
    assert record.divergent_fields == ["weights_sha256"]  # type: ignore[attr-defined]
    assert record.resolution == "manifest_wins"  # type: ignore[attr-defined]
    # Both answers, so the log line is enough to diagnose without re-reading the disk.
    assert record.detail["weights_sha256"] == {  # type: ignore[attr-defined]
        "index": "9" * 64,
        "manifest": "b" * 64,
    }


def test_the_row_is_corrected_so_the_line_is_not_emitted_forever(
    engine, root: Path, caplog
) -> None:
    """Logged once, then fixed. A divergence that re-reports every read is noise."""
    support.write_checkpoint(root, support.checkpoint_manifest(step=100, domain="support"))
    with support.session_for(engine) as session:
        reindex(session, root=root)
        session.get(Checkpoint, NAME).base_model = "wrong"  # type: ignore[union-attr]
        session.flush()

    with caplog.at_level(logging.WARNING, logger="miforge.reindex"):
        with support.session_for(engine) as session:
            resolve_checkpoint(session, NAME, root=root)
        first = len(_divergence_records(caplog))
        caplog.clear()
        with support.session_for(engine) as session:
            resolve_checkpoint(session, NAME, root=root)
        second = len(_divergence_records(caplog))

    assert (first, second) == (1, 0)
    with support.session_for(engine) as session:
        assert session.get(Checkpoint, NAME).base_model == "LFM2-1.2B"  # type: ignore[union-attr]


def test_a_divergent_domain_state_resolves_to_the_manifests_three_valued_answer(
    engine, root: Path, caplog
) -> None:
    """The divergence that matters most, because the index can express a state the
    manifest does not.

    The manifest says *this run has no domain* with an explicit null. An index row
    claiming a value is not a stale string - it is a claim about the run's purpose. Both
    columns resolve together, because either alone would violate the CHECK.
    """
    support.write_checkpoint(root, support.checkpoint_manifest(step=100, domain=None))
    with support.session_for(engine) as session:
        reindex(session, root=root)
        row = session.get(Checkpoint, NAME)
        assert row is not None
        row.domain = "support"
        row.domain_state = DomainState.RECORDED.value
        session.flush()

    with (
        caplog.at_level(logging.WARNING, logger="miforge.reindex"),
        support.session_for(engine) as session,
    ):
        payload = resolve_checkpoint(session, NAME, root=root)

    assert payload["domain"] is None
    assert payload["domain_state"] == DomainState.ASSERTED_NULL.value
    assert payload["domain_display"] == "no domain"
    records = _divergence_records(caplog)
    assert len(records) == 1
    assert set(records[0].divergent_fields) == {"domain", "domain_state"}  # type: ignore[attr-defined]


def test_an_indexed_checkpoint_whose_manifest_is_gone_is_refused_and_logged(
    engine, root: Path, caplog
) -> None:
    """The authority is missing, so the row must not be served as though it were backed.

    Serving it would be the index outliving the artifact - the exact inversion of ADR-004,
    and a stale answer a caller cannot tell from a live one.
    """
    directory = support.write_checkpoint(root, support.checkpoint_manifest(step=100))
    with support.session_for(engine) as session:
        reindex(session, root=root)
    (directory / "manifest.json").unlink()

    with (
        caplog.at_level(logging.WARNING, logger="miforge.reindex"),
        pytest.raises(Refusal) as raised,
        support.session_for(engine) as session,
    ):
        resolve_checkpoint(session, NAME, root=root)

    assert raised.value.code == "manifest_missing_for_indexed_checkpoint"
    records = _divergence_records(caplog)
    assert len(records) == 1
    assert records[0].divergence == "manifest_missing"  # type: ignore[attr-defined]


def test_an_unknown_name_is_refused_by_name(engine, root: Path) -> None:
    with pytest.raises(Refusal) as raised, support.session_for(engine) as session:
        resolve_checkpoint(session, "nothing:like:this", root=root)
    assert raised.value.code == "checkpoint_not_indexed"


def test_every_indexed_field_is_compared(engine, root: Path, caplog) -> None:
    """The comparison must cover the columns the builder fills, not a hand-picked subset.

    A divergence check whose field list drifted from the row builder would stop checking a
    column with no test going red - the shape of defect this project keeps finding. So the
    two lists are asserted to be the same list.
    """
    from services.reindex import (
        CHECKPOINT_INDEXED_FIELDS,
        ArtifactKind,
        checkpoint_row_values,
        scan_artifact,
    )

    directory = support.write_checkpoint(root, support.checkpoint_manifest(step=100))
    artifact = scan_artifact(directory / "manifest.json", ArtifactKind.CHECKPOINT, root)
    built = set(checkpoint_row_values(artifact))

    # The builder fills these three from outside the manifest's compared fields: identity,
    # the manifest's own digest, and where the directory is. Everything else is compared.
    not_compared = {"name", "manifest_sha256", "path", "created_at"}
    assert built - not_compared == set(CHECKPOINT_INDEXED_FIELDS), (
        built - not_compared - set(CHECKPOINT_INDEXED_FIELDS),
        set(CHECKPOINT_INDEXED_FIELDS) - (built - not_compared),
    )
