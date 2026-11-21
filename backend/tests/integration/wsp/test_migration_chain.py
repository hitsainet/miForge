"""The WSP migration's place in miForge's Alembic chain (operator decision 2026-11-16).

miForge 017 (``0027_run_hub_access``) pushes before 016, so 016's migration is ``0028_wsp_provider``
and its parent is 017's ``0027_run_hub_access``. The chain is read from the versions directory
through Alembic's own ``ScriptDirectory`` (the authority ``upgrade head`` uses), not source text.

Until 017-P1 is on 016's base, ``0027_run_hub_access`` is absent and 0028 still descends from
``0026_legacy_name_uses``; the parent pin is then an expected failure (strict, so it cannot pass
by accident). The moment the stack is rebased onto a base carrying 017 the marker switches off:
the pin applies, and with the parent not yet changed both it and the one-head test go red, which
is the reminder to make the one-line re-parent commit (``wsp-forge`` NEEDS 11 records the steps).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

BACKEND = Path(__file__).resolve().parents[3]
WSP = "0028_wsp_provider"
PARENT = "0027_run_hub_access"
#: Feature 018's migration, on 016's (FTID §4); the chain's head from 018-P2 to 2026-11-20.
ENGINE = "0029_wsp_engine"
#: Live progress (2026-11-20): the worker heartbeat on the run, each call's outcome.
LIVE = "0030_wsp_live_progress"
#: An agent's editor endpoint proposal (2026-11-20): the wsp_endpoint_register action.
REGISTER = "0031_wsp_endpoint_register"
#: The endpoint's thinking and the engine's reasoning count (2026-11-21).
THINKING = "0032_wsp_thinking"
#: The served calls' prompt evidence (2026-11-21).
EVIDENCE = "0033_wsp_prompt_evidence"
#: The endpoint's paging and the engine's paging counters (2026-11-21); the head.
PAGING = "0034_wsp_paging"


def _scripts() -> ScriptDirectory:
    cfg = Config()
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    return ScriptDirectory.from_config(cfg)


def _ids() -> set[str]:
    return {r.revision for r in _scripts().walk_revisions()}


def test_the_wsp_migration_is_numbered_0028() -> None:
    ids = _ids()
    assert WSP in ids
    assert "0027_wsp_provider" not in ids
    assert (BACKEND / "alembic" / "versions" / f"{WSP}.py").is_file()


def test_the_chain_has_one_head_and_it_is_the_paging_migration() -> None:
    assert _scripts().get_heads() == [PAGING]


def test_the_paging_migration_follows_the_prompt_evidence_migration() -> None:
    revision = _scripts().get_revision(PAGING)
    assert revision is not None and revision.down_revision == EVIDENCE


def test_the_prompt_evidence_migration_follows_the_thinking_migration() -> None:
    revision = _scripts().get_revision(EVIDENCE)
    assert revision is not None and revision.down_revision == THINKING


def test_the_thinking_migration_follows_the_endpoint_register_migration() -> None:
    revision = _scripts().get_revision(THINKING)
    assert revision is not None and revision.down_revision == REGISTER


def test_the_endpoint_register_migration_follows_the_live_progress_migration() -> None:
    revision = _scripts().get_revision(REGISTER)
    assert revision is not None and revision.down_revision == LIVE


def test_the_live_progress_migration_follows_the_engine_migration() -> None:
    revision = _scripts().get_revision(LIVE)
    assert revision is not None and revision.down_revision == ENGINE


def test_the_engine_migration_follows_the_wsp_provider_migration() -> None:
    revision = _scripts().get_revision(ENGINE)
    assert revision is not None and revision.down_revision == WSP


@pytest.mark.xfail(
    PARENT not in _ids(),
    reason=f"017's {PARENT} is not on this base yet; the pin applies once the stack is rebased "
    "onto it",
    strict=True,
)
def test_the_wsp_migration_follows_017s_run_hub_access() -> None:
    revision = _scripts().get_revision(WSP)
    assert revision is not None and revision.down_revision == PARENT
