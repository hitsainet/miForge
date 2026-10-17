"""This feature has no review queue, asserted as an absence  (FR-007.21, FAC-11).

**The in-run human-review queue is miDataworks'** (R-02.36, BRD-03 R-03.41), and the
pack-authoring dry-run is *a different activity* - *"conflating the two is a mistake this
project already made once."*

|                | **Pack-authoring dry-run (F7)**      | **In-run human review (miDataworks)** |
|----------------|--------------------------------------|---------------------------------------|
| When           | Before any run exists                | While a run is stepping               |
| Who            | The pack's author, synchronously     | A reviewer, asynchronously            |
| What is judged | The **selector's** verdicts          | **Candidate completions**             |
| Queue          | **None**                             | miDataworks'                          |

So F7 must not build: a queue, an inbox, a pending-decision store, a reviewer identity, or
any notion of a decision coming back later. Every assertion below is over the **live** route
table, the **live** Celery registry, the real model metadata and the feature's own source -
because *"we did not build one"* is a claim and these are checks.

Mutation control (task 14.3c): add a ``/pack-drafts/{id}/queue`` route to
``packs_drafts.py``. ``test_no_queue_route_is_mounted`` must go red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import packs
from api.app import create_app

#: Words that would betray a queue if they appeared on this feature's surface - **including
#: ``review``**, because in this repository that word names miDataworks' queue. The step-7
#: route is ``/validity`` for exactly that reason; it was ``/review`` until F10's own guard
#: went red, which is the cross-feature version of this same assertion.
QUEUE_WORDS = ("queue", "inbox", "pending", "adjudicat", "decision", "review")

SOURCE_DIR = Path(packs.__file__).resolve().parent


def pack_routes() -> list[tuple[str, str]]:
    app = create_app()
    return sorted(
        (method, getattr(route, "path", ""))
        for route in app.routes
        for method in getattr(route, "methods", set())
        if "pack" in getattr(route, "path", "")
    )


class TestNoQueueSurface:
    def test_no_queue_route_is_mounted(self) -> None:
        """Over the **mounted** route table, not by reading the endpoint modules."""
        for method, path in pack_routes():
            for word in QUEUE_WORDS:
                assert word not in path.lower(), f"{method} {path} names a queue"

    def test_no_pack_route_mentions_review_at_all(self) -> None:
        """Step 7 is *"Review and save"* in the copy and ``/validity`` in the path.

        In this repository *review* names miDataworks' human-review queue, and F10 ships a
        guard asserting the only route mentioning it is its own. This feature's step-7 route
        was ``/review`` until that guard went red - the guard was right, and the endpoint
        reports validity rather than reviewing anything.
        """
        assert [path for _method, path in pack_routes() if "review" in path.lower()] == []
        assert ("GET", "/api/v1/pack-drafts/{draft_id}/validity") in pack_routes()

    def test_no_route_accepts_a_decision(self) -> None:
        """A decision-recording endpoint here would be a second answer to miDataworks' question."""
        app = create_app()
        for route in app.routes:
            path = getattr(route, "path", "")
            if "pack" not in path:
                continue
            for method in getattr(route, "methods", set()):
                assert not (method == "POST" and "decision" in path.lower())


class TestNoQueueStorage:
    def test_this_feature_declares_no_table_at_all(self) -> None:
        """Every pack table is the foundation's; F7 adds none and therefore adds no queue.

        A queue would need a table, so the absence of a table is the strongest form of this
        assertion available. (It is also a **reported gap**: ``mf_pack_versions``,
        ``mf_external_parts`` and ``mf_pack_drafts`` were specified by F7's documents and are
        not created here - drafts live in the pack store and external provenance lives in
        F10's index. The owed DDL is reported rather than quietly skipped.)
        """
        from db.base import Base

        pack_tables = {name for name in Base.metadata.tables if name.startswith("mf_pack")}
        assert pack_tables == {"mf_packs", "mf_pack_parts"}

    def test_no_pack_module_names_a_queue_concept(self) -> None:
        """The source, checked - a queue is tempting exactly where the panel is all undecided."""
        for path in SOURCE_DIR.glob("*.py"):
            text = path.read_text("utf-8").lower()
            # The docstrings explain at length why there is no queue, so the check is for a
            # queue being *used*: an attribute, a call or an assignment.
            for forbidden in (
                "queue.put",
                "queue.get",
                "self.queue",
                "_queue =",
                "pending_decisions",
                "review_queue(",
            ):
                assert forbidden not in text, f"{path.name} contains {forbidden!r}"

    def test_no_pack_module_imports_f10s_review_client(self) -> None:
        """F10 asserts the mirror of this from its own side, on the call count."""
        for path in SOURCE_DIR.glob("*.py"):
            text = path.read_text("utf-8")
            assert "midataworks_review" not in text, path.name
            assert "services.review" not in text, path.name


class TestNoWorker:
    def test_this_feature_registers_no_celery_task(self) -> None:
        """The dry-run returns within the request. *Shrink the sample; do not add a queue.*"""
        from workers.celery_app import celery_app

        assert [name for name in celery_app.tasks if "pack" in name] == []

    def test_no_pack_module_imports_a_broker(self) -> None:
        for path in SOURCE_DIR.glob("*.py"):
            text = path.read_text("utf-8")
            assert "celery" not in text.lower()
            assert "redis" not in text.lower()


class TestTheFenceAroundTheHotUpdateMode:
    """ADR-022: the §5.10 mode is fenced, and a pack has nothing to do with it."""

    #: **Code-shaped** patterns, not words. The modules' prose legitimately mentions miLLM -
    #: ``packs/contracts.py`` explains why the adapter manifest must not be edited, which is
    #: *because* miLLM is owed a byte-identical copy of it - and a check that banned the word
    #: would fire on the comment that keeps the rule and teach the next person to delete it.
    #: What ADR-022 forbids is a core-path module *reaching* the mode: an import, an
    #: endpoint path, or a protocol field.
    FORBIDDEN_CODE = (
        "training.hotupdate",
        "training/hotupdate",
        "import millm",
        "from millm",
        "millm_base_url",
        "/api/weights",
        "weights_state",
        "commit-begin",
        "commit_begin",
        "X-miLLM-Attachment",
    )

    @pytest.mark.parametrize("forbidden", FORBIDDEN_CODE)
    def test_no_pack_module_reaches_the_attachment_protocol(self, forbidden: str) -> None:
        for path in SOURCE_DIR.glob("*.py"):
            text = path.read_text("utf-8")
            assert forbidden not in text, f"{path.name} reaches {forbidden!r}"

    def test_no_pack_module_imports_a_sibling(self) -> None:
        """ADR-001, over this feature's own code (the repository-wide guard is foundation 1.5)."""
        for path in SOURCE_DIR.glob("*.py"):
            text = path.read_text("utf-8")
            for sibling in ("import millm", "import mistudio", "import midataworks"):
                assert sibling not in text, f"{path.name} imports a sibling"
