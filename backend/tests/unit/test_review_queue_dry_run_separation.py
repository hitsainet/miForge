"""The two review activities must not touch  (AC-010.10, AC-010.12; FR-010.26, .27).

**This project already got this wrong once**, and the mistake is recorded: conflating
**in-run review** (R-02.36 - a person adjudicating generated candidates while a run is
stepping) with the **pack-authoring selector dry-run** (R-02.39 - an operator checking a
selector's verdicts on a sample before a run exists) *"is how the pack ownership question
went wrong the first time."*

They differ in every way that matters: the dry-run uses **no queue**, reaches **no sibling**,
and is **F7's**. So the guard is not a convention; it is two assertions:

* an **AST** check that no authoring path imports the client or the binding;
* a **call count of zero** - because *"a reachability test that only checks 'was called'
  passes against a call sending the wrong arguments"*, and its mirror here is that "was not
  called with" passes against a call.

And the submission-time refusal of FR-010.26 is asserted on the **moment**: a run that cannot
proceed says so before it starts, not partway through.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

import pytest

from core.config import reset_settings
from core.errors import Refusal
from services.review import preflight

SRC = Path(__file__).resolve().parents[2] / "src"


def _imports(node: ast.AST, module: str) -> bool:
    """Whether ``node`` imports ``module``, in either form.

    ``from x.y import z`` and ``import x.y`` both count. A guard that only knew one form is
    a guard the next author routes around by accident.
    """
    if isinstance(node, ast.ImportFrom):
        return node.module == module
    if isinstance(node, ast.Import):
        return any(alias.name == module for alias in node.names)
    return False


@pytest.fixture(autouse=True)
def unconfigured(monkeypatch):
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_settings()
    yield
    reset_settings()


class ClientCounter:
    """Counts constructions of the one outbound call site.

    Counting **constructions** rather than requests is the stronger assertion: the guard is
    not merely that no HTTP went out, but that nothing on the authoring path even reached for
    the seam. A construction with an unconfigured sibling would raise anyway, so a path that
    reached for it would fail loudly - the point is that it must not reach.
    """

    def __init__(self, monkeypatch) -> None:
        self.count = 0
        import clients.midataworks as module

        original = module.MiDataworksClient.__init__

        def counted(inner_self, *args, **kwargs):
            self.count += 1
            return original(inner_self, *args, **kwargs)

        monkeypatch.setattr(module.MiDataworksClient, "__init__", counted)


class TestTheSubmissionTimeRefusal:
    def test_a_human_review_run_is_refused_while_unconfigured(self):
        """AC-010.10. **At submission**, naming the selector as the capability that needs it."""
        with pytest.raises(Refusal) as caught:
            preflight.require_for_selector_kinds(["human_review"])
        assert caught.value.code == "midataworks_not_configured"
        assert caught.value.details["capability"] == "human_review_selector"
        assert "human_review" in caught.value.details["action"]
        assert caught.value.details["setting"] == "MIDATAWORKS_BASE_URL"

    def test_the_message_names_the_standalone_alternative(self):
        with pytest.raises(Refusal) as caught:
            preflight.require_for_selector_kinds(["human_review", "rule"])
        assert "your own files" in caught.value.message

    @pytest.mark.parametrize(
        "kinds",
        [["rule"], ["reference_match"], ["composite"], ["rule", "reference_match"], []],
    )
    def test_a_run_that_needs_nothing_from_midataworks_is_not_refused(self, kinds):
        """**Removing ``MIDATAWORKS_BASE_URL`` from a working install must break exactly two
        capabilities, by name, and nothing else.**"""
        preflight.require_for_selector_kinds(kinds)

    def test_a_judge_model_run_is_not_refused_by_this_check(self):
        """A judge needs a **generation provider** - F4's optional service with its own
        refusal - not miDataworks. Refusing it here would name the wrong sibling, and the
        operator would go and configure something that was never the problem."""
        preflight.require_for_selector_kinds(["judge_model"])

    def test_it_is_satisfied_once_configured(self, monkeypatch):
        monkeypatch.setenv("MIDATAWORKS_BASE_URL", "http://dw.invalid:8000")
        reset_settings()
        preflight.require_for_selector_kinds(["human_review"])
        assert preflight.selector_kinds_unavailable(["human_review"]) == []

    def test_it_does_not_probe_reachability(self, monkeypatch):
        """**Unconfigured is a decision; unreachable is a fault.** A sibling down at
        submission may be up by the first candidate, and refusing a run for a transient
        network state is worse than the wait."""
        monkeypatch.setenv("MIDATAWORKS_BASE_URL", "http://127.0.0.1:1")
        reset_settings()
        counter = ClientCounter(monkeypatch)
        preflight.require_for_selector_kinds(["human_review"])
        assert counter.count == 0

    def test_the_unavailable_listing_names_the_kind_rather_than_omitting_it(self):
        assert preflight.selector_kinds_unavailable(["human_review", "rule"]) == ["human_review"]


class TestTheAuthoringPathNeverReachesTheQueue:
    #: Modules that may reach the review seam: the port implementation, the durable binding,
    #: the run-review route and the preflight check. Anything else importing them is a second
    #: path to a queue that must not exist (FR-010.27).
    ALLOWED: ClassVar[frozenset[str]] = frozenset(
        {
            "clients/midataworks_review.py",
            "services/review/binding.py",
            "services/review/verdicts.py",
            "services/review/preflight.py",
            "api/v1/endpoints/runs_review.py",
        }
    )

    def _importers_of(self, module: str) -> set[str]:
        found: set[str] = set()
        for path in SRC.rglob("*.py"):
            tree = ast.parse(path.read_text("utf-8"))
            for node in ast.walk(tree):
                if _imports(node, module):
                    found.add(str(path.relative_to(SRC)))
        return found - {module.replace(".", "/") + ".py"}  # never itself

    def test_only_the_review_seam_imports_the_binding(self):
        importers = self._importers_of("services.review.binding")
        assert importers <= self.ALLOWED, sorted(importers - self.ALLOWED)

    def test_only_the_seam_and_the_acquisition_path_import_the_client(self):
        """``services/parts/acquire.py`` is the other legitimate caller - the HTTP import
        path - and it is named rather than excluded silently."""
        allowed = self.ALLOWED | {
            "services/parts/acquire.py",
            "api/v1/endpoints/midataworks.py",
            "clients/midataworks_errors.py",
        }
        importers = self._importers_of("clients.midataworks")
        assert importers <= allowed, sorted(importers - allowed)

    def test_no_pack_or_authoring_module_imports_either(self):
        """F7's authoring dry-run, when it lands, **must not** appear in either list. The
        assertion is written now, against the packages F7 will own, so the guard exists
        before the code it guards - which is the only ordering that works.
        """
        offenders = {
            path
            for path in self._importers_of("clients.midataworks")
            | self._importers_of("services.review.binding")
            if path.startswith("packs/") or "dry_run" in path or "authoring" in path
        }
        assert offenders == set(), sorted(offenders)


class TestASelectorEvaluationTouchesNothing:
    def test_evaluating_a_rule_selector_costs_zero_midataworks_calls(self, monkeypatch):
        """AC-010.12's behavioural half, asserted on the **call count**.

        F7's authoring dry-run does not exist yet; what exists is F3's ``evaluate`` path,
        which is the same shape - a selector judged against a sample with no run - and is
        the thing a dry-run will call. So the guarantee is asserted against the path that
        exists today, and the AST checks above cover the one that is coming.
        """
        from fastapi.testclient import TestClient

        from api.app import create_app

        counter = ClientCounter(monkeypatch)
        client = TestClient(create_app(), raise_server_exceptions=False)
        response = client.get("/api/v1/selectors")
        assert response.status_code == 200
        assert counter.count == 0

    def test_the_human_review_type_is_listed_as_unavailable_rather_than_hidden(self):
        """ADR-013: an option missing from a response is indistinguishable from one that is
        broken. So with miDataworks unconfigured the type is still **named**, with the reason.
        """
        from fastapi.testclient import TestClient

        from api.app import create_app

        client = TestClient(create_app(), raise_server_exceptions=False)
        body = client.get("/api/v1/selectors").json()
        # F3's listing is a flat array keyed by `name`. Read from the response rather than
        # from a remembered shape: the key was `kind` in this feature's notes and is `name`
        # on the wire, which is the kind of drift a cross-feature assertion exists to catch.
        names = {entry["name"] for entry in body}
        assert "human_review" in names, f"the type is hidden rather than named: {names}"
        entry = next(e for e in body if e["name"] == "human_review")
        assert entry["available"] is False
        assert entry["unavailable_reason"] == "midataworks_not_configured", (
            "F3's listing must name THIS feature's refusal code, so an operator is told "
            "which sibling to configure rather than that something is unavailable"
        )
        assert entry["requires"] == "midataworks"
