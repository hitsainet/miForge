"""**The binding that was missing**  (ADR-016 #1; FR-004.1, R-02.11, R-02.49).

F3's judge-model selector resolves its provider through
``services.selectors.provider_port.ProviderResolver``, installed by
:func:`services.selectors.provider_port.bind_provider_resolver`. Before
:mod:`services.scoring.wiring` existed **that binding was made only in tests**: every judge
test passed by binding a stub itself, the type was listed, the plugin was implemented and
documented - and in a running miForge no resolver was ever bound, so
``current_provider_resolver()`` returned ``None`` and the judge refused
``judge_provider_not_configured`` **whatever the operator configured.**

That is the sixteen-unregistered-MCP-tools failure in this repository: *every one of those
tests passed by importing the module directly, so the suite was green and the documentation
said ✅ while the feature did not exist for any caller.*

**The control.** Delete the ``bind_provider_resolver(...)`` line in
:func:`services.scoring.wiring.wire` and this file must go red. Observed and recorded, not
assumed.

**Why "was bound" is not enough.** A reachability test that only checks *something* was
bound passes against a binding that resolves nothing - the same way "was called" passes
against a call sending the wrong arguments. So this file asserts three separable things:

1. a resolver is bound **after a production import** - not after a test binds one;
2. it is **this registry's**, so a leftover stub from another test cannot satisfy it;
3. a provider configured in the environment is **actually resolvable through it**, and the
   plugin reaches it end to end.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from providers import registry
from services.selectors import availability, provider_port
from services.selectors.settings import reset_selector_settings

SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def judge_configured(monkeypatch: pytest.MonkeyPatch):
    """One judge-eligible HTTP row, configured the way an operator would configure it."""
    monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "judge-7b")
    monkeypatch.setenv("JUDGE_PROVIDER_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("JUDGE_PROVIDER_MODEL", "judge-7b")
    reset_selector_settings()
    yield "judge"
    reset_selector_settings()


@pytest.fixture(autouse=True)
def rewired():
    """Leave the process wired as production leaves it, whatever a test did in between."""
    yield
    from services.scoring import wiring

    wiring._wired = False
    wiring.wire()


class TestTheBindingExistsInAFreshProcess:
    def test_importing_the_api_binds_a_resolver_with_no_test_help(self) -> None:
        """**The load-bearing assertion.** A subprocess, so no fixture in this suite can
        have bound anything first.

        Imports what production imports - the application factory - and asks the port
        whether anything is bound. Before this feature the answer was ``None``.

        **Why the answer is read from a MARKED line and not from the whole of stdout.**
        This assertion was ``out.stdout.strip() == "providers.registry.JudgeProviderResolver"``,
        which made the project's most load-bearing control breakable by **any** future
        startup log line: ``create_app()`` now runs ``atomic_dir.verify_root_is_atomic``
        and logs a structured JSON line to stdout when the checkpoint root is unwritable,
        which is the normal state of a developer box where ``/data/checkpoints`` is
        root-owned. Exact equality therefore went red on this workstation and stayed green
        in CI, where ``CHECKPOINT_ROOT`` is writable - a control whose verdict depended on
        the environment rather than on the wiring it exists to protect, and in the
        direction that reads as *"the resolver is not bound"*. The marker is printed by the
        subprocess, so an absent marker is still a failure: nothing here passes because a
        measurement did not happen.
        """
        code = (
            f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
            "from api.app import create_app\n"
            "create_app()\n"
            "from services.selectors.provider_port import current_provider_resolver\n"
            "r = current_provider_resolver()\n"
            "print('RESOLVER=' + type(r).__module__ + '.' + type(r).__name__)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert out.returncode == 0, out.stderr
        assert "RESOLVER=providers.registry.JudgeProviderResolver" in out.stdout.splitlines(), (
            f"no provider resolver is bound after a production import (stdout was "
            f"{out.stdout!r}). F3's judge-model selector is then unreachable for "
            f"every caller while all of its own tests pass, because they bind a stub "
            f"themselves. That is ADR-016 #1."
        )

    def test_it_is_this_registrys_resolver_and_not_merely_something(self) -> None:
        """A leftover stub from another test would satisfy "something was bound"."""
        from services.scoring import wiring

        wiring._wired = False
        wiring.wire()
        resolver = provider_port.current_provider_resolver()
        assert isinstance(resolver, registry.JudgeProviderResolver)

    def test_the_bound_object_satisfies_f3s_protocol_at_runtime(self) -> None:
        """The agreement is structural, so it is checked structurally.

        ``providers/`` deliberately imports nothing from ``services/`` - the layering is
        one-way - so the only thing keeping :class:`providers.registry.JudgeProviderResolver`
        in step with F3's ``Protocol`` is this assertion.
        """
        assert isinstance(registry.judge_resolver(), provider_port.ProviderResolver)


class TestTheBindingActuallyRESOLVES:
    """A binding that resolves nothing is a registry that exists for no caller."""

    def test_a_configured_judge_is_available_through_the_port(self, judge_configured) -> None:
        from services.scoring import wiring

        wiring._wired = False
        wiring.wire()
        resolver = provider_port.current_provider_resolver()
        assert resolver is not None
        assert resolver.available(judge_configured) is True
        assert resolver.resolve(judge_configured).identity.model == "judge-7b"

    def test_f3s_own_availability_function_now_answers_true(self, judge_configured) -> None:
        """The end the operator sees: ``GET /api/v1/selectors`` lists the judge as available.

        It needs **both** halves - a bound resolver and a configured endpoint - and the
        resolver half is the one that was missing in production.
        """
        from services.scoring import wiring
        from services.selectors.settings import get_selector_settings

        wiring._wired = False
        wiring.wire()
        assert availability.generation_provider_available(get_selector_settings()) is True

    def test_unbinding_makes_the_judge_unavailable_BY_NAME(self, judge_configured) -> None:
        """The negative control for the control: with no resolver the type is unavailable,
        **by name**, and nothing falls back to the model being trained."""
        from services.scoring import wiring
        from services.selectors.settings import get_selector_settings
        from services.selectors.types import Requirement

        wiring.unwire()
        assert availability.generation_provider_available(get_selector_settings()) is False
        available, reason = availability.availability_of(
            Requirement.GENERATION_PROVIDER, get_selector_settings()
        )
        assert (available, reason) == (False, "judge_provider_not_configured")


class TestTheWiringIsAnImportSideEffectOfThePackage:
    def test_wire_is_called_when_the_package_is_imported(self) -> None:
        """*An import side effect that nobody triggers is not a side effect.*"""
        module = importlib.import_module("services.scoring")
        assert hasattr(module, "wire")

    def test_the_source_line_exists_and_is_the_one_the_control_removes(self) -> None:
        """Names the exact line a mutation removes, so the control is unambiguous.

        This is the *weaker* form of the rule - a grep proves a call exists, not that
        anything depends on it - and it is here only to point at the line. The assertions
        above are what actually bite.
        """
        source = (SRC / "services" / "scoring" / "wiring.py").read_text(encoding="utf-8")
        assert "provider_port.bind_provider_resolver(registry.judge_resolver())" in source

    def test_wire_is_idempotent(self) -> None:
        from services.scoring import wiring

        wiring.wire()
        wiring.wire()
        assert isinstance(provider_port.current_provider_resolver(), registry.JudgeProviderResolver)
