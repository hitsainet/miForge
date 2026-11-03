"""R-02.19's activation source: off by default, named when absent, and *unreachable is a
different code*  (F4 task 8.6; FR-004.33-36, AC-004.28, E-20, E-21).

**Codes and ``details`` keys are asserted; message text is not.** Messages are copy and
will be improved; codes are contract, and something matches on them.

**Every HTTP case goes through the real ``httpx`` client over a mock transport**, not past
it. A double that cannot express the defect makes its assertion vacuous - the trap F11
found in its own MUT-19a and the one that let two ``hot_update_step`` call sites pass
``object()`` as a model. So ``Probe.statistics`` is exercised by the same code path
production takes, with only the socket replaced.
"""

from __future__ import annotations

import httpx
import pytest

from clients import activations
from core.errors import Refusal
from services.scoring.settings import (
    ScoringSettings,
    get_scoring_settings,
    reset_scoring_settings,
)

A_ROUTE = "/api/probe/activations"


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch):
    monkeypatch.delenv("ACTIVATION_SOURCE_BASE_URL", raising=False)
    reset_scoring_settings()
    yield
    reset_scoring_settings()


def configure(monkeypatch, url: str = "http://probes.invalid:9000") -> None:
    monkeypatch.setenv("ACTIVATION_SOURCE_BASE_URL", url)
    reset_scoring_settings()


def transport_returning(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestOffByDefault:
    """AC-004.28's first clause, and R-02.50's whole posture."""

    def test_the_setting_is_unset_in_a_default_install(self):
        assert get_scoring_settings().activation_source_base_url is None
        assert activations.is_configured() is False
        assert activations.base_url() is None

    def test_an_empty_string_is_not_a_configured_source(self, monkeypatch):
        """``ACTIVATION_SOURCE_BASE_URL=`` in a ``.env`` means unset, not the empty URL.

        Without this the standalone path is never taken on a machine with a blank line in
        its env file, and the failure appears as a connection error to ``""`` at some
        arbitrary later moment instead of a named refusal at the boundary.
        """
        monkeypatch.setenv("ACTIVATION_SOURCE_BASE_URL", "")
        reset_scoring_settings()
        assert activations.base_url() is None
        assert activations.is_configured() is False
        assert activations.resolve("recording activation statistics") is None
        # And the refusal is the named one, not a connection error to "" an hour later.
        with pytest.raises(Refusal) as caught:
            activations.require_activation_source("x", "activation_statistics")
        assert caught.value.code == "activation_source_not_configured"

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_every_blank_spelling_is_unset(self, monkeypatch, blank):
        """Asserted **through the reader**, so it holds whichever layer coerces it.

        *This docstring said ``ScoringSettings.activation_source_base_url`` has "no
        empty-string validator, unlike ``core.config``'s two sibling URLs - so this
        coercion is the reader's". It had none when that was written and it has
        ``_empty_string_is_unset`` now, so the claim is false and the layer it named is no
        longer the only one doing the work.* Both layers coerce, deliberately: the settings
        validator stops ``.env``'s deliberately-blank line configuring the empty URL, and
        :func:`activations.base_url` refuses to be fooled by a value assigned past
        validation. Found by running the case, not by reading."""
        monkeypatch.setenv("ACTIVATION_SOURCE_BASE_URL", blank)
        reset_scoring_settings()
        assert activations.base_url() is None

    @pytest.mark.parametrize("through", ["the environment", "a keyword argument"])
    def test_a_credential_in_the_url_is_REFUSED_not_masked(self, monkeypatch, through):
        """**A base URL is a location**, and the refusal is a *settings-layer* refusal.

        The same hole on ``millm_base_url`` defeated every field-by-field mask and reached
        any log line or traceback; masking hides the symptom and leaves the credential live
        on the wire. ``ScoringSettings`` now carries the validator ``core.config``'s two
        sibling URLs always had, so the value never becomes settings at all and no consumer
        can be reached with it - which is strictly stronger than refusing at one boundary.
        Same shape as, and asserted the same way as,
        ``test_no_secrets_in_artifacts.TestACredentialInABaseUrlIsRefused``: Pydantic's
        ``ValidationError`` **is** a ``ValueError``.

        *This test asserted a ``Refusal`` with code
        ``activation_source_url_carries_credentials`` - a code that exists nowhere under
        ``src/`` - and its docstring said ``ScoringSettings`` has no validator for this
        field. Both were true when written and the settings validator made them false; the
        client-boundary branch it drove is no longer reachable from any ordinary path.*

        **The secret-absent assertion is the regression test for the leak itself**, not
        decoration: the validator's message carries the host only, but Pydantic appends
        ``input_value=...`` to ``str(ValidationError)`` regardless, so without
        ``hide_input_in_errors=True`` on ``model_config`` the refusal reproduces the
        credential. That is reachable from a request: ``core.config``'s
        ``configured_integrations()`` reads this settings object and ``api/health.py``
        calls it, so ``GET /api/health`` wrote the credential into the application log.
        Seen red against the missing flag before being trusted.
        """
        url = "http://user:s3cret@probes.invalid:9000"
        with pytest.raises(ValueError) as caught:
            if through == "the environment":
                monkeypatch.setenv("ACTIVATION_SOURCE_BASE_URL", url)
                reset_scoring_settings()
                get_scoring_settings()
            else:
                ScoringSettings(activation_source_base_url=url)

        message = str(caught.value)
        assert "credential" in message.lower()
        assert "s3cret" not in message, (
            "the refusal reproduced the credential it was refusing. A validation error is "
            "an output - this one reaches the application log through GET /api/health - so "
            "the message must name the problem without echoing the secret. Check that "
            "ScoringSettings.model_config sets hide_input_in_errors=True, as "
            "core.config.Settings does for the same reason."
        )
        # Or an operator is told a credential is in "a URL" and not told which one.
        assert "probes.invalid" in message, "the refusal must still name the host"

    def test_a_bare_username_is_refused_too(self):
        """It carries no secret and is refused anyway, exactly as ``core.config``'s two are:
        it means the URL is being used to carry identity, which is the pattern being kept
        out. A refusal scoped to ``password`` would admit the pattern and leak the next one."""
        with pytest.raises(ValueError) as caught:
            ScoringSettings(activation_source_base_url="http://trainer@probes.invalid:9000")
        assert "credential" in str(caught.value).lower()

    def test_an_ordinary_url_is_still_accepted(self):
        """The other direction, because a refusal that is too broad is worse than none: if
        this went red every deployment with an activation source would fail to start, and
        the fix would be to delete the check - which is how a correct guard gets removed."""
        url = "http://probes.invalid:9000/probe"
        assert ScoringSettings(activation_source_base_url=url).activation_source_base_url == url

    def test_resolve_returns_None_rather_than_raising(self):
        """A caller doing optional extra work needs a branch, not an exception."""
        assert activations.resolve("recording activation statistics") is None


class TestTheBoundaryGuardIsDefenceInDepthAndREACHABLE:
    """``_reject_credentials_in`` after the settings validator landed in front of it.

    **An unreachable guard reads as protection while protecting nothing**, and this
    project's precedent is to delete one rather than keep it "for safety" - F8's duplicate
    code check was deleted for exactly that, because the validator ran first and left it
    unreachable for the two cases it existed to catch. So this guard is kept **only**
    because the tests below genuinely reach it, through the two routes that bypass a
    Pydantic validator rather than through a monkeypatched reader:

    * ``model_construct``, which skips validation by design;
    * **in-place assignment**, because ``ScoringSettings`` does not set
      ``validate_assignment`` - asserted below rather than assumed, so if that config ever
      changes, the test that justifies this guard's existence says so.

    Neither is exotic: a settings object is a process-wide singleton that any code can
    assign to, and the client is the one reader of this field.
    """

    def test_assignment_is_genuinely_unvalidated_so_this_guard_has_work_to_do(self):
        """The **positive control** for the whole class: if assignment were validated the
        three tests below would be testing an unreachable branch, and the guard should be
        deleted instead of kept."""
        assert not ScoringSettings.model_config.get("validate_assignment"), (
            "ScoringSettings now validates assignment, so clients/activations.py's "
            "_reject_credentials_in may be unreachable. Re-check its reachability and "
            "delete it if nothing can reach it - an unreachable guard reads as protection "
            "while protecting nothing."
        )
        settings = ScoringSettings(activation_source_base_url="http://probes.invalid:9000")
        settings.activation_source_base_url = "http://u:s3cret@probes.invalid:9000"
        assert settings.activation_source_base_url.startswith("http://u:")

    @pytest.mark.parametrize("bypass", ["assignment", "model_construct"])
    def test_a_credential_assigned_past_validation_is_refused_at_the_boundary(
        self, monkeypatch, bypass
    ):
        """Reached through the real settings module, not by faking :func:`base_url`.

        Asserts the code, that the secret is absent from **args and details both**, and
        that the host is present - a refusal that cannot name the host cannot be acted on.
        """
        from services.scoring import settings as settings_module

        url = "http://u:s3cret@probes.invalid:9000"
        if bypass == "assignment":
            smuggled = ScoringSettings(activation_source_base_url="http://probes.invalid:9000")
            smuggled.activation_source_base_url = url
        else:
            smuggled = ScoringSettings.model_construct(activation_source_base_url=url)
        monkeypatch.setattr(settings_module, "_settings", smuggled)

        assert activations.base_url() == url, "the bypass did not reach the client's reader"
        with pytest.raises(Refusal) as caught:
            activations.require_activation_source("x", "activation_statistics")

        assert caught.value.code == "activation_source_url_carries_credentials"
        blob = f"{caught.value.args} {caught.value.details}"
        assert "s3cret" not in blob, "the refusal reproduced the credential it was refusing"
        assert "probes.invalid" in blob, "the refusal must still say which host"

    def test_probe_construction_refuses_on_the_same_smuggled_value(self, monkeypatch):
        """The constructor runs the guard through ``require_activation_source``, so an
        object is never built around a credentialled base URL that would then be sent."""
        from services.scoring import settings as settings_module

        monkeypatch.setattr(
            settings_module,
            "_settings",
            ScoringSettings.model_construct(
                activation_source_base_url="http://u:s3cret@probes.invalid:9000"
            ),
        )
        with pytest.raises(Refusal) as caught:
            activations.Probe(action="x")
        assert caught.value.code == "activation_source_url_carries_credentials"
        assert "s3cret" not in f"{caught.value.args} {caught.value.details}"


class TestUnconfiguredIsNamed:
    """E-20. Named with the capability it costs - never omitted, never substituted."""

    def test_the_guard_refuses_with_ADR_010s_spelling(self):
        with pytest.raises(Refusal) as caught:
            activations.require_activation_source(
                "recording activation statistics for step 40", "activation_statistics"
            )
        assert caught.value.code == "activation_source_not_configured"

    def test_the_refusal_names_the_service_the_setting_and_the_capability(self):
        with pytest.raises(Refusal) as caught:
            activations.require_activation_source("x", "activation_statistics")
        details = caught.value.details
        assert details["service"] == "activation_source"
        assert details["setting"] == "ACTIVATION_SOURCE_BASE_URL"
        assert details["capability"] == "activation_statistics"
        # The capability made unavailable, in words, not merely its key.
        assert details["capabilities_unavailable"] == [
            activations.CAPABILITIES["activation_statistics"]
        ]

    def test_constructing_a_probe_refuses_at_the_boundary(self):
        """The guard runs in the constructor, so an unconfigured install never builds an
        object with a ``None`` base URL that fails later somewhere else."""
        with pytest.raises(Refusal) as caught:
            activations.Probe(action="recording activation statistics")
        assert caught.value.code == "activation_source_not_configured"

    def test_an_unregistered_capability_is_refused_not_defaulted(self):
        """A typo would otherwise make a status response report on a capability nothing
        gates - a fabricated value, and silent (ADR-018)."""
        with pytest.raises(Refusal) as caught:
            activations.describe("activationz")
        assert caught.value.code == "activation_capability_unknown"
        assert caught.value.details["known"] == ["activation_statistics"]

    def test_resolve_checks_the_capability_even_on_the_absent_path(self):
        """Caught on a default install rather than first appearing when somebody
        configures the service."""
        with pytest.raises(Refusal) as caught:
            activations.resolve("x", "activationz")
        assert caught.value.code == "activation_capability_unknown"


class TestUnreachableIsNotUnconfigured:
    """E-21, and AC-004.28's sharpest clause. **Two states, two codes, two fixes.**"""

    def test_a_transport_failure_is_activation_source_unreachable(self, monkeypatch):
        configure(monkeypatch)

        def refuse_to_connect(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host", request=request)

        with (
            activations.Probe(
                action="recording activation statistics",
                transport=transport_returning(refuse_to_connect),
            ) as probe,
            pytest.raises(Refusal) as caught,
        ):
            probe.statistics(A_ROUTE, {"step": 40})

        assert caught.value.code == "activation_source_unreachable"
        assert caught.value.code != activations.NOT_CONFIGURED
        assert caught.value.details["setting"] == "ACTIVATION_SOURCE_BASE_URL"
        assert "ConnectError" in caught.value.details["detail"]

    def test_a_5xx_is_unreachable_too(self, monkeypatch):
        configure(monkeypatch)
        with (
            activations.Probe(
                action="recording activation statistics",
                transport=transport_returning(lambda r: httpx.Response(503, text="down")),
            ) as probe,
            pytest.raises(Refusal) as caught,
        ):
            probe.statistics(A_ROUTE, {"step": 40})
        assert caught.value.code == "activation_source_unreachable"
        assert "503" in caught.value.details["detail"]

    def test_a_4xx_is_REFUSED_which_is_a_THIRD_code(self, monkeypatch):
        """It answered. Reporting that as a network failure sends the operator to the
        wrong place, and reporting it as not-configured sends them further still.

        **The body is a well-formed JSON object on purpose.** Written first with
        ``text="no such feature"``, this test passed against a mutation that disabled the
        4xx branch entirely - the non-JSON fallback raised the same code with the same
        ``status``, so the two paths agreed by construction and the assertion was vacuous.
        Control MUT-A5 is what found it. With a JSON object body the *only* way to a
        refusal is the status check, and disabling it returns the error envelope as
        statistics instead.
        """
        configure(monkeypatch)
        envelope = {"error": {"code": "no_such_feature", "message": "unknown feature id"}}
        with (
            activations.Probe(
                action="recording activation statistics",
                transport=transport_returning(lambda r: httpx.Response(422, json=envelope)),
            ) as probe,
            pytest.raises(Refusal) as caught,
        ):
            probe.statistics(A_ROUTE, {"step": 40})
        assert caught.value.code == "activation_source_refused"
        assert caught.value.code not in (activations.NOT_CONFIGURED, activations.UNREACHABLE)
        assert caught.value.details["status"] == 422

    def test_a_4xx_error_envelope_is_never_returned_as_statistics(self, monkeypatch):
        """The failure direction that matters: a refusal body read as a measurement.

        ``{"error": {...}}`` is a conforming JSON object, so a status check that stopped
        looking would hand it back and the run log would record an error envelope as a
        step's activation statistics - a fabricated value wearing a 200 (ADR-018).
        """
        configure(monkeypatch)
        for status in (400, 401, 403, 404, 409, 422):
            with activations.Probe(
                action="x",
                transport=transport_returning(
                    lambda r, s=status: httpx.Response(s, json={"feature_mean": 0.9})
                ),
            ) as probe:
                with pytest.raises(Refusal) as caught:
                    probe.statistics(A_ROUTE, {})
                assert caught.value.code == "activation_source_refused", status

    def test_the_three_codes_are_distinct(self):
        """The property the three cases above exist to protect, asserted directly so a
        rename that collapses two of them fails here and says so."""
        codes = {activations.NOT_CONFIGURED, activations.UNREACHABLE, activations.REFUSED}
        assert len(codes) == 3, codes


class TestTheOneCallSite:
    def test_a_successful_probe_returns_the_body_unreshaped(self, monkeypatch):
        """**Not reshaped, not defaulted and not written anywhere** (FR-004.34)."""
        configure(monkeypatch)
        seen: list[httpx.Request] = []

        def answer(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"feature_mean": 0.41, "clusters": [1, 2]})

        with activations.Probe(
            action="recording activation statistics", transport=transport_returning(answer)
        ) as probe:
            body = probe.statistics(A_ROUTE, {"step": 40, "completions": ["a"]})

        assert body == {"feature_mean": 0.41, "clusters": [1, 2]}
        # Payload AND call count: "was called" passes against a call sending the wrong
        # arguments (ADR-016 #1).
        assert len(seen) == 1
        assert seen[0].url.path == A_ROUTE
        assert seen[0].method == "POST"

    def test_a_route_without_a_leading_slash_still_addresses_the_route(self, monkeypatch):
        configure(monkeypatch, "http://probes.invalid:9000/")
        seen: list[str] = []

        def answer(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(200, json={})

        with activations.Probe(action="x", transport=transport_returning(answer)) as probe:
            probe.statistics("api/probe/activations", {})
        assert seen == [A_ROUTE], "a trailing slash on the base URL must not produce //"

    def test_an_empty_route_is_REFUSED_rather_than_defaulted(self, monkeypatch):
        """**The invented-contract guard.** R-02.19 names no route, so there must be no
        default: a constant here would be a sibling endpoint invented in this repository
        and asserted as that sibling's contract."""
        configure(monkeypatch)
        with activations.Probe(
            action="recording activation statistics",
            transport=transport_returning(lambda r: httpx.Response(200, json={})),
        ) as probe:
            for route in ("", "   "):
                with pytest.raises(Refusal) as caught:
                    probe.statistics(route, {"step": 1})
                assert caught.value.code == "activation_source_route_unspecified"

    def test_a_non_json_2xx_is_named_rather_than_guessed_at(self, monkeypatch):
        configure(monkeypatch)
        with (
            activations.Probe(
                action="x",
                transport=transport_returning(lambda r: httpx.Response(200, text="<html/>")),
            ) as probe,
            pytest.raises(Refusal) as caught,
        ):
            probe.statistics(A_ROUTE, {})
        assert caught.value.code == "activation_source_refused"

    def test_a_json_array_is_not_accepted_as_a_statistics_object(self, monkeypatch):
        configure(monkeypatch)
        with (
            activations.Probe(
                action="x",
                transport=transport_returning(lambda r: httpx.Response(200, json=[1, 2, 3])),
            ) as probe,
            pytest.raises(Refusal) as caught,
        ):
            probe.statistics(A_ROUTE, {})
        assert caught.value.code == "activation_source_refused"

    def test_resolve_hands_back_a_usable_probe_once_configured(self, monkeypatch):
        configure(monkeypatch)
        probe = activations.resolve(
            "recording activation statistics",
            transport=transport_returning(lambda r: httpx.Response(200, json={"ok": True})),
        )
        assert probe is not None
        with probe:
            assert probe.statistics(A_ROUTE, {}) == {"ok": True}


class TestTheStatusRow:
    """``reachable=None`` is not ``reachable=False`` (E-21; F10's FR-010.6)."""

    def test_unconfigured_reports_every_capability_unavailable_by_key(self):
        row = activations.integration_status()
        assert row["configured"] is False
        assert row["code"] == "activation_source_not_configured"
        assert row["reachable"] is None
        assert row["capabilities_unavailable"] == ["activation_statistics"]

    def test_configured_but_unprobed_reports_reachable_None(self, monkeypatch):
        configure(monkeypatch)
        row = activations.integration_status()
        assert row["configured"] is True
        assert row["code"] is None
        assert row["reachable"] is None, "unprobed is not unreachable"
        assert row["capabilities_unavailable"] == []

    def test_configured_and_down_reports_reachable_False(self, monkeypatch):
        configure(monkeypatch)
        row = activations.integration_status(reachable=False)
        assert row["configured"] is True
        assert row["reachable"] is False
        # The pair that must never collapse: unset -> (False, None); down -> (True, False).
        reset_scoring_settings()
        monkeypatch.delenv("ACTIVATION_SOURCE_BASE_URL", raising=False)
        reset_scoring_settings()
        unset = activations.integration_status(reachable=False)
        assert (unset["configured"], unset["reachable"]) == (False, None)

    def test_the_row_is_built_by_LOOPING_the_registry(self):
        """So a capability cannot be added without appearing in the status row."""
        row = activations.integration_status()
        assert sorted(row["capabilities"]) == activations.all_keys()
        assert set(row["capabilities"].values()) == set(activations.CAPABILITIES.values())


class TestTheFenceAndTheSingleCallSite:
    """FR-004.35 and ADR-010/ADR-017's one-module rule, asserted structurally."""

    def test_this_is_the_only_module_outside_clients_that_may_speak_to_a_probe(self):
        """``httpx`` appears in ``clients/`` and nowhere near the scoring path."""
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src"
        offenders: list[str] = []
        for path in sorted((src / "services" / "scoring").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] in ("httpx", "clients"):
                        offenders.append(f"{path.name}:{node.lineno} -> {name}")
        assert offenders == [], (
            f"the scoring path reaches a client or an HTTP library: {offenders}. R-02.19's "
            f"source is reached through clients/activations.py, by configuration, or not "
            f"at all (FR-004.35, ADR-010)."
        )

    def test_this_module_names_no_sibling_by_name(self):
        """ADR-022. The whole reason R-02.19 was rerouted here is that its original text
        named miLLM's probe endpoints; the replacement must not reintroduce the name."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "src" / "clients" / "activations.py"
        ).read_text(encoding="utf-8")
        code_lines = [
            line
            for line in source.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        body = "\n".join(code_lines)
        # The docstrings legitimately explain the history, so only *code* is searched for a
        # hard-coded sibling host or endpoint.
        for forbidden in ("/api/weights/", "millm:", "mistudio:", "midataworks:"):
            assert forbidden not in body, f"{forbidden!r} is hard-coded in the client"

    def test_no_default_route_constant_exists(self):
        """The guard against the invented contract, asserted on the module rather than on
        a behaviour: a default added later would make the refusal above unreachable."""
        import inspect

        signature = inspect.signature(activations.Probe.statistics)
        assert signature.parameters["route"].default is inspect.Parameter.empty
