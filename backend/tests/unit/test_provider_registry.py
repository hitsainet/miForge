"""The one provider resolution path  (FR-004.1, 4, 6, 10; E-1, E-4; M-9).

What this file is really about: **`get()` never returns a different provider.** Everything
else here supports that - the rows come from live settings so a configuration change is
visible without a restart, the in-process row is listed-but-unavailable where no parameters
are held, and a mid-run identity change refuses unless the operator re-pins it.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from providers import registry
from providers.base import ProviderIdentity
from providers.fakes import FakeModelHandle
from providers.settings import http_provider_rows


@pytest.fixture(autouse=True)
def unbound_handle():
    """Every test starts with no parameters held, which is the API process's state."""
    registry.bind_model_handle(None)
    yield
    registry.bind_model_handle(None)


@pytest.fixture
def no_providers(monkeypatch: pytest.MonkeyPatch):
    """No ``PROVIDER_*`` row configured - the standalone world (AC20)."""
    for key in list(registry.http_provider_rows.__globals__["os"].environ):
        if key.startswith("PROVIDER_"):
            monkeypatch.delenv(key, raising=False)
    return None


@pytest.fixture
def one_http_row(monkeypatch: pytest.MonkeyPatch, no_providers):
    monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
    monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "judge-7b")
    return "judge"


class TestTheRowsComeFromLiveSettings:
    def test_with_nothing_configured_in_process_is_the_only_row(self, no_providers) -> None:
        """The standalone case: no endpoint, no credential, no sibling, one provider."""
        rows = registry.list_rows()
        assert [row.name for row in rows] == [registry.IN_PROCESS_NAME]

    def test_an_endpoint_configured_after_import_appears_without_a_restart(
        self, one_http_row
    ) -> None:
        """Computed each call, not cached at import.

        F3's capability listing does the same, and for the same reason: an operator who
        configures an endpoint and is told it does not exist goes looking for a bug.
        """
        assert one_http_row in {row.name for row in registry.list_rows()}

    def test_a_base_url_with_no_model_refuses_rather_than_guessing(
        self, monkeypatch: pytest.MonkeyPatch, no_providers
    ) -> None:
        """A guessed model name makes every score it touches incomparable while looking fine."""
        monkeypatch.setenv("PROVIDER_HALF_BASE_URL", "http://half.test")
        with pytest.raises(Refusal) as caught:
            http_provider_rows()
        assert caught.value.code == "inference_provider_misconfigured"
        assert "PROVIDER_HALF_MODEL" in caught.value.details["missing"]

    def test_an_empty_value_counts_as_unset(
        self, monkeypatch: pytest.MonkeyPatch, no_providers
    ) -> None:
        """`.env.example`'s deliberately blank lines must not configure a provider."""
        monkeypatch.setenv("PROVIDER_BLANK_BASE_URL", "   ")
        assert [row.name for row in registry.list_rows()] == [registry.IN_PROCESS_NAME]

    def test_an_unparseable_timeout_refuses_rather_than_defaulting(
        self, monkeypatch: pytest.MonkeyPatch, one_http_row
    ) -> None:
        """A discarded setting the operator chose is a configuration they believe is in force."""
        monkeypatch.setenv("PROVIDER_JUDGE_READ_TIMEOUT_S", "soon")
        with pytest.raises(Refusal) as caught:
            http_provider_rows()
        assert caught.value.code == "inference_provider_misconfigured"


class TestTheCapabilityVocabulary:
    def test_activations_is_not_a_capability(self) -> None:
        """R-02.19's own amended text: *"not a capability of R-02.49's inference provider,
        whose scope is generation and which exposes no activations."*

        Asserted because **a declared-but-unused capability slot is how the next reader
        concludes the provider is the route**, which is precisely what DEF-10 reported.
        """
        assert "activations" not in registry.KNOWN_CAPABILITIES

    def test_the_vocabulary_is_generation_and_embeddings(self) -> None:
        assert registry.KNOWN_CAPABILITIES == frozenset({"generation", "embeddings"})

    def test_no_registered_provider_declares_embeddings_today(self, one_http_row) -> None:
        """`embeddings` stays in the vocabulary while nothing declares it.

        R-02.16 asks for *"exact or fuzzy match"* and F4 adds no embeddings provider
        (DEF-3); whether the capability belongs on this interface at all is **F3's** open
        question, not F4's to close. So the name survives and no implementation claims it -
        and that is asserted rather than assumed, because a capability that quietly acquired
        an implementation would be F4 widening R-02.49.
        """
        for row in registry.list_rows():
            assert "embeddings" not in row.capabilities


class TestGetNeverReturnsADifferentProvider:
    def test_an_unknown_name_refuses_and_names_what_is_configured(self, no_providers) -> None:
        with pytest.raises(Refusal) as caught:
            registry.get("nope")
        assert caught.value.code == "inference_provider_unavailable"
        assert registry.IN_PROCESS_NAME in caught.value.details["configured"]

    def test_in_process_without_parameters_refuses_rather_than_substituting(
        self, no_providers
    ) -> None:
        """E-1. The API process holds no parameters; it says so by name."""
        with pytest.raises(Refusal) as caught:
            registry.get(registry.IN_PROCESS_NAME)
        assert caught.value.code == "inference_provider_unavailable"
        assert "will not substitute" in caught.value.message

    def test_in_process_resolves_once_a_handle_is_bound(self, no_providers) -> None:
        registry.bind_model_handle(FakeModelHandle())
        provider = registry.get(registry.IN_PROCESS_NAME)
        assert provider.identity.kind == "in_process"
        assert provider.identity.model == "fake-model"

    def test_the_in_process_row_is_listed_unavailable_rather_than_omitted(
        self, no_providers
    ) -> None:
        """ADR-013: *an option missing from a response is indistinguishable from a broken one.*"""
        row = registry.row_for(registry.IN_PROCESS_NAME)
        assert row is not None
        assert row.available is False
        assert row.unavailable_reason == "inference_provider_unavailable"

    def test_M9_AN_UNKNOWN_NAME_REFUSES_EVEN_WITH_A_HANDLE_BOUND(self, no_providers) -> None:
        """**Written after mutation control M-9 SURVIVED the suite.**

        The earlier assertions resolved an unknown name with **no handle bound**, so a
        ``return get(IN_PROCESS_NAME)`` fallback at the end of :func:`providers.registry.get`
        was caught by nothing: with no parameters held it refused anyway, for the wrong
        reason, and the test read as protection. The mutation that matters is the one taken
        **in the ``train`` worker**, which is the only process that holds a handle - and
        therefore the only process where a fallback would silently succeed.

        That is also the shape in which the defect would actually ship: an operator
        configures an endpoint, the name is typo'd or the row is unset, and the run scores
        itself through the model it is training while every surface says the configured
        provider answered. *A silent provider substitution rewrites the curve's meaning
        mid-run.*
        """
        registry.bind_model_handle(FakeModelHandle())
        with pytest.raises(Refusal) as caught:
            registry.get("judge")
        assert caught.value.code == "inference_provider_unavailable"
        assert "will not fall back" in caught.value.message

    def test_M9_A_CONFIGURED_ROW_WHOSE_KIND_IS_UNREGISTERED_REFUSES_WITH_A_HANDLE_BOUND(
        self, monkeypatch: pytest.MonkeyPatch, no_providers
    ) -> None:
        """The second route to the same substitution, and the one a "robustness" edit takes.

        The row exists and is configured; its implementation is missing. With a handle bound,
        falling back here would score through the trained model under the configured
        provider's name.
        """
        monkeypatch.setenv("PROVIDER_JUDGE_BASE_URL", "http://judge.test:8000")
        monkeypatch.setenv("PROVIDER_JUDGE_MODEL", "judge-7b")
        monkeypatch.delitem(registry._FACTORIES, "openai_compatible")
        monkeypatch.setattr(registry, "ensure_registered", lambda: None)
        registry.bind_model_handle(FakeModelHandle())
        with pytest.raises(Refusal) as caught:
            registry.get("judge")
        assert caught.value.code == "inference_provider_unavailable"
        assert caught.value.details["provider"] == "judge"
        registry.ensure_registered()

    def test_M9_the_resolved_identity_is_never_in_process_for_an_http_name(
        self, one_http_row
    ) -> None:
        """The property stated positively: resolving a configured HTTP name gives an HTTP
        provider, never the in-process one, whatever is bound."""
        registry.bind_model_handle(FakeModelHandle())
        assert registry.get(one_http_row).identity.kind == "openai_compatible"

    def test_an_http_row_resolves_to_f3s_implementation_unforked(self, one_http_row) -> None:
        """Imported and registered. **No subclass, no wrapper, no copy.**"""
        from providers.openai_compatible import OpenAICompatibleProvider

        provider = registry.get(one_http_row)
        assert type(provider) is OpenAICompatibleProvider
        assert provider.identity.base_url == "http://judge.test:8000"


class TestPinning:
    """FR-004.10: *a run's test curve compares each number with the one before it.*"""

    def test_an_unchanged_identity_is_not_a_discontinuity(self, no_providers) -> None:
        registry.bind_model_handle(FakeModelHandle())
        pin = registry.pin(registry.IN_PROCESS_NAME)
        assert registry.assert_pin_unchanged(pin, pin.identity) is False

    def test_a_changed_identity_refuses_by_name(self, no_providers) -> None:
        registry.bind_model_handle(FakeModelHandle())
        pin = registry.pin(registry.IN_PROCESS_NAME)
        moved = ProviderIdentity(kind="openai_compatible", model="other", base_url="http://x")
        with pytest.raises(Refusal) as caught:
            registry.assert_pin_unchanged(pin, moved)
        assert caught.value.code == "provider_changed_mid_run"
        # BOTH identities, because a refusal that named only one would leave the operator
        # guessing which half moved.
        assert caught.value.details["pinned"]["model"] == "fake-model"
        assert caught.value.details["current"]["model"] == "other"

    def test_an_explicit_repin_records_a_discontinuity_instead_of_refusing(
        self, no_providers
    ) -> None:
        registry.bind_model_handle(FakeModelHandle())
        pin = registry.pin(registry.IN_PROCESS_NAME)
        moved = ProviderIdentity(kind="openai_compatible", model="other", base_url="http://x")
        assert registry.assert_pin_unchanged(pin, moved, repin=True) is True

    def test_a_moving_parameter_digest_is_not_a_discontinuity_for_in_process(
        self, no_providers
    ) -> None:
        """The in-process digest changes at **every step** by design.

        If it counted, every step of every run would report a discontinuity and F9 would
        break the curve at every point - which trains every reader to ignore the marker, and
        then the one that matters is invisible too.
        """
        registry.bind_model_handle(FakeModelHandle())
        pin = registry.pin(registry.IN_PROCESS_NAME)
        later = ProviderIdentity(
            kind="in_process", model="fake-model", engine_version="fake-parameter-set@step900"
        )
        assert registry.assert_pin_unchanged(pin, later) is False

    def test_the_same_change_on_an_http_provider_IS_a_discontinuity(self, no_providers) -> None:
        """The symmetric case, asserted so the exemption above cannot widen by accident."""
        pin = registry.Pin(
            name="judge",
            identity=ProviderIdentity(
                kind="openai_compatible", model="judge-7b", base_url="http://j", engine_version="a"
            ),
        )
        moved = ProviderIdentity(
            kind="openai_compatible", model="judge-7b", base_url="http://j", engine_version="b"
        )
        with pytest.raises(Refusal):
            registry.assert_pin_unchanged(pin, moved)


class TestTheJudgeViewCannotReachTheInProcessProvider:
    """R-02.11: **a model grading itself is not a measurement.**"""

    def test_the_in_process_row_is_never_judge_eligible(self, no_providers) -> None:
        registry.bind_model_handle(FakeModelHandle())
        row = registry.row_for(registry.IN_PROCESS_NAME)
        assert row is not None
        assert row.available is True
        assert row.judge_eligible is False, (
            "there is no configuration under which the in-process provider may judge; "
            "R-02.11 is enforced by what the judge view cannot reach"
        )

    def test_available_is_false_for_an_in_process_key(self, no_providers) -> None:
        registry.bind_model_handle(FakeModelHandle())
        assert registry.judge_resolver().available(registry.IN_PROCESS_NAME) is False

    def test_resolving_it_refuses_with_the_SAME_code_the_listing_publishes(
        self, no_providers
    ) -> None:
        """The spelling is load-bearing.

        F3's ``availability.JUDGE_PROVIDER_NOT_CONFIGURED`` is the ``unavailable_reason`` its
        listing publishes **and** the code its plugin refuses with. If this registry refused
        under a different code, an operator would be told two different things about one
        state and would fix the wrong one first.
        """
        from services.selectors import availability

        registry.bind_model_handle(FakeModelHandle())
        with pytest.raises(Refusal) as caught:
            registry.judge_resolver().resolve(registry.IN_PROCESS_NAME)
        assert caught.value.code == availability.JUDGE_PROVIDER_NOT_CONFIGURED
        assert "a model grading itself is not a measurement" in caught.value.message

    def test_an_unconfigured_key_refuses_under_the_same_code(self, no_providers) -> None:
        with pytest.raises(Refusal) as caught:
            registry.judge_resolver().resolve("judge")
        assert caught.value.code == "judge_provider_not_configured"
        assert "will NOT fall back" in caught.value.message

    def test_a_configured_http_key_is_judge_eligible_and_resolves(self, one_http_row) -> None:
        resolver = registry.judge_resolver()
        assert resolver.available(one_http_row) is True
        assert resolver.resolve(one_http_row).identity.model == "judge-7b"


class TestRegistration:
    def test_a_duplicate_kind_refuses(self) -> None:
        """Two implementations of one kind is two answers to *which provider produced this*."""
        with pytest.raises(Refusal) as caught:
            registry.register("openai_compatible", lambda row: None)  # type: ignore[arg-type,return-value]
        assert caught.value.code == "inference_provider_duplicate_kind"

    def test_ensure_registered_is_idempotent(self) -> None:
        """A repeated import is normal and must not fail."""
        registry.ensure_registered()
        registry.ensure_registered()
        assert "openai_compatible" in registry.registered_kinds()


class TestNoCredentialLeaves:
    def test_the_row_payload_has_no_credential(
        self, monkeypatch: pytest.MonkeyPatch, one_http_row
    ) -> None:
        """ADR-020, E-23. Not even redacted-by-value: ``has_api_key`` says the same thing
        without implying the value is somewhere in the payload."""
        monkeypatch.setenv("PROVIDER_JUDGE_API_KEY", "sk-secret-value")
        payloads = [row.payload() for row in registry.list_rows()]
        assert all("sk-secret-value" not in repr(payload) for payload in payloads)
        assert all("api_key" not in payload for payload in payloads)

    def test_provider_identity_cannot_hold_one(self) -> None:
        """The cheapest enforcement is a type that cannot carry a secret."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProviderIdentity(kind="openai_compatible", model="m", api_key="sk-x")  # type: ignore[call-arg]
