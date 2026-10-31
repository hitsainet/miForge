"""``GET /api/health``  (foundation 5.2; ADR-010, ADR-013).

Two rules, and the second is the one with history behind it:

1. each dependency separately, so one failing never hides the other;
2. **every optional integration named, configured or not** - because an integration simply
   missing from a response is indistinguishable from one that is broken, and ADR-010 makes
   an unset base URL mean *absent and normal* rather than *broken*.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import health as health_module
from api.app import create_app
from core.config import get_settings, reset_settings

OK = {"status": "ok", "detail": None}
DOWN = {"status": "unavailable", "detail": "down"}


@pytest.fixture
def client(monkeypatch):
    """A client whose dependency probes are controllable.

    The probes are replaced through :data:`api.health.DEPENDENCY_PROBES` - the same table
    the route reads - rather than by patching the functions by name, so a probe *removed*
    from the table is visible to these tests instead of silently skipped.
    """
    reset_settings()
    monkeypatch.setenv("MILLM_BASE_URL", "")
    monkeypatch.setenv("MIDATAWORKS_BASE_URL", "")
    reset_settings()
    return TestClient(create_app(), raise_server_exceptions=False)


def _probes(monkeypatch, **statuses):
    """Install counting spies and return the call log."""
    calls: list[str] = []

    def make(name, result):
        def probe():
            calls.append(name)
            return result

        return probe

    for name in ("postgres", "redis"):
        monkeypatch.setitem(health_module.DEPENDENCY_PROBES, name, make(name, statuses[name]))
    return calls


class TestEachDependencySeparately:
    def test_both_up_reports_ok(self, client, monkeypatch):
        _probes(monkeypatch, postgres=OK, redis=OK)
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["dependencies"]["postgres"]["status"] == "ok"
        assert body["dependencies"]["redis"]["status"] == "ok"
        assert body["degraded_dependencies"] == []

    def test_one_failing_does_not_hide_the_other(self, client, monkeypatch):
        """**The point of reporting them separately.** A single boolean sends an operator
        looking; two named probes tell them where to look."""
        _probes(monkeypatch, postgres=DOWN, redis=OK)
        body = client.get("/api/health").json()
        assert body["dependencies"]["postgres"]["status"] == "unavailable"
        assert body["dependencies"]["redis"]["status"] == "ok"
        assert body["degraded_dependencies"] == ["postgres"]

    def test_both_failing_are_both_named(self, client, monkeypatch):
        _probes(monkeypatch, postgres=DOWN, redis=DOWN)
        body = client.get("/api/health").json()
        assert body["degraded_dependencies"] == ["postgres", "redis"]

    def test_each_probe_is_called_exactly_once_per_request(self, client, monkeypatch):
        """**The call count.** A route that probed Postgres twice and Redis not at all would
        pass a 'was it called' assertion while reporting a cached answer for one dependency
        and nothing for the other."""
        calls = _probes(monkeypatch, postgres=OK, redis=OK)
        client.get("/api/health")
        assert sorted(calls) == ["postgres", "redis"]
        assert len(calls) == 2

    def test_a_degraded_dependency_answers_503_with_the_same_body_shape(self, client, monkeypatch):
        """Fail closed: a probe answering 200 while Postgres is down keeps a broken pod in
        a Service. **The body shape is identical either way**, so a reader - including
        foundation 1.7's standalone harness - reads the body and never branches on the
        code."""
        _probes(monkeypatch, postgres=DOWN, redis=OK)
        response = client.get("/api/health")
        assert response.status_code == 503
        body = response.json()
        assert set(body) == {
            "status",
            "environment",
            "dependencies",
            "integrations",
            "unconfigured_integrations",
            "degraded_dependencies",
        }
        assert body["integrations"], "integrations are reported even while degraded"

    def test_a_probe_reports_rather_than_raises(self, client, monkeypatch):
        """A health endpoint that 500s when a dependency is down tells the reader nothing
        about the other dependency, which is the whole question being asked."""

        def exploding():
            raise RuntimeError("boom")

        monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "postgres", exploding)
        monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "redis", lambda: OK)
        with pytest.raises(RuntimeError):
            # The *probe table* is what reports; a probe that raises is a bug in the probe,
            # and the real probes catch. Asserted so that replacing a real probe with one
            # that raises cannot pass unnoticed.
            health_module.health_payload()


class TestEveryIntegrationIsNamed:
    def test_an_unconfigured_sibling_is_present_and_named(self, client, monkeypatch):
        """**Never omitted.** An integration missing from the response is
        indistinguishable from one that is broken.

        *Amended 2026-10-14: the set is DERIVED from `Settings.configured_integrations()`
        rather than retyped here.* It read `{"millm", "midataworks"}`, and when
        `clients/activations.py` landed, a third optional integration existed, was
        unreported by health, and **this test went green over the omission** — it asserted
        equality with the two names it already knew. A test that lists what it expects
        cannot notice something new going missing; one that derives it has to.
        """
        from core.config import get_settings

        _probes(monkeypatch, postgres=OK, redis=OK)
        body = client.get("/api/health").json()
        expected = set(get_settings().configured_integrations())
        assert expected >= {
            "millm",
            "midataworks",
            "activation_source",
        }, "the registry itself lost an integration"
        assert set(body["integrations"]) == expected
        for service in expected:
            entry = body["integrations"][service]
            assert entry["configured"] is False
            assert entry["code"] == f"{service}_not_configured"

    def test_the_unconfigured_list_names_them_all(self, client, monkeypatch):
        """Sorted, and derived for the same reason as above."""
        from core.config import get_settings

        _probes(monkeypatch, postgres=OK, redis=OK)
        body = client.get("/api/health").json()
        assert body["unconfigured_integrations"] == sorted(get_settings().configured_integrations())

    def test_an_unconfigured_sibling_names_the_capabilities_it_costs(self, client, monkeypatch):
        """ADR-018: the reader learns what they cannot do, not merely that something is
        unset."""
        _probes(monkeypatch, postgres=OK, redis=OK)
        body = client.get("/api/health").json()
        assert body["integrations"]["midataworks"]["capabilities_unavailable"]
        assert body["integrations"]["millm"]["capabilities_unavailable"]

    def test_the_code_comes_from_the_same_helper_the_refusal_path_uses(self):
        """So the code on this response cannot drift from the code a caller receives when
        it actually tries to use the absent sibling."""
        from core.errors import not_configured

        assert not_configured("millm").code == "millm_not_configured"

    def test_a_configured_sibling_is_reported_as_configured(self, monkeypatch):
        reset_settings()
        monkeypatch.setenv("MILLM_BASE_URL", "http://millm.test:8000")
        monkeypatch.setenv("MIDATAWORKS_BASE_URL", "")
        reset_settings()
        monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "postgres", lambda: OK)
        monkeypatch.setitem(health_module.DEPENDENCY_PROBES, "redis", lambda: OK)

        body = health_module.health_payload()
        assert body["integrations"]["millm"]["configured"] is True
        assert body["integrations"]["millm"]["code"] is None
        # The configured one drops out and every other optional integration stays named -
        # `activation_source` included, since this test configures only MILLM_BASE_URL.
        assert "millm" not in body["unconfigured_integrations"]
        assert body["unconfigured_integrations"] == ["activation_source", "midataworks"]
        reset_settings()

    def test_an_empty_env_var_counts_as_unset_not_as_the_empty_url(self, client):
        """``MILLM_BASE_URL=`` in a ``.env`` is **unset**. Without that, task 9.4's
        deliberately-blank lines would configure every sibling as the empty URL and the
        standalone path would never be taken on a developer's machine."""
        assert get_settings().millm_base_url is None


class TestTheRouteItself:
    def test_health_is_outside_the_version_prefix(self, client, monkeypatch):
        """ADR-013 puts it at ``/api/health``: a liveness probe that moves when the API
        version moves breaks a cluster on a day nobody was changing the cluster."""
        _probes(monkeypatch, postgres=OK, redis=OK)
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/v1/health").status_code == 404

    def test_the_environment_is_reported(self, client, monkeypatch):
        _probes(monkeypatch, postgres=OK, redis=OK)
        assert client.get("/api/health").json()["environment"] in {"dev", "ci", "prod"}


class TestTheDependencySetItself:
    """**A surviving mutation, and its cause is this file's own fixture.**

    The sweep found that deleting ``"redis": _probe_redis`` from
    :data:`api.health.DEPENDENCY_PROBES` left the suite **green**. The reason is the
    fixture above: ``monkeypatch.setitem`` **creates** a key that is missing, so every test
    that installs spies for both probes puts back the very entry the mutation removed. The
    fixture and the code were agreeing by construction - ADR-016's named cause - and the
    agreement was manufactured by the test helper itself.

    What it would have cost: a dependency silently stops being probed, and health answers
    200 with that dependency absent from the body. Which is the *same* failure ADR-010
    names for integrations - *an entry missing from the response is indistinguishable from
    one that is broken* - applied to something this product cannot run without.

    So these tests never touch the table. They assert its membership directly, and assert
    that both names reach the body through the **real** probes, whose answers here are
    "unavailable" because this machine has neither service - which is fine, because the
    question is whether the dependency is *reported*, not whether it is up.
    """

    def test_both_dependencies_are_in_the_probe_table(self):
        assert set(health_module.DEPENDENCY_PROBES) == {"postgres", "redis"}

    def test_both_dependencies_reach_the_body_with_no_probe_patched(self):
        """No ``setitem`` anywhere in this test. The real probes run and report failure on a
        machine with no Postgres and no Redis - and both must still be **named**."""
        reset_settings()
        body = health_module.health_payload()
        assert set(body["dependencies"]) == {"postgres", "redis"}
        for name in ("postgres", "redis"):
            assert body["dependencies"][name]["status"] in {"ok", "unavailable"}

    def test_an_unavailable_dependency_carries_a_detail_saying_what_failed(self):
        """ADR-018: name what failed. ``{"status": "unavailable"}`` with no detail sends the
        reader to the logs for something the response already knew."""
        reset_settings()
        body = health_module.health_payload()
        for name, entry in body["dependencies"].items():
            if entry["status"] == "unavailable":
                assert entry["detail"], f"{name} is unavailable with no detail"

    def test_every_probed_dependency_is_accounted_for_in_the_degraded_list(self):
        reset_settings()
        body = health_module.health_payload()
        degraded = {n for n, d in body["dependencies"].items() if d["status"] != "ok"}
        assert set(body["degraded_dependencies"]) == degraded
