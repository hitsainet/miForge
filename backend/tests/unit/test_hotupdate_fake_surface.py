"""The fake must be able to misbehave - task 3.8's test of the fake itself.

**Why this file exists at all.** *A fake that can only succeed makes every refusal path in
F11 agree with its test by construction*, which is exactly how a suite stays green over a
real bug (ADR-016). So each of the five misbehaviours the TDD requires is produced here and
asserted to be producible - and so is the **happy** path, because a fake that could only
refuse would make the positive assertions vacuous in the other direction.

**And a second job:** every route cites a BRD-01 requirement, and this file asserts that
each one does. *The fake is F11's reading of a contract whose implementation does not exist;
the citation is the only thing a reviewer can check it against*, so a route with no citation
is an invention and must be red.
"""

from __future__ import annotations

import re

import httpx
import pytest

from tests.support.fake_weights_surface import FakeWeightsSurface

CITATION = re.compile(r"R-01\.\d+")


def client(fake: FakeWeightsSurface) -> httpx.Client:
    return httpx.Client(base_url="http://millm.test", transport=fake.transport())


def headers(fake: FakeWeightsSurface, *, attachment: str | None = None) -> dict[str, str]:
    out = {"authorization": f"Bearer {fake.valid_token}"}
    if attachment is not None:
        out["x-millm-attachment"] = attachment
    return out


class TestEveryRouteCitesItsRequirement:
    def test_every_op_handler_names_a_BRD_01_requirement(self) -> None:
        handlers = {
            name: getattr(FakeWeightsSurface, name)
            for name in dir(FakeWeightsSurface)
            if name.startswith("_op_")
        }
        assert len(handlers) == 15, f"BRD-01 §4 defines fifteen endpoints; found {len(handlers)}"
        uncited = [
            name for name, handler in handlers.items() if not CITATION.search(handler.__doc__ or "")
        ]
        assert uncited == [], f"routes with no citation are inventions: {uncited}"


class TestMisbehaviourOneRefusals:
    def test_already_attached_with_a_holder(self) -> None:
        fake = FakeWeightsSurface()
        fake.make_stale("trainer:someone-else")
        response = client(fake).post("/api/weights/attach", headers=headers(fake))
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ALREADY_ATTACHED"
        assert response.json()["error"]["details"]["holder"] == "trainer:someone-else"

    def test_not_attachable_on_a_quantized_model(self) -> None:
        fake = FakeWeightsSurface(dtype="int8")
        response = client(fake).post("/api/weights/attach", headers=headers(fake))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "NOT_ATTACHABLE"

    @pytest.mark.parametrize("op", ["quiesce", "resume"])
    def test_not_clean_refuses_the_barrier_calls(self, op: str) -> None:
        fake = FakeWeightsSurface(weights_state="torn", attachment_id="a")
        response = client(fake).post(f"/api/weights/{op}", headers=headers(fake, attachment="a"))
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "WEIGHTS_NOT_CLEAN"

    def test_unauthorized_without_the_credential(self) -> None:
        fake = FakeWeightsSurface()
        response = client(fake).post("/api/weights/attach")
        assert response.status_code == 401

    def test_not_attached_and_not_the_holder_are_different_refusals(self) -> None:
        """R-01.31's two: ``409 NOT_ATTACHED`` with no attachment, ``403
        NOT_ATTACHMENT_HOLDER`` with the wrong id. Collapsing them would send an operator to
        the wrong place - one means "attach first", the other "you lost the id"."""
        fake = FakeWeightsSurface()
        no_attachment = client(fake).post("/api/weights/quiesce", headers=headers(fake))
        assert no_attachment.status_code == 409
        assert no_attachment.json()["error"]["code"] == "NOT_ATTACHED"
        fake.attachment_id = "right"
        wrong = client(fake).post("/api/weights/quiesce", headers=headers(fake, attachment="wrong"))
        assert wrong.status_code == 403
        assert wrong.json()["error"]["code"] == "NOT_ATTACHMENT_HOLDER"

    def test_clear_stale_is_refused_to_a_trainer_role(self) -> None:
        """**R-01.29 is operator-only**, which is what makes R-02.34's prohibition protocol
        rather than policy. A fake that let a trainer clear it would make F11's two-layer
        argument untestable."""
        fake = FakeWeightsSurface(role="trainer")
        fake.make_stale()
        response = client(fake).post("/api/weights/clear-stale", headers=headers(fake))
        assert response.status_code == 403


class TestMisbehaviourTwoStalls:
    def test_a_stall_raises_a_timeout_rather_than_sleeping(self) -> None:
        """*A test that really waits out a watchdog is a test nobody runs.*"""
        fake = FakeWeightsSurface()
        fake.stall_next("status")
        with pytest.raises(httpx.ReadTimeout):
            client(fake).get("/api/weights/status")
        # Consumed: the next call succeeds, so a stall is one event rather than a mode.
        assert client(fake).get("/api/weights/status").status_code == 200


class TestMisbehaviourThreeTheLateCommitEnd:
    def test_the_watchdog_expires_and_the_late_commit_end_is_still_accepted(self) -> None:
        """R-01.22 then R-01.23a: ``torn``, serving refused, and then a late ``commit-end``
        from the same attachment returns it to ``clean`` with the overrun recorded."""
        fake = FakeWeightsSurface()
        http = client(fake)
        http.post("/api/weights/attach", headers=headers(fake))
        hdr = headers(fake, attachment=str(fake.attachment_id))
        http.post("/api/weights/quiesce", headers=hdr)
        http.post("/api/weights/commit-begin", headers=hdr)
        fake.trip_commit_watchdog()
        assert fake.weights_state == "torn"
        assert fake.serve_completion() == "", "miLLM never serves a torn set"
        late = http.post("/api/weights/commit-end", headers=hdr)
        assert late.status_code == 200
        assert late.json()["overran"] is True
        assert fake.weights_state == "clean"


class TestMisbehaviourFourTheRestart:
    def test_a_restart_resets_the_barrier_state_but_not_the_resident_identity(self) -> None:
        """FPRD AMB-J: the barrier state is **process memory** and a restart begins
        ``clean``, while the resident identity goes back to the last *load*. That
        combination is the only way to test FR-011.37."""
        fake = FakeWeightsSurface()
        fake.weights_state = "torn"
        fake.steps_since_load = 9
        fake.attachment_id = "a"
        fake.restart()
        assert fake.weights_state == "clean"
        assert fake.steps_since_load == 0
        assert fake.attachment_id is None, "a restart loses the attachment, so the id is gone"
        assert fake.resident_model == "LFM2-1.2B", "the resident identity survives"


class TestMisbehaviourFiveStale:
    def test_a_stale_attachment_blocks_an_attach_until_an_operator_clears_it(self) -> None:
        fake = FakeWeightsSurface(role="operator")
        fake.make_stale()
        http = client(fake)
        assert http.post("/api/weights/attach", headers=headers(fake)).status_code == 409
        assert http.post("/api/weights/clear-stale", headers=headers(fake)).status_code == 200
        assert http.post("/api/weights/attach", headers=headers(fake)).status_code == 200

    def test_clearing_a_non_stale_attachment_is_refused(self) -> None:
        """R-01.29's own refusal, so the fake cannot be used to prove a clear always works."""
        fake = FakeWeightsSurface(role="operator")
        response = client(fake).post("/api/weights/clear-stale", headers=headers(fake))
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ATTACHMENT_NOT_STALE"


class TestTheHappyPathAlsoWorks:
    def test_a_full_bracket_succeeds_and_the_counter_moves(self) -> None:
        """The other direction of the vacuity check: a fake that could only refuse would
        make every positive assertion in this feature meaningless."""
        fake = FakeWeightsSurface()
        http = client(fake)
        attach = http.post("/api/weights/attach", headers=headers(fake))
        assert attach.status_code == 200
        hdr = headers(fake, attachment=str(fake.attachment_id))
        for path in ("quiesce", "commit-begin", "commit-end", "resume"):
            assert http.post(f"/api/weights/{path}", headers=hdr).status_code == 200
        assert fake.steps_since_load == 1
        assert fake.serve_completion().startswith("served by")
        assert fake.served_while_not_clean == []

    def test_status_and_lineage_are_not_gated(self) -> None:
        """R-01.17's list contains neither, and R-01.20 says status *"is how a restarting
        trainer discovers a torn set it did not create"* - which it could not do if a
        credential were required to ask."""
        fake = FakeWeightsSurface()
        http = client(fake)
        assert http.get("/api/weights/status").status_code == 200
        assert http.get("/api/weights/lineage").status_code == 200

    def test_the_attach_response_ties_two_parameters_to_one_storage(self) -> None:
        """So a rebuild that produced two allocations is observable. Tied embeddings are the
        common case, and two copies of one tensor means the trained model differs from the
        served one in exactly that tensor while everything else passes."""
        fake = FakeWeightsSurface()
        body = client(fake).post("/api/weights/attach", headers=headers(fake)).json()
        keys = [entry["storage_key"] for entry in body["parameters"]]
        assert len(keys) == 2 and len(set(keys)) == 1
