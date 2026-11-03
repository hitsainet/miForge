"""The GPU lease  (foundation 6.7; ADR-008).

**One run per card, with a holder and an expiry.** A bare boolean is the named
anti-pattern, so the tests that matter are the three a boolean cannot pass: that a second
claimant is *refused and told who holds the card*, that a lapsed lease is reclaimable
without a human at a SQL prompt, and that a run whose lease lapsed is **refused a renewal**
rather than silently put back on a card someone else may now be using.

The device is never resolved here, so the first class below asserts the delegation itself -
the **call count and the argument**, not merely that something was called. ADR-016: a test
that only checks "was called" passes against a call sending the wrong arguments, and the
wrong argument in this case is a GPU index.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from core.config import reset_settings
from core.errors import Refusal
from core.gpu import Device
from tests.support.jobs import memory_engine, read_leases, session_for
from workers import lease
from workers.run_record import utcnow

CARD_A = "GPU-11111111-2222-3333-4444-555555555555"
CARD_B = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

DEVICE_A = Device(uuid=CARD_A, name="NVIDIA GeForce RTX 3090", index=0)
DEVICE_B = Device(uuid=CARD_B, name="NVIDIA GeForce RTX 3080 Ti", index=1)


@pytest.fixture
def engine():
    reset_settings()
    return memory_engine()


class TestItDelegatesDeviceResolution:
    """ADR-008's rule is only true if **one** place can break it, and that place is
    ``core.gpu.resolve_device``."""

    def test_the_request_reaches_resolve_device_verbatim_exactly_once(self, engine, monkeypatch):
        """**The payload and the call count.** A lease that passed ``"0"`` or a bus address
        through would be refused inside ``resolve_device`` - but only if it actually calls
        it with what the caller asked for. "It called something" would pass against a lease
        that resolved ``auto`` while the operator named a specific card."""
        seen: list[str] = []

        def spy(request: str) -> Device:
            seen.append(request)
            return DEVICE_A

        monkeypatch.setattr(lease, "resolve_device", spy)
        with session_for(engine) as session:
            held = lease.acquire(session, "run-1", CARD_A)

        assert seen == [CARD_A], "exactly one resolution, with the request as given"
        assert held.gpu_uuid == CARD_A

    def test_auto_is_passed_through_rather_than_interpreted_here(self, engine, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(lease, "resolve_device", lambda r: (seen.append(r), DEVICE_A)[1])
        with session_for(engine) as session:
            lease.acquire(session, "run-1", "auto")
        assert seen == ["auto"]

    def test_a_prefresolved_device_skips_resolution_entirely(self, engine, monkeypatch):
        """For a caller that resolved once at submit and leases after a budget check -
        R-02.42's guard runs *before any allocation*, so resolving twice would be two
        answers to which card this run is on."""
        monkeypatch.setattr(
            lease,
            "resolve_device",
            lambda r: pytest.fail("must not resolve when a device was supplied"),
        )
        with session_for(engine) as session:
            held = lease.acquire(session, "run-1", "auto", device=DEVICE_A)
        assert held.gpu_uuid == CARD_A

    def test_an_index_is_refused_by_core_gpu_not_tolerated_here(self, engine):
        """No monkeypatch: the real ``resolve_device`` runs, and refuses before any
        database row exists. The lease must not have its own, softer opinion."""
        with session_for(engine) as session:
            with pytest.raises(Refusal) as caught:
                lease.acquire(session, "run-1", "0")
            assert caught.value.code == "gpu_request_is_an_index"
            assert read_leases(session, "0") == []


class TestTheHolderAndTheExpiry:
    def test_a_lease_records_its_holder_and_its_expiry(self, engine):
        with session_for(engine) as session:
            now = utcnow()
            held = lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=3600, now=now)
            assert held.holder_run_id == "run-1"
            assert held.expires_at == now + timedelta(seconds=3600)

            rows = read_leases(session, CARD_A)
            assert len(rows) == 1
            assert rows[0]["holder_run_id"] == "run-1"
            assert rows[0]["expires_at"] is not None
            assert rows[0]["released_at"] is None

    def test_a_lease_with_no_holder_is_refused(self, engine):
        """The anti-pattern, refused at the door: a holder-less lease cannot say what to
        ask when the card is busy."""
        with session_for(engine) as session:
            with pytest.raises(Refusal) as caught:
                lease.acquire(session, "", device=DEVICE_A)
            assert caught.value.code == "gpu_lease_holder_required"

    def test_the_ttl_defaults_to_settings(self, engine):
        from core.config import get_settings

        with session_for(engine) as session:
            now = utcnow()
            held = lease.acquire(session, "run-1", device=DEVICE_A, now=now)
        expected = now + timedelta(seconds=get_settings().gpu_lease_seconds)
        assert held.expires_at == expected

    def test_seconds_remaining_counts_down(self, engine):
        with session_for(engine) as session:
            now = utcnow()
            held = lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=600, now=now)
        assert held.seconds_remaining(now=now) == pytest.approx(600, abs=1)
        assert held.seconds_remaining(now=now + timedelta(seconds=599)) == pytest.approx(1, abs=1)


class TestContention:
    """**One run per card.** The assertion is not only that the second claim fails but that
    the refusal *names the holder and the expiry* - a refusal saying "busy" leaves the
    operator unable to decide whether to wait or intervene."""

    def test_a_second_run_is_refused_and_told_who_holds_the_card(self, engine):
        with session_for(engine) as session:
            lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=3600)

            with pytest.raises(Refusal) as caught:
                lease.acquire(session, "run-2", device=DEVICE_A, lease_seconds=3600)

            refusal = caught.value
            assert refusal.code == "gpu_lease_held"
            assert refusal.details["holder_run_id"] == "run-1"
            assert refusal.details["gpu_uuid"] == CARD_A
            assert refusal.details["expires_at"] is not None

    def test_the_losing_claim_leaves_no_row_behind(self, engine):
        """The savepoint's job. A failed claim that left a row would make the card
        permanently unleasable, and the symptom would appear on the *next* submit."""
        with session_for(engine) as session:
            lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=3600)
            with pytest.raises(Refusal):
                lease.acquire(session, "run-2", device=DEVICE_A)
            assert len(read_leases(session, CARD_A)) == 1

    def test_the_caller_transaction_survives_a_losing_claim(self, engine):
        """A refused lease must not poison the submitting transaction - the caller is
        mid-submit with a run row to finish. Without the savepoint the session would be in
        a failed state and the next statement would raise something unrelated."""
        with session_for(engine) as session:
            lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=3600)
            with pytest.raises(Refusal):
                lease.acquire(session, "run-2", device=DEVICE_A)
            # The session is still usable, and a DIFFERENT card still leases.
            other = lease.acquire(session, "run-2", device=DEVICE_B)
            assert other.gpu_uuid == CARD_B

    def test_two_cards_are_leased_independently(self, engine):
        with session_for(engine) as session:
            lease.acquire(session, "run-1", device=DEVICE_A)
            lease.acquire(session, "run-2", device=DEVICE_B)
            assert lease.holder_of(session, CARD_A) == "run-1"
            assert lease.holder_of(session, CARD_B) == "run-2"

    def test_the_same_run_re_acquiring_extends_rather_than_refusing(self, engine):
        """A retried submit must not refuse against itself. Safe as a check-then-act
        *because* it is not the contention case: nothing can transfer this run's lease to
        another holder without this run releasing it."""
        with session_for(engine) as session:
            now = utcnow()
            first = lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=600, now=now)
            again = lease.acquire(
                session,
                "run-1",
                device=DEVICE_A,
                lease_seconds=600,
                now=now + timedelta(seconds=10),
            )
            assert again.expires_at > first.expires_at
            assert len(read_leases(session, CARD_A)) == 1, "extended, not a second row"


class TestExpiry:
    def test_an_expired_lease_does_not_block_a_new_holder(self, engine):
        """ADR-008's whole argument for an expiry: a process that dies holding the card must
        not hold it forever, with the only recovery being a human at a SQL prompt."""
        with session_for(engine) as session:
            start = utcnow() - timedelta(hours=2)
            lease.acquire(session, "dead-run", device=DEVICE_A, lease_seconds=60, now=start)
            taken = lease.acquire(session, "new-run", device=DEVICE_A)
            assert taken.holder_run_id == "new-run"
            assert lease.holder_of(session, CARD_A) == "new-run"

    def test_the_expired_row_is_marked_released_rather_than_left_unreleased(self, engine):
        """**The consequence of 4.2's partial unique index that is easy to miss.**
        Exclusivity is over the rows where ``released_at IS NULL``, and expiry is a fact
        about ``expires_at`` - so a lapsed lease is still unreleased and still blocks the
        index. Reclaiming it is what makes the next insert legal."""
        with session_for(engine) as session:
            start = utcnow() - timedelta(hours=2)
            lease.acquire(session, "dead-run", device=DEVICE_A, lease_seconds=60, now=start)
            lease.acquire(session, "new-run", device=DEVICE_A)

            rows = sorted(read_leases(session, CARD_A), key=lambda r: r["id"])
            assert len(rows) == 2, "append-and-release keeps the history"
            assert rows[0]["holder_run_id"] == "dead-run"
            assert rows[0]["released_at"] is not None, "the lapsed row was reclaimed"
            assert rows[1]["holder_run_id"] == "new-run"
            assert rows[1]["released_at"] is None

    def test_holder_of_reports_no_holder_once_a_lease_has_lapsed(self, engine):
        """The question is "is this card in use", and a lease nobody renewed is not in use -
        even though its row is still unreleased."""
        with session_for(engine) as session:
            now = utcnow()
            lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=60, now=now)
            assert lease.holder_of(session, CARD_A, now=now) == "run-1"
            assert lease.holder_of(session, CARD_A, now=now + timedelta(seconds=61)) is None

    def test_reclaim_is_idempotent(self, engine):
        """Two racing acquirers both run it; the second must simply match no row."""
        with session_for(engine) as session:
            start = utcnow() - timedelta(hours=2)
            lease.acquire(session, "dead", device=DEVICE_A, lease_seconds=60, now=start)
            assert lease.reclaim_expired(session, CARD_A) == 1
            assert lease.reclaim_expired(session, CARD_A) == 0

    def test_reclaim_does_not_touch_a_live_lease(self, engine):
        with session_for(engine) as session:
            lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=3600)
            assert lease.reclaim_expired(session, CARD_A) == 0
            assert lease.holder_of(session, CARD_A) == "run-1"


class TestRenew:
    def test_a_live_lease_is_extended(self, engine):
        with session_for(engine) as session:
            now = utcnow()
            held = lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=600, now=now)
            later = now + timedelta(seconds=300)
            renewed = lease.renew(session, "run-1", CARD_A, lease_seconds=600, now=later)
            assert renewed.expires_at > held.expires_at

    def test_a_lapsed_lease_is_refused_rather_than_silently_re_acquired(self, engine):
        """**The reason renew is not just another acquire.** A run whose lease lapsed may
        already be sharing the card with whoever reclaimed it. Re-taking it silently would
        put two trainers on one device, which is the outcome the lease exists to prevent."""
        with session_for(engine) as session:
            now = utcnow()
            lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=60, now=now)
            with pytest.raises(Refusal) as caught:
                lease.renew(session, "run-1", CARD_A, now=now + timedelta(seconds=61))
            assert caught.value.code == "gpu_lease_lost"

    def test_renewing_a_lease_held_by_another_run_is_refused(self, engine):
        with session_for(engine) as session:
            lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=3600)
            with pytest.raises(Refusal) as caught:
                lease.renew(session, "run-2", CARD_A)
            assert caught.value.code == "gpu_lease_lost"
            assert caught.value.details["holder_run_id"] == "run-1"
            assert lease.holder_of(session, CARD_A) == "run-1"


class TestRelease:
    def test_releasing_frees_the_card(self, engine):
        with session_for(engine) as session:
            lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=3600)
            assert lease.release(session, "run-1", CARD_A) is True
            assert lease.holder_of(session, CARD_A) is None
            taken = lease.acquire(session, "run-2", device=DEVICE_A)
            assert taken.holder_run_id == "run-2"

    def test_one_run_cannot_release_another_runs_card(self, engine):
        """The same hazard as R-02.34's prohibition on clearing another process's
        attachment: that is how two writers end up on one set of weights."""
        with session_for(engine) as session:
            lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=3600)
            assert lease.release(session, "run-2", CARD_A) is False
            assert lease.holder_of(session, CARD_A) == "run-1"

    def test_releasing_an_unheld_card_returns_false_rather_than_refusing(self, engine):
        """Release is called on the way out of a run **including from a failure path**, and
        a refusal there would mask the original error with a lease error."""
        with session_for(engine) as session:
            assert lease.release(session, "run-1", CARD_A) is False

    def test_release_is_idempotent(self, engine):
        with session_for(engine) as session:
            lease.acquire(session, "run-1", device=DEVICE_A, lease_seconds=3600)
            assert lease.release(session, "run-1", CARD_A) is True
            assert lease.release(session, "run-1", CARD_A) is False
