"""The GPU lease: one run per card, with a holder and an expiry  (task 6.7; ADR-008).

**The named anti-pattern is a bare boolean.** miLLM's ``locked`` flag has neither a holder
nor an expiry, and both omissions have the same shape of consequence:

* **no holder** - when the lock is held, nothing can say *by what*. The operator cannot
  decide whether to wait or to intervene, and the error message can only say "busy".
* **no expiry** - a process that dies holding the lock holds it forever. The only recovery
  is a human with a SQL prompt, and what they do at that prompt is indistinguishable from
  the thing that puts two trainers on one set of weights.

So ``mf_leases`` (task 4.2) has ``holder_run_id`` and ``expires_at``, both ``NOT NULL``.

**The device is never resolved here.** :func:`core.gpu.resolve_device` is the only thing in
this codebase that turns a request into a card, because ADR-008's rule - pin by UUID from
live ``nvidia-smi -L``, never an index, never a bus address - is only true if there is one
place that can break it. The node's indices inverted between two boots and the work
silently ran on the other card; a copied bus address once unbound the root NVMe.

**Exclusivity is a partial unique index, and that shapes everything below.**
4.2 models a lease as **append-and-release**: a row per claim, released by setting
``released_at``, with a unique index over ``resource`` restricted to the rows where
``released_at IS NULL``. 4.2's own note says why - *"two workers racing for the same card
both pass an application-level check; only one passes this"* - and it is the right model.
It has one consequence that is easy to miss and expensive to miss:

    **A lapsed lease is still unreleased, so it still blocks the index.**

Expiry is a fact about ``expires_at``; the index only knows about ``released_at``. So
acquiring is two statements: *reclaim* every unreleased row for that card whose expiry has
passed, then *insert*. If the insert loses on the index, the card is genuinely held by a
live holder - and that loss is the authoritative answer, arrived at by the database rather
than by a ``SELECT`` this process hoped was still true.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.config import get_settings
from core.errors import refuse
from core.gpu import Device, resolve_device
from workers.run_record import LEASES, as_aware, rowcount_of, utcnow

log = logging.getLogger(__name__)

_C = LEASES.c


@dataclass(frozen=True)
class Lease:
    """A held lease. Returned by :func:`acquire`, carrying what a run must record.

    ``gpu_uuid`` is the value ADR-008 requires on the run row. ``expires_at`` is on the
    object because a holder that does not know when its own lease dies cannot renew it in
    time, and a lease that lapses under a live run is how two runs end up on one card.
    """

    gpu_uuid: str
    holder_run_id: str
    acquired_at: datetime
    expires_at: datetime
    device_name: str | None = None

    def seconds_remaining(self, *, now: datetime | None = None) -> float:
        return (self.expires_at - (now or utcnow())).total_seconds()


def _ttl(lease_seconds: int | None) -> int:
    return get_settings().gpu_lease_seconds if lease_seconds is None else lease_seconds


def _live_row(session: Session, resource: str, now: datetime) -> sa.Row | None:  # type: ignore[type-arg]
    """The unreleased, unexpired lease on ``resource``, if there is one."""
    return session.execute(
        sa.select(_C.id, _C.holder_run_id, _C.acquired_at, _C.expires_at)
        .where(_C.resource == resource)
        .where(_C.released_at.is_(None))
        .where(_C.expires_at > now)
    ).one_or_none()


def reclaim_expired(session: Session, resource: str, *, now: datetime | None = None) -> int:
    """Release every unreleased row for ``resource`` whose expiry has passed.

    Separate and public because it is the step that makes a crashed holder recoverable
    *without a human at a SQL prompt*, which is half of ADR-008's argument for an expiry at
    all. Idempotent, so two racing acquirers can both run it: the second simply matches no
    row.
    """
    moment = now or utcnow()
    released = rowcount_of(
        session.execute(
            sa.update(LEASES)
            .where(_C.resource == resource)
            .where(_C.released_at.is_(None))
            .where(_C.expires_at <= moment)
            .values(released_at=moment)
        )
    )
    if released:
        log.warning(
            "reclaimed an expired gpu lease",
            extra={
                "code": "gpu_lease_reclaimed",
                "gpu_uuid": resource,
                "rows": released,
                "reason": "the holder's lease expired without being released",
            },
        )
    return released


def acquire(
    session: Session,
    run_id: str,
    gpu_request: str = "auto",
    *,
    lease_seconds: int | None = None,
    now: datetime | None = None,
    device: Device | None = None,
) -> Lease:
    """Resolve ``gpu_request`` to a card and take the lease on it.

    Args:
        session: the session to work in. The caller commits.
        run_id: the holder. Recorded on the row so a refusal can name who holds the card.
        gpu_request: ``"auto"`` (most free memory) or an explicit ``GPU-...`` UUID. Passed
            straight to :func:`core.gpu.resolve_device`; an index or a bus address is
            refused *there*, by name.
        lease_seconds: override the TTL.
        now: injectable clock.
        device: a pre-resolved device, for a caller that resolved once and is leasing after
            a budget check. **Still produced by** :func:`core.gpu.resolve_device` - this
            parameter exists so a device is resolved once per submit, not so one can be
            constructed by hand.

    Returns:
        The held :class:`Lease`.

    Raises:
        Refusal: ``gpu_lease_held`` when another live run holds the card - naming the holder
            and when its lease expires, which is what makes the refusal actionable. Plus
            every refusal :func:`core.gpu.resolve_device` can raise.
    """
    if not run_id:
        raise refuse(
            "gpu_lease_holder_required",
            "A lease must name its holder: when the card is busy, nothing can say what to ask.",
        )

    resolved = device if device is not None else resolve_device(gpu_request)
    moment = now or utcnow()
    expires = moment + timedelta(seconds=_ttl(lease_seconds))

    # Already ours? Extend rather than refuse against ourselves. This is a check-then-act,
    # and it is safe *because* it is not the contention case: no other process can turn a
    # lease held by this run into one held by someone else without this run releasing it.
    mine = _live_row(session, resolved.uuid, moment)
    if mine is not None and mine.holder_run_id == run_id:
        session.execute(sa.update(LEASES).where(_C.id == mine.id).values(expires_at=expires))
        return Lease(
            gpu_uuid=resolved.uuid,
            holder_run_id=run_id,
            acquired_at=as_aware(mine.acquired_at) or moment,
            expires_at=expires,
            device_name=resolved.name,
        )

    # A lapsed lease is still unreleased, so it still blocks the partial unique index.
    reclaim_expired(session, resolved.uuid, now=moment)

    # The claim. A savepoint so that losing the index race does not poison the caller's
    # transaction - the caller is mid-submit and has a run row to finish.
    savepoint = session.begin_nested()
    try:
        session.execute(
            sa.insert(LEASES).values(
                resource=resolved.uuid,
                holder_run_id=run_id,
                acquired_at=moment,
                expires_at=expires,
            )
        )
        savepoint.commit()
    except IntegrityError:
        savepoint.rollback()
        held = _live_row(session, resolved.uuid, moment)
        holder = None if held is None else str(held.holder_run_id)
        held_until = None if held is None else as_aware(held.expires_at)
        raise refuse(
            "gpu_lease_held",
            f"Device {resolved.uuid} is leased to run {holder} until "
            f"{held_until.isoformat() if held_until else 'an unknown time'}. One run per "
            f"card. Wait for that run, stop it, or name a different device.",
            gpu_uuid=resolved.uuid,
            holder_run_id=holder,
            expires_at=held_until.isoformat() if held_until else None,
        ) from None

    log.info(
        "gpu lease acquired",
        extra={
            "run_id": run_id,
            "gpu_uuid": resolved.uuid,
            "gpu_request": gpu_request,
            "expires_at": expires.isoformat(),
        },
    )
    return Lease(
        gpu_uuid=resolved.uuid,
        holder_run_id=run_id,
        acquired_at=moment,
        expires_at=expires,
        device_name=resolved.name,
    )


def renew(
    session: Session,
    run_id: str,
    gpu_uuid: str,
    *,
    lease_seconds: int | None = None,
    now: datetime | None = None,
) -> Lease:
    """Extend a lease this run still holds.

    **Refuses rather than re-acquiring when the lease has already lapsed**, and that is the
    reason this is not just another :func:`acquire`. A run whose lease lapsed may already be
    sharing the card with whoever reclaimed it; silently re-taking it would put two trainers
    on one device, which is the outcome the whole lease exists to prevent. The run must stop.
    """
    moment = now or utcnow()
    expires = moment + timedelta(seconds=_ttl(lease_seconds))
    rows = rowcount_of(
        session.execute(
            sa.update(LEASES)
            .where(_C.resource == gpu_uuid)
            .where(_C.holder_run_id == run_id)
            .where(_C.released_at.is_(None))
            .where(_C.expires_at > moment)
            .values(expires_at=expires)
        )
    )
    if rows == 0:
        held = _live_row(session, gpu_uuid, moment)
        raise refuse(
            "gpu_lease_lost",
            f"Run {run_id} no longer holds a live lease on {gpu_uuid}; it lapsed or was "
            f"taken. Refusing to re-acquire silently - two trainers on one card is the "
            f"outcome the lease exists to prevent. Stop this run.",
            gpu_uuid=gpu_uuid,
            run_id=run_id,
            holder_run_id=None if held is None else str(held.holder_run_id),
        )
    return Lease(gpu_uuid=gpu_uuid, holder_run_id=run_id, acquired_at=moment, expires_at=expires)


def release(session: Session, run_id: str, gpu_uuid: str, *, now: datetime | None = None) -> bool:
    """Release a lease. Returns whether this run held one.

    Scoped to the holder: ``WHERE holder_run_id = :run_id``. A release matching on the card
    alone would let any run free another run's card, which is the same hazard as R-02.34's
    prohibition on clearing another process's attachment - that is how two writers end up on
    one set of weights.

    Returns ``False`` rather than refusing, deliberately: release is called on the way out
    of a run **including from a failure path**, and a refusal there would mask the original
    error with a lease error.
    """
    moment = now or utcnow()
    rows = rowcount_of(
        session.execute(
            sa.update(LEASES)
            .where(_C.resource == gpu_uuid)
            .where(_C.holder_run_id == run_id)
            .where(_C.released_at.is_(None))
            .values(released_at=moment)
        )
    )
    if rows:
        log.info("gpu lease released", extra={"run_id": run_id, "gpu_uuid": gpu_uuid})
    else:
        log.info(
            "gpu lease release was a no-op: this run did not hold it",
            extra={"code": "gpu_lease_not_held", "run_id": run_id, "gpu_uuid": gpu_uuid},
        )
    return bool(rows)


def release_held_by(session: Session, run_id: str, *, now: datetime | None = None) -> list[str]:
    """Release every unreleased lease ``run_id`` holds; return the cards freed.

    For the janitor (D49, 2026-10-18): a run it has declared dead cannot be using a card,
    and before this the card stayed blocked until the lease's TTL - found live in F11 14.5,
    where a killed worker's run was reaped at 921 s and every new run on the 3090 was then
    refused ``gpu_lease_held`` for most of an hour. Scoped to the holder exactly as
    :func:`release` is: it never frees another run's card.
    """
    moment = now or utcnow()
    held = [
        str(row.resource)
        for row in session.execute(
            sa.select(_C.resource).where(_C.holder_run_id == run_id).where(_C.released_at.is_(None))
        )
    ]
    for gpu_uuid in held:
        release(session, run_id, gpu_uuid, now=moment)
    return held


def holder_of(session: Session, gpu_uuid: str, *, now: datetime | None = None) -> str | None:
    """Which run holds this card right now, or ``None`` if it is free, released or lapsed.

    A lapsed lease reports ``None`` even though its row is still unreleased: the question
    asked here is *"is this card in use"*, and a lease nobody renewed is not in use. The
    row's survival is :func:`reclaim_expired`'s business, not this answer's.
    """
    moment = now or utcnow()
    row = _live_row(session, gpu_uuid, moment)
    return None if row is None else str(row.holder_run_id)
