"""The read-only attached base  (FR-011.60; BRD criterion 14's ``(§5.10 only)`` clause).

**R-02.43 is F8's requirement and is not implemented here.** F11 supplies a base the
adapter run *reads*; F8 supplies the adapter, its configuration and its artifacts. *A
requirement lives in exactly one feature; an acceptance clause may be owned by the feature
that implements the mode it describes.*

## No bracket. Ever.

R-02.43's own words: *"the resident tensors are not written at all, so an adapter run needs
no commit bracket and cannot tear the parameter set"*, and *"an adapter run never interrupts
serving."* So a ``quiesce``, a ``commit-begin`` or a ``resume`` on this path is a **defect,
not a configuration** - and the symptom is **latency rather than an error**: serving is held
for a write that never happens. That is why this module **does not import ``barrier``** and
a structural test asserts the absence, and why AC-011.18 asserts the client's ``quiesce``,
``commit-begin`` and ``resume`` call counts are **zero** rather than asserting only that the
base is unchanged.

## The hash alone does not catch the thing it is for

BRD criterion 14 is *"the resident base is byte-unchanged"*, and the honest check is a hash
before and after **through the real rebuild path** - never over a copy, because *a fixture
that hashes its own copy agrees with itself by construction.* But a before/after hash
**passes whenever no gradient step happens to land on the base**, which is most of the time.
So the assertion that bites is ``requires_grad`` being set on the adapter's parameters and
**not** on any rebuilt base tensor, plus the zero call count. **MUT-19 targets both halves
and the hash catches neither reliably.**

## Cross-device is impossible, not inadvisable

A CUDA IPC handle is valid **only on the device that owns the tensor**. An adapter run
computing on another card cannot attach at all, so that is a refusal naming **both** devices
and naming F8's default copy path as the alternative - never a silent fallback to it
(ADR-018). On the card holding the served weights, attaching is strictly better than
copying: it avoids a second 2.4 GB allocation on one card.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.errors import refuse
from training.hotupdate import rebuild

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadOnlyBase:
    """A base read over the attachment's handles. **Nothing here writes back.**"""

    module: Any
    tensors: dict[str, Any]
    device_uuid: str | None
    #: The digest taken **before** anything has had a chance to touch the base, through the
    #: real rebuild path. Compared after the run by :func:`assert_unchanged`.
    digest_before: str

    def digest_now(self) -> str:
        return base_digest(self.module)


def base_digest(module: Any) -> str:
    """SHA-256 over the base parameters, **through F8's one digest function**.

    Reused rather than re-implemented: ``training/runtypes/adapter.base_parameter_digest``
    is the codebase's single definition of this hash, it excludes the adapter's own tensors
    by name, and it **refuses** on a parameter whose bytes cannot be read rather than
    skipping it - *a digest over nothing compares equal to another digest over nothing, and
    criterion 14 would pass against a run that rewrote the whole base.* A second
    implementation here would be a second answer to one question.
    """
    from training.runtypes.adapter import base_parameter_digest

    return base_parameter_digest(module)


def attach_base(
    acquirer: Any, *, compute_device_uuid: str | None = None, response: Any = None
) -> ReadOnlyBase:
    """The resident base, read-only, for an adapter run in the ``attach`` mode.

    Args:
        acquirer: the ``attach`` acquirer, already attached. Its client is the one call
            site; this module opens none of its own.
        compute_device_uuid: the card this run will compute on, or ``None`` to accept the
            device the attachment names. A mismatch is refused.
        response: the attach response, when the caller already holds it.

    Raises:
        Refusal: ``attachment_cross_device`` naming **both** devices;
            ``weights_rebuild_failed`` from the rebuild, naming what could not be rebuilt.
    """
    attachment = getattr(acquirer, "attachment", None)
    raw = response if response is not None else (attachment.raw if attachment else None)
    if raw is None:
        raise refuse(
            "attachment_missing",
            "A read-only base needs a live attachment, and this acquirer holds none. The "
            "adapter run's alternative is F8's default path, which holds its own copy of "
            "the base - refusing rather than silently taking it (ADR-018).",
        )
    owning_device = rebuild._optional_str(raw.get("device_uuid"))
    if (
        compute_device_uuid is not None
        and owning_device is not None
        and compute_device_uuid != owning_device
    ):
        raise refuse(
            "attachment_cross_device",
            f"This adapter run computes on {compute_device_uuid} and the served tensors "
            f"live on {owning_device}. A CUDA IPC handle is valid only on the device that "
            f"owns the tensor, so attaching is impossible rather than merely inadvisable. "
            f"Use F8's default adapter path, which loads its own copy of the base, or pin "
            f"this run to the card holding the served weights.",
            compute_device=compute_device_uuid,
            serving_device=owning_device,
            alternative="the default (non-attached) adapter run type",
        )

    # read_only=True AND requires_grad=False, together: the base cannot receive a gradient
    # even if one were computed, which is the one thing R-02.43 guarantees it cannot.
    base = rebuild.build(raw, read_only=True, requires_grad=False)
    from training.hotupdate.attach import build_module

    module = build_module(raw, base)
    digest = base_digest(module)
    log.info(
        "attached a READ-ONLY base for an adapter run",
        extra={
            "code": "adapter_read_base_attached",
            "parameters": len(base.tensors),
            "device_uuid": base.device_uuid,
            "base_digest": digest,
            "bracket": "none - R-02.43 writes no resident tensor",
        },
    )
    return ReadOnlyBase(
        module=module, tensors=base.tensors, device_uuid=base.device_uuid, digest_before=digest
    )


def assert_only_the_adapter_requires_grad(module: Any) -> list[Any]:
    """The adapter's trainable parameters, **through F8's own assertion**.

    One definition of "only the adapter requires grad", in the feature that owns the
    adapter. F11 calls it rather than restating it, so there is one place for the rule and
    one place to mutate (MUT-19's ``requires_grad`` half).
    """
    from training.runtypes.adapters.build import assert_only_adapter_requires_grad

    return assert_only_adapter_requires_grad(module)


def assert_unchanged(base: ReadOnlyBase) -> str:
    """Criterion 14 as a **check** rather than an intention.

    Raises:
        Refusal: ``resident_base_changed`` when the digest moved. The run has already
            happened at this point, so this is the honest report of a guarantee that was
            broken rather than a prevention - the prevention is ``requires_grad`` and the
            absent bracket, and this is what makes the guarantee *reachable by a test*.
    """
    after = base.digest_now()
    if after != base.digest_before:
        raise refuse(
            "resident_base_changed",
            "An adapter run in the attach mode changed the serving process's resident base "
            "parameters. R-02.43 guarantees it cannot: the base is read-only, no bracket is "
            "opened, and requires_grad is set on the adapter's parameters only. The digest "
            "before and after this run disagree, so one of those three did not hold.",
            digest_before=base.digest_before,
            digest_after=after,
        )
    return after
