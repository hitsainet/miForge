"""Identity, the two headers, and stop-don't-retry  (R-02.32; FR-011.20 - FR-011.26).

**Two tables, in one place each.** :data:`GATED` decides the bearer credential and
:data:`HOLDER_ONLY` decides the attachment header. Both are cited from BRD-01 and neither
is reasoned about here - that document is normative for the wire contract, and three of
this project's recorded cross-document defects came from restating another repository's
field list instead of citing it.

## Why ``heartbeat`` being in ``GATED`` has its own mutation control

A heartbeat refused for a missing token is **indistinguishable from a dead trainer.** miLLM
marks the attachment stale (BRD-01 R-01.4) and the run dies one interval later looking like
a crash - the right failure arriving as the wrong diagnosis. BRD-01 R-01.17 was amended on
2026-10-13 to gate it for exactly that reason, and R-02.32's own list omitted it until the
same review. **MUT-5 reverts that fix and must fail loudly**; if it does not, the round
that made it produced an unpinned fix.

## Three secrets, one rule

The bearer token, the audit label and the attachment id never reach a log line, a step
record, a run log, a manifest or an API response (ADR-020, TID §12). :func:`digest_ref` is
the **only** way any of them is referenced in a message, which is why it is one function:
one function is one mutation target (MUT-13).
"""

from __future__ import annotations

from core.canonical import sha256_of
from core.errors import refuse

#: BRD-01 **R-01.17**'s gated set, cited rather than reasoned about. Every one of these
#: carries ``Authorization: Bearer <token>``.
#:
#: ``clear-stale`` is in the set because the set is R-01.17's and faithfulness to the cited
#: source is the point - **and miForge never calls it** (R-02.34, FR-011.34). The client
#: asserts that separately, so the absence is checkable in two places inside miForge and
#: enforced by the role gate outside it.
GATED: frozenset[str] = frozenset(
    {
        "attach",
        "detach",
        "heartbeat",
        "quiesce",
        "commit-begin",
        "commit-end",
        "resume",
        "save",
        "load",
        "instantiate",
        "adapter/load",
        "adapter/unload",
        "clear-stale",
    }
)

#: BRD-01 **R-01.31**'s holder-only set: exactly seven. These additionally carry
#: ``X-miLLM-Attachment: <attachment id>``.
#:
#: **Settled upstream 2026-10-13, and it used to be three answers.** BRD-01 R-01.21
#: restricted *two* calls, miLLM's Feature 031 FR-31.14.6 named *seven*, and BRD-02 R-02.32
#: said "the holder-only calls" and enumerated *none*. R-01.31 fixes the set at these seven,
#: declares FR-31.14.6's seven normative, and makes R-01.21's two an application of the rule
#: rather than a narrower set. So this membership is the contract's rather than a safe
#: over-approximation. **Verified in the committed upstream file, not in a note about it.**
HOLDER_ONLY: frozenset[str] = frozenset(
    {
        "detach",
        "heartbeat",
        "quiesce",
        "resume",
        "commit-begin",
        "commit-end",
        "save",
    }
)

#: R-01.31's complement, recorded because a later reader needs the other half of the rule:
#: ``load`` and ``instantiate`` **read** the header when it is present in order to recognise
#: the holder, but do not require it; ``adapter/load``, ``adapter/unload``, ``clear-stale``
#: and every ``/v1/*`` call ignore it entirely.
READS_HEADER_WHEN_PRESENT: frozenset[str] = frozenset({"load", "instantiate"})
IGNORES_HEADER: frozenset[str] = frozenset(
    {"adapter/load", "adapter/unload", "clear-stale", "lineage", "status"}
)

#: The call miForge must never make (R-02.34, BRD-01 R-01.29 - **operator-only**).
FORBIDDEN_OP = "clear-stale"

#: BRD-01 **R-01.31**'s header, spelled as that requirement spells it.
ATTACHMENT_HEADER = "X-miLLM-Attachment"

#: **There is no audit-label header, and that is deliberate.** R-02.32 as amended says the
#: identity is *"a bearer token from a deployment secret, carrying a role of `trainer` or
#: `operator` **and a label for miLLM's audit log**"*, and BRD-01's wire detail names
#: exactly two headers: ``Authorization`` and ``X-miLLM-Attachment``. So the label travels
#: **inside the token**; ``MILLM_AUDIT_LABEL`` is miForge's declaration of which label its
#: token carries, which is what lets a half-configured identity be refused by name
#: (FR-011.25) and what a post-mortem references by digest. Sending it as a header of our
#: own invention would be a field no requirement defines - the restatement that produced
#: three of this project's recorded cross-document defects. Reported rather than invented.
AUDIT_LABEL_IS_CARRIED_BY_THE_TOKEN = True


def digest_ref(secret: str | None) -> str | None:
    """A short digest reference to a secret. **The only way one reaches a log line.**

    miLLM does the same for the attachment id - it returns the id once and logs only a
    reference to it - so this is the sibling's own discipline rather than an invention.
    ``None`` in, ``None`` out: a reference to an absent secret would read as a present one.
    """
    if secret is None:
        return None
    return sha256_of(secret)[:12]


def headers_for(
    op: str, *, token: str | None, label: str | None, attachment_id: str | None
) -> dict[str, str]:
    """The headers for one op. Bearer on every gated op; the attachment header on the seven.

    Raises:
        Refusal: ``weights_identity_incomplete`` for a gated op with no token or no audit
            label; ``attachment_id_missing`` for a holder-only op with no id. **Both refuse
            locally, before any HTTP call is made** - sending the call without them wastes a
            round trip and produces a refusal whose cause is less clear than the local one.
    """
    headers: dict[str, str] = {"content-type": "application/json"}
    if op in GATED:
        if not token:
            raise refuse(
                "weights_identity_incomplete",
                f"The {op!r} call to miLLM's weights surface is identity-gated "
                f"and MILLM_TRAINER_TOKEN is not set. Set it from the deployment "
                f"secret carrying the 'trainer' role; this is refused here rather than sent "
                f"unauthenticated, because a rejected heartbeat is indistinguishable from a "
                f"dead trainer and the run would die one interval later as a crash.",
                op=op,
                missing="millm_trainer_token",
            )
        if not label:
            raise refuse(
                "weights_identity_incomplete",
                f"The {op!r} call needs an audit label (MILLM_AUDIT_LABEL). It is not "
                f"decoration: it is what makes a weights change attributable in miLLM's "
                f"audit log, so a token without one is a change nobody can attribute.",
                op=op,
                missing="millm_audit_label",
            )
        headers["authorization"] = f"Bearer {token}"
    if op in READS_HEADER_WHEN_PRESENT and attachment_id:
        # **Sent although it is not required, and the distinction matters.** R-01.31: these
        # two "read the header when present to recognise the holder, but do not require
        # it." A recovery `load` is refused while *another* process holds an open commit
        # (R-01.25) - so the holder recovering its OWN interrupted commit must be
        # recognisable, and the header is the only thing that says so. Omitting it would
        # make R-02.30(b)'s primary recovery path refuse itself.
        headers[ATTACHMENT_HEADER] = attachment_id
    if op in HOLDER_ONLY:
        if not attachment_id:
            raise refuse(
                "attachment_id_missing",
                f"The {op!r} call is holder-only and this run holds no "
                f"attachment id. miLLM returns the id once and no read path returns it "
                f"again, so it is kept with the run's state; losing it loses the ability "
                f"to commit. Refusing locally rather than sending a call that would be "
                f"refused 403 NOT_ATTACHMENT_HOLDER with a less specific cause.",
                op=op,
            )
        headers[ATTACHMENT_HEADER] = attachment_id
    return headers


def identity_rejected(
    op: str, status: int, *, label: str | None, upstream: dict[str, object] | None = None
) -> Exception:
    """A ``401``/``403`` from a gated call. **The run stops. No retry.** (FR-011.24.)

    The message names the identity as the cause and says the credential must be fixed in
    deployment configuration. Deliberately **not** "authentication failed", which reads as
    transient and invites a retry: a rejected identity will not become valid by repetition
    (ADR-020), and a retried heartbeat would burn the staleness window as well.

    The audit label is **not** in the message - it is one of the three never-logged values -
    but its digest reference is, so two runs with different labels are distinguishable in a
    post-mortem without the label itself appearing anywhere.
    """
    return refuse(
        "weights_identity_rejected",
        f"miLLM refused the {op!r} call with HTTP {status}: the 'trainer' credential was "
        f"rejected. The run is stopped and NOT retried - a rejected identity does not "
        f"become valid by repetition. Fix MILLM_TRAINER_TOKEN (and MILLM_AUDIT_LABEL) in "
        f"the deployment configuration and start the run again.",
        op=op,
        status=status,
        audit_label_ref=digest_ref(label),
        retried=False,
        upstream=upstream or {},
    )
