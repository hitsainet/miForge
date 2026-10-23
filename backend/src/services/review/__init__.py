"""In-run human review: the queue is miDataworks', the pointer is miForge's  (F10; R-02.36).

**miForge keeps no review queue and no review screen.** No table of items awaiting a
decision, no decision store, no adjudication endpoint, no review component, no review route
in the live router. The queue, the decision history and the screen are miDataworks' (R-02.36,
BRD-03 R-03.41), and building a second one here would put two answers in the suite to one
question.

What miForge stores is exactly three things, as **durable run state**: the queue id, the
candidates' external ids, and the decision ids and decisions it was told. Losing any of them
loses the ability to read decisions back. **Holding a queue id is not holding a queue** - and
R-02.36's own closing sentence *requires* the ``external_id`` -> decision-id map, because the
decisions' identifiers are recorded with the batch they admitted.

## Not the authoring dry-run

This package serves **in-run review** (R-02.36): a person adjudicating generated candidates
while a run is stepping. It is **not** the pack-authoring selector dry-run of R-02.39, which
is F7's, uses no queue and reaches no sibling. Conflating the two is how the pack-ownership
question went wrong the first time, and the guard is a test asserting the client's call count
is **zero** during a dry-run - a count, because "was not called with" passes against a call.
"""

from __future__ import annotations

# ⚠ IMPORT FOR ITS SIDE EFFECT, AND THE SIDE EFFECT IS THE WHOLE POINT — added 2026-10-16.
# `clients.midataworks_review.wire()` binds the run-scoped review-queue port into the single
# process-global slot `HumanReviewSelector` reads. Until this line existed, **nothing in
# `backend/src/` imported that module at all**, so `current_review_queue()` was always `None`
# and a `human_review` selector raised `midataworks_not_configured` *even with miDataworks
# fully configured and reachable* — naming an absent service that is present.
#
# It lives HERE, in F10's own package, and not at any of the three sites that actually use the
# port, because four separate fences forbid those layers from importing `clients` — and all
# four are correct. `services/selectors/**` must depend on the port rather than F10's concrete
# client, *"because depending on F10's concrete client instead of the port is how the port
# stops being a port"*; `services/scoring/**` and `training/**` must have no outbound sibling
# call site at all (ADR-010). The agent that found this moved its seam three times rather than
# weaken a guard, which was the right call.
#
# `services.scoring.wiring` imports this package, so workers reach it through `TASK_MODULES`
# and the API through `endpoints/scoring.py`. Removing this line makes in-run human review
# silently unavailable again; `test_review_queue_wiring.py` is the control that goes red.
from clients import midataworks_review as _review_port_wiring  # noqa: F401
