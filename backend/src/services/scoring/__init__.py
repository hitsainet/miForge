"""Test and retention scoring - **F4**  (R-02.16, R-02.17, R-02.18, R-02.19, R-02.49).

The one place in miForge that produces a number an operator will trust without being able
to check it. Everything here that looks like over-recording - the provider identity, the
decode parameters, the aggregation method, the subsample seed, the split markers - exists
because **a number without them is not a measurement** (R-02.16: *"a score is only
comparable against another taken the same way"*).

## What it does, and the one thing it does not

F6 calls :func:`services.scoring.service.evaluate` on a cadence; it resolves a set,
generates through a pinned provider, scores each completion through **F3's** registry,
aggregates by a named method, writes a durable record, publishes a breach signal, and hands
F5 two ScoreRecord payloads at save time. It **does not stop the run** - R-02.25 is F6's -
and :mod:`services.scoring.retention` imports nothing that could, which is how that boundary
is kept structurally rather than by discipline.

## THE WIRING BELOW IS THE POINT

:func:`services.scoring.wiring.wire` runs as an import side effect of this package, and it
does two things no other module does:

* registers the provider kinds (:mod:`providers.registry`);
* **binds F4's registry into F3's ``ProviderResolver`` port**, which until now was bound
  only in tests - so the judge-model selector was fully implemented, fully unit-tested, and
  **unreachable in production.** ADR-016 #1 exactly: *a capability is not shipped until a
  test fails when its wiring is removed.* Deleting the call in
  :func:`services.scoring.wiring.wire` must make
  ``backend/tests/unit/test_scoring_provider_resolver_reachability.py`` red, and that red is
  recorded rather than assumed.
"""

from __future__ import annotations

from services.scoring.wiring import wire

wire()
