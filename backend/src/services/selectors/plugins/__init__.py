"""The five built-in selector/grader implementations  (R-02.9).

One file per type, named for its canonical kind, so the module name and the registered name
are one substitution apart and a mismatch is visible.

**Nothing here registers itself.** Registration is the one explicit block in
:mod:`services.selectors`, because a plugin that registered on import would be reachable
without appearing in that block - and then the block would be a partial list that reads as
exhaustive.

**No plugin reads a model parameter, a checkpoint, a manifest or a run row, and none
imports from ``training/``** (FR-003.13), asserted by a static test. That is what keeps *"a
selector is a pure function of a completion"* true rather than aspirational - and it is why
F3 depends on nothing from F1.
"""

from __future__ import annotations
