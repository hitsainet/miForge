"""Chaining: a curriculum of stages, advanced by one sweep  (R-02.27; FR-006.39 - .46).

Three modules, and the split is FR-006.46's deletability property made structural:

* :mod:`services.chains.tables` - ``mf_chains`` and ``mf_chain_stages``, on F6's own
  ``MetaData``.
* :mod:`services.chains.service` - create, advance, halt. The orchestrator.
* :mod:`workers.chain_tasks` - ``miforge.maintenance.chain_sweep``, the **one** advance
  path, plus :mod:`api.v1.endpoints.chains`, the four routes.

**Nothing outside those modules may depend on a chain existing** (FR-006.46, AMB-5), and
``test_chain_deletability.py`` is the assertion rather than the intention. BRD §9 has not
decided whether chaining is a first-release feature or a script on top of single runs, so
the shape is what keeps either answer cheap.

Re-exports are deliberately **thin**: a caller wanting the tables imports them from
:mod:`services.chains.tables`, so there is one path to each name rather than two spellings
of it.
"""

from __future__ import annotations

from services.chains.service import (
    advance_one,
    create,
    detail,
    halt,
    listing,
    start_next,
    stop_chain,
    sweep,
)

__all__ = [
    "advance_one",
    "create",
    "detail",
    "halt",
    "listing",
    "start_next",
    "stop_chain",
    "sweep",
]
