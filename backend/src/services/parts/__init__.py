"""Dataset parts: the contract, the mapping, ingest and the index  (F10; R-02.37).

A *part* is one input a run draws on - a prompt set, a corpus, preference pairs, a test
set, a retention set, or a selector or grader configuration - taken in under
``midataworks.dataset-version/v1``.

**Two choke points live in this package and nothing may route around either:**

* :mod:`services.parts.contract_validator` is **the only way a part comes in**. Every
  acquisition path converges there before a row is written. There is no "trusted" path -
  a version that fails the contract is a defect wherever it came from, and a bypass for
  "parts we fetched ourselves" is how an invalid part gets a part row.
* :mod:`services.parts.reader` is **the only place the held-out rule lives**. F2 and F4
  both call it, and if the rule appears in a loader as well, one of them will drift and
  miForge will be the code path that merges a held-out split into training material -
  which miDataworks guarantees it never does.
"""

from __future__ import annotations
