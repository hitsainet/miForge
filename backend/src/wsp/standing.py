"""A verdict's standing: FU-7b, an ``improved`` verdict stands only after a fresh held-out
confirmation (operator decision 2026-11-24; 018 FTASKS FU-7b).

A held-out set scored again and again stops being held out: each verdict on it is feedback, and
the next editing round can fit to it. The D128 bundle's sets had each been scored 18 times when
the operator decided this. So ``improved`` is read in two ways:

- **``improved``**, confirmed: at least one PRIMARY held-out set reads ``improved`` in this verdict
  and was scored for the first time by it. Its ``heldout_uses`` after this evaluation is 1, so no
  earlier verdict had seen it. The count is miForge's ``mf_wsp_heldout_uses``, kept per owner, set
  name and row-id digest across every bundle, so 1 means never scored in this bundle's history or
  any other.
- **``improved_unconfirmed``**: every primary set it beat the control on had been scored before.
  Do not act on it (import it, build on it, call the probe better) without a confirmation on a
  held-out set never scored before.

Every other value stands as it is. Only the verdict document is read: ``value`` and each
``per_set`` entry's ``primary``, ``value`` and ``heldout_uses``. So the rule adds no field to the
run contract (``miforge-wsp-run-v1.json``), a stored verdict needs no migration, and any reader
of the run document (miWorkbench's import, the run page) can apply it. The run page applies the
same rule in ``frontend/src/components/panels/wsp/standing.ts``; both are held to one case table,
``frontend/src/components/panels/wsp/standing.cases.json``.
"""

from __future__ import annotations

from typing import Any

UNCONFIRMED = "improved_unconfirmed"


def confirmed_by(document: dict[str, Any]) -> list[str]:
    """The primary sets that confirm an ``improved`` verdict: read ``improved`` here, and scored
    for the first time by it (``heldout_uses == 1``). Empty for any other value."""
    if document.get("value") != "improved":
        return []
    out = []
    for entry in document.get("per_set") or []:
        uses = entry.get("heldout_uses")
        # A count, never a bool (True == 1 in Python); 1.0 is 1, as the run page reads JSON.
        fresh = isinstance(uses, int | float) and not isinstance(uses, bool) and uses == 1
        if entry.get("primary") is True and entry.get("value") == "improved" and fresh:
            out.append(str(entry.get("name")))
    return out


def standing(document: dict[str, Any]) -> str:
    """``improved`` only when confirmed (above), else ``improved_unconfirmed``; any other
    value as it is."""
    value = str(document.get("value"))
    if value != "improved":
        return value
    return "improved" if confirmed_by(document) else UNCONFIRMED


def reading(document: dict[str, Any]) -> dict[str, Any]:
    """What the candidate routes show beside a verdict's id and value."""
    return {"standing": standing(document), "confirmed_by": confirmed_by(document)}
