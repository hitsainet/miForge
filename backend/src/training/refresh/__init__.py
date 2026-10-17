"""The refresh registry - **exactly two modes**  (R-02.26; FR-006.31, FR-006.32).

Two explicit registrations, for the same reason the stop registry has five: a scan makes
*"is it registered?"* untestable and a missing mode silent. An unknown mode is refused
``refresh_mode_unknown`` **listing the two**, because R-02.26 names two and **there is no
third**.

## A refresh on a non-self-generation run is refused, never ignored  (FR-006.31)

R-02.26 opens *"For self-generation runs."* A refresh configured on a ``corpus``,
``preference-pairs`` or ``reward`` run is refused at submit, ``refresh_not_applicable``,
naming the run's source. **Not ignored**: an ignored refresh leaves the operator believing
the prompt set rotated when it did not, which is a wrong answer delivered quietly.

## A failed refresh fails the run; it is not a sixth stop condition  (FR-006.38)

A refresh resolving to an empty or unresolvable prompt set terminates the run **``failed``**
with ``refresh_produced_no_prompts``. It does **not** continue on the old set: the run log
would then record a refresh that did not happen, and every subsequent batch's part reference
would be wrong. The loop owns that transition - see :mod:`training.boundary` - and
FR-006.15's set stays closed at five.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.errors import refuse
from training.refresh import new_prompt_set, template_seeds
from training.refresh.base import (
    MODE_NAMES,
    SELF_GENERATION,
    AcceptedSeedSource,
    RefreshContext,
    RefreshMode,
    RefreshModeName,
    RefreshOutcome,
    Seed,
    rule_hash,
)

#: **THE WIRING.** Two entries, one per mode R-02.26 names. Deleting one must make
#: ``backend/tests/unit/test_refresh_registry.py`` red.
_REGISTRATIONS: tuple[tuple[str, RefreshMode], ...] = (
    (new_prompt_set.NAME, new_prompt_set.MODE),
    (template_seeds.NAME, template_seeds.MODE),
)

_MODES: dict[str, RefreshMode] = dict(_REGISTRATIONS)

assert set(_MODES) == set(MODE_NAMES), (
    "the registry must hold exactly R-02.26's two: "
    f"missing {sorted(set(MODE_NAMES) - set(_MODES))}, "
    f"unknown {sorted(set(_MODES) - set(MODE_NAMES))}"
)


def names() -> tuple[str, ...]:
    """The two registered mode names, in R-02.26's order."""
    return tuple(name for name, _mode in _REGISTRATIONS)


def resolve(name: str) -> RefreshMode:
    """The mode registered as ``name``.

    Raises:
        Refusal: ``refresh_mode_unknown``, listing the two.
    """
    mode = _MODES.get(name)
    if mode is None:
        raise refuse(
            "refresh_mode_unknown",
            f"No refresh mode named {name!r}. R-02.26 names two: {list(names())} - a new "
            f"prompt set is loaded, or the selector's accepted completions become seeds "
            f"through a templating rule. There is no third.",
            mode=name,
            known=list(names()),
        )
    return mode


def assert_applicable(signal_source: str) -> None:
    """Refuse a refresh configured on anything but self-generation (FR-006.31, E-12).

    Raises:
        Refusal: ``refresh_not_applicable``, naming the run's source.
    """
    if signal_source != SELF_GENERATION:
        raise refuse(
            "refresh_not_applicable",
            f"A prompt refresh applies to self-generation runs only (R-02.26 opens 'For "
            f"self-generation runs'), and this run's signal source is "
            f"{signal_source!r}. Refused rather than ignored: an ignored refresh leaves "
            f"you believing the prompt set rotated when it did not. Remove the refresh "
            f"rule, or submit the run as a self-generation run.",
            signal_source=signal_source,
            applicable_to=SELF_GENERATION,
        )


def run(ctx: RefreshContext, mode: str, rule: Mapping[str, Any]) -> RefreshOutcome:
    """Resolve the mode, refresh, and refuse an empty result (FR-006.38, E-13).

    Raises:
        Refusal: ``refresh_mode_unknown``; ``refresh_rule_incomplete``;
            ``refresh_produced_no_prompts`` when the refreshed set is empty. The last is
            raised **here** rather than inside each mode so the rule holds for both and a
            new mode cannot forget it.
    """
    outcome = resolve(mode).refresh(ctx, rule)
    if outcome.size <= 0:
        raise refuse(
            "refresh_produced_no_prompts",
            f"The {mode!r} refresh resolved to a prompt set of {outcome.size} prompts. The "
            f"run fails here rather than continuing on the old set: the run log would "
            f"otherwise record a refresh that did not happen, and every batch after it "
            f"would carry the wrong part reference.",
            mode=mode,
            rule_sha256=outcome.rule_sha256,
            round=outcome.round,
            size=outcome.size,
        )
    return outcome


__all__ = [
    "MODE_NAMES",
    "SELF_GENERATION",
    "AcceptedSeedSource",
    "RefreshContext",
    "RefreshMode",
    "RefreshModeName",
    "RefreshOutcome",
    "Seed",
    "assert_applicable",
    "names",
    "resolve",
    "rule_hash",
    "run",
]
