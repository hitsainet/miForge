"""The judge is never the resident model  (FR-003.17, FR-003.18; **R-02.11**).

R-02.11 forbids the judge being the model under training. It is the stated mitigation for
PPRD §8's recursion risk, and the reason is not procedural: **a model that grades its own
outputs and trains on the winners converges on whatever it already likes.** Loss falls, the
test score does not move, and nothing errors. So this is a **refusal from the code path,
before the first step** - a constraint made unreachable beats one merely stated.

## In-process is refused unconditionally, and no comparison is written

ADR-009 defines the in-process provider as generating from the parameters miForge holds,
which *is* the resident model by definition. So there is no comparison to make, and none
should be written: **a comparison implies a passing case.** If a later edit adds one, it
has invented a way for the in-process provider to be an acceptable judge.

## HTTP is compared on a normalised target, and the normaliser is the guard

An unnormalised comparison is a guard that ``http://judge:8000/`` defeats against
``http://judge:8000``. The normaliser therefore lives **in this module with its own test**;
it is not a shared utility (FTID I-12), because a shared helper gets "tidied" by someone
with a different caller in mind and this one is load-bearing.

## Three call sites, one function

Configuration resolution, pre-flight before a run's first step, and again on a
resolved-identity change mid-run. One implementation, three thin call sites, **one test
each** - because a guard called from two of three places is a guard that is absent in the
third, and the third is the one a long run reaches.

**That warning described this module's own state until 2026-10-16.** Only
:func:`guard_at_resolution` had a production caller, and it is called with
``run_target=None`` (F2 builds its ``RunContext`` with ``provider=None``), so the comparison
below never ran in production at all - the in-process refusal was carrying the whole
requirement. :func:`guard_at_preflight` is now wired into R-02.53's validator;
:func:`guard_on_identity_change` is still owed and says why at its own definition.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from core.errors import refuse
from providers.base import ProviderIdentity

#: The caveat recorded when the judge is a *different* parameter set of the same lineage -
#: the run's own untrained base, or an earlier checkpoint (E-7, FPRD OQ-3). **Returned and
#: recorded, never only logged**: FR-003.18 is about auditability after the run, and a log
#: line is not a record.
CAVEAT_SHARES_LINEAGE = "judge_shares_lineage"

#: ADR-009's name for the provider that generates from the parameters miForge holds.
IN_PROCESS_KIND = "in_process"

#: Default ports, so ``http://host`` and ``http://host:80`` compare equal.
_DEFAULT_PORTS = {"http": 80, "https": 443}


def normalise_base_url(base_url: str | None) -> str | None:
    """Canonical form of a base URL for comparison. ``None`` stays ``None``.

    Lower-cases the scheme and host, makes the port explicit, strips a trailing slash from
    the path, and drops query and fragment - none of which identify an endpoint. Every one
    of these is a way two spellings of one endpoint would otherwise compare unequal and
    defeat the guard.
    """
    if base_url is None:
        return None
    raw = base_url.strip()
    if not raw:
        return None
    parts = urlsplit(raw if "//" in raw else f"//{raw}", scheme="http")
    scheme = (parts.scheme or "http").lower()
    host = (parts.hostname or "").lower()
    port = parts.port if parts.port is not None else _DEFAULT_PORTS.get(scheme)
    netloc = f"{host}:{port}" if port is not None else host
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def normalise_model(model: str | None) -> str:
    """Canonical form of a model name: trimmed and case-folded.

    Case-folded because an endpoint that serves ``LFM2-1.2B`` and one that serves
    ``lfm2-1.2b`` are serving the same model, and a guard that let the second judge the
    first would be defeated by a capital letter.
    """
    return (model or "").strip().casefold()


def normalise_target(identity: ProviderIdentity) -> tuple[str, str, str | None]:
    """``(kind, model, base_url)`` in comparable form. The guard's one key."""
    return (
        identity.kind.strip().casefold(),
        normalise_model(identity.model),
        normalise_base_url(identity.base_url),
    )


def check_judge_identity(
    judge: ProviderIdentity, run_target: ProviderIdentity | None
) -> tuple[str, ...]:
    """**The R-02.11 guard.** Refuses a self-judging configuration; returns any caveats.

    Args:
        judge: The resolved judge provider's identity.
        run_target: The run's own inference target, when there is a run. ``None`` for F7's
            dry-run, which happens before a run exists - and the in-process check below
            still applies, because that one needs no comparison.

    Returns:
        A tuple of caveat names, empty when there are none. :data:`CAVEAT_SHARES_LINEAGE`
        when the judge serves the **same model name from a different endpoint**: almost
        certainly a different parameter set of the same lineage, which FPRD OQ-3 permits
        while recording that it happened.

    Raises:
        Refusal: ``judge_is_resident_model``, carrying **both identities** in ``details``.
    """
    if judge.kind.strip().casefold() == IN_PROCESS_KIND:
        # UNCONDITIONAL. No comparison with `run_target` is written here, deliberately:
        # ADR-009's in-process provider generates from the parameters miForge holds, so it
        # IS the resident model, and a comparison would imply there is a case where it is
        # not.
        raise refuse(
            "judge_is_resident_model",
            "The judge is configured as the in-process provider, which generates from the "
            "parameters miForge is training - a model grading itself is not a measurement. "
            "Point the judge at an OpenAI-compatible endpoint serving a "
            "different model.",
            judge=judge.model_dump(mode="json"),
            run_target=run_target.model_dump(mode="json") if run_target else None,
        )

    if run_target is None:
        return ()

    if normalise_target(judge) == normalise_target(run_target):
        raise refuse(
            "judge_is_resident_model",
            "The judge resolves to the same (base_url, model) as the run's own inference "
            "target, so the model would be grading its own output. Point the "
            "judge at a different model or a different endpoint.",
            judge=judge.model_dump(mode="json"),
            run_target=run_target.model_dump(mode="json"),
        )

    if normalise_model(judge.model) == normalise_model(run_target.model):
        # Same model name, different endpoint: a different parameter set of the same
        # lineage (the run's untrained base, or an earlier checkpoint served elsewhere).
        # FPRD OQ-3 takes the lenient reading - refuse exact identity, caveat the rest -
        # because the strict reading removes a cheap and genuinely useful judge. The caveat
        # is what makes the choice auditable instead of invisible.
        return (CAVEAT_SHARES_LINEAGE,)

    return ()


# --------------------------------------------------------------------------------------
# The three call sites. Thin on purpose: each exists so there is something to test, and
# so a future reader can see that all three go through one implementation.
# --------------------------------------------------------------------------------------


def guard_at_resolution(
    judge: ProviderIdentity, run_target: ProviderIdentity | None
) -> tuple[str, ...]:
    """Call site 1: configuration resolution, before any completion is evaluated."""
    return check_judge_identity(judge, run_target)


def guard_at_preflight(
    judge: ProviderIdentity, run_target: ProviderIdentity | None
) -> tuple[str, ...]:
    """Call site 2: pre-flight, **before the run's first step** (AC-003.5).

    Separate from resolution because a configuration can be resolved long before a run
    starts - during pack authoring, for instance - and the run's inference target may not
    have been chosen then. Refusing at the first judged completion instead would mean the
    run had already taken a step.

    **Wired 2026-10-16** into :func:`services.run_definition.validate` as its
    ``judge_identity`` step, which is R-02.53's validator and the one path a run reaches a
    card through. It had no caller before that, and neither did call site 3 - so the only
    site in production was :func:`guard_at_resolution`, which F2's source pre-flight calls
    with ``run_target=None`` because ``training/run_driver._resolve_source`` builds its
    ``RunContext`` with ``provider=None``. **R-02.11's comparison was therefore never once
    made:** only the unconditional in-process refusal fired, and the module docstring's own
    warning came true - *a guard called from two of three places is a guard that is absent
    in the third.*

    The judge's identity is taken from ``providers.registry.row_for``'s **row** rather than
    from a constructed provider, so the check costs no allocation and contacts no endpoint -
    which is what lets it live inside a validator whose defining property is that it
    allocates nothing.
    """
    return check_judge_identity(judge, run_target)


def guard_on_identity_change(
    judge: ProviderIdentity, run_target: ProviderIdentity | None
) -> tuple[str, ...]:
    """Call site 3: a resolved identity changed mid-run.

    The one a long run reaches. An endpoint re-pointed at the model under training, or a
    run whose own target moved, becomes self-judging without any configuration having
    changed - so the comparison is re-made rather than cached from resolution.

    .. warning::

       **STILL UNWIRED, and the reason is that the mechanism it needs does not exist.**
       Checked rather than assumed, 2026-10-16: ``JudgeModelSelector.prepare`` resolves the
       provider once and caches a ``PreparedJudge``, and nothing anywhere re-resolves a
       judge or re-reads a run's target mid-run. So there is no event for this call site to
       hang on - wiring it would mean *building* the detection, not connecting it, and the
       seam is ``services/selectors/plugins/judge_model.py`` plus whatever re-reads the
       provider row, neither of which belongs to this module.

       The consequence, stated so it is not rediscovered: a judge endpoint re-pointed at the
       model under training **after** the run starts is never noticed, for the whole life of
       the run. R-02.11's failure mode is silent by construction - the loss falls, the test
       score does not move, and nothing errors - so the run looks like it is working.
       Recorded as owed rather than papered over with a call nothing triggers.
    """
    return check_judge_identity(judge, run_target)
