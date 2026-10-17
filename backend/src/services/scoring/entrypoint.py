"""From a run id to an evaluation pass  (FR-004.14a, FR-004.22, FR-004.23; E-12, E-16).

**The convergence point for the three callers.** F6's loop calls
:func:`services.scoring.service.evaluate` directly with a spec it already holds; the Celery
task and the out-of-cadence route have only a ``run_id`` and a step, so they come through
here, which loads the run, resolves its scoring configuration and calls the **same** service
function. One implementation, three callers, asserted by call count.

## Where a run's scoring configuration lives, and why it is read from the definition

``mf_runs`` has no column for a test slot, a retention slot, a floor, a cadence, a pinned
provider or an aggregation method, and ``models/run.py`` is single-owner so this feature adds
none. It does not need to: R-02.53's ``definition_resolved`` is *"after resolution: the GPU
UUID, pinned part versions, the computed budget"* - the run's resolved inputs - and the
scoring configuration is exactly that. F6 writes it; F4 reads it.

**With no scoring configuration the run is refused by name**, not scored with defaults. A
defaulted test set is a number about a set nobody chose, and a defaulted floor is the
fabricated 0.0 that passes every run forever (M-3).

## The two submission guards live here  (E-12, E-16)

:func:`check_submission` runs **before a run starts**, which is R-02.48's rule - *"a run that
cannot proceed says so before it starts, not partway through"* - and refuses:

* a missing retention set, or one with no floor (E-12). R-02.17 states the set and R-02.38
  makes the retention slot **with its floor** a required pack slot;
* a part version bound as a test or retention set **and** as a training signal source in the
  same run (E-16), naming the part. F10's ``reader.splits_for`` keeps marked splits out of
  training material at import; this is the **composition-time** half, because a part can be
  imported cleanly and then bound twice. Training on the test set raises the test score for
  the wrong reason, which is indistinguishable from success on every surface this feature
  produces (M-8).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from core.errors import refuse
from providers import registry
from providers.base import GenerationParams
from services.scoring import expectations, retention, service, sets
from services.scoring.settings import (
    ScoringSettings,
    effective_max_prompts,
    get_scoring_settings,
)
from services.scoring.tables import KIND_RETENTION, KIND_TEST

log = logging.getLogger("miforge.scoring")

#: Where a run's scoring configuration sits inside ``definition_resolved``. One key, named
#: here so F6 writes what F4 reads rather than two documents each describing it.
DEFINITION_KEY = "scoring"


def check_submission(
    *,
    test_part_versions: Iterable[str],
    retention_part_versions: Iterable[str],
    signal_part_versions: Iterable[str],
    retention_set_present: bool,
    retention_floor: float | None,
) -> None:
    """The two guards R-02.48 requires **before the run starts**.

    Raises:
        Refusal: ``retention_set_missing`` / ``retention_floor_missing`` (E-12);
            ``held_out_set_used_for_training`` naming the part (E-16).
    """
    if not retention_set_present:
        raise refuse(
            "retention_set_missing",
            "This run defines no retention set. R-02.17 requires one: the retention set is "
            "how a run notices that a domain is being bought with general capability, and "
            "without it the only visible signals are a falling loss and a rising test "
            "score - both of which a model losing general capability still shows.",
        )
    if retention_floor is None:
        raise refuse(
            "retention_floor_missing",
            "This run's retention set has no floor. R-02.38 makes the retention slot with "
            "its floor a required pack slot, and miForge will not choose one: a fabricated "
            "floor of 0.0 passes every run forever while nothing says a thing. Set the "
            "floor on the pack's retention slot, or on the run.",
        )

    evaluation = set(test_part_versions) | set(retention_part_versions)
    overlap = sorted(evaluation & set(signal_part_versions))
    if overlap:
        raise refuse(
            "held_out_set_used_for_training",
            f"Part version {overlap[0]!r} is bound both as evaluation material and as a "
            f"training signal source in this run. Training on the test set raises the test "
            f"score for the wrong reason, and the rise is indistinguishable from success on "
            f"every surface miForge produces. Bind it to one or the other.",
            parts=overlap,
        )


def spec_from_definition(
    definition: Mapping[str, Any],
    *,
    run_id: str,
    settings: ScoringSettings | None = None,
    load_rows: Any = None,
) -> service.EvaluationSpec:
    """Build an :class:`~services.scoring.service.EvaluationSpec` from a run definition.

    Args:
        definition: ``definition_resolved[DEFINITION_KEY]``.
        run_id: The run.
        settings: Injected for tests.
        load_rows: ``(slot) -> (rows, markers, reference)`` for a slot, injected so this
            function is testable without a parts directory. The production loader is
            :func:`load_slot`.

    Raises:
        Refusal: ``run_scoring_not_configured`` when the definition carries no scoring
            block; plus anything set resolution, expectation resolution or floor resolution
            refuses.
    """
    settings = settings or get_scoring_settings()
    loader = load_rows or load_slot

    provider_name = str(definition.get("provider") or settings.scoring_provider)
    aggregation = str(definition.get("aggregation") or "mean")
    decode = _decode_from(definition.get("decode"), settings)

    floor = retention.resolve_floor(
        kind=KIND_RETENTION,
        pack_slot_floor=_optional_float(definition.get("pack_retention_floor")),
        run_override_floor=_optional_float(definition.get("retention_floor_override")),
    )

    test_set = _resolve_slot(definition.get("test_slot"), KIND_TEST, loader, settings)
    retention_set = _resolve_slot(
        definition.get("retention_slot"), KIND_RETENTION, loader, settings
    )

    return service.EvaluationSpec(
        run_id=run_id,
        provider_name=provider_name,
        aggregation=aggregation,
        decode=decode,
        test_set=test_set,
        retention_set=retention_set,
        test_expectation=_resolve_expectation(definition.get("test_expectation")),
        retention_expectation=_resolve_expectation(definition.get("retention_expectation")),
        floor=floor,
        batch_bound=int(definition.get("batch_bound") or settings.scoring_batch_bound),
        requires=frozenset(definition.get("requires") or ()),
    )


def _decode_from(raw: Mapping[str, Any] | None, settings: ScoringSettings) -> GenerationParams:
    """The decode parameters, from the definition or the stated defaults.

    **Greedy by default** (FR-004.5), which is the one case BRD-02's criterion 2 is written
    against. The seed rides here because ``GenerationParams.seed`` is documented as *the
    subsample seed that determines which ``n`` prompts were used*, and a score over an
    unnamed subsample is not reproducible.
    """
    raw = raw or {}
    return GenerationParams(
        max_tokens=int(raw.get("max_tokens") or settings.scoring_decode_max_tokens),
        temperature=float(raw.get("temperature", settings.scoring_decode_temperature)),
        top_p=float(raw.get("top_p", settings.scoring_decode_top_p)),
        stop=tuple(raw.get("stop") or settings.decode_stop()),
        seed=raw.get("seed"),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _resolve_expectation(raw: Mapping[str, Any] | None) -> expectations.ResolvedExpectation | None:
    if not raw:
        return None
    config = raw.get("config")
    if not isinstance(config, Mapping):
        raise refuse(
            "expectation_not_configured",
            "An expectation entry carries no selector configuration body. F4 authors none "
            "and invents none: the body comes from miDataworks through F10, or from the "
            "pack slot, cited by hash.",
        )
    return expectations.resolve_expectation(
        config,
        form=str(raw.get("form") or expectations.FORM_SELECTOR),
        catalog=raw.get("catalog"),
        source=str(raw.get("source") or "pack"),
        run_target=None,
    )


def _resolve_slot(
    raw: Mapping[str, Any] | None, kind: str, loader: Any, settings: ScoringSettings
) -> sets.ResolvedSet | None:
    if not raw:
        return None
    rows, markers, reference = loader(raw)
    sample = None
    if raw.get("sample_size") is not None:
        seed = raw.get("sample_seed")
        if seed is None:
            raise refuse(
                "evaluation_sample_seed_missing",
                "A subsample was requested with no seed. miForge will not choose one: an "
                "unseeded sample makes the score irreproducible while reading as "
                "reproducible, and the seed is recorded with the score for exactly that "
                "reason.",
                kind=kind,
            )
        sample = sets.SampleConfig(size=int(raw["sample_size"]), seed=int(seed))
    return sets.resolve(
        kind,
        reference,
        rows,
        sample=sample,
        markers=markers,
        part_versions=raw.get("part_versions") or (),
        # The LOWER of this feature's ceiling and F3's synchronous bound. See
        # `effective_max_prompts`: two features held two numbers for one bound, and the
        # symptom was a pass that generated 201 completions and then refused.
        max_prompts=effective_max_prompts(settings),
    )


def load_slot(
    slot: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...], sets.SetReference]:
    """The production slot loader: rows, split markers and the pinned reference.

    Reads an imported part's directory through F10's **validated** view. F4 never calls
    miDataworks and never parses a handoff manifest itself - F10's ``parse_and_validate`` is
    the one validator and its client the one call site.
    """
    local_path = slot.get("local_path")
    if not local_path:
        raise refuse(
            "evaluation_set_unresolvable",
            "This evaluation slot names no imported part directory, so there is nothing to "
            "read. Import the part through F10 and pin its version on the pack slot.",
            slot=dict(slot),
        )
    from services.parts.contract_validator import MANIFEST_FILENAME, parse_and_validate

    manifest_path = Path(str(local_path)) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise refuse(
            "evaluation_set_unresolvable",
            f"{manifest_path} does not exist, so this part's identity cannot be read. The "
            f"manifest is the record (ADR-004); reindex and re-import rather than scoring "
            f"the files that remain.",
            slot=dict(slot),
            path=str(manifest_path),
        )
    version, _digest = parse_and_validate(manifest_path.read_bytes(), source=str(manifest_path))
    rows, markers = sets.prompt_rows_from_part(
        local_path, version.raw, split_name=slot.get("split")
    )
    contract = version.contract_reference
    structured = contract.as_dict() if contract is not None else slot.get("external_provenance")
    return (
        rows,
        markers,
        sets.SetReference(
            rendered=sets.render_reference(structured, fallback=slot.get("rendered")),
            structured=structured,
            part_id=slot.get("part_id"),
        ),
    )


def evaluate_run(
    *,
    run_id: str,
    step: int,
    out_of_cadence: bool = False,
    kinds: Sequence[str] = (KIND_TEST, KIND_RETENTION),
    session: Any = None,
    log_path: Path | None = None,
    emit: Any = None,
) -> dict[str, Any]:
    """Load the run, build the spec and call the **one** service function.

    Returns the evaluation block that was written, which is what the route returns and what
    the Celery task returns.

    Raises:
        Refusal: ``run_not_found``; ``run_scoring_not_configured`` when the run's definition
            carries no scoring block.
    """
    from db.session import session_scope
    from models.run import Run
    from training.tasks import run_log_root

    if session is not None:
        return _evaluate_with(
            session,
            run_id=run_id,
            step=step,
            out_of_cadence=out_of_cadence,
            kinds=kinds,
            log_path=log_path,
            emit=emit,
        )
    with session_scope() as scoped:
        del Run, run_log_root  # imported above so an import failure is loud at call time
        return _evaluate_with(
            scoped,
            run_id=run_id,
            step=step,
            out_of_cadence=out_of_cadence,
            kinds=kinds,
            log_path=log_path,
            emit=emit,
        )


def _evaluate_with(
    session: Any,
    *,
    run_id: str,
    step: int,
    out_of_cadence: bool,
    kinds: Sequence[str],
    log_path: Path | None,
    emit: Any,
) -> dict[str, Any]:
    from models.run import Run
    from training.run_log import run_log_path
    from training.tasks import run_log_root

    run = session.get(Run, run_id)
    if run is None:
        raise refuse("run_not_found", f"No run with id {run_id!r}.", run_id=run_id)

    definition = (run.definition_resolved or {}).get(DEFINITION_KEY)
    if not definition:
        raise refuse(
            "run_scoring_not_configured",
            f"Run {run_id!r} carries no scoring configuration in its resolved definition, "
            f"so there is no test set, no retention set and no floor to score against. "
            f"Refusing rather than scoring with defaults: a defaulted test set is a number "
            f"about a set nobody chose, and a defaulted floor passes every run forever.",
            run_id=run_id,
        )

    spec = spec_from_definition(definition, run_id=run_id)
    # Pin, and record the identity the run started with. `get` never returns a different
    # provider: an unreachable endpoint refuses rather than becoming the in-process one.
    pinned = registry.pin(spec.provider_name)
    outcome = service.evaluate(
        spec,
        step,
        session=session,
        log_path=log_path or run_log_path(run_log_root(), run_id),
        kinds=kinds,
        out_of_cadence=out_of_cadence,
        emit=emit,
    )
    del pinned
    return outcome.block
