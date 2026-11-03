"""The lifecycle all four sources share  (FR-002.2, FR-002.4, FR-002.30).

**This is not a ``utils`` module and must not become one.** The FTID's rule - *resist
``signals/utils.py``, because each of the four sources adds exactly one hard thing and the
hard things are not alike* - is about collecting unrelated helpers. What is here is the
**one lifecycle** every source walks:

    preflight:  resolve the parts -> resolve the selector or grader -> name the objective
    next():     fill the material -> count the outcomes -> assemble -> record -> hand over

The four differ only in *fill the material*, which is the one hard thing each of them adds.
Writing that walk four times would give four chances for one of them to skip the recording
step or to record a different set of counts - and a batch that reached a step without a
record is exactly the provenance hole R-02.8 exists to close.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.errors import refuse
from providers.base import ProviderIdentity
from services.selectors.resolve import ResolvedSelector
from services.selectors.types import SelectionHealth
from training.signals import batch_record, parts_reader
from training.signals.base import (
    BatchCounts,
    PartSpec,
    ResolvedPart,
    ResolvedSource,
    RunContext,
    SourceCapability,
    SourceSpec,
    TrainingBatch,
    example_prompt_ids,
)
from training.signals.settings import SignalSettings, get_signal_settings


class BaseSource:
    """Shared pre-flight and recording. Each source implements :meth:`fill`.

    Subclasses declare :attr:`NAME` and :attr:`CAPABILITY`, and override :meth:`fill` and,
    where they need one, :meth:`resolve_extra`.
    """

    NAME: str = ""
    CAPABILITY: SourceCapability

    def __init__(self, context: RunContext | None = None) -> None:
        self.context = context or RunContext()
        self.settings: SignalSettings = get_signal_settings()
        self.resolved: ResolvedSource | None = None
        #: The order each side effect happened in, for the ordering test. Not telemetry:
        #: an inverted order changes nothing observable until the machine dies.
        self.effects: list[str] = []
        self._closed = False

    # ------------------------------------------------------------------ the surface
    def capability(self) -> SourceCapability:
        return self.CAPABILITY

    def close(self) -> None:
        """Release readers. Idempotent, because a loop may close a source it never opened."""
        self._closed = True

    # ------------------------------------------------------------------ pre-flight
    def preflight(self, spec: SourceSpec) -> ResolvedSource:
        """Validate everything decidable without a GPU and without a step (FR-002.4).

        Returns a :class:`ResolvedSource` or refuses. **Never partially succeeds**: a
        half-resolved source is one whose failure surfaces at step 37, by which point the
        operator is reading a transport error instead of a configuration error.
        """
        if spec.source != self.NAME:
            raise refuse(
                "signal_source_spec_mismatch",
                f"This specification names source {spec.source!r} and was handed to "
                f"{self.NAME!r}. Refusing rather than running anyway: the two would "
                f"disagree about which source the record names.",
                spec_source=spec.source,
                source=self.NAME,
            )
        # Refuse a source the gate contract cannot record, BEFORE anything is allocated.
        # R-02.48: "a run that cannot proceed says so before it starts, not partway
        # through" - and a batch that cannot be written down is a run that cannot proceed.
        batch_record.record_source(self.NAME)

        parts = self.resolve_parts(spec)
        selector = self.resolve_selector(spec)
        grader = self.resolve_grader(spec)
        from training.objectives import SOURCE_OBJECTIVE

        resolved = ResolvedSource(
            source=self.NAME,
            spec=spec,
            parts=parts,
            selector=selector,
            grader=grader,
            objective=SOURCE_OBJECTIVE[self.NAME],
            run_target=self.run_target(),
        )
        self.resolve_extra(resolved)
        self.resolved = resolved
        return resolved

    def resolve_parts(self, spec: SourceSpec) -> tuple[ResolvedPart, ...]:
        """Resolve every part through F10's index and this feature's three refusals."""
        if not spec.parts:
            raise refuse(
                "signal_source_no_part",
                f"Source {self.NAME!r} needs at least one "
                f"{parts_reader.SOURCE_PART_KIND[self.NAME]!r} part and none was given. "
                f"Import a miDataworks version, or a contract-conformant export of one, "
                f"and name it on the run.",
                source=self.NAME,
                required_kind=parts_reader.SOURCE_PART_KIND[self.NAME],
            )
        return tuple(
            parts_reader.resolve(
                self.lookup_part(part_spec),
                part_spec,
                source=self.NAME,
                required_kind=self.required_kind(part_spec),
            )
            for part_spec in spec.parts
        )

    def required_kind(self, part_spec: PartSpec) -> str | None:
        """Which kind a given part must be. ``None`` means the source's own default."""
        del part_spec
        return None

    def lookup_part(self, part_spec: PartSpec) -> Mapping[str, Any]:
        """F10's row for one part. **Read-only, through F10's one resolver.**

        F2 performs no HTTP to miDataworks, imports no ``httpx`` and adds no second
        validator: ADR-010 permits exactly one outbound call site per service and F10 owns
        miForge's.
        """
        if self.context.session is None:
            raise refuse(
                "part_index_unavailable",
                "No database session is bound, so the part index cannot be read. Parts are "
                "imported and resolved from the part index; a source cannot read a "
                "directory on its own, because that would be a second path into training "
                "material with no provenance behind it.",
                source=self.NAME,
            )
        from services.parts.resolve import reference_from, resolve_one

        target: Any = part_spec.part_id
        if target is None and part_spec.reference is not None:
            target = reference_from(dict(part_spec.reference))
        resolved = resolve_one(self.context.session, target)
        return dict(resolved.payload)

    def resolve_selector(self, spec: SourceSpec) -> ResolvedSelector | None:
        """F3's resolved selector, or ``None`` when this source does not select.

        **F2 writes no rule, no threshold, no judge call and no configuration hash.** It
        resolves through F3's one path so a configuration error is refused once, before the
        run, rather than once per completion.
        """
        if not self.CAPABILITY.needs_selector:
            return None
        if spec.selector is None:
            raise refuse(
                "selector_not_configured",
                f"Source {self.NAME!r} admits completions through a selector and none was "
                f"configured. Choose one of the five types; four of them need no "
                f"other miSuite app at all.",
                source=self.NAME,
            )
        return self._resolve_config(spec, dict(spec.selector))

    def resolve_grader(self, spec: SourceSpec) -> Any | None:
        """F3's grader form of a resolved configuration, or ``None``.

        Through ``services.graders.grader_for`` so the range check lives in one place and a
        grader and a selector over one configuration can never disagree about what a
        completion scored. **F2 implements no grader and declares no range.**
        """
        if spec.grader is None:
            if self.CAPABILITY.needs_grader:
                raise refuse(
                    "grader_not_configured",
                    f"Source {self.NAME!r} scores completions and no grader was "
                    f"configured. A grader configuration is authored in miDataworks and "
                    f"cited by hash, or supplied locally with its own provenance.",
                    source=self.NAME,
                )
            return None
        from services.graders import grader_for

        return grader_for(self._resolve_config(spec, dict(spec.grader)))

    def _resolve_config(self, spec: SourceSpec, config: dict[str, Any]) -> ResolvedSelector:
        from services.selectors import ensure_registered
        from services.selectors.resolve import resolve as resolve_config

        ensure_registered()
        return resolve_config(
            config,
            source=spec.config_source,
            run_target=self.run_target(),
        )

    def run_target(self) -> ProviderIdentity | None:
        """The run's own inference target, published for F3's R-02.11 identity guard.

        **Self-generation's generation target is the model under training, and that is
        required** - R-02.5 says *"prompt the model"*. R-02.11 forbids only the **judge**
        being that model. The two constrain different calls, and a reader who conflates them
        will either break self-generation or break the guard. F2 asserts nothing about
        judges; it supplies the value the guard compares against (FR-002.14, F3 FR-003.17).
        """
        provider = self.context.provider
        identity = getattr(provider, "identity", None)
        return identity if isinstance(identity, ProviderIdentity) else None

    def resolve_extra(self, resolved: ResolvedSource) -> None:
        """A hook for a source's own pre-flight checks. Default: nothing."""
        return None

    # ------------------------------------------------------------------ next()
    def require_resolved(self) -> ResolvedSource:
        """The pre-flight result, refusing a source that was never pre-flighted.

        Refused rather than pre-flighting lazily here: a lazy pre-flight would move every
        configuration refusal from submit time to step 1, which is precisely what
        FR-002.4 exists to prevent.
        """
        if self.resolved is None:
            raise refuse(
                "signal_source_not_preflighted",
                f"Source {self.NAME!r} was asked for a batch before pre-flight. Pre-flight "
                f"is what refuses an unresolvable part, an inadmissible split or an absent "
                f"selector BEFORE a run starts rather than at step 1.",
                source=self.NAME,
            )
        return self.resolved

    def check_cancelled(self) -> None:
        """The cooperative cancellation boundary (ADR-006, ADR-007).

        A batch is assembled and recorded, or it is not handed to a step. Never half.
        """
        if self.context.is_cancelled is not None and self.context.is_cancelled():
            raise refuse(
                "batch_assembly_cancelled",
                "The run was cancelled while this batch was being assembled. The partial "
                "batch is discarded and no step starts: a step completes and is recorded, "
                "or it has not started.",
                run_id=self.context.run_id,
            )

    def handle_shortfall(self, counts: BatchCounts, *, requested: int) -> None:
        """Apply the configured shortfall behaviour. **Neither value admits a pending one.**

        OQ-1 left *refuse the batch* or *take a short one* open, and this is where that
        answer is applied - once, from configuration, with no default (FR-002.13).
        """
        from training.signals.settings import require_shortfall_behaviour

        if counts.admitted >= requested:
            return
        behaviour = require_shortfall_behaviour(self.settings)
        if behaviour == "refuse_batch":
            raise refuse(
                "batch_could_not_fill",
                f"This batch admitted {counts.admitted} of the {requested} examples asked "
                f"for ({counts.pending} still awaiting review, {counts.rejected} rejected, "
                f"{counts.ties} unorderable, {counts.dropped} malformed"
                f"{', material exhausted' if counts.exhausted else ''}). "
                f"SIGNAL_PENDING_SHORTFALL is 'refuse_batch'. No pending candidate is "
                f"admitted to make up the number - set it to 'short_batch' to train on what "
                f"was admitted and record the shortfall.",
                requested=requested,
                admitted=counts.admitted,
                pending=counts.pending,
                rejected=counts.rejected,
                ties=counts.ties,
                dropped=counts.dropped,
                exhausted=counts.exhausted,
            )

    def record(
        self,
        *,
        examples: Sequence[Any],
        counts: BatchCounts,
        resolved_model: str | None = None,
        health: SelectionHealth | None = None,
        hashes: Any | None = None,
        signals: Sequence[str] = (),
        provider: ProviderIdentity | None = None,
        review_ids: Sequence[str] = (),
        chat_template: str | None = None,
        masking_policy: str | None = None,
        reference_source: str | None = None,
    ) -> TrainingBatch:
        """Assemble the record, write it **durably first**, and return the batch.

        The batch is returned **only after** its record exists: that is what makes R-02.8's
        guarantee true rather than intended, and it is why ``next()`` has no path that
        returns material without recording it.
        """
        resolved = self.require_resolved()
        record = batch_record.assemble(
            source=self.NAME,
            prompt_ids=example_prompt_ids(examples),
            parts=[part.reference for part in resolved.parts],
            counts=counts,
            resolved_model=resolved_model,
            selector_config_sha256=_hash_of(hashes, "selector_hash"),
            grader_config_sha256=_hash_of(hashes, "grader_hash"),
        )
        batch_record.write(
            record,
            run_id=self.context.run_id,
            step=self._current_step(),
            counts=counts,
            objective=resolved.objective,
            log_root=None if self.context.log_root is None else Path(str(self.context.log_root)),
            session=self.context.session,
            emit=self.context.emit,
            heartbeat=self.context.heartbeat,
            signals=signals,
            effects=self.effects,
        )
        return TrainingBatch(
            objective=resolved.objective,
            record=record,
            examples=tuple(examples),
            counts=counts,
            health=health,
            hashes=hashes,
            masking_policy=masking_policy,
            reference_source=reference_source,
            tokenizer=self.context.tokenizer,
            chat_template=chat_template,
            provider=provider,
            binding=_binding_of(resolved),
            review_ids=tuple(review_ids),
        )

    def _current_step(self) -> int | None:
        if self.context.current_step is None:
            return None
        return int(self.context.current_step())


def _hash_of(hashes: Any, attribute: str) -> str | None:
    """One published hash, or ``None``. F3 publishes them; F2 copies and never computes one."""
    value = getattr(hashes, attribute, None)
    return value if isinstance(value, str) else None


def _binding_of(resolved: ResolvedSource) -> dict[str, str]:
    """``role -> semantic`` across the resolved parts, for the batch object.

    Carried on the batch rather than in the record because the gate contract has no field
    for it - and it matters: *a corpus trained from ``messages`` and one trained from
    ``text`` are not the same run.*
    """
    binding: dict[str, str] = {}
    for part in resolved.parts:
        binding.update(part.binding.as_record())
    return binding
