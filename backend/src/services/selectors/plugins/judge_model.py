"""The judge-model selector  (FR-003.5, FR-003.17, FR-003.18; R-02.9, **R-02.11**, R-02.49).

A second model scores the completion. **It goes through the provider abstraction, not a
named transport.** R-02.9 originally said this selector *"calls a second model on the CPU
nodes via llama.cpp"* - a hardcoded transport, which is the exact shape that made the whole
document depend on one sibling. That phrasing was not swept with the three "through miLLM"
paths and §9 already contradicted it. **miLLM is one row beside vLLM, Ollama and the OpenAI
API**, reached by ``base_url``; a llama.cpp server on a CPU node is one deployment of this
requirement, not its definition.

## R-02.11 makes the in-process provider unusable AS A JUDGE

ADR-009's in-process provider generates from the parameters miForge holds, so a judge using
it would be **the model grading itself** - which is not a measurement. Therefore:

* the in-process provider is refused **unconditionally** by the identity guard;
* and with **no endpoint configured this type reports itself unavailable BY NAME**
  (``judge_provider_not_configured``) rather than falling back to anything. There is
  nothing to fall back *to*, and that is deliberate: a silent fallback here would be the
  recursion failure PPRD §8 names and R-02.11 exists to mitigate.

## No default verdict, on any failure

A transport failure is **not a verdict**. An unreachable endpoint, a timeout, an
unparseable reply and a score outside its range each refuse the **evaluation**; none of
them produces an accept or a reject. Defaulting any of them to accept is the plausible-
looking "robustness" change that converts this feature into the accept-everything selector
it exists to prevent - and it is mutation control 3 for exactly that reason.

## The prompt template, and a field the contract does not have

``selector-config-v1.json``'s ``JudgeModelSelector`` is ``{kind, name, config_sha256,
provider, threshold}`` with ``additionalProperties: false``. **There is no template field**,
while FR-003.5 says *"the prompt template and the accept threshold come from the
configuration"*. F3 does not add a field to that schema (FR-003.2), so the template is
explicit operator configuration (``JUDGE_PROMPT_TEMPLATE``) and this type **refuses without
it** - see :mod:`services.selectors.settings`. Reported as a contract gap, not papered over
with a hard-coded template, which would be a verdict policy chosen by whoever wrote the line.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable
from typing import Any

from core.errors import refuse
from providers.base import GenerationParams, InferenceProvider, ProviderIdentity
from schemas.selector import SelectorKind, Verdict, VerdictValue
from services.selectors import availability
from services.selectors.identity import guard_at_resolution
from services.selectors.provider_port import current_provider_resolver
from services.selectors.resolve import ResolvedSelector
from services.selectors.settings import get_selector_settings

#: The placeholders a judge template may use. Listed rather than discovered, so a template
#: naming something else is refused at ``prepare`` instead of rendering a literal
#: ``{answer}`` into the prompt and judging that.
TEMPLATE_FIELDS = frozenset({"prompt", "completion"})

#: The reply form, fixed and strict. The judge must emit a ``SCORE:`` line with a number in
#: **0..1**, optionally followed by a ``REASON:`` line.
#:
#: Strict, because *"a regex that scrapes the first integer anywhere in the reply will
#: happily read a score out of the model's own prose"* - and a score read out of prose is a
#: verdict with no relationship to what the judge meant.
#:
#: The range is 0..1 rather than configurable because the contract fixes it at both ends:
#: ``Verdict.score`` is ``ge=0.0, le=1.0`` and ``Grader.grade`` is documented as *"a score
#: in 0..1"*. That also answers FPRD OQ-5 - the range is the contract's, not a per-
#: configuration choice.
_SCORE_LINE = re.compile(r"^\s*SCORE\s*:\s*(?P<score>[-+]?\d+(?:\.\d+)?)\s*$", re.I | re.M)
_REASON_LINE = re.compile(r"^\s*REASON\s*:\s*(?P<reason>.+?)\s*$", re.I | re.M)

SCORE_RANGE = (0.0, 1.0)


class PreparedJudge:
    """What ``prepare`` produced: the provider, the template, the threshold, the caveats."""

    def __init__(
        self,
        provider: InferenceProvider,
        template: str,
        threshold: float,
        caveats: tuple[str, ...],
    ) -> None:
        self.provider = provider
        self.template = template
        self.threshold = threshold
        self.caveats = caveats


def render_template(template: str, *, prompt: str, completion: str) -> str:
    """Render the judge prompt. **Refuses an unknown placeholder.**

    ``str.format`` would raise ``KeyError`` on an unknown field at the first judged
    completion; this refuses at ``prepare`` time instead, naming the field - and a template
    with a stray brace is a configuration error, not a per-completion one.
    """
    try:
        return template.format(prompt=prompt, completion=completion)
    except (KeyError, IndexError) as exc:
        raise refuse(
            "judge_template_invalid",
            f"The judge prompt template names a placeholder this release does not supply: "
            f"{exc}. Use only {sorted(TEMPLATE_FIELDS)}.",
            available=sorted(TEMPLATE_FIELDS),
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise refuse(
            "judge_template_invalid",
            f"The judge prompt template is malformed: {exc}. Escape a literal brace as "
            f"'{{{{' and '}}}}'.",
            detail=str(exc),
        ) from exc


def parse_judge_reply(text: str) -> tuple[float, str | None]:
    """``(score, rationale)`` from a judge reply. **Refuses anything else.**

    Raises:
        Refusal: ``judge_reply_unparseable`` with the reply **truncated** in ``details`` -
            a judge reply can be long and a refusal is not a transcript.
        Refusal: ``grader_score_out_of_range`` with the value and the range, for a score
            outside 0..1. **Never clamped**: a clamp turns a judge that misunderstood the
            scale into a confident verdict.
    """
    match = _SCORE_LINE.search(text)
    if match is None:
        raise refuse(
            "judge_reply_unparseable",
            "The judge's reply contains no 'SCORE: <number>' line, so there is no score to "
            "read. Refusing rather than scraping a number out of the reply's prose, which "
            "would produce a verdict unrelated to what the judge meant. No verdict is "
            "defaulted.",
            reply=_truncate(text, 500),
            expected="a line of the form 'SCORE: 0.80'",
        )
    score = float(match.group("score"))
    low, high = SCORE_RANGE
    if not (low <= score <= high):
        raise refuse(
            "grader_score_out_of_range",
            f"The judge returned {score:g}, outside the {low:g}..{high:g} range a score "
            f"must lie in. Refusing rather than clamping: a clamp turns a judge that "
            f"misread the scale into a confident verdict. Fix the prompt template so it "
            f"asks for a score in {low:g}..{high:g}.",
            value=score,
            range=[low, high],
        )
    reason_match = _REASON_LINE.search(text)
    rationale = reason_match.group("reason") if reason_match else None
    return score, rationale


class JudgeModelSelector:
    """Satisfies :class:`schemas.selector.Selector` and offers a grader form."""

    kind: SelectorKind = "judge_model"

    def __init__(self, resolved: ResolvedSelector) -> None:
        self._resolved = resolved
        #: Judge replies fetched ahead by :meth:`prefetch`, keyed by the exact
        #: ``(prompt, completion)`` pair, one entry per occurrence. **Raw reply text, not a
        #: parsed score**: parsing happens when the item is evaluated, so an unparseable
        #: reply refuses at the same item, in the same order, as it would without prefetch.
        #: A pair with no entry is a MISS and is judged live - so a prefetch that stopped
        #: early (cancellation) or never ran changes latency, never a verdict.
        self._prefetched: dict[tuple[str, str], deque[str]] = {}

    # ------------------------------------------------------------------ batch prefetch

    def prefetch(
        self,
        prompts: list[str],
        completions: list[str],
        *,
        batch_id: str,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Judge the batch ahead of the per-item loop, ``judge_concurrency`` at a time.

        **Suite tracker 4.28: ``SelectorSettings.judge_concurrency`` was read by nothing.**
        Per-item judging calls ``generate([rendered])`` with ONE prompt, so the provider's
        bounded fan-out (``HttpProviderRow.concurrency``) never engaged and the setting
        published a lever that moved nothing. Here each window of ``judge_concurrency``
        rendered prompts goes to the provider in one ``generate`` call, which the provider
        parallelises up to its own bound.

        **Cancellation stays prompt.** ``is_cancelled`` is checked before every window, so a
        cancelled run makes at most ``judge_concurrency`` judge calls after the request
        (one window already in flight) and none after that; the per-item loop then raises
        ``evaluation_cancelled`` at its first boundary with nothing recorded. The invariant
        *"evaluated and recorded, or not started"* is about records, and no record is
        written here - a prefetched reply that is never consumed is discarded.

        **A text over the bound is left as a miss**, not refused here: the per-item path
        refuses it at its own index, after any earlier cancellation check, exactly as it
        would without prefetch.
        """
        del batch_id
        self._prefetched = {}
        prepared = self._prepared()
        settings = get_selector_settings()
        limit = settings.selector_max_text_chars
        window = max(1, int(settings.judge_concurrency))
        params = GenerationParams(temperature=0.0, max_tokens=256)
        pairs = [
            (prompt, completion)
            for prompt, completion in zip(prompts, completions, strict=True)
            if len(prompt) <= limit and len(completion) <= limit
        ]
        for start in range(0, len(pairs), window):
            if is_cancelled is not None and is_cancelled():
                return
            chunk = pairs[start : start + window]
            rendered = [
                render_template(prepared.template, prompt=prompt, completion=completion)
                for prompt, completion in chunk
            ]
            replies = prepared.provider.generate(rendered, params)
            if len(replies) != len(chunk):
                # A short answer is not a verdict for the missing items; leave them as
                # misses rather than guess which reply belongs to which prompt.
                continue
            for pair, reply in zip(chunk, replies, strict=True):
                self._prefetched.setdefault(pair, deque()).append(reply.text)

    def release_batch(self) -> None:
        """Drop every unconsumed prefetched reply. Called by the service when a batch ends."""
        self._prefetched = {}

    @property
    def name(self) -> str:
        return self._resolved.name

    @property
    def provider_key(self) -> str:
        return str(self._resolved.body["provider"])

    def prepare(self, resolved: ResolvedSelector, *, run_target: Any = None) -> None:
        """Resolve the provider, apply the **R-02.11** guard, and require a threshold.

        Everything that can be refused once per configuration is refused here, so a batch
        of 64 completions does not produce 64 copies of one configuration error - and does
        not produce them *after* half the batch has been judged and paid for.
        """
        settings = get_selector_settings()

        threshold = resolved.body.get("threshold")
        if threshold is None:
            # NOT defaulted. ADR-018's named shape for this feature is
            # `getattr(config, "threshold", 0.5)` - a fabricated verdict policy. A
            # threshold decides which completions train the model; nobody gets to pick it
            # by writing a line.
            raise refuse(
                "judge_threshold_missing",
                "A judge-model selector needs an explicit threshold: it decides which "
                "completions are admitted into training, and miForge will not choose one. "
                "Set 'threshold' in the configuration (0..1).",
                selector=resolved.name,
            )

        # THE PROVIDER IS RESOLVED BEFORE THE TEMPLATE IS CHECKED, deliberately. Both are
        # configuration refusals, but on the standalone path BOTH are missing - and the
        # refusal the caller gets must be the same one `GET /api/v1/selectors` gives as the
        # type's `unavailable_reason`, or the operator is told two different things about
        # one state and fixes the wrong one first.
        provider = self._resolve_provider()

        template = settings.judge_prompt_template
        if not template:
            raise refuse(
                "judge_template_not_configured",
                "No judge prompt template is configured, so there is nothing to ask the "
                "judge. Set JUDGE_PROMPT_TEMPLATE to a template using {prompt} and "
                "{completion} that asks for a line 'SCORE: <0..1>'. Refusing rather than "
                "judging with a template nobody chose.",
                expects=sorted(TEMPLATE_FIELDS),
                reply_form="SCORE: <number in 0..1>",
            )
        # Render once with placeholder text so a malformed template is refused here rather
        # than at the first judged completion.
        render_template(template, prompt="", completion="")

        caveats = guard_at_resolution(provider.identity, _as_identity(run_target))

        resolved.caveats = tuple(dict.fromkeys((*resolved.caveats, *caveats)))
        resolved.bindings["provider"] = provider
        resolved.prepared = PreparedJudge(provider, template, float(threshold), caveats)

    def _resolve_provider(self) -> InferenceProvider:
        """Resolve the configured provider key from **F4's registry**. Constructs nothing.

        Raises:
            Refusal: ``judge_provider_not_configured``, naming the service - which is the
                answer to *"what happens with no endpoint configured?"*. The type is
                unavailable, by name, and **nothing falls back to the model being trained.**
        """
        resolver = current_provider_resolver()
        if resolver is None or not resolver.available(self.provider_key):
            raise refuse(
                availability.JUDGE_PROVIDER_NOT_CONFIGURED,
                f"No generation provider is available for key {self.provider_key!r}, so "
                f"the judge-model selector cannot run. Configure an OpenAI-compatible "
                f"endpoint (JUDGE_PROVIDER_BASE_URL and JUDGE_PROVIDER_MODEL). miForge "
                f"will NOT fall back to the model it is training: R-02.11 forbids it, "
                f"because a model grading itself is not a measurement.",
                provider_key=self.provider_key,
                service="judge_provider",
                capabilities_unavailable=["judge_model selector", "judge_model grader"],
            )
        return resolver.resolve(self.provider_key)

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, *, prompt: str, completion: str, context: dict[str, Any]) -> Verdict:
        """One verdict, whose reason records **both numbers**.

        ``judge 0.20/1.00, threshold 0.60`` - the score and the line it was compared
        against, because either alone is unreadable.
        """
        del context
        prepared = self._prepared()
        score, rationale = self._judge(prompt, completion, prepared)
        admitted = score >= prepared.threshold
        reason = f"judge {score:.2f}/1.00, threshold {prepared.threshold:.2f}"
        if rationale:
            reason = f"{reason} - {rationale}"
        return Verdict(
            value=VerdictValue.ACCEPT if admitted else VerdictValue.REJECT,
            selector=self.name,
            label="judge_above_threshold" if admitted else "judge_below_threshold",
            reason=reason,
            score=round(score, 6),
        )

    def grade(self, *, prompt: str, completion: str, context: dict[str, Any]) -> float:
        """The judge's score, as a scalar in 0..1 (FR-003.9)."""
        del context
        prepared = self._prepared()
        score, _ = self._judge(prompt, completion, prepared)
        return score

    def judge_identity(self) -> ProviderIdentity:
        """Who judged (FR-003.18). Recorded on **every** judged verdict.

        Without it, BRD-02 acceptance criterion 3 - *"a self-generation run **with the
        judge-model selector**"* - is not auditable after the fact.
        """
        return self._prepared().provider.identity

    def _judge(
        self, prompt: str, completion: str, prepared: PreparedJudge
    ) -> tuple[float, str | None]:
        settings = get_selector_settings()
        self._bound_text(prompt, "prompt", settings.selector_max_text_chars)
        self._bound_text(completion, "completion", settings.selector_max_text_chars)
        queued = self._prefetched.get((prompt, completion))
        if queued:
            # A HIT: judged ahead by `prefetch`, same template and parameters. Consumed, so
            # the same pair later in the batch takes its own reply rather than this one.
            return parse_judge_reply(queued.popleft())
        rendered = render_template(prepared.template, prompt=prompt, completion=completion)
        # Deterministic by default: a judge sampled at temperature makes a run
        # irreproducible and turns R-02.10's hash into a claim the verdicts do not support.
        params = GenerationParams(temperature=0.0, max_tokens=256)
        completions = prepared.provider.generate([rendered], params)
        if not completions:
            raise refuse(
                "judge_provider_unavailable",
                "The judge provider returned no completion for a prompt it was given. "
                "Refusing: a transport failure is not a verdict.",
                base_url=prepared.provider.identity.base_url,
            )
        return parse_judge_reply(completions[0].text)

    @staticmethod
    def _bound_text(text: str, field: str, limit: int) -> None:
        """Refuse an oversized text **with the bound named**, rather than truncating.

        Truncation changes the verdict: a judge shown the first 65,536 characters of a
        longer answer is judging a different answer.
        """
        if len(text) > limit:
            raise refuse(
                "selector_text_too_long",
                f"The {field} is {len(text)} characters, over the {limit}-character bound. "
                f"Refusing rather than truncating: a judge shown a truncated answer is "
                f"judging a different answer.",
                field=field,
                length=len(text),
                limit=limit,
            )

    def _prepared(self) -> PreparedJudge:
        prepared = self._resolved.prepared
        if not isinstance(prepared, PreparedJudge):  # pragma: no cover - resolve() guarantees it
            raise refuse(
                "selector_not_prepared",
                "A judge-model selector was evaluated without being resolved. Resolve the "
                "configuration first; an unprepared judge has no provider and no threshold.",
                selector=self.name,
            )
        return prepared


def _as_identity(run_target: Any) -> ProviderIdentity | None:
    """Coerce a run target into a :class:`ProviderIdentity`, or ``None``.

    Accepts an identity or a mapping, because the run's target comes from F1/F6 and the
    shape is theirs. **Refuses anything else** rather than silently treating an
    unrecognised object as "no target" - which would disable the R-02.11 comparison while
    looking like it ran.
    """
    if run_target is None:
        return None
    if isinstance(run_target, ProviderIdentity):
        return run_target
    if isinstance(run_target, dict):
        return ProviderIdentity.model_validate(run_target)
    raise refuse(
        "judge_run_target_unreadable",
        f"The run's inference target was supplied as {type(run_target).__name__}, which "
        f"this release cannot read, so R-02.11's comparison could not be made. Refusing "
        f"rather than proceeding with the guard disabled.",
        got=type(run_target).__name__,
    )


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text)} chars]"


def factory(resolved: ResolvedSelector) -> JudgeModelSelector:
    return JudgeModelSelector(resolved)
