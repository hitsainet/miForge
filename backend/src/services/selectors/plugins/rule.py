"""The rule selector  (FR-003.4; R-02.9).

Pure, in-process, no model, no network. **The only type guaranteed available on the
standalone path with nothing configured** (AC20), which is why it is also the one every
other test leans on.

## The ``expression`` field, and why it holds JSON

``selector-config-v1.json`` gives this type a single ``expression`` string described as
*"the rule body, interpreted by F3. Opaque to this contract."* So the interpretation is
this module's to define, and it is defined as **a JSON object**, not as a mini-language:

* **Not evaluated code.** The TDD's security section is explicit - *configuration bodies
  are data, never code* - so no ``eval``, no expression compiler, nothing that turns an
  authored string into behaviour beyond the checks enumerated below.
* **It keeps FR-003.4's five check kinds.** Regular expressions, length bounds, required
  terms, forbidden terms and structure. A one-line boolean language would have had to grow
  all five anyway, and each would then have had two spellings.
* **It is refused when it does not parse**, naming the problem, rather than being treated
  as a literal substring to look for - which is the guess ADR-018 forbids.

## Evaluation order is fixed, so the reason is stable

forbidden terms -> required terms -> length -> patterns -> structure. The reason names the
**first** thing that fired, and because the order is fixed the same completion gives the
same reason every time. A reason that depends on dict ordering is a reason a UI cannot map
and an operator cannot trust.

**The reason names the term or pattern that fired** - ``forbidden term "guarantee"`` - not
``rule failed``. A verdict whose reason is the word "failed" is indistinguishable from a
selector that is not really deciding.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.errors import refuse
from schemas.selector import SelectorKind, Verdict, VerdictValue
from services.selectors.resolve import ResolvedSelector
from services.selectors.settings import get_selector_settings

#: Constructs that make a regex catastrophically slow on an adversarial input: a quantifier
#: applied to a group that is itself quantified. Checked at ``prepare`` and **refused
#: there**, not run against every completion - a pathological pattern that stalls a batch
#: is a configuration error, and a configuration error belongs at configuration time.
_NESTED_QUANTIFIER = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*{]")


class PatternRule(BaseModel):
    """One regular expression and whether it must match."""

    model_config = ConfigDict(extra="forbid")

    regex: str = Field(min_length=1)
    #: ``True`` requires a match, ``False`` requires its absence. **No default**: "regex"
    #: alone does not say which, and guessing picks a verdict policy.
    must_match: bool


class StructureRule(BaseModel):
    """Structural checks over the whole completion."""

    model_config = ConfigDict(extra="forbid")

    #: The completion must parse as JSON. Useful for a domain whose answers are structured,
    #: and the only structure check with an unambiguous meaning today.
    json_parsable: bool = False
    #: Minimum number of non-empty lines.
    min_lines: int | None = Field(default=None, ge=1)


class RuleBody(BaseModel):
    """The parsed ``expression``. ``extra="forbid"`` so a misspelled key is a refusal.

    A silently-ignored key in a *selector* configuration is worse than elsewhere: the
    operator believes a forbidden-term list is in force, every completion is admitted, and
    the run drifts toward whatever the model already produces. That is the first of the two
    silent failures this whole feature exists to make visible.
    """

    model_config = ConfigDict(extra="forbid")

    forbidden_terms: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=1)
    patterns: tuple[PatternRule, ...] = ()
    structure: StructureRule | None = None
    #: Explicit rather than assumed. Case-sensitive matching means a term list means
    #: exactly what it says; an operator who wants otherwise says so.
    case_sensitive: bool = True

    @model_validator(mode="after")
    def _at_least_one_check(self) -> RuleBody:
        """A rule with no checks accepts everything, which is not selecting.

        Refused at the type, because the failure mode is the headline one: *"a selector
        that accepts everything is not selecting"*. An empty rule body would produce a
        green run, a falling loss and a test score that never moves.
        """
        if not any(
            (
                self.forbidden_terms,
                self.required_terms,
                self.min_length is not None,
                self.max_length is not None,
                self.patterns,
                self.structure is not None,
            )
        ):
            raise ValueError(
                "a rule selector with no checks accepts every completion, which is not "
                "selecting; give it at least one of forbidden_terms, required_terms, "
                "min_length, max_length, patterns or structure"
            )
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError(
                f"min_length {self.min_length} exceeds max_length {self.max_length}, "
                f"so no completion can satisfy this rule"
            )
        return self


class PreparedRule:
    """What ``prepare`` produced: the parsed body and the compiled patterns."""

    def __init__(self, body: RuleBody, compiled: list[tuple[PatternRule, re.Pattern[str]]]):
        self.body = body
        self.compiled = compiled


def compile_bounded_regex(pattern: str, *, max_chars: int) -> re.Pattern[str]:
    """Compile ``pattern``, refusing a pathological one. Called at ``prepare`` only.

    Raises:
        Refusal: ``rule_pattern_invalid`` for a pattern that is too long, uses a nested
            quantifier, or does not compile. All three are configuration errors, and
            refusing once beats timing out once per completion.
    """
    if len(pattern) > max_chars:
        raise refuse(
            "rule_pattern_invalid",
            f"A rule pattern of {len(pattern)} characters exceeds the {max_chars}-character "
            f"ceiling. Split it into several patterns, or raise "
            f"RULE_MAX_PATTERN_CHARS deliberately.",
            pattern_chars=len(pattern),
            limit=max_chars,
        )
    if _NESTED_QUANTIFIER.search(pattern):
        raise refuse(
            "rule_pattern_invalid",
            "A rule pattern applies a quantifier to an already-quantified group, which can "
            "take exponential time on an adversarial completion and would stall a batch. "
            "Rewrite the pattern without the nested quantifier.",
            pattern=pattern,
            construct="nested quantifier",
        )
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise refuse(
            "rule_pattern_invalid",
            f"A rule pattern does not compile: {exc}. Correct the pattern; miForge will "
            f"not interpret an invalid regular expression as a literal string.",
            pattern=pattern,
            detail=str(exc),
        ) from exc


def parse_expression(expression: str) -> RuleBody:
    """Parse the ``expression`` field into a :class:`RuleBody`. **Refuses, never guesses.**"""
    try:
        raw = json.loads(expression)
    except json.JSONDecodeError as exc:
        raise refuse(
            "rule_expression_invalid",
            f"A rule selector's 'expression' must be a JSON object describing the checks "
            f"(forbidden_terms, required_terms, min_length, max_length, patterns, "
            f"structure); this one does not parse as JSON: {exc}. Refusing rather than "
            f"treating it as a literal substring, which would silently change every "
            f"verdict.",
            detail=str(exc),
        ) from exc
    if not isinstance(raw, dict):
        raise refuse(
            "rule_expression_invalid",
            "A rule selector's 'expression' must be a JSON **object**; a bare "
            f"{type(raw).__name__} does not say which check it is.",
            parsed_type=type(raw).__name__,
        )
    try:
        return RuleBody.model_validate(raw)
    except ValueError as exc:
        raise refuse(
            "rule_expression_invalid",
            f"A rule selector's 'expression' is not a valid rule body: {exc}",
            detail=str(exc),
        ) from exc


class RuleSelector:
    """Satisfies :class:`schemas.selector.Selector`. Deterministic by construction."""

    kind: SelectorKind = "rule"

    def __init__(self, resolved: ResolvedSelector) -> None:
        self._resolved = resolved

    @property
    def name(self) -> str:
        return self._resolved.name

    def prepare(self, resolved: ResolvedSelector, *, run_target: Any = None) -> None:
        """Parse the expression and compile every pattern, **once**."""
        del run_target  # a rule needs no model, which is why it is always available
        body = parse_expression(str(resolved.body["expression"]))
        limit = get_selector_settings().rule_max_pattern_chars
        compiled = [
            (rule, compile_bounded_regex(rule.regex, max_chars=limit)) for rule in body.patterns
        ]
        resolved.prepared = PreparedRule(body, compiled)

    def evaluate(self, *, prompt: str, completion: str, context: dict[str, Any]) -> Verdict:
        """One verdict, with a reason naming the first check that fired."""
        del prompt, context  # a rule is a function of the completion alone
        prepared = self._resolved.prepared
        assert isinstance(prepared, PreparedRule)  # prepare() ran; resolve() guarantees it
        body = prepared.body
        haystack = completion if body.case_sensitive else completion.casefold()

        def needle(term: str) -> str:
            return term if body.case_sensitive else term.casefold()

        for term in body.forbidden_terms:
            if needle(term) in haystack:
                return self._reject("rule_forbidden_term", f'forbidden term "{term}"')

        for term in body.required_terms:
            if needle(term) not in haystack:
                return self._reject("rule_required_term_missing", f'missing required term "{term}"')

        length = len(completion)
        if body.min_length is not None and length < body.min_length:
            return self._reject("rule_too_short", f"{length} characters, minimum {body.min_length}")
        if body.max_length is not None and length > body.max_length:
            return self._reject("rule_too_long", f"{length} characters, maximum {body.max_length}")

        for rule, pattern in prepared.compiled:
            matched = pattern.search(completion) is not None
            if rule.must_match and not matched:
                return self._reject(
                    "rule_pattern_not_matched", f'no match for pattern "{rule.regex}"'
                )
            if not rule.must_match and matched:
                return self._reject(
                    "rule_pattern_matched", f'matched forbidden pattern "{rule.regex}"'
                )

        if body.structure is not None:
            refusal = self._check_structure(body.structure, completion)
            if refusal is not None:
                return refusal

        return Verdict(
            value=VerdictValue.ACCEPT,
            selector=self.name,
            label="rule_passed",
            reason="every configured rule check passed",
            score=1.0,
        )

    def _check_structure(self, structure: StructureRule, completion: str) -> Verdict | None:
        if structure.json_parsable:
            try:
                json.loads(completion)
            except json.JSONDecodeError as exc:
                return self._reject(
                    "rule_structure_invalid", f"completion does not parse as JSON: {exc.msg}"
                )
        if structure.min_lines is not None:
            lines = [line for line in completion.splitlines() if line.strip()]
            if len(lines) < structure.min_lines:
                return self._reject(
                    "rule_structure_invalid",
                    f"{len(lines)} non-empty lines, minimum {structure.min_lines}",
                )
        return None

    def _reject(self, code: str, reason: str) -> Verdict:
        return Verdict(
            value=VerdictValue.REJECT, selector=self.name, label=code, reason=reason, score=0.0
        )


def factory(resolved: ResolvedSelector) -> RuleSelector:
    return RuleSelector(resolved)
