"""Stub servers and builders for F3's tests  (ADR-016; FTID I-7).

**Stub servers, not mocks**, for both HTTP surfaces. A mock never runs the real request
builder, so a serialisation bug, a wrong header, a missing field or a mishandled status
passes it; ADR-016 names over-mocked stubs among the reasons a suite stays green over a real
bug. These are ``httpx.MockTransport`` handlers, which means the real
:class:`providers.openai_compatible.OpenAICompatibleProvider` builds the real request and
parses the real response - only the socket is replaced.

**Builders go through the real resolver.** A hand-built ``ResolvedSelector`` has fields that
agree by construction, which is the recorded reason suites stay green over real bugs.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from providers.openai_compatible import OpenAICompatibleProvider
from services.selectors.resolve import ResolvedSelector, resolve
from services.selectors.review_port import ReviewCandidate, ReviewDecision, ReviewItem

# --------------------------------------------------------------------------------------
# The OpenAI-compatible judge endpoint
# --------------------------------------------------------------------------------------


class StubJudgeServer:
    """An OpenAI-compatible ``/v1/chat/completions`` stub that records what it was sent.

    ``calls`` is the whole point: ADR-016 requires the **payload and the call count** to be
    asserted, because *"was called" passes against a call sending the wrong template* - and
    the template is what the judge actually judges.
    """

    def __init__(
        self,
        *,
        reply: str = "SCORE: 0.80\nREASON: coherent and on topic",
        status: int = 200,
        body: Any | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.status = status
        self.body = body
        self.raise_error = raise_error
        self.calls: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.headers.append(dict(request.headers))
        self.calls.append(json.loads(request.content.decode()))
        if self.raise_error is not None:
            raise self.raise_error
        if self.body is not None:
            return httpx.Response(self.status, json=self.body)
        if self.status >= 400:
            return httpx.Response(self.status, text="upstream says no")
        return httpx.Response(
            self.status,
            json={
                "id": "cmpl-stub",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": self.reply}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def provider(
        self, *, base_url: str = "http://judge.test:8000", model: str = "judge-7b"
    ) -> OpenAICompatibleProvider:
        """A **real** provider wired to this stub through ``httpx.MockTransport``."""
        return OpenAICompatibleProvider(
            base_url=base_url,
            model=model,
            connect_timeout_s=1.0,
            read_timeout_s=2.0,
            client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )


class StubProviderResolver:
    """A :class:`services.selectors.provider_port.ProviderResolver` over one provider.

    Stands in for F4's registry, which does not exist yet. F3 resolves from a registry and
    constructs nothing, so a test must supply the registry rather than the provider.
    """

    def __init__(self, provider: Any, *, key: str = "judge", available: bool = True) -> None:
        self._provider = provider
        self._key = key
        self._available = available
        self.resolve_calls: list[str] = []

    def available(self, key: str) -> bool:
        return self._available and key == self._key

    def resolve(self, key: str) -> Any:
        from core.errors import refuse

        self.resolve_calls.append(key)
        if key != self._key:
            raise refuse("judge_provider_not_configured", f"no provider for {key!r}")
        return self._provider


# --------------------------------------------------------------------------------------
# miDataworks' review queue
# --------------------------------------------------------------------------------------


class StubReviewQueue:
    """A :class:`services.selectors.review_port.ReviewQueuePort` with the shipped shapes.

    Records every call so the **outbound payload can be asserted field by field with the
    call count** - the one data-egress path in this feature, where prompt and completion
    text leaves the process. *A privacy guarantee verified by reading is verified by
    nothing.*
    """

    def __init__(
        self,
        *,
        available: bool = True,
        decisions: dict[str, tuple[str, str | None]] | None = None,
        unreachable: bool = False,
    ) -> None:
        self._available = available
        self._unreachable = unreachable
        #: ``external_id -> (decision kind, label)``
        self._decisions = decisions or {}
        self.ensure_calls: list[dict[str, Any]] = []
        self.posted: list[list[ReviewCandidate]] = []
        self.read_calls: list[list[str] | None] = []
        #: Every ``external_id`` ever posted, so an idempotent repost is observable.
        self.items: dict[str, str] = {}

    # -- the port
    def available(self) -> bool:
        return self._available

    def ensure_queue(self, *, name: str, question: str, label_set: list[str]) -> str:
        self._guard()
        self.ensure_calls.append({"name": name, "question": question, "label_set": label_set})
        return f"q-{name}"

    def post_candidates(self, queue_id: str, candidates: list[ReviewCandidate]) -> list[ReviewItem]:
        self._guard()
        if len(candidates) > 500:
            raise AssertionError(
                f"{len(candidates)} candidates in one post; miDataworks caps it at 500"
            )
        self.posted.append(list(candidates))
        result: list[ReviewItem] = []
        for candidate in candidates:
            # Idempotent per external_id: a repeated id returns its EXISTING item, which is
            # miDataworks' verified behaviour and what makes a retried post safe.
            item_id = self.items.setdefault(candidate.external_id, f"rv_{len(self.items):04x}")
            result.append(ReviewItem(external_id=candidate.external_id, item_id=item_id))
        return result

    def read_decisions(
        self, queue_id: str, *, external_ids: list[str] | None = None
    ) -> list[ReviewDecision]:
        self._guard()
        self.read_calls.append(list(external_ids) if external_ids else None)
        wanted = set(external_ids or self._decisions)
        return [
            ReviewDecision(
                external_id=external_id,
                decision_id=f"d-{external_id}",
                decision=kind,
                label=label,
            )
            for external_id, (kind, label) in self._decisions.items()
            if external_id in wanted
        ]

    # -- helpers
    def _guard(self) -> None:
        from core.errors import refuse

        if self._unreachable:
            raise refuse(
                "midataworks_unreachable",
                "the stub review API is configured but unreachable",
            )

    @property
    def post_count(self) -> int:
        return len(self.posted)

    @property
    def all_posted(self) -> list[ReviewCandidate]:
        return [candidate for batch in self.posted for candidate in batch]

    def decide(self, external_id: str, decision: str, label: str | None = None) -> None:
        self._decisions[external_id] = (decision, label)


# --------------------------------------------------------------------------------------
# Configuration builders - through the REAL resolver
# --------------------------------------------------------------------------------------


def rule_config(name: str = "house-rules", **checks: Any) -> dict[str, Any]:
    """A ``rule`` configuration whose ``expression`` is the JSON rule body F3 defines."""
    return {
        "kind": "rule",
        "name": name,
        "config_sha256": None,
        "expression": json.dumps(checks or {"min_length": 1}),
    }


def judge_config(
    name: str = "judge", *, threshold: float | None = 0.6, provider: str = "judge"
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "kind": "judge_model",
        "name": name,
        "config_sha256": None,
        "provider": provider,
    }
    if threshold is not None:
        config["threshold"] = threshold
    return config


def reference_config(
    name: str = "ref", *, mode: str = "fuzzy", threshold: float | None = 0.6
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "kind": "reference_match",
        "name": name,
        "config_sha256": None,
        "mode": mode,
    }
    if threshold is not None:
        config["threshold"] = threshold
    return config


def human_config(
    name: str = "human", *, queue: str = "legal-review", timeout_seconds: int | None = None
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "kind": "human_review",
        "name": name,
        "config_sha256": None,
        "queue": queue,
    }
    if timeout_seconds is not None:
        config["timeout_seconds"] = timeout_seconds
    return config


def composite_config(name: str, children: list[str], combine: str = "all") -> dict[str, Any]:
    return {
        "kind": "composite",
        "name": name,
        "config_sha256": None,
        "children": children,
        "combine": combine,
    }


def resolved(
    config: dict[str, Any],
    *,
    catalog: list[dict[str, Any]] | None = None,
    source: str = "inline",
    **kwargs: Any,
) -> ResolvedSelector:
    """Resolve through the **real** resolver. Never hand-build a ``ResolvedSelector``.

    A hand-built one has a hash that agrees with its body by construction, a capability
    that agrees with its kind by construction, and a ``prepared`` that never ran - which is
    three of the four ways ADR-016 says a suite stays green over a real bug.
    """
    by_name = {body["name"]: body for body in (catalog or [])}
    return resolve(config, catalog=by_name, source=source, **kwargs)


def handoff_version(
    sha256: str,
    *,
    name: str = "house-rules",
    version: int = 3,
    kind: str = "selector_config",
) -> dict[str, Any]:
    """A ``midataworks.dataset-version/v1``-shaped version carrying a config hash.

    Only the fields F3 reads: the hash it copies and the reference it records beside it.
    Shaped from the vendored schema rather than imagined, which is the check this project
    keeps finding it needed.
    """
    return {
        "target": {"kind": kind, "name": name, "version": version},
        "content": {
            "content_kind": "config",
            "config": {
                "kind": "selector",
                "name": name,
                "number": version,
                "plugin": "rule",
                "body_format": "json",
                "sha256": sha256,
                "path": f"configs/{name}.json",
            },
        },
    }
