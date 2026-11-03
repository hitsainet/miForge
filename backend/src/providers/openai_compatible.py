"""The OpenAI-compatible HTTP provider  (ADR-009, R-02.49; F3's module, **F4 reuses it**).

One implementation of :class:`providers.base.InferenceProvider` covers **miLLM, vLLM,
Ollama and the OpenAI API** with one ``base_url``. That is the whole point of R-02.49:
*"a test path wired to one sibling by name is what made the whole document depend on it."*
So **miLLM is one row beside the others, never a named default** - the endpoint is
configuration, and if it happens to point at miLLM that is the operator's choice, not an
import, a default or a name in this code.

## Why this module is F3's and not F4's

R-02.11 forbids the judge being the resident model, and ADR-009 defines the in-process
provider as generating from the parameters miForge holds - so **the judge can only ever use
the HTTP provider.** F3 therefore cannot wait on F4 for it. The ratified split (F4's
FR-004.1, foundation task 3.8): the interface is the foundation's, this HTTP implementation
is F3's, and the registry, the provider identity and the in-process path are F4's.

**F4 reuses this module and does not fork it.** Said here so the second copy never gets
written: two HTTP providers would be two sets of timeout semantics and two answers to
``capabilities()``.

## It is synchronous, because the interface is

:meth:`providers.base.InferenceProvider.generate` is a plain ``def``. That is not an
oversight to work around: every worker in this project runs ``--pool=solo -c 1`` (ADR-006),
and an async provider called from a sync Celery task is how a cross-queue ``.get()`` ends
up deadlocking a worker that will never read its control queue.

## No fabricated reply, ever

A transport failure raises a named ``Refusal``. There is **at most one retry, on a
connection error only, never on a 4xx**, and never a fallback completion. A retry policy
that masked a systematic failure as latency would turn a judge outage into an
accept-everything run one layer up - the single failure this whole feature exists to
prevent.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import httpx

from core.errors import refuse
from providers.base import Completion, GenerationParams, InferenceProvider, ProviderIdentity

#: ADR-009's name for this provider family. Never a sibling's name: the *endpoint* may
#: happen to be miLLM, which belongs in ``base_url``, not in the type system.
PROVIDER_KIND = "openai_compatible"

#: Capability names this provider may declare. **Generation only.** There is no
#: ``embeddings`` row: R-02.49's abstraction is scoped to generation, the selector contract
#: cut embedding similarity for want of a mechanism, and a capability declared here with
#: nothing behind it would be the prose-without-mechanism shape again.
CAPABILITY_GENERATION = "generation"
#: Next-token log-probabilities over an explicit token set, via the legacy
#: ``/v1/completions`` scoring mode (``logprobs`` + ``allowed_token_ids``). vLLM's
#: ``processed_logprobs`` semantics, which miLLM implements (its Feature 25 scoring mode,
#: ``millm/services/next_token_scores.py``): the distribution is renormalised over the set.
#: R-02.2's reconstruction check reads it (F11 FR-011.10: through the provider, never a
#: second private HTTP path).
CAPABILITY_NEXT_TOKEN_LOGPROBS = "next_token_logprobs"
#: OpenAI's (and miLLM's) cap on returned alternatives. A token set larger than this would
#: come back truncated, and the comparison's domain would depend on the model.
MAX_SCORED_TOKENS = 20


class OpenAICompatibleProvider(InferenceProvider):
    """Generate completions over an OpenAI-compatible ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        connect_timeout_s: float,
        read_timeout_s: float,
        concurrency: int = 1,
        client: httpx.Client | None = None,
    ) -> None:
        """Build the provider.

        ``connect_timeout_s`` and ``read_timeout_s`` are **required keyword arguments with
        no defaults**. A missing timeout is how a judge outage becomes a hung step instead
        of a named refusal, and a default here would mean every caller that forgot got the
        hang.

        Args:
            base_url: The endpoint root, e.g. ``http://judge:8000``. ``/v1`` is appended
                when it is not already present.
            model: The model to request. Recorded on every completion this produces.
            api_key: Bearer credential, when the endpoint wants one. **Never logged, never
                hashed, never recorded** (ADR-020).
            concurrency: How many completions of one :meth:`generate` call may be in
                flight at once. **Defaults to 1 — sequential, the behaviour this class had
                before the bound was wired.** The configured value arrives from
                ``HttpProviderRow.concurrency``, which ``GET /api/v1/providers`` already
                published as a fact before anything enforced it.
            client: An injected ``httpx.Client``, for a test against a stub server. One
                client per process, pooled: a client per call pays connection setup per
                completion.
        """
        if not base_url:
            raise refuse(
                "judge_provider_not_configured",
                "An OpenAI-compatible provider needs a base URL. Set the judge provider's "
                "base URL to a reachable endpoint; miForge will not fall back to the "
                "model it is training.",
            )
        if not model:
            raise refuse(
                "judge_provider_not_configured",
                "An OpenAI-compatible provider needs a model name: it is recorded on every "
                "score this provider produces, and a score whose provider is unnamed is "
                "not comparable with any other.",
                base_url=base_url,
            )
        if concurrency < 1:
            raise refuse(
                "judge_provider_not_configured",
                f"A provider's concurrency must be at least 1, not {concurrency}. Refusing "
                f"rather than clamping: a clamped bound is a figure the operator did not "
                f"set and GET /api/v1/providers would publish it as though they had.",
                base_url=base_url,
                concurrency=concurrency,
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._connect_timeout_s = connect_timeout_s
        self._read_timeout_s = read_timeout_s
        self._concurrency = concurrency
        self._client = client
        #: True when this instance must close the client it uses. Decided here, from
        #: whether one was injected, and **never recomputed from ``self._client``** - the
        #: client is created lazily, so ``self._client is None`` stops meaning "not owned"
        #: the moment it is built. Conflating the two is what made :meth:`close`
        #: unreachable in both branches.
        self._owns_client = client is None
        self._client_lock = threading.Lock()

    # ---------------------------------------------------------------- identity
    @property
    def identity(self) -> ProviderIdentity:
        """Who this is, recorded on every score it produces (ADR-009, R-02.16)."""
        return ProviderIdentity(kind=PROVIDER_KIND, model=self._model, base_url=self._base_url)

    @property
    def base_url(self) -> str:
        return self._base_url

    def capabilities(self) -> frozenset[str]:
        """What this provider offers. **Declared, never inferred from a 404.**

        A 404 means "that path is not here", which is not the same statement as "this
        engine cannot do that" - a reverse proxy, a wrong path prefix and an unsupported
        capability all produce one. Guessing between them would be the fabricated value
        ADR-018 forbids, so the set is declared.
        """
        return frozenset({CAPABILITY_GENERATION, CAPABILITY_NEXT_TOKEN_LOGPROBS})

    def available(self) -> bool:
        """Whether this provider can be used right now.

        True once it is constructed with an endpoint and a model: those are the only
        preconditions this implementation has. It deliberately does **not** probe the
        network - a liveness probe in an availability check would make listing the selector
        types depend on a remote host being up, and a slow endpoint would make
        ``GET /api/v1/selectors`` slow.
        """
        return bool(self._base_url and self._model)

    # ---------------------------------------------------------------- scoring
    def next_token_logprobs(
        self,
        prompt: str,
        token_ids: list[int],
        *,
        add_special_tokens: bool = True,
        precision: str | None = None,
    ) -> dict[int, float]:
        """The next token's log-probabilities over EXACTLY ``token_ids`` (R-02.2).

        Deterministic (``temperature`` 0, one token), keyed by token id - the request asks
        for ``return_tokens_as_token_ids`` because a decoded token is ambiguous. The values
        are renormalised over the set by the server, so the caller must compute its own the
        same way. Every requested id must come back; a missing one is refused, not skipped,
        because skipping is how a comparison over an empty intersection reads "matched".
        """
        if not token_ids or len(token_ids) > MAX_SCORED_TOKENS:
            raise refuse(
                "provider_logprobs_token_set_invalid",
                f"A scored token set holds 1..{MAX_SCORED_TOKENS} ids; got {len(token_ids)}. "
                f"Above {MAX_SCORED_TOKENS} the endpoint truncates the alternatives.",
                size=len(token_ids),
            )
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": len(token_ids),
            "allowed_token_ids": [int(i) for i in token_ids],
            "return_tokens_as_token_ids": True,
            "add_special_tokens": add_special_tokens,
            "stream": False,
        }
        if precision is not None:
            # F11 14.3b: ask the endpoint to score in this precision. NOT trusted unless
            # echoed: miLLM accepts unknown fields silently (`extra="allow"`), so a server
            # that ignores the request answers in its native dtype and looks identical.
            payload["scoring_precision"] = precision
        data = self._post(
            "/v1/completions",
            payload,
            on_error=None if precision is None else _scoring_precision_refusal(precision),
        )
        if precision is not None and data.get("scoring_precision") != precision:
            raise refuse(
                "provider_scoring_precision_unconfirmed",
                f"The endpoint was asked to score in {precision} and did not confirm it "
                f"(scoring_precision={data.get('scoring_precision')!r}). Its answer may be in "
                f"its native dtype, and comparing that against a {precision} forward would "
                f"measure the precision difference, not the reconstruction. Refusing rather "
                f"than comparing. The serving side needs a {precision} scoring mode; "
                f"until then use the native verification precision.",
                base_url=self._base_url,
                requested=precision,
                echoed=data.get("scoring_precision"),
            )
        try:
            top = data["choices"][0]["logprobs"]["top_logprobs"][0]
            parsed = {
                int(str(key).removeprefix("token_id:")): float(value) for key, value in top.items()
            }
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise refuse(
                "provider_logprobs_unavailable",
                "The endpoint answered without next-token log-probabilities in the "
                "/v1/completions scoring shape (choices[0].logprobs.top_logprobs[0], keyed "
                "token_id:<id>). It may not implement scoring mode.",
                base_url=self._base_url,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        missing = sorted({int(i) for i in token_ids} - set(parsed))
        if missing:
            raise refuse(
                "provider_logprobs_incomplete",
                f"The endpoint returned no log-probability for token ids {missing} of the "
                f"requested set; a comparison over a partial set is refused, not narrowed.",
                missing=missing,
            )
        return {int(i): parsed[int(i)] for i in token_ids}

    # ---------------------------------------------------------------- generation
    def generate(
        self, prompts: list[str], params: GenerationParams, **options: Any
    ) -> list[Completion]:
        """One completion per prompt, **in input order** (the interface's contract).

        A provider that reordered or dropped one would silently misattribute every score
        after the gap, so this builds the result list positionally and refuses rather than
        returning a short list.

        Raises:
            Refusal: ``judge_provider_unavailable`` for any transport failure, timeout or
                non-2xx status; ``judge_reply_unparseable`` for a body that is not an
                OpenAI-compatible completion. **No fabricated completion in either case.**
        """
        system = options.pop("system", None)
        if options:
            # Refused rather than ignored: a silently-dropped option is a request the
            # caller believes it made.
            raise refuse(
                "judge_provider_option_unknown",
                f"Unknown generation options {sorted(options)} were passed to the "
                f"OpenAI-compatible provider. Refusing rather than ignoring them: a "
                f"silently dropped option is a request the caller believes it made.",
                unknown=sorted(options),
            )
        if not prompts:
            return []
        workers = min(self._concurrency, len(prompts))
        if workers <= 1:
            return [self._generate_one(prompt, params, system=system) for prompt in prompts]
        return self._generate_concurrently(prompts, params, system=system, workers=workers)

    def _generate_concurrently(
        self, prompts: list[str], params: GenerationParams, *, system: str | None, workers: int
    ) -> list[Completion]:
        """Fan out over ``workers`` threads, **returning results in input order**.

        Threads rather than ``asyncio`` because the interface is synchronous and every
        worker in this project is ``--pool=solo -c 1`` (ADR-006): an event loop started
        inside a Celery task is how a worker stops reading its control queue. ``httpx``
        clients are thread-safe and the pooled client is shared, which is the whole reason
        the connection setup is paid once rather than per completion.

        **The first failure by prompt index is the one raised**, not whichever thread
        happened to fail first. A refusal that varied run to run would make the same broken
        endpoint report a different prompt each time. Pending work is cancelled as soon as
        a failure is seen, so a dead endpoint still fails fast instead of retrying once per
        prompt.
        """
        # Built on THIS thread, before any fan-out: a lazily-created client raced by four
        # workers would open four.
        self._ensure_client()
        futures: list[Future[Completion]] = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mf-provider") as pool:
            for prompt in prompts:
                futures.append(pool.submit(self._generate_one, prompt, params, system=system))
            completions: list[Completion] = []
            for index, future in enumerate(futures):
                try:
                    completions.append(future.result())
                except BaseException:
                    for pending in futures[index + 1 :]:
                        pending.cancel()
                    raise
        return completions

    def _generate_one(
        self, prompt: str, params: GenerationParams, *, system: str | None
    ) -> Completion:
        payload = self._request_body(prompt, params, system=system)
        data = self._post("/v1/chat/completions", payload)
        text = self._text_from(data)
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            provider=self.identity,
            params=params,
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=_int_or_none(usage.get("completion_tokens")),
        )

    def _request_body(
        self, prompt: str, params: GenerationParams, *, system: str | None
    ) -> dict[str, Any]:
        """The wire body. A named method so a test can assert it **field by field**.

        ADR-016: *"was called" passes against a call sending the wrong arguments.* Building
        the body here rather than inline in the request call is what makes the payload
        assertion possible without a live endpoint.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": False,
        }
        if params.stop:
            body["stop"] = list(params.stop)
        if params.seed is not None:
            # Sent only when set. `seed: null` is accepted by some engines and rejected by
            # others, and sending a null the caller did not ask for is a request they did
            # not make.
            body["seed"] = params.seed
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        on_error: Callable[[httpx.Response], None] | None = None,
    ) -> dict[str, Any]:
        """POST with one connection-error retry and **no fabricated reply**.

        ``on_error`` sees a >= 400 response first and may raise a more specific refusal (the
        float32 scoring refusal, which its caller handles differently from a dead endpoint).
        """
        url = f"{self._base_url}{path}"
        response = self._send_with_one_retry(self._ensure_client(), url, payload)

        if response.status_code >= 400 and on_error is not None:
            on_error(response)
        if response.status_code >= 400:
            # A 4xx is NOT retried: it is a statement about the request, and retrying it
            # would turn a wrong model name into a slow failure instead of a fast one.
            raise refuse(
                "judge_provider_unavailable",
                f"The judge endpoint answered HTTP {response.status_code}. Refusing the "
                f"evaluation: a transport failure is not a verdict, and defaulting it to "
                f"accept would turn this into an accept-everything selector. Check the "
                f"endpoint and the model name.",
                status=response.status_code,
                base_url=self._base_url,
                model=self._model,
                body=_truncate(response.text, 500),
            )
        try:
            parsed = response.json()
        except ValueError as exc:
            raise refuse(
                "judge_reply_unparseable",
                f"The judge endpoint's reply is not JSON: {exc}. No verdict is defaulted.",
                base_url=self._base_url,
                body=_truncate(response.text, 500),
            ) from exc
        if not isinstance(parsed, dict):
            raise refuse(
                "judge_reply_unparseable",
                "The judge endpoint's reply is JSON but not an object, so it is not an "
                "OpenAI-compatible completion. No verdict is defaulted.",
                base_url=self._base_url,
                parsed_type=type(parsed).__name__,
            )
        return parsed

    def _send_with_one_retry(
        self, client: httpx.Client, url: str, payload: dict[str, Any]
    ) -> httpx.Response:
        """Send, retrying **once** on a connection error only.

        One retry, because a single dropped connection is common and a systematic failure
        must not be smoothed into latency. A timeout is **not** retried: the read timeout
        has already elapsed, and retrying doubles the time the caller waits before learning
        the endpoint is unhealthy.
        """
        last: Exception | None = None
        for attempt in (1, 2):
            try:
                return client.post(url, json=payload, headers=self._headers())
            except httpx.TimeoutException as exc:
                raise refuse(
                    "judge_provider_unavailable",
                    f"The judge endpoint did not answer within "
                    f"{self._read_timeout_s:g}s (connect {self._connect_timeout_s:g}s). "
                    f"Refusing the evaluation rather than defaulting a verdict.",
                    base_url=self._base_url,
                    read_timeout_s=self._read_timeout_s,
                ) from exc
            except httpx.TransportError as exc:
                last = exc
                if attempt == 2:
                    break
        raise refuse(
            "judge_provider_unavailable",
            f"The judge endpoint at {self._base_url} is unreachable after one retry: "
            f"{last}. Refusing the evaluation; no verdict is defaulted.",
            base_url=self._base_url,
            detail=str(last),
        ) from last

    @staticmethod
    def _text_from(data: dict[str, Any]) -> str:
        """Pull the completion text, refusing a reply that does not carry one."""
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise refuse(
                "judge_reply_unparseable",
                "The judge endpoint's reply carries no 'choices', so there is no "
                "completion to read. No verdict is defaulted.",
                keys=sorted(data),
            )
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if content is None and isinstance(first, dict):
            content = first.get("text")  # the legacy /v1/completions shape
        if not isinstance(content, str):
            raise refuse(
                "judge_reply_unparseable",
                "The judge endpoint's first choice carries no text content. No verdict is "
                "defaulted.",
                choice_keys=sorted(first) if isinstance(first, dict) else None,
            )
        return content

    # ---------------------------------------------------------------- the client
    def _ensure_client(self) -> httpx.Client:
        """The pooled client, built once per instance on first use.

        **One client per provider, not one per completion.** The old code opened a fresh
        ``httpx.Client`` inside every ``_post``, so a judged batch paid connection setup
        per candidate: measured against miLLM over a same-cluster pod-to-pod link,
        66.6 ms median per completion with a client per call against 59.3 ms with one
        pooled client - 7.3 ms, 11.0 % of the round trip, 20 reps each. On that link it is
        the cheapest hop available; over TLS to a remote judge the handshake makes it far
        worse, which is exactly the topology an operator reaches for when they want a
        bigger judge.

        Lazy rather than built in ``__init__`` so constructing a provider still costs
        nothing - ``GET /api/v1/providers`` builds one per configured row to report
        availability and must not open sockets to do it.
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = httpx.Client(
                        timeout=httpx.Timeout(
                            connect=self._connect_timeout_s,
                            read=self._read_timeout_s,
                            write=self._read_timeout_s,
                            pool=self._connect_timeout_s,
                        )
                    )
        return self._client

    def close(self) -> None:
        """Close the client **this instance owns**. A no-op when one was injected.

        Reachable in the owning case, which it was not before: the guard read
        ``self._client is not None and self._owns_client`` while ``_owns_client`` was
        ``client is None``, so the two conditions could never both hold - an injected
        client made ``_owns_client`` False and an owned one made ``_client`` None. It read
        as resource management and closed nothing in either branch. The client is now
        created lazily, so an owned one is non-``None`` once it has been used.
        """
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None


def _int_or_none(value: Any) -> int | None:
    """``None`` rather than ``0`` for an absent count: zero is a measurement, absence is not."""
    return int(value) if isinstance(value, (int, float)) else None


def _truncate(text: str, limit: int) -> str:
    """Bounded error detail. A judge reply can be long; a refusal is not a transcript."""
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text)} chars]"


#: The serving side's float32 scoring refusal, in the OpenAI error envelope: HTTP 409 with
#: ``error.code`` ``scoring_precision_unavailable`` (the vendored scoring contract,
#: ``docs/schemas/millm-scoring-api-v1.json`` -> ``scoring_precision.refusal``). The reason
#: travels in a response header the envelope has no field for; this generic provider does not
#: name any engine's header, so it hands back every ``*scoring-precision*`` header and the
#: caller that knows the engine reads its own (``training.hotupdate.verify``).
SCORING_PRECISION_REFUSAL_STATUS = 409
SCORING_PRECISION_REFUSAL_CODE = "scoring_precision_unavailable"


def _scoring_precision_refusal(precision: str) -> Callable[[httpx.Response], None]:
    """Turn a 409 ``scoring_precision_unavailable`` into its own refusal, by name.

    The serving side said it CANNOT score in ``precision`` (and answered nothing) - a different
    fact from an endpoint that is down, and the one fact a caller may act on by choosing another
    precision. Every other error response falls through to the generic refusal.
    """

    def check(response: httpx.Response) -> None:
        if response.status_code != SCORING_PRECISION_REFUSAL_STATUS:
            return
        try:
            error = response.json().get("error") or {}
        except (ValueError, AttributeError):
            return
        if not isinstance(error, dict) or error.get("code") != SCORING_PRECISION_REFUSAL_CODE:
            return
        message = str(error.get("message") or "")
        found = re.search(r"\(reason: ([a-z0-9_]+)\)", message)
        raise refuse(
            "provider_scoring_precision_unavailable",
            f"The endpoint refused to score in {precision} and answered nothing: "
            f"{_truncate(message, 300)}",
            requested=precision,
            reason=found.group(1) if found else None,
            status=response.status_code,
            response_headers={
                key.lower(): value
                for key, value in response.headers.items()
                if "scoring-precision" in key.lower()
            },
        )

    return check
