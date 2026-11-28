"""The served window: three readers, and the one resolution a push uses (feature 019; FR-019.1,
FR-019.2, FR-019.15; F19-D10, F19-D11, F19-D12, F19-D30; TID §3.2).

**The readers** (confirmed from each project's source, ``0xcc/reviews/019_window_readers.md``):

- ``serving: miserver``: ``wsp.tasks.read_served_window``, miServer's ``GET /api/health/inference``
  ``context_length``, unchanged from 018 (FR-019.22).
- ``serving: generic``: ``GET <origin>/v1/models``, the entry whose ``id`` is the endpoint's
  ``model``, field ``max_model_len`` (vLLM, SGLang; a LoRA card carries ``null``). When that gives
  nothing, ``GET <origin>/props?model=<model>``, ``default_generation_settings.n_ctx`` (llama.cpp's
  PER-SLOT context, the figure one request may use; a router proxies the named model's, and a
  single-model server ignores the query; a router asked without it answers 0, nothing). Never a
  total divided by ``total_slots``, and never ``n_ctx_train`` or ``context_length`` (training
  lengths).
- A hosted (external) endpoint is never read in 019 (FR-019.4): no request is made.

Every read is 5 s, follows no redirect and ignores the proxy environment. It sends no key (the
backend cannot open one), so a keyed server reads as nothing. A read never raises to its caller:
any failure is logged once (``wsp_window_unread``) and gives ``None``. Only a positive ``int``
counts; a ``bool``, zero, a negative, a float, a string or a missing field is nothing.

**Answered or not** (operator decision D1, 2026-11-27): :func:`read_window_answer` also says when a
``generic`` server gave NO answer at all (:data:`NOT_ANSWERED`): every request it made failed in
transport (connection refused, a timeout, a reset) or got a 5xx. Any other status (a keyed
server's 401, a 404, a redirect) is an answer that states no window. :func:`read_window` (the
registration read) folds the two into ``None``.

**The resolution** (:func:`resolve`, pure): ``stated`` (the endpoint's ``context_window_tokens``)
wins over ``reported`` (this push's read), which wins over ``assumed`` (8,192). The ceiling is the
endpoint's, else 32,000, lowered to the REPORTED window when that is smaller (F19-D12); a stated
window never lowers it (registration and PATCH already bound it). A run never uses the
registration read (F19-D11).

**A miServer endpoint is never assumed** (2026-11-27, run ``wsp_4fbf5a18758e``). miServer omits
``context_length`` from ``GET /api/health/inference`` while no model is resident, so for a miServer
endpoint that states no window an empty read means "not loaded yet", not "this server does not
say": that run was pushed 18 s before miServer finished loading a 126,976-token model, resolved
8,192 ``assumed`` and failed ``editor_context_window_too_small`` with no editor request sent. Such a
run (one that calls the editor: :func:`awaits_resident`) now WAITS queued: the driver's claim reads
the window before it consumes anything, skips the run while the read is empty and fails it
``editor_not_resident`` after ``wsp_editor_resident_wait_s`` (``wsp.dispatch``); a push whose own
read comes back empty (the model went away between the claim and the push) puts the run back in
the queue. :func:`resolve` refuses (:class:`NotResidentError`) rather than guess, so no path can
reach the assumed window for one. A ``generic`` server that ANSWERS without a window stays
``assumed`` with its warning (F19-D10, F19-D11): a keyed server, a LoRA card and a llama.cpp
router all answer without a window while serving. One that gives NO answer at all (connection
refused, a timeout, a 5xx) waits on the same clock and fails the same way (operator decision D1,
2026-11-27; :func:`waits`). A hosted endpoint must state its window (``endpoints.run_refusal``),
so it never reaches the fallback.

**No answer backs off** (debt D5, 2026-11-28). While a generic server gives no answer, the claim
would read it again at every pass (every 5 s) and log ``wsp_window_unread`` each time, and a
black-holed host (no refusal, only a timeout) would hold each pass for up to ``READ_TIMEOUT_S``.
:class:`NoAnswerBackoff` (the claim's only, on the pass's clock) holds an endpoint that gave no
answer for 10 s, doubling at each further no answer to at most 60 s; while held, the claim takes
the last answer (:data:`NOT_ANSWERED`) without a request, so the run still waits and still fails
``editor_not_resident`` on time. Any answer clears the hold, so a recovered server is read again
at most 60 s after it came back. The push's read and the registration read are never held: each
is one live read.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from wsp_worker import editor_limits

if TYPE_CHECKING:
    import httpx

    from wsp.models import WspEndpoint

log = logging.getLogger(__name__)

READ_TIMEOUT_S = 5.0
MODELS_PATH = "/v1/models"
PROPS_PATH = "/props"

#: The two warnings (FR-019.2, F19-D10, F19-D11): their codes are the transcript's, the driver
#: log's and the list rows' (TDD §5.5).
WINDOW_ASSUMED = "window_assumed"
WINDOW_STATED_ABOVE_REPORTED = "window_stated_above_reported"

WindowSource = Literal["stated", "reported", "assumed"]


@dataclass(frozen=True)
class Resolved:
    """The window and the ceiling one push uses, and where they came from."""

    window_tokens: int
    window_source: WindowSource
    #: The window the server reported at this push, stored in ``served_context_tokens``.
    reported: int | None
    #: The ceiling after F19-D12's lowering.
    ceiling: int
    ceiling_lowered: bool
    warnings: tuple[str, ...]


class NotAnswered:
    """The read's answer when a generic server answered none of its requests (D1)."""

    _instance: NotAnswered | None = None

    def __new__(cls) -> NotAnswered:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NOT_ANSWERED"


#: :class:`NotAnswered`'s one value.
NOT_ANSWERED = NotAnswered()
#: What a push's (or a claim's) read gives: a window, nothing stated, or no answer at all.
WindowAnswer = int | None | NotAnswered


#: Debt D5: the first hold after a no answer, and the longest (the pass runs every 5 s).
NO_ANSWER_HOLD_S = 10.0
NO_ANSWER_HOLD_MAX_S = 60.0


class NoAnswerBackoff:
    """The claim's hold on a generic endpoint whose server gave no answer (debt D5): after a
    :data:`NOT_ANSWERED` read the endpoint is not read again for ``first_s``, doubling at each
    further no answer to at most ``max_s``; any answer clears it. Keyed by the endpoint's id,
    base URL and model, so an endpoint whose address is changed is read at once. One per process
    (``wsp.tasks.CLAIM_BACKOFF``; the solo ``worker-wsp`` consumer runs every pass)."""

    def __init__(
        self, first_s: float = NO_ANSWER_HOLD_S, max_s: float = NO_ANSWER_HOLD_MAX_S
    ) -> None:
        self.first_s, self.max_s = first_s, max_s
        self._holds: dict[tuple[str, str, str], tuple[dt.datetime, float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(endpoint: WspEndpoint) -> tuple[str, str, str]:
        return (endpoint.id, endpoint.base_url, endpoint.model)

    def held(self, endpoint: WspEndpoint, now: dt.datetime) -> bool:
        """True while ``endpoint``'s last no answer still holds it at ``now``."""
        with self._lock:
            hold = self._holds.get(self._key(endpoint))
        return hold is not None and now < hold[0]

    def record(self, endpoint: WspEndpoint, answer: WindowAnswer, now: dt.datetime) -> None:
        """Record one live read: a no answer starts (or doubles) the hold, anything else ends it."""
        key = self._key(endpoint)
        with self._lock:
            if answer is not NOT_ANSWERED:
                self._holds.pop(key, None)
                return
            previous = self._holds.get(key)
            hold_s = self.first_s if previous is None else min(previous[1] * 2, self.max_s)
            self._holds[key] = (now + dt.timedelta(seconds=hold_s), hold_s)
        log.info(
            "wsp window read held after no answer",
            extra={
                "code": "wsp_window_read_held",
                "endpoint_id": endpoint.id,
                "hold_s": hold_s,
            },
        )


class NotResidentError(Exception):
    """A miServer endpoint's read gave no window for a run that calls the editor
    (:func:`awaits_resident`): no model is resident, so there is no window to resolve."""


def may_wait(endpoint: WspEndpoint, style: str) -> bool:
    """True when the run's read can make it wait (:func:`waits`): an endpoint that states no
    window, not hosted (never read), and a run that calls the editor (a session or a program; a
    replay or a rerun sends no editor request, so its window is only a record and may be
    assumed)."""
    from wsp import runs

    return (
        _positive(endpoint.context_window_tokens) is None
        and runs.takes_gateway(style)
        and runs.endpoint_locality(endpoint) != "external"
    )


def awaits_resident(endpoint: WspEndpoint, style: str) -> bool:
    """True when an EMPTY read already means the editor's model is not resident yet: a miServer
    endpoint for which :func:`may_wait` holds."""
    return endpoint.serving == "miserver" and may_wait(endpoint, style)


def waits(endpoint: WspEndpoint, style: str, read: WindowAnswer) -> bool:
    """True when this read leaves the run waiting instead of resolving a window: no window was
    read for a run that :func:`may_wait`, and either the server gave no answer at all (D1, any
    serving) or it is miServer, whose empty answer means no model is resident. A generic server
    that answered without a window is ``assumed`` (F19-D10)."""
    if _positive(read) is not None or not may_wait(endpoint, style):
        return False
    return read is NOT_ANSWERED or awaits_resident(endpoint, style)


def _positive(value: object) -> int | None:
    """A positive ``int``, else ``None`` (a ``bool`` is refused first: it is an ``int``)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def parse_openai_models(body: object, model: str) -> int | None:
    """``max_model_len`` of the ``data`` entry whose ``id`` equals ``model`` (vLLM, SGLang)."""
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if not isinstance(data, list):
        return None
    for entry in data:
        if isinstance(entry, dict) and entry.get("id") == model:
            return _positive(entry.get("max_model_len"))
    return None


def parse_llamacpp_props(body: object) -> int | None:
    """``default_generation_settings.n_ctx``: llama.cpp's per-slot context (0 in router mode)."""
    if not isinstance(body, dict):
        return None
    settings = body.get("default_generation_settings")
    if not isinstance(settings, dict):
        return None
    return _positive(settings.get("n_ctx"))


def origin(base_url: str) -> str:
    """``scheme://host:port`` of the endpoint's base URL, its ``/v1`` path dropped (the miServer
    read builds it the same way, ``wsp.tasks.read_served_window``)."""
    parts = urlsplit(base_url)
    return f"{parts.scheme}://{parts.netloc}"


def _get_json(
    client: httpx.Client, url: str, params: dict[str, str] | None = None
) -> tuple[bool, Any]:
    """``(answered, body)``: not answered on a 5xx (D1); answered with no body on any other status
    but 200, or a 200 whose body is not JSON. A transport failure (connection refused, a timeout,
    a reset) raises ``httpx.TransportError`` to :func:`_read_generic`."""
    response = client.get(url, params=params)
    if response.status_code >= 500:
        return False, None
    if response.status_code != 200:
        return True, None
    try:
        return True, response.json()
    except ValueError:
        return True, None


def _read_generic(endpoint: WspEndpoint, client: httpx.Client) -> WindowAnswer:
    import httpx

    root = origin(endpoint.base_url)
    try:
        answered, body = _get_json(client, root + MODELS_PATH)
    except httpx.TransportError:
        # D1: the server's origin is not answering (refused, timed out, reset). ``/props`` is on
        # the same origin, so it is not asked: one request, as before D1, and a black-holed host
        # holds the claim's pass for one timeout, not two.
        return NOT_ANSWERED
    found = parse_openai_models(body, endpoint.model)
    if found is not None:
        return found
    # Feature 019 P2 (12.4, operator 2026-11-24): ``?model=`` names the endpoint's model, so a
    # llama.cpp ROUTER proxies the read to that model's instance (its slot figure) instead of
    # answering its placeholder 0; a single-model server ignores the query.
    try:
        props_answered, props = _get_json(
            client, root + PROPS_PATH, params={"model": endpoint.model}
        )
    except httpx.TransportError:
        props_answered, props = False, None
    found = parse_llamacpp_props(props)
    if found is not None:
        return found
    return None if answered or props_answered else NOT_ANSWERED


def read_window(endpoint: WspEndpoint, *, client: httpx.Client | None = None) -> int | None:
    """The window the endpoint's server reports now, or ``None`` (no answer included: the
    registration read). Never raises (FR-019.1)."""
    found = read_window_answer(endpoint, client=client)
    return found if isinstance(found, int) else None


def read_window_answer(
    endpoint: WspEndpoint, *, client: httpx.Client | None = None
) -> WindowAnswer:
    """The window the endpoint's server reports now, ``None`` when it answered without one (or
    is miServer, whose read never distinguishes), or :data:`NOT_ANSWERED` when a generic server
    answered none of the requests (D1). The claim's and the push's reader. Never raises."""
    from wsp import runs

    if runs.endpoint_locality(endpoint) == "external":
        return None
    try:
        if endpoint.serving == "miserver":
            # Imported here: wsp.tasks imports Celery, which the API must not load just to read.
            from wsp.tasks import read_served_window

            return _positive(read_served_window(endpoint))
        if client is not None:
            found = _read_generic(endpoint, client)
        else:
            import httpx

            with httpx.Client(
                timeout=READ_TIMEOUT_S, follow_redirects=False, trust_env=False
            ) as owned:
                found = _read_generic(endpoint, owned)
        if found is NOT_ANSWERED:
            log.warning(
                "wsp window read got no answer",
                extra={
                    "code": "wsp_window_unread",
                    "endpoint_id": endpoint.id,
                    "reason": "no_answer",
                },
            )
        return found
    except Exception as exc:
        log.warning(
            "wsp window read failed",
            extra={
                "code": "wsp_window_unread",
                "endpoint_id": endpoint.id,
                "reason": type(exc).__name__,
            },
        )
        return None


def resolve(endpoint: WspEndpoint, read: WindowAnswer, *, style: str) -> Resolved:
    """The one window and ceiling a push uses (FR-019.2, F19-D10, F19-D11, F19-D12). Raises
    :class:`NotResidentError` instead of assuming a window when the read leaves the run waiting
    (:func:`waits`: miServer with nothing read, or a server that gave no answer, D1)."""
    if waits(endpoint, style, read):
        raise NotResidentError(endpoint.id)
    reported = _positive(read)
    stated = _positive(endpoint.context_window_tokens)
    source: WindowSource
    if stated is not None:
        window, source = stated, "stated"
    elif reported is not None:
        window, source = reported, "reported"
    else:
        window, source = editor_limits.DEFAULT_WINDOW_TOKENS, "assumed"
    ceiling = editor_limits.ceiling(endpoint.output_ceiling_tokens)
    lowered = False
    if reported is not None and reported < ceiling:
        ceiling, lowered = reported, True
    warnings: tuple[str, ...] = ()
    if source == "assumed":
        warnings += (WINDOW_ASSUMED,)
    if stated is not None and reported is not None and reported < stated:
        warnings += (WINDOW_STATED_ABOVE_REPORTED,)
    return Resolved(window, source, reported, ceiling, lowered, warnings)


def warnings_of(
    window_source: str | None, window_tokens: int | None, reported: int | None
) -> list[str]:
    """The warnings of a stored run, from its columns (the pure twin of :func:`resolve`'s rule,
    TID §3.4). A run from before 019 (no source) has none."""
    found: list[str] = []
    if window_source == "assumed":
        found.append(WINDOW_ASSUMED)
    read, used = _positive(reported), _positive(window_tokens)
    if window_source == "stated" and read is not None and used is not None and read < used:
        found.append(WINDOW_STATED_ABOVE_REPORTED)
    return found


def run_fields(row: Any) -> dict[str, Any]:
    """A run's window and ceiling as its list row and its progress read serve them (feature 019,
    FR-019.2, FR-019.17; TID §3.4). A run from before 019 serves nulls and no warning."""
    return {
        "window_tokens": row.window_tokens,
        "window_source": row.window_source,
        "window_reported_tokens": row.served_context_tokens,
        "output_ceiling_tokens": row.output_ceiling_tokens,
        "output_ceiling_lowered": row.output_ceiling_lowered,
        "window_warnings": warnings_of(
            row.window_source, row.window_tokens, row.served_context_tokens
        ),
    }
