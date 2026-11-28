"""One editor call's time limits: the one source of truth (fix 2026-11-22, D128 round 7).

Production, 2026-11-21: a reply ran away on miServer (32,000 tokens, 46 minutes). The
``weightspace`` OpenAI client gave up first (the SDK's 600 s default), the gateway's upstream and
``miwsp_engine``'s client both waited 900 s and raced, and the two runs queued behind it failed
on their first call. Three literals, no order between them. This module orders them:

- the **gateway's upstream timeout** (``wsp_gateway.settings``' default and its ceiling:
  ``WSP_GATEWAY_UPSTREAM_TIMEOUT_S`` may lower it, never raise it);
- the **deadline** the gateway asks miServer for (``max_time_s``), :data:`DEADLINE_HEADROOM_S`
  inside that timeout, so miServer ends the generation (a length stop) and answers before the
  gateway would hang up;
- the **worker's client timeout** (``miwsp_engine.client``; until 018-P4 also the
  ``weightspace`` adapter's OpenAI and Anthropic clients), :data:`CLIENT_MARGIN_S` beyond the
  gateway's timeout, so the gateway always answers first: a refusal, or the cut-off reply, never
  a dropped connection with the gateway still holding the session's one in-flight reservation.

**Standard library only, and no import at all:** the worker image carries this file (it is part
of ``wsp_worker``), and the gateway (the backend image, which carries the whole tree) imports it,
never the other way round (``test_wsp_fences.py``).
"""

from __future__ import annotations

#: The gateway's upstream timeout for one editor call, and the ceiling its setting may not pass.
GATEWAY_UPSTREAM_TIMEOUT_S = 900.0
#: miServer is asked to end a free generation this long before the gateway's upstream timeout.
DEADLINE_HEADROOM_S = 60.0
#: A call is not sent with less time than this to generate in; it is refused through the budget
#: path (``wsp_budget_exhausted``, "time budget"), never left to time out.
MIN_CALL_TIME_S = 30.0
#: Kept between a call's deadline and the session's end (the supervisor's kill), for the answer
#: to travel back and the session to write its outputs.
SESSION_MARGIN_S = 10.0
#: miServer's own bound on ``max_time_s`` (a number above 0, at most 3600).
MISERVER_MAX_TIME_S = 3600.0
#: How much longer the worker's clients wait than the gateway's upstream: the gateway's own
#: calls to the backend (reserve, settle: 30 s each at most) fit inside it.
CLIENT_MARGIN_S = 120.0
#: Every worker client's timeout for one editor call through the gateway.
CLIENT_TIMEOUT_S = GATEWAY_UPSTREAM_TIMEOUT_S + CLIENT_MARGIN_S


def call_deadline(upstream_timeout_s: float, session_left_s: float | None) -> int | None:
    """The ``max_time_s`` one call may take, in whole seconds, or ``None`` when that is under
    :data:`MIN_CALL_TIME_S` (the call is then refused, not sent).

    The smallest of the gateway's upstream timeout less :data:`DEADLINE_HEADROOM_S`, the session's
    time left less :data:`SESSION_MARGIN_S` (``None``: not known, so not a bound) and miServer's
    own bound."""
    bounds = [upstream_timeout_s - DEADLINE_HEADROOM_S, MISERVER_MAX_TIME_S]
    if session_left_s is not None:
        bounds.append(session_left_s - SESSION_MARGIN_S)
    deadline = min(bounds)
    if deadline < MIN_CALL_TIME_S:
        return None
    return int(deadline)
