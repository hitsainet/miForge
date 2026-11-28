"""The editor's shared limits: the one source for the engine and the gateway (feature 019;
FR-019.15; W3-D2, W3-D6; F19-D34; TID §3.1).

Standard library only, and no import at all: the worker image carries ``wsp_worker/`` alone, and
the gateway (which runs the backend image) imports this module as it imports ``call_time``. Shared
constants flow from the worker to the gateway, never back (miForge memory
``editor-call-deadline-2026-11-22``). ``miwsp_engine.context`` and ``wsp_gateway.meter`` alias these
names; ``test_engine_room.py`` pins each alias by identity, so a literal re-introduced in either
copy goes red.
"""

#: The window when the endpoint states none and the server reports none (F18-D12, W3-D2).
DEFAULT_WINDOW_TOKENS = 8_192
#: An endpoint's output ceiling when it sets none (W3-D6): the gateway's cap before 019.
DEFAULT_OUTPUT_CEILING_TOKENS = 32_000
#: The least ceiling an endpoint may set (W3-D6): the gateway's floor for one request.
MIN_OUTPUT_CEILING_TOKENS = 256
#: The largest ceiling accepted: an INTEGER column, and the window's own maximum.
MAX_OUTPUT_CEILING_TOKENS = 10_000_000


def ceiling(value: object) -> int:
    """The endpoint's (or the run's) ceiling, or :data:`DEFAULT_OUTPUT_CEILING_TOKENS` when it is
    null or anything but a positive integer (a ``bool`` is not one)."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return DEFAULT_OUTPUT_CEILING_TOKENS
