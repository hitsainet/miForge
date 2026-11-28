"""The answers of the WSP editor-endpoint routes (feature 021; ``wsp_endpoints.py``).

Documentation only: the routes still return plain dicts; these models describe them (MF9).
One endpoint row (``wsp.endpoints.public_document``) is the list's item and the whole answer of
a registration and of a change; it carries exactly ``tests/support/wsp_list_keys.py``'s
``ENDPOINT_ROW_KEYS``.
"""

from pydantic import BaseModel


class WspEndpointRunsRefused(BaseModel):
    """``runs_refused`` (``wsp/endpoints.py:runs_refused``): what a start would be refused with."""

    code: str
    message: str


class WspEndpointRegistrationWindow(BaseModel):
    """``registration_window`` (``wsp/endpoints.py:window_facts``): the read at registration."""

    tokens: int | None
    read_at: str | None


class WspEndpointLatestRunWindow(BaseModel):
    """``latest_run_window`` (``wsp/endpoints.py:window_facts``): the newest run's window read."""

    tokens: int | None
    source: str
    run_id: str
    at: str


class WspEndpointWindowNow(BaseModel):
    """``window_now`` (``wsp/endpoints.py:window_now``): the next run's expected window. ``source``
    ``waits`` with ``tokens`` 0 (no window) for a miServer endpoint that states none and has no
    run's read yet (2026-11-27): its next run waits for miServer to report one."""

    tokens: int
    source: str
    run_id: str | None
    at: str | None


class WspEndpointWindowIfReadFails(BaseModel):
    """``window_if_read_fails`` (``wsp/endpoints.py:window_if_read_fails``). ``source`` ``waits``
    with ``tokens`` 0 (no window) for a miServer endpoint that states none (2026-11-27): a run
    whose read fails waits, then fails ``editor_not_resident``; no window is assumed. For a
    generic endpoint the two fields are the case of a server that ANSWERS without a window, and
    ``no_answer`` (operator decision D1, 2026-11-27) the case of one that gives no answer at all:
    ``stated``, ``waits`` (then ``editor_not_resident``) or ``not_read`` (hosted)."""

    tokens: int
    source: str
    no_answer: str


class WspEndpointProtocolTested(BaseModel):
    """``protocol_test.tested`` (``wsp/protocol_tests.py:pass_state``): the values last tested."""

    base_url: str | None
    model: str | None
    action_protocol: str | None


class WspEndpointProtocolTest(BaseModel):
    """``protocol_test`` (``wsp/endpoints.py:protocol_test_document``): the latest protocol test."""

    result: str | None
    run_id: str | None
    at: str | None
    tested: WspEndpointProtocolTested
    current: bool
    cleared: bool
    required: bool


class WspEndpointRow(BaseModel):
    """One editor endpoint (``wsp/endpoints.py:public_document``): never the key or ciphertext."""

    id: str
    kind: str
    base_url: str
    model: str
    locality: str
    serving: str
    output_cap_field: str
    price_in_per_mtok: str | None
    price_out_per_mtok: str | None
    price_cache_write_per_mtok: str | None
    price_cache_read_per_mtok: str | None
    has_key: bool
    key_fingerprint: str | None
    created_at: str
    created_by: str
    action_protocol: str
    context_window_tokens: int | None
    thinking: str
    thinking_tokens: int | None
    output_cap: str
    paging: str
    output_ceiling_tokens: int | None
    history_mode: str
    runs_refused: WspEndpointRunsRefused | None
    output_ceiling_effective: int
    registration_window: WspEndpointRegistrationWindow
    latest_run_window: WspEndpointLatestRunWindow | None
    window_now: WspEndpointWindowNow
    window_if_read_fails: WspEndpointWindowIfReadFails
    window_warnings: list[str]
    protocol_test: WspEndpointProtocolTest


class WspEndpointListEndpointsAnswer(BaseModel):
    """``GET /wsp/endpoints`` (``api/v1/endpoints/wsp_endpoints.py:list_endpoints``)."""

    endpoints: list[WspEndpointRow]


class WspEndpointRemoveEndpointAnswer(BaseModel):
    """``DELETE /wsp/endpoints/{id}`` (``api/v1/endpoints/wsp_endpoints.py:remove_endpoint``)."""

    id: str
    removed_at: str


class WspEndpointProtocolTestRunAnswer(BaseModel):
    """``POST /wsp/endpoints/{id}/protocol-test`` (``wsp_endpoints.py:protocol_test_run``)."""

    run_id: str
    status: str
    thinking: str | None
