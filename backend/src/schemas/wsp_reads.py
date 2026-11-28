"""Typed answers of the WSP run reads added for agents on 2026-11-26 (the 019 P4 MCP gaps).

* :class:`WspRunRecord` - ``GET /wsp/runs/{run_id}/record``: one run's row exactly as the WSP run
  list serves it (``GET /wsp/runs``, miForge's own fields: window, refused turns, usage, metered
  cost, live progress) beside its ``miforge.wsp-run/v1`` document (``GET /wsp/runs/{run_id}``).
  The row is the list's own builder, so it stays the list's shape; the document is the vendored
  contract (``docs/schemas/miforge-wsp-run-v1.json``), so neither is restated here.
* :class:`WspRunLedger` - ``GET /wsp/runs/{run_id}/ledger``: the gateway's reservation and usage
  rows for one run (``mf_wsp_reservations``, ``mf_wsp_usage``): endpoint, tokens, cost and times.
  Counts and codes only, as the ledger stores: never a prompt, an answer, a key or a price.

Feature 021 (typed answers) owns the answer contract; these models are written so it can cover
them as they are.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .answers_documents import WSP_RUN
from .contract_tools import EXTERNAL, MODEL_REF


class WspRunRecord(BaseModel):
    """One WSP run: its list row and its run document."""

    run_id: str
    # Feature 021: the answer contract declares both fields by reference (the OpenAPI document
    # only; the served model and its serializer are unchanged).
    row: dict[str, Any] = Field(
        description="The run's row as GET /api/v1/wsp/runs serves it (miForge's own fields).",
        json_schema_extra={MODEL_REF: "WspRunRow"},
    )
    document: dict[str, Any] = Field(
        description="The run as miforge.wsp-run/v1 (GET /api/v1/wsp/runs/{run_id}).",
        json_schema_extra={EXTERNAL: WSP_RUN},
    )


class WspLedgerReservation(BaseModel):
    """One gateway reservation: the worst case held before an editor call, and its settlement."""

    id: int
    endpoint_id: str
    external: bool
    worst_case_tokens: int
    worst_case_usd: float
    created_at: str
    settled_at: str | None
    settled_cost_usd: float | None


class WspLedgerUsage(BaseModel):
    """One editor call the gateway saw (``mf_wsp_usage``), with the endpoint that served it."""

    seq: int
    endpoint_id: str
    at: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    outcome: str | None
    http_status: int | None
    latency_ms: int | None


class WspLedgerTotals(BaseModel):
    """Every row of the run, not only this page: what its sums are, and which endpoints appear."""

    reservations: int
    reservations_open: int
    settled_cost_usd: float
    usage_rows: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    endpoint_ids: list[str]


class WspRunLedger(BaseModel):
    """A page of one run's reservation and usage rows, oldest first, and the run's totals."""

    run_id: str
    status: str
    reservations: list[WspLedgerReservation]
    usage: list[WspLedgerUsage]
    more_reservations: bool
    more_usage: bool
    totals: WspLedgerTotals
