"""The typed answer of ``POST /wsp/runs/validate`` (2026-11-26), the route that
``forge_wsp_validate_run`` calls: a start body frozen exactly as ``POST /wsp/runs`` will freeze it,
with nothing written.

Documentation only, as feature 021's answers are (MF9): the route still returns its dict
(``api/v1/endpoints/wsp_ops.py:validate_run``) and this model describes it, so the MCP tool's
output schema is typed (``backend/scripts/generate_output_schemas.py``). The frozen request is the
run contract's ``FrozenRequest`` (``docs/schemas/miforge-wsp-run-v1.json``, checked by
``wsp.freeze.validate_frozen_request``), so it is not restated here. Feature 021 owns the answer
contract; this model is written so it can cover the route as it is.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .answers_documents import WSP_RUN, part
from .contract_tools import EXTERNAL


class WspValidateSensitivity(BaseModel):
    """``sensitivity``: the bundle's mark, and what the run is treated as."""

    value: str = Field(description="The bundle's own mark.")
    effective: str = Field(description="sensitive or not_sensitive.")


class WspRunValidation(BaseModel):
    """``POST /wsp/runs/validate``: what ``POST /wsp/runs`` would freeze and file."""

    request: dict[str, Any] = Field(
        description="The frozen request: the run contract's FrozenRequest "
        "(miforge-wsp-run-v1.json), exactly what POST /wsp/runs stores.",
        json_schema_extra={EXTERNAL: part(WSP_RUN, "/$defs/FrozenRequest")},
    )
    request_sha256: str = Field(description="RFC 8785 sha256 of request.")
    approval_action: str | None = Field(
        description="The approval POST /wsp/runs would file (wsp_session, or "
        "wsp_session_external for an external endpoint); null when the start runs at once."
    )
    locality: str = Field(description="The endpoint's locality: local or external.")
    sensitivity: WspValidateSensitivity
    budgets: dict[str, Any] = Field(
        description="request.budgets, repeated.",
        json_schema_extra={EXTERNAL: part(WSP_RUN, "/$defs/Budgets")},
    )
    caller_ref: str | None = Field(description="The body's caller_ref, or null.")
