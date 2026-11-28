"""The answer models of the chain routes (feature 021, P3): ``api/v1/endpoints/chains.py``.

Documentation only (MF9): the routes still return their dicts, and these models change what the
OpenAPI document (and so the published answer contract) says about them, nothing else. Each class
names the function that builds the shape it describes.
"""

from pydantic import BaseModel

from .answers_common import may_be_absent


class ChainRow(BaseModel):
    """One chain (``services/chains/service.py:_chain_payload``)."""

    chain_id: str
    name: str
    status: str
    current_stage: int
    stage_count: int
    gpu_uuid: str | None
    halt_code: str | None
    halt_detail: str | None
    created_at: str | None
    updated_at: str | None
    finished_at: str | None


class ChainAdvanceDecision(BaseModel):
    """A finished stage's advance decision (``services/chains/service.py:_stage_payload``)."""

    stop_reason: str
    advances: bool
    halt_code: str | None


class ChainStage(BaseModel):
    """One stage of a chain (``services/chains/service.py:_stage_payload``)."""

    stage_index: int
    run_id: str | None
    advance_on: list[str]
    started_at: str | None
    finished_at: str | None
    run_status: str | None
    advance_decision: ChainAdvanceDecision | None


class ChainListChainsAnswer(BaseModel):
    """``GET /api/v1/chains`` (``services/chains/service.py:listing``)."""

    chains: list[ChainRow]


class ChainReadChainAnswer(ChainRow):
    """``GET /api/v1/chains/{chain_id}`` (``services/chains/service.py:detail``)."""

    stages: list[ChainStage]


class ChainCreateChainAnswer(ChainReadChainAnswer):
    """``POST /api/v1/chains`` (``services/chains/service.py:create``, which answers ``detail``)."""


class ChainStageStop(BaseModel):
    """The running stage's stop: ``services/stop_service.py:request_stop``'s outcome, or
    ``{run_id, already_terminal}`` when the stage had already finished
    (``services/chains/service.py:stop_chain``)."""

    run_id: str
    stop_pending: bool | None = may_be_absent(description="Sent when the stop was requested.")
    cancel_requested_at: str | None = may_be_absent(description="Sent when the stop was requested.")
    collapsed: bool | None = may_be_absent(description="Sent when the stop was requested.")
    stopped_now: bool | None = may_be_absent(description="Sent when the stop was requested.")
    already_terminal: bool | None = may_be_absent(
        description="Sent, true, when the stage had already finished."
    )


class ChainStopChainAnswer(BaseModel):
    """``POST /api/v1/chains/{chain_id}/stop`` (``services/chains/service.py:stop_chain``,
    ``halt``, ``_outcome``)."""

    chain_id: str
    action: str
    reason: str
    detail: str | None
    stage_index: int
    requested_by: str | None
    stage_stop: ChainStageStop | None
