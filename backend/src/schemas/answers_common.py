# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/schemas/answers_common.py @ 6b50a8f4
# Mode: adapt (docs/REUSE.md). Kept: may_be_absent, external_ref, ErrorBody, ErrorEnvelope. Changed:
# no closed_enum or literal_of (021 publishes no enum), and the error code is lower snake case here.
"""Shared pieces of the answer models (feature 021, FTID §3.3).

The markers below change only what the OpenAPI document (and so the published answer contract) says
about a field; none of them changes what a route serves (MF9):

- ``may_be_absent()``: a field some answers leave out. The contract does not list it in ``required``
  ("required" means always sent, MF10).
- ``external_ref(schema_id)``: a field whose value is a published ``docs/schemas`` document; the
  contract references that document's ``$id`` (MF15).

``ErrorBody`` and ``ErrorEnvelope`` describe the one error shape every refusal leaves the API in
(``core/errors.py``: ``{"error": {"code", "message", "details"}}``); the contract publishes it as
``errors.app``. No route declares it, so the served OpenAPI components do not move. The pending
approval and the ``gated_responses`` helper arrive with the gate declaration (FTASKS 3.1, 3.2).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel

from .contract_tools import (
    EXTERNAL,
    EXTERNAL_WITHOUT,
    HISTORY_FORM,
    MAY_BE_ABSENT,
    MEDIA_ALTERNATIVES,
)

#: A field whose value is genuinely any JSON (a stored request body, a setting's value). Its
#: schema is ``{}``, so every use must carry :func:`free_form`'s reason; a structure the code fixes
#: is declared.
FREE_FORM = "x-free-form"

__all__ = [
    "FREE_FORM",
    "ApprovalObject",
    "ApprovalPending",
    "ErrorBody",
    "ErrorEnvelope",
    "JsonValue",
    "NextStep",
    "PublishedDocument",
    "Unavailable",
    "binary_answer",
    "document_config",
    "external_ref",
    "free_form",
    "gated_responses",
    "history_form",
    "may_be_absent",
    "media_alternatives",
]


def may_be_absent(**kwargs: Any) -> Any:
    """A field some answers leave out: default ``None``, and never ``required`` in the contract."""
    extra = dict(kwargs.pop("json_schema_extra", None) or {})
    extra[MAY_BE_ABSENT] = True
    kwargs.setdefault("default", None)
    return Field(json_schema_extra=extra, **kwargs)


def free_form(reason: str, **kwargs: Any) -> Any:
    """A field (or a map's values) whose content is any JSON, and why. Declared, not guessed: the
    reason names what the content is (for example "the stored request body, as the caller sent it").
    """
    if not reason:
        raise ValueError("free_form needs the reason the content is free-form")
    extra = dict(kwargs.pop("json_schema_extra", None) or {})
    extra[FREE_FORM] = reason
    return Field(json_schema_extra=extra, **kwargs)


def external_ref(schema_id: str, **kwargs: Any) -> Any:
    """A field whose value is the published document ``schema_id`` names (a ``docs/schemas`` file's
    ``$id``); the contract replaces the field's schema with a ``$ref`` to that ``$id``."""
    extra = dict(kwargs.pop("json_schema_extra", None) or {})
    extra[EXTERNAL] = schema_id
    return Field(json_schema_extra=extra, **kwargs)


class PublishedDocument(RootModel[dict[str, JsonValue]]):
    """The base of a route answer (or a field) that IS a published ``docs/schemas`` document (021
    FR-021.9). A subclass sets ``model_config = document_config(<$id>)``; its contract schema is
    then a ``$ref`` to that ``$id``. Documentation only: no route serves through it."""


def document_config(schema_id: str, without: tuple[str, ...] = ()) -> ConfigDict:
    """The ``model_config`` of a :class:`PublishedDocument` subclass for document ``schema_id``
    (a ``$id``, or ``<$id>#<pointer>`` for one part of a document).

    ``without`` names top-level properties the answer leaves out (a stored snapshot of the
    document taken before they are added): the contract then declares the document's other
    properties, each a ``$ref`` to the document's own, and requires what the document requires
    except those (``answers_contract._resolve_external_markers``)."""
    extra: dict[str, Any] = {EXTERNAL: schema_id}
    if without:
        extra[EXTERNAL_WITHOUT] = sorted(without)
    return ConfigDict(json_schema_extra=extra)


def history_form(description: str, **kwargs: Any) -> Any:
    """An approval's stored ``result`` (miServer TD14): the contract replaces its schema with one
    branch per ``action``, each the HISTORY FORM of that action's routes' own answers (their types
    kept, no ``required`` at any depth: a stored row can predate a field), or null."""
    extra = dict(kwargs.pop("json_schema_extra", None) or {})
    extra[HISTORY_FORM] = True
    kwargs.setdefault("default", None)
    return Field(json_schema_extra=extra, description=description, **kwargs)


def media_alternatives(**by_type: str) -> dict[str, Any]:
    """The rest of a ``responses`` entry whose JSON answer has other media types (021 FT4): each
    media type, with the request that selects it, e.g.
    ``{"application/octet-stream": "name = edit.npz"}``. Merge it beside ``"model"``."""
    return {
        "content": {media: {"schema": {"type": "string", "format": "binary"}} for media in by_type},
        MEDIA_ALTERNATIVES: dict(sorted(by_type.items())),
    }


def binary_answer(media_type: str, reason: str) -> dict[int | str, dict[str, Any]]:
    """The ``responses=`` of a route that answers only bytes: the contract lists it under
    ``excluded`` with ``media_type`` and ``reason``. Pair it with ``response_class=Response`` so
    FastAPI documents no JSON body beside it (the endpoint returns its ``Response`` itself, so
    nothing it serves changes)."""
    return {
        200: {
            "description": reason,
            "content": {media_type: {"schema": {"type": "string", "format": "binary"}}},
        }
    }


class ErrorBody(BaseModel):
    """``error`` in every refusal (``core/errors.py``)."""

    model_config = ConfigDict(title="ErrorBody")

    code: str = Field(
        description="The refusal's code, lower snake case (for example run_not_found)."
    )
    message: str = Field(description="One line for the person reading it.")
    details: dict[str, JsonValue] = free_form("the refusal's facts, which each refusal names")


class ErrorEnvelope(BaseModel):
    """The one error shape: ``{"error": {"code", "message", "details"}}`` (ADR-013)."""

    model_config = ConfigDict(title="ErrorEnvelope")

    error: ErrorBody


class NextStep(BaseModel):
    """One ``next_steps`` item (``services/next_step.py`` ``next_for``): the tool to call next."""

    model_config = ConfigDict(title="NextStep")

    tool: str = Field(description="A miForge MCP tool name (forge_*).")
    args: dict[str, JsonValue] = free_form("the tool's arguments, as that tool takes them")
    why: str = Field(description="Why this call is next (1 - 300 characters).")


class ApprovalObject(BaseModel):
    """``services/result_language.py:approval_view``: the suite's ``approval`` object (the vendored
    result language's ``approval`` definition, as miForge fills it)."""

    model_config = ConfigDict(title="ApprovalObject")

    id: str
    status: str
    decide_in: str
    then: str
    summary: str
    action: str
    expires_at: str
    reason: str | None


class ApprovalPending(BaseModel):
    """``services/approval_service.py:pending_body``: the 202 an agent's call to a gated route
    answers while the operator decides (``api/agent_origin.py``'s gate)."""

    model_config = ConfigDict(title="ApprovalPending")

    approval_id: str
    status: Literal["pending"]
    action: str
    summary: dict[str, JsonValue] = free_form(
        "the summary the gate's summariser built for the operator; its keys depend on the action"
    )
    request_digest: str
    expires_at: str
    hint: str
    approval: ApprovalObject
    next_steps: list[NextStep]


class Unavailable(BaseModel):
    """``mcp_server/health_gate.py``: a tool's answer when a dependency is down (nothing sent)."""

    model_config = ConfigDict(title="Unavailable")

    unavailable: str
    reason: str | None


_PENDING_OR: dict[type[BaseModel], type[RootModel[Any]]] = {}


def _pending_or(own: type[BaseModel]) -> type[RootModel[Any]]:
    """``PendingOr<Own>``: the one 202 of a gated route whose own success is also 202, told apart by
    the pending branch's required ``approval_id`` (no own model declares one). Cached by ``own``,
    so a component name never changes between builds."""
    if own not in _PENDING_OR:
        # Built at run time from the route's own model, so the type checker cannot see it.
        union: Any = Annotated[ApprovalPending | own, Field(union_mode="left_to_right")]
        model = type(
            f"PendingOr{own.__name__}",
            (RootModel[union],),
            {"__doc__": f"The gate's pending approval, or {own.__name__}.", "__module__": __name__},
        )
        _PENDING_OR[own] = model
    return _PENDING_OR[own]


def gated_responses(own_202: type[BaseModel] | None = None) -> dict[int | str, dict[str, Any]]:
    """The ``responses=`` entry a gated route declares for the gate's pending approval (021 FT3).

    Without ``own_202`` the pending approval is the route's 202. With it, the route's own success
    is also a 202, and the entry is the union of the two (``PendingOr<Own>``). Merge it into the
    route's own declaration: ``responses={200: {"model": Own}, **gated_responses()}``."""
    if own_202 is None:
        return {202: {"model": ApprovalPending, "description": "The operator decides first."}}
    return {
        202: {
            "model": _pending_or(own_202),
            "description": "The route's own answer, or the pending approval when an agent asks.",
        }
    }
