"""Whose credential a Hugging Face Hub request may use (017 FR-017.26 - .29, .33, .34).

**An agent never borrows the operator's Hugging Face credential** (suite CHAIN_PLAN A17, A28).
miForge stores no Hub token of its own, but ``huggingface_hub`` sends whatever implicit
credential the process has - ``HF_TOKEN`` or a token file under ``HF_HOME`` - whenever a call
passes ``token=None``. Only ``token=False`` sends nothing. So every Hub helper in
``training/acquisition/hub.py`` takes ``token`` as a **required** keyword, and this module is the
one place that decides its value:

- ``False`` for an agent's request: the request's actor, the approval executor's
  :func:`api.agent_origin.executing_actor` (an approved request runs AS its agent), or a run
  whose ``mf_runs.hub_access`` is ``anonymous``;
- ``None`` for the operator's request (today's behaviour, unchanged);
- ``False`` when nothing names the caller. **The default fails closed** (017-D24): a path that
  forgot to say who asked fetches anonymously, which can only fail loudly
  (``model_fetch_failed`` on a private repository), never leak.

For an agent, validation fetches metadata only (FR-017.34, A42): :func:`fetch_for` answers
``"metadata"`` whenever the token is ``False``, so no weight file leaves the Hub before the
operator has approved the run and its worker holds the lease.

Two scopes carry the decision to code that cannot take an argument:

- :func:`requesting` - the API process. The approval gate wraps the route's validator and the
  route body in it (``api/agent_origin.py``); the three ungated Hub routes name the actor
  themselves.
- :func:`run_scope` - the worker. ``training.tasks.drive_run`` enters it with the run's own
  ``hub_access`` for the run's whole life, so the acquisition, a ``create`` run's tokenizer and
  an attach run's verification tokenizer all fetch under it. ``acquire_for_run(spec)`` keeps its
  one-argument signature (tests double it by name) and reads :func:`worker_hub_token`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from api.agent_origin import Actor

#: The value of ``token`` every Hub helper receives: ``False`` (send nothing) or ``None`` (the
#: process's implicit credential). A string is never produced here.
HubToken = bool | None
Fetch = Literal["metadata", "snapshot"]

#: The one value ``mf_runs.hub_access`` takes; ``NULL`` is today's behaviour (017-D23).
ANONYMOUS = "anonymous"

#: The sentinel for a run object with no ``hub_access`` attribute at all (fails closed).
_MISSING: Any = object()

_requesting: ContextVar[Actor | None] = ContextVar("mf_requesting_actor", default=None)


class _Unset:
    pass


_UNSET = _Unset()
_run_token: ContextVar[HubToken | _Unset] = ContextVar("mf_run_hub_token", default=_UNSET)
_inherited: ContextVar[str | _Unset | None] = ContextVar("mf_inherited_hub_access", default=_UNSET)


@contextmanager
def requesting(actor: Actor | None) -> Iterator[None]:
    """Run the block as a request from ``actor``. Reset in ``finally``."""
    token = _requesting.set(actor)
    try:
        yield
    finally:
        _requesting.reset(token)


def requesting_actor() -> Actor | None:
    """The approval executor's requester while it runs, else the request's actor, else ``None``."""
    from api.agent_origin import executing_actor

    return executing_actor() or _requesting.get()


def hub_token(actor: Actor | None = None) -> HubToken:
    """``False`` for an agent (or for nobody), ``None`` for the operator."""
    who = actor or requesting_actor()
    if who is None:
        return False  # FAIL CLOSED: nothing says this is the operator's request
    return False if who.origin == "agent" else None


def fetch_for(token: HubToken) -> Fetch:
    """What an API-process pre-flight may fetch for a request carrying ``token`` (A42)."""
    return "metadata" if token is False else "snapshot"


def hub_token_for_run(run: Any) -> HubToken:
    """The worker's token for a run row (or anything with ``hub_access``).

    Only an explicit null is the operator's run (``None``: the process's own credential). Every
    other value - ``anonymous``, a value nothing writes today, or a row with no ``hub_access``
    at all - answers ``False``: FAIL CLOSED, never the operator's credential by default (R2
    review fix OR2-017-4, 2026-11-16; migration ``0027``'s CHECK narrows the column too).
    """
    value = getattr(run, "hub_access", _MISSING)
    return None if value is None else False


def access_of(token: HubToken) -> str | None:
    """The ``mf_runs.hub_access`` value for a requester whose token is ``token``."""
    return ANONYMOUS if token is False else None


@contextmanager
def run_scope(token: HubToken) -> Iterator[None]:
    """The worker's scope for one run: every Hub fetch inside it uses ``token``."""
    reset = _run_token.set(token)
    try:
        yield
    finally:
        _run_token.reset(reset)


def worker_hub_token() -> HubToken:
    """The token of the run this worker is driving; ``False`` outside any run (fail closed)."""
    value = _run_token.get()
    return False if isinstance(value, _Unset) else value


@contextmanager
def inheriting(access: str | None) -> Iterator[None]:
    """A chain stage after stage 0 takes the previous stage's ``hub_access`` (017-D25)."""
    reset = _inherited.set(access)
    try:
        yield
    finally:
        _inherited.reset(reset)


def access_for_launch(*, parent_access: str | None, existing_access: str | None) -> str | None:
    """``mf_runs.hub_access`` for a run :func:`services.run_launch.launch` records (FR-017.27).

    - the requester: ``anonymous`` for an agent (or nobody), else null - unless the run is a
      chain stage after stage 0, which takes the previous stage's value instead;
    - ORed with the parent's value for a branch or a continue (a run started from another run's
      checkpoint), so an operator branching an anonymous run stays anonymous;
    - never clearing a value the row already holds (a continue reopens the same row).
    """
    inherited = _inherited.get()
    if isinstance(inherited, _Unset):
        own = access_of(hub_token())
    else:
        own = inherited
    if ANONYMOUS in (own, parent_access, existing_access):
        return ANONYMOUS
    return None


__all__ = [
    "ANONYMOUS",
    "Fetch",
    "HubToken",
    "access_for_launch",
    "access_of",
    "fetch_for",
    "hub_token",
    "hub_token_for_run",
    "inheriting",
    "requesting",
    "requesting_actor",
    "run_scope",
    "worker_hub_token",
]
