"""Request-scoped dependencies  (task 5.1).

Only one so far: a database session. It exists as a FastAPI dependency rather than as a
direct call inside each route so that a test can override it with
``app.dependency_overrides`` - which is what lets the reachability test prove a route is
mounted and answers, on a machine with no Postgres.

The import of ``db.session`` is deferred for the reason stated in
:mod:`workers.run_record`: foundation 4.1 owns that module and this one is written beside
it, so a module-level import would make every route in the application unimportable while
that file is in flight.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from core.errors import refuse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session


def session_dependency() -> Iterator[Session]:
    """Yield a request-scoped session, committing on a clean exit.

    Raises:
        Refusal: ``database_unavailable``, named rather than an ``ImportError`` escaping
            as a 500 with a stack trace. ADR-018's rule applies to infrastructure too.
    """
    try:
        from db.session import session_scope
    except ImportError as exc:  # pragma: no cover - only before 4.1 lands
        raise refuse(
            "database_unavailable",
            "db.session.session_scope is not importable, so no request touching the "
            "database can be served. Foundation task 4.1 owns it.",
            detail=str(exc),
        ) from exc

    with session_scope() as session:
        yield session
