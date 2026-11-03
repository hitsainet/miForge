"""The text the New run form shows from the backend is plain operator copy.

The New run screen (``/#/runs/new``) renders strings this backend writes: each registry's
one-line summaries, an unavailable mode's message, and the refusals a check or an estimate
returns. Those strings reached the screen carrying requirement ids (``R-02.25's first
condition.``, ``(R-02.11)``, ``BRD-02 §5.10``, ``ADR-020``) and Markdown bold that rendered as
literal asterisks (operator copy review, 2026-10-20). A requirement id means nothing to the
person choosing a stop condition.

Two scans, so neither has a blind spot:

1. **The live registry payloads**, read through the mounted app - every string anywhere in
   the JSON, so a new field is covered without an edit here.
2. **The messages a payload only carries in some states** (attach with miLLM absent or
   half-configured, a budget request with neither input, a definition with no pack), built
   by calling the code that raises them.

The stop-condition summary used to be the first line of each class's docstring, which is
where requirement ids belong. It is now ``SUMMARY``; a condition registered without one fails
here, not on the screen.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app

#: Copy that names a document, a requirement or a feature instead of saying something to
#: the operator. Mirrors the frontend's ``PLACEHOLDER_PATTERNS`` jargon entries.
JARGON = (
    re.compile(r"\bR-\d{2}\.\d+"),
    re.compile(r"\bFR-\d{3}\.\d+"),
    re.compile(r"\b(?:ADR|BRD|PRD|TDD|DEF|AMB)-\d+"),
    re.compile(r"§\s?\d"),
    re.compile(r"\bF\d{1,2}(?:'s|\u2019s)\b"),
    re.compile(r"\*\*"),
)

#: The registries the New run form reads, by path.
REGISTRIES = (
    "/api/v1/models/sources",
    "/api/v1/run-types",
    "/api/v1/signal-sources",
    "/api/v1/selectors",
    "/api/v1/stop-conditions",
)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def jargon_in(text: str) -> str | None:
    for pattern in JARGON:
        match = pattern.search(text)
        if match is not None:
            return match.group(0)
    return None


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.parametrize("path", REGISTRIES)
def test_registry_payload_carries_no_jargon(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200, response.text
    found = [(s, hit) for s in _strings(response.json()) if (hit := jargon_in(s)) is not None]
    assert found == [], f"{path} renders jargon on the New run form: {found}"


def test_each_stop_condition_summary_is_its_own_plain_sentence(client: TestClient) -> None:
    """Not the docstring's first line - the payload's summary is ``SUMMARY``, verbatim."""
    from training.stop import names, resolve

    rows = {
        row["condition"]: row for row in client.get("/api/v1/stop-conditions").json()["conditions"]
    }
    assert set(rows) == set(names())
    for name in names():
        summary = resolve(name).SUMMARY
        assert rows[name]["summary"] == summary
        assert summary.endswith("."), summary
        assert jargon_in(summary) is None, summary


def test_state_dependent_messages_carry_no_jargon() -> None:
    from training.acquisition import ATTACH_UNAVAILABLE
    from training.hotupdate.settings import (
        CAUSE_IDENTITY_INCOMPLETE,
        CAUSE_NOT_CONFIGURED,
        unavailable_message,
    )

    messages = {
        "attach, package absent": ATTACH_UNAVAILABLE.message or "",
        "attach, miLLM not configured": unavailable_message(CAUSE_NOT_CONFIGURED),
        "attach, identity incomplete": unavailable_message(CAUSE_IDENTITY_INCOMPLETE),
    }
    found = {key: hit for key, text in messages.items() if (hit := jargon_in(text)) is not None}
    assert found == {}, found
    assert "MILLM_BASE_URL" in messages["attach, miLLM not configured"]
    assert "MILLM_AUDIT_LABEL" in messages["attach, identity incomplete"]


def test_refusals_the_form_shows_carry_no_jargon() -> None:
    from core.errors import Refusal
    from schemas.run_definition import RunDefinition
    from training.acquisition.base import AcquisitionSpec
    from training.budget import require_exactly_one_input

    texts: list[str] = []
    try:
        RunDefinition.model_validate(
            {
                "model_source": {"mode": "create"},
                "run_type": "create",
                "run_configuration": {"learning_rate": 1e-5, "batch_size": 4},
                "signal_source": {"source": "corpus"},
                "cadences": {"test_every": 5, "retention_every": 5, "save_every": 10},
                "stop_conditions": [{"condition": "operator_stop"}],
            }
        )
    except Exception as error:  # pydantic's ValidationError
        texts.append(str(error))
    assert texts, "a definition with neither a pack nor parts must be refused"
    assert "choose a pack version" in texts[0]

    with pytest.raises(Refusal) as caught:
        require_exactly_one_input(AcquisitionSpec(path=Path("/tmp/x")))
    texts.append(caught.value.message)

    found = [(t, hit) for t in texts if (hit := jargon_in(t)) is not None]
    assert found == [], found
