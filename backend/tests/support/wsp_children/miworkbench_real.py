"""miWorkbench's REAL scorer and importer code, run for miForge's cross-repo tests (feature 016
phase 2; FTASKS 13.4, 16.4a). ``python -I miworkbench_real.py <miworkbench backend> <case>``
reads one JSON object on stdin and prints one JSON object.

miWorkbench's packages are loaded from its checkout WITHOUT their ``__init__`` files (which pull
in the whole application: async SQLAlchemy, Celery), so the functions run are exactly
miWorkbench's own modules, untouched. Cases:

- ``metrics``: ``services.wsp_score.metrics_of`` per set and head (float32 scores, labels, bar),
  and ``services.wsp_patch.jcs_sha256`` of the ``heldout_ref``;
- ``refusals``: the body miWorkbench's route answers for each named refusal -
  ``HTTPException(status_code=refused.status, detail=refused.detail())`` (``api/v1/endpoints/
  wsp.py``'s two lines) rendered by FastAPI's own handler;
- ``run_sha256``: ``services.wsp_import.run_sha256_of`` and ``validate_run_document``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path


def _load(backend: Path) -> None:
    src = backend / "src"
    for name, sub in (
        ("src", src),
        ("src.services", src / "services"),
        ("src.core", src / "core"),
        ("src.schemas", src / "schemas"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(sub)]
        sys.modules[name] = module


def metrics(payload: dict) -> dict:
    import numpy as np
    from src.services.wsp_patch import jcs_sha256
    from src.services.wsp_score import metrics_of

    out = {}
    for name, one in payload["sets"].items():
        labels = np.asarray(one["labels"], dtype=np.int8)
        out[name] = {
            label: metrics_of(
                np.asarray(scores, dtype=np.float32), labels, one["thresholds"][label]
            )
            for label, scores in one["scores"].items()
        }
    return {"metrics": out, "heldout_ref_sha256": jcs_sha256(payload["heldout_ref"])}


def refusals(payload: dict) -> dict:
    from fastapi import HTTPException
    from fastapi.exception_handlers import http_exception_handler
    from src.services.wsp_score import ScoreRefused
    from starlette.requests import Request

    out = {}
    for code, extra in payload["refusals"].items():
        refused = ScoreRefused(code, "the scorer refused", **extra)
        exc = HTTPException(status_code=refused.status, detail=refused.detail())
        scope = {"type": "http", "method": "POST", "path": "/api/v1/wsp/score", "headers": []}
        response = asyncio.run(http_exception_handler(Request(scope), exc))
        out[code] = {"status": response.status_code, "body": json.loads(bytes(response.body))}
    return out


def run_sha256(payload: dict) -> dict:
    from src.services.wsp_import import ImportRefused, run_sha256_of, validate_run_document

    document = payload["document"]
    try:
        validate_run_document(document)
        valid = None
    except ImportRefused as refused:
        valid = str(refused)
    return {"run_sha256": run_sha256_of(document), "invalid": valid}


def main() -> None:
    backend, case = Path(sys.argv[1]), sys.argv[2]
    _load(backend)
    payload = json.loads(sys.stdin.read())
    print(
        json.dumps(
            {"metrics": metrics, "refusals": refusals, "run_sha256": run_sha256}[case](payload)
        )
    )


if __name__ == "__main__":
    main()
