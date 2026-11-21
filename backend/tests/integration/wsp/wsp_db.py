"""Shared helpers for the WSP integration tests: an engine at head, and minimal valid rows."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from alembic.config import Config

from tests.support import db as support

NOW = dt.datetime(2026, 11, 16, 5, 0, tzinfo=dt.UTC)
HEX = "a" * 64


def config(url: str) -> Config:
    cfg = Config(str(support.REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(support.REPO_ROOT / "backend" / "alembic"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = support._XArgs(url)  # type: ignore[assignment]
    return cfg


def engine_at_head(tmp_path: Path) -> sa.Engine:
    return support.migrated_engine(tmp_path / "db")


def insert(conn: sa.Connection, table: str, **values: Any) -> None:
    columns = ", ".join(values)
    params = ", ".join(f":{k}" for k in values)
    conn.execute(sa.text(f"INSERT INTO {table} ({columns}) VALUES ({params})"), values)


def bundle_row(conn: sa.Connection, sha: str = HEX) -> str:
    insert(
        conn,
        "mf_wsp_bundles",
        bundle_sha256=sha,
        owner="miworkbench",
        bundle_id="wsb_abc123",
        bundle_kind="linear_probe",
        artifact_kind="probe",
        artifact_id="p1",
        detector_digest=None,
        model="m",
        layer=1,
        sensitivity_value="not_sensitive",
        manifest="{}",
        bytes=1,
        import_status="imported",
        received_at=NOW,
        received_by="operator",
    )
    return sha


def run_row(conn: sa.Connection, run_id: str = "wsp_000000000001", **overrides: Any) -> str:
    values: dict[str, Any] = {
        "run_id": run_id,
        "operation": "repair",
        "style": "session",
        "status": "queued",
        "stop_reason": None,
        "outcome": None,
        "request": "{}",
        "request_sha256": HEX,
        "bundle_sha256": HEX,
        "endpoint_id": None,
        "endpoint_locality": None,
        "sensitivity_effective": "not_sensitive",
        "provider_id": "weightspace",
        "provider_version": "0.3.0",
        "caller": "operator",
        "usage": "{}",
        "created_at": NOW,
        "patch_sha256": None,
        "gateway_token_sha256": None,
    }
    values.update(overrides)
    insert(conn, "mf_wsp_runs", **values)
    return run_id


def candidate_row(conn: sa.Connection, run_id: str, candidate_id: str = "wspc_000000000001") -> str:
    insert(
        conn,
        "mf_wsp_candidates",
        candidate_id=candidate_id,
        run_id=run_id,
        parent_kind="bundle",
        parent_ref=HEX,
        bundle_base_sha256=HEX,
        patch_sha256="c" * 64,
        changed_entries=1,
        tensors="{}",
        created_at=NOW,
    )
    return candidate_id


def usage_row(conn: sa.Connection, run_id: str, seq: int = 0) -> None:
    insert(
        conn,
        "mf_wsp_usage",
        run_id=run_id,
        seq=seq,
        endpoint_id="ep",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0,
        at=NOW,
    )
