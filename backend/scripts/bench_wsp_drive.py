"""Benchmark one WSP driver pass (feature 019 P3; D-23, FTASKS 19.2).

The pass is what the beat runs every 5 s (``miforge.wsp.drive``). P3 made it read every worker's
``/ready``, record it, poll each running run on its own worker and try a claim, so its cost now
grows with the pool. This times the steady state the bound is about: every worker answering and
busy (one running run each, polled every pass) and runs queued behind them, so each pass does the
whole read, record, poll and claim attempt.

The same script runs against ``32b6fa8`` (before P3: ``drive(factory, worker, ...)``, one worker)
and against P3 (``drive(factory, workers, ...)``): it reads the signature and the schema and does
what each takes. The workers are in-process fakes answering at once, so the figure is miForge's own
work (database reads and writes per pass), not the network's.

Run from ``backend/`` with ``PYTHONPATH=src:.``::

    python scripts/bench_wsp_drive.py --maxima 1 2 4 --passes 50

Prints one JSON object. Uses a throwaway SQLite database under ``--root`` (a temporary directory by
default); never a configured one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import secrets
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

HEX = "a" * 64
NOW = dt.datetime(2026, 11, 25, 6, 0, tzinfo=dt.UTC)


class Worker:
    """A worker answering at once: idle until pushed, then running its run on its boot."""

    def __init__(self) -> None:
        self.boot = secrets.token_hex(16)
        self.run_id: str | None = None

    def ready(self) -> dict[str, Any]:
        return {"boot_id": self.boot, "state": "running" if self.run_id else "idle"}

    def push(self, document: dict[str, Any], files: Any, **kwargs: Any) -> dict[str, Any]:
        self.run_id = document["run_id"]
        return {"state": "running"}

    def status(self) -> dict[str, Any]:
        return {"boot_id": self.boot, "run_id": self.run_id, "state": "running", "elapsed_s": 1.0}

    def result(self) -> bytes:  # pragma: no cover - never finished here
        raise AssertionError("not finished")

    def ack(self, boot_id: str, run_id: str, result_sha256: str) -> None:  # pragma: no cover
        return

    def recycle(self, boot_id: str) -> None:  # pragma: no cover
        return

    def close(self) -> None:
        return


def _engine(root: Path) -> sa.Engine:
    from tests.support import db as support

    root.mkdir(parents=True, exist_ok=True)
    url = f"sqlite+pysqlite:///{root / 'bench.db'}"
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(support.REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(support.REPO_ROOT / "backend" / "alembic"))
    cfg.attributes["configure_logger"] = False
    cfg.cmd_opts = support._XArgs(url)  # type: ignore[assignment]
    command.upgrade(cfg, "head")
    return support.make_engine(url)


def _seed(engine: sa.Engine, workers: list[Worker], queued: int) -> None:
    columns = {c["name"] for c in sa.inspect(engine).get_columns("mf_wsp_runs")}
    pool = "worker_index" in columns

    def insert(conn: sa.Connection, table: str, **values: Any) -> None:
        names = ", ".join(values)
        params = ", ".join(f":{k}" for k in values)
        conn.execute(sa.text(f"INSERT INTO {table} ({names}) VALUES ({params})"), values)

    request = json.dumps({"budgets": {"tokens": 1000, "time_s": 86_400}})
    with engine.begin() as conn:
        insert(
            conn,
            "mf_wsp_bundles",
            bundle_sha256=HEX,
            owner="miworkbench",
            bundle_id="wsb_0123456789ab",
            bundle_kind="linear_probe",
            artifact_kind="probe",
            artifact_id="p",
            model="m",
            layer=12,
            sensitivity_value="not_sensitive",
            manifest="{}",
            bytes=1,
            import_status="imported",
            received_at=NOW,
            received_by="operator",
        )
        for i in range(len(workers) + 1):
            insert(
                conn,
                "mf_wsp_endpoints",
                id=f"gen-{i}",
                kind="openai_compatible",
                base_url=f"http://gen-{i}.lan:8000/v1",
                model="m",
                locality="local",
                output_cap_field="max_completion_tokens",
                serving="generic",
                created_at=NOW,
                created_by="operator",
            )
        base = {
            "operation": "repair",
            "style": "session",
            "request": request,
            "request_sha256": HEX,
            "bundle_sha256": HEX,
            "endpoint_locality": "local",
            "sensitivity_effective": "not_sensitive",
            "provider_id": "miwsp_engine",
            "provider_version": "0.1.0",
            "caller": "operator",
            "usage": "{}",
        }
        for i, worker in enumerate(workers):
            run_id = f"wsp_{i:012x}"
            worker.run_id = run_id
            insert(
                conn, "mf_wsp_worker_boots", boot_id=worker.boot, first_seen_at=NOW, run_id=run_id
            )
            extra = {"worker_index": i, "schedule_key": f"endpoint:gen-{i}"} if pool else {}
            insert(
                conn,
                "mf_wsp_runs",
                run_id=run_id,
                status="running",
                endpoint_id=f"gen-{i}",
                boot_id=worker.boot,
                created_at=NOW,
                started_at=NOW,
                pushed_at=dt.datetime.now(dt.UTC),
                last_status_at=dt.datetime.now(dt.UTC),
                **base,
                **extra,
            )
        for q in range(queued):
            # Behind the busy keys, so a claim is tried and finds nothing to take.
            insert(
                conn,
                "mf_wsp_runs",
                run_id=f"wsp_{0xF00000 + q:012x}",
                status="queued",
                endpoint_id=f"gen-{q % max(1, len(workers))}",
                created_at=NOW + dt.timedelta(seconds=q + 1),
                **base,
            )


def bench(maximum: int, passes: int, queued: int, root: Path) -> dict[str, Any]:
    from wsp import dispatch
    from wsp.settings import WSPSettings

    workers = [Worker() for _ in range(maximum)]
    engine = _engine(root / f"max{maximum}")
    _seed(engine, workers, queued)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    settings = (
        WSPSettings(
            wsp_worker_url="http://w-{index}.bench:9000",
            wsp_gateway_url="http://g:8080",
            wsp_worker_pool_max=maximum,
        )
        if "wsp_worker_pool_max" in WSPSettings.model_fields
        else WSPSettings(wsp_worker_url="http://w.bench:9000", wsp_gateway_url="http://g:8080")
    )
    params = inspect.signature(dispatch.drive).parameters
    pooled = "workers" in params

    def one() -> list[str]:
        common: dict[str, Any] = {
            "read_sensitivity": lambda run, bundle: "not_sensitive",
            "files_for": lambda run, bundle: [],
        }
        if pooled:
            the_slots = [dispatch.WorkerSlot(i, f"w-{i}", w) for i, w in enumerate(workers)]
            done = dispatch.drive(
                factory,
                the_slots,
                settings,
                pool_size=maximum,
                enqueue_push=lambda r: None,
                **common,
            )
        else:
            done = dispatch.drive(factory, workers[0], settings, **common)
        return list(done.actions)

    first = one()
    times = []
    for _ in range(passes):
        started = time.perf_counter()
        actions = one()
        times.append(time.perf_counter() - started)
    engine.dispose()
    return {
        "maximum": maximum,
        "pooled_driver": pooled,
        "passes": passes,
        "median_s": round(statistics.median(times), 6),
        "p95_s": round(sorted(times)[max(0, int(len(times) * 0.95) - 1)], 6),
        "max_s": round(max(times), 6),
        "first_pass_actions": first,
        "steady_actions": actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--maxima", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument("--queued", type=int, default=20)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="bench-wsp-drive-"))
    from wsp import dispatch

    pooled = "workers" in inspect.signature(dispatch.drive).parameters
    maxima = args.maxima if pooled else [1]
    out = {"results": [bench(m, args.passes, args.queued, root) for m in maxima], "root": str(root)}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
