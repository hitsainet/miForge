"""What the supervisor and its child agree on (FTID §3.2; FTDD §5.4a; FR-016.10).

Standard library only, so the child can import it before anything else. It holds:

- the push the backend sends to ``POST /session`` (its first multipart part), validated whole:
  :func:`parse_push`;
- the one JSON line the supervisor writes to the child's stdin once the child's filter is
  proved (:func:`child_line`), and its parse (:func:`parse_child_line`);
- ``CHILD_ENV``, the child's **exact** environment: a literal, so the live test compares a set
  (FR-016.10; the gateway token travels only on stdin, never in the environment);
- the signals the supervisor ignores and the dispositions a fresh interpreter starts with
  (R3A-21, R4A-3), and the process exit codes.
"""

from __future__ import annotations

import ipaddress
import json
import re
import signal
from dataclasses import dataclass
from typing import Any

#: ``POST /session`` modes, and the child's ``__main__`` mode for each.
CHILD_MODE = {"session": "run", "program": "run", "program_rerun": "rerun", "replay": "replay"}
#: Modes that never reach the gateway: no token is minted for them (R3A-13).
NO_GATEWAY_MODES = frozenset({"program_rerun", "replay"})

#: The child's whole environment, minus ``WSP_MODE`` (added per session by :func:`child_env`).
CHILD_ENV: dict[str, str] = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

#: Exit codes. 0 is the only clean exit; the rest name who stopped and why.
EXIT_OK = 0
EXIT_SPEC_INVALID = 65
EXIT_MODE_UNAVAILABLE = 69
EXIT_WATCHDOG = 70
EXIT_FILTER_FAILED = 71
EXIT_REFUSED = 78

#: Session bounds the push may not exceed (a malformed or hostile spec is refused, not clamped).
MAX_TIME_BUDGET_S = 24 * 3600
MAX_GRACE_S = 3600
MAX_PULL_TIMEOUT_S = 3600
DEFAULT_LOG_MAX_BYTES = 1_048_576
MAX_LOG_MAX_BYTES = 64 * 1_048_576
DEFAULT_WORK_MAX_BYTES = 8 * 1024**3
MAX_FILES = 64
MAX_PUSH_JSON_BYTES = 4 * 1_048_576
MAX_CHILD_LINE_BYTES = 8 * 1_048_576

SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")
ROLE = re.compile(r"^[a-z_]{1,64}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
BOOT_ID = re.compile(r"^[0-9a-f]{32}$")
RUN_ID = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


class SpecError(ValueError):
    """A push or child line that is not exactly what the contract allows."""

    def __init__(self, message: str, code: str = "worker_spec_invalid") -> None:
        super().__init__(message)
        self.code = code


def child_env(child_mode: str) -> dict[str, str]:
    """``CHILD_ENV`` plus ``WSP_MODE``: the exact environment of one child."""
    if child_mode not in set(CHILD_MODE.values()):
        raise SpecError(f"unknown child mode {child_mode!r}")
    return {**CHILD_ENV, "WSP_MODE": child_mode}


# ----------------------------------------------------------------------------- signals


def supervisor_ignored() -> list[signal.Signals]:
    """Every catchable signal but ``SIGCHLD``: the supervisor sets each to ``SIG_IGN`` (P59).

    ``SIGCHLD`` keeps ``SIG_DFL`` with no handler, so ``Popen.wait`` reads the child's real exit
    code (R3A-21)."""
    skip = {signal.SIGKILL, signal.SIGSTOP, signal.SIGCHLD}
    return sorted((s for s in signal.valid_signals() if s not in skip), key=int)


def restore_fresh_dispositions() -> None:
    """In the child, first: undo the supervisor's ``SIG_IGN`` to a fresh interpreter's state.

    Ignored dispositions survive ``exec``, and ``Popen``'s ``restore_signals`` resets only
    ``SIGPIPE`` and ``SIGXFSZ``. CPython itself starts with ``SIGPIPE`` and ``SIGXFSZ`` ignored
    and ``SIGINT`` at ``default_int_handler`` (R3A-21, R4A-3)."""
    for sig in supervisor_ignored():
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (OSError, ValueError):
            continue  # a number the platform reserves (glibc's own real-time signals)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.default_int_handler)


# ----------------------------------------------------------------------------- the push


@dataclass(frozen=True)
class PushFile:
    name: str
    role: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Push:
    boot_id: str
    run_id: str
    mode: str
    spec: dict[str, Any]
    files: tuple[PushFile, ...]
    gateway: dict[str, Any] | None

    @property
    def child_mode(self) -> str:
        return CHILD_MODE[self.mode]

    @property
    def time_budget_s(self) -> float:
        return float(self.spec["time_budget_s"])

    @property
    def lifetime_s(self) -> float:
        """``time_budget + grace + pull timeout`` (FR-016.10 step 8), from the session's start (the
        push received); while the push is still being received, from its acceptance."""
        return self.time_budget_s + float(self.spec["grace_s"]) + float(self.spec["pull_timeout_s"])

    @property
    def log_max_bytes(self) -> int:
        return int(self.spec.get("log_max_bytes", DEFAULT_LOG_MAX_BYTES))

    @property
    def work_max_bytes(self) -> int:
        return int(self.spec.get("work_max_bytes", DEFAULT_WORK_MAX_BYTES))


def _number(spec: dict[str, Any], key: str, low: float, high: float, *, required: bool) -> None:
    if key not in spec:
        if required:
            raise SpecError(f"spec.{key} is required")
        return
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low < value <= high:
        raise SpecError(f"spec.{key} must be a number in ({low}, {high}]")


def parse_push(raw: bytes) -> Push:
    """The first multipart part of ``POST /session``, refused unless exactly the contract."""
    if len(raw) > MAX_PUSH_JSON_BYTES:
        raise SpecError("the push is larger than its bound")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecError(f"the push is not JSON: {exc}") from exc
    keys = {"boot_id", "run_id", "mode", "spec", "files", "gateway"}
    if not isinstance(doc, dict) or set(doc) != keys:
        raise SpecError(f"the push must hold exactly {sorted(keys)}")
    if not isinstance(doc["boot_id"], str) or not BOOT_ID.match(doc["boot_id"]):
        raise SpecError("boot_id must be 32 hex digits")
    if not isinstance(doc["run_id"], str) or not RUN_ID.match(doc["run_id"]):
        raise SpecError("run_id is malformed")
    if doc["mode"] not in CHILD_MODE:
        raise SpecError(f"mode must be one of {sorted(CHILD_MODE)}")
    spec = doc["spec"]
    if not isinstance(spec, dict):
        raise SpecError("spec must be an object")
    _number(spec, "time_budget_s", 0, MAX_TIME_BUDGET_S, required=True)
    _number(spec, "grace_s", 0, MAX_GRACE_S, required=True)
    _number(spec, "pull_timeout_s", 0, MAX_PULL_TIMEOUT_S, required=True)
    _number(spec, "log_max_bytes", 0, MAX_LOG_MAX_BYTES, required=False)
    _number(spec, "work_max_bytes", 0, 1024**4, required=False)
    files = doc["files"]
    if not isinstance(files, list) or len(files) > MAX_FILES:
        raise SpecError(f"files must be a list of at most {MAX_FILES}")
    parsed: list[PushFile] = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"name", "role", "sha256", "bytes"}:
            raise SpecError("each file is exactly {name, role, sha256, bytes}")
        name, role, sha, size = entry["name"], entry["role"], entry["sha256"], entry["bytes"]
        if not isinstance(name, str) or not SAFE_NAME.match(name) or name in (".", ".."):
            raise SpecError(f"file name {name!r} is not a safe flat name")
        if not isinstance(role, str) or not ROLE.match(role):
            raise SpecError(f"file role {role!r} is malformed")
        if not isinstance(sha, str) or not HEX64.match(sha):
            raise SpecError(f"file {name} sha256 is not 64 hex digits")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SpecError(f"file {name} bytes must be a non-negative integer")
        parsed.append(PushFile(name, role, sha, size))
    if len({f.name for f in parsed}) != len(parsed):
        raise SpecError("two files share a name")
    gateway = doc["gateway"]
    if doc["mode"] in NO_GATEWAY_MODES and gateway is not None:
        raise SpecError(f"a {doc['mode']} push carries no gateway (it gets no egress)")
    if gateway is not None:
        if not isinstance(gateway, dict) or set(gateway) != {"ip", "port", "token"}:
            raise SpecError("gateway is null or exactly {ip, port, token}")
        try:
            ipaddress.ip_address(gateway["ip"])
        except (ValueError, TypeError) as exc:
            raise SpecError("gateway.ip must be an IP literal (the worker has no DNS)") from exc
        port = gateway["port"]
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise SpecError("gateway.port must be 1..65535")
        token = gateway["token"]
        if not isinstance(token, str) or not 0 < len(token) <= 512:
            raise SpecError("gateway.token must be a non-empty string")
    return Push(doc["boot_id"], doc["run_id"], doc["mode"], spec, tuple(parsed), gateway)


# ----------------------------------------------------------------------------- the child line


def child_line(push: Push, paths: dict[str, str], time_left_s: float) -> bytes:
    """The one line written to the child's stdin, after its filter is proved. ``time_left_s`` is
    the session's time budget left when the line is written (fix 2026-11-21): the supervisor
    kills the child when it reaches 0, so the engine measures its own end against it."""
    line = {
        "mode": push.mode,
        "run_id": push.run_id,
        "spec": push.spec,
        "files": [f.__dict__ for f in push.files],
        "gateway": push.gateway,
        "paths": paths,
        "time_left_s": round(max(0.0, float(time_left_s)), 3),
    }
    return (json.dumps(line, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def parse_child_line(raw: bytes) -> dict[str, Any]:
    """The child's read of that line (bounded, exactly the keys :func:`child_line` writes)."""
    if len(raw) > MAX_CHILD_LINE_BYTES:
        raise SpecError("the child line is larger than its bound")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecError(f"the child line is not JSON: {exc}") from exc
    keys = {"mode", "run_id", "spec", "files", "gateway", "paths", "time_left_s"}
    if not isinstance(doc, dict) or set(doc) != keys:
        raise SpecError(f"the child line must hold exactly {sorted(keys)}")
    if not isinstance(doc["paths"], dict) or set(doc["paths"]) != {"work", "out", "tmp"}:
        raise SpecError("paths must be exactly {work, out, tmp}")
    left = doc["time_left_s"]
    if isinstance(left, bool) or not isinstance(left, int | float) or not 0 <= left <= 1e7:
        raise SpecError("time_left_s must be a number of seconds, at least 0")
    return doc
