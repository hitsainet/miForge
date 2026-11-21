"""The worker's result archive, read as HOSTILE input (feature 016; FR-016.19; FTID §3.6; P63,
R2A-15, R3A-14): member by member, never extracted; only regular files from a fixed set, each
capped; the supervisor's record checked against the run.

Standard library and ``core.errors`` only, on purpose: the private-CI image job imports this
module on a host that has pytest and nothing else, to put the tar the REAL supervisor builds in
the image through the same reader the intake uses (``test_wsp_worker_image.py::
test_intake_accepts_a_real_worker_tar``). ``wsp.runs.intake_result`` is its one production
caller.
"""

from __future__ import annotations

import json
from typing import Any

from core.errors import refuse

RESULT_MEMBERS = frozenset(
    {
        "result.json",
        "transcript.jsonl",
        "patch/patch.json",
        "patch/edit.npz",
        "logs/stdout.txt",
        "logs/stderr.txt",
        "program.py",
        "supervisor.json",
    }
)
MEMBER_MAX_BYTES = {
    "result.json": 4 << 20,
    "transcript.jsonl": 64 << 20,
    "patch/patch.json": 1 << 20,
    "patch/edit.npz": 512 << 20,
    "logs/stdout.txt": 2 << 20,
    "logs/stderr.txt": 2 << 20,
    "program.py": 1 << 20,
    "supervisor.json": 64 << 10,
}
#: How many of the supervisor's ``unsafe`` reasons a refusal names.
UNSAFE_SHOWN = 3
#: The driver never reads more than this from ``GET /result``.
RESULT_MAX_BYTES = sum(MEMBER_MAX_BYTES.values())
#: Stored beside the run (never served raw): what the run page and the replay read.
STORED_MEMBERS = ("transcript.jsonl", "result.json", "logs/stdout.txt", "logs/stderr.txt")


def _invalid(check: str, message: str) -> Exception:
    return refuse(
        "wsp_result_invalid", f"The worker's result failed the intake: {message}", check=check
    )


def read_result_archive(data: bytes) -> dict[str, bytes]:
    """The archive's members by name, refusing anything outside the fixed set (P63)."""
    import io
    import tarfile

    if len(data) > RESULT_MAX_BYTES:
        raise _invalid("archive_size", "the archive is larger than its bound")
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r|") as tar:
            for info in tar:
                name = info.name
                if name not in RESULT_MEMBERS:
                    raise _invalid("member_name", f"an unknown member {name[:80]!r}")
                if not info.isreg():
                    raise _invalid("member_type", f"{name} is not a regular file")
                if name in members:
                    raise _invalid("member_duplicate", f"{name} appears twice")
                if info.size > MEMBER_MAX_BYTES[name]:
                    raise _invalid("member_size", f"{name} is larger than its bound")
                handle = tar.extractfile(info)
                if handle is None:  # pragma: no cover - isreg() members always have data
                    raise _invalid("member_type", f"{name} has no data")
                members[name] = handle.read(MEMBER_MAX_BYTES[name] + 1)
    except tarfile.TarError as exc:
        raise _invalid("archive_format", "the archive is not a readable tar") from exc
    if "supervisor.json" not in members:
        raise _invalid("supervisor_record", "the supervisor's record is missing")
    return members


def check_record(members: dict[str, bytes], run_id: str, boot_id: str | None) -> dict[str, Any]:
    """The supervisor's record, for THIS run and boot, with no unsafe ``/out``."""
    record = json_object(members["supervisor.json"], "supervisor_record")
    if record.get("run_id") != run_id or record.get("boot_id") != boot_id:
        raise _invalid("supervisor_record", "the record names another run or boot")
    if record.get("unsafe_reason"):
        # R3 review fix (OR3-016-5), 2026-11-16: name what the supervisor found, so the operator
        # can tell a link or a FIFO from an oversized file. Child-chosen names: capped, as data.
        found = record.get("unsafe")
        reasons = [str(r)[:160] for r in found[:UNSAFE_SHOWN]] if isinstance(found, list) else []
        more = len(found) - len(reasons) if isinstance(found, list) else 0
        named = (
            (": " + "; ".join(reasons) + (f"; and {more} more" if more > 0 else ""))
            if reasons
            else ""
        )
        raise _invalid(
            "worker_result_unsafe",
            f"the worker reported an unsafe /out{named}. The editor (or its program) wrote a "
            "link, a special file or an oversized file into /out; only regular files within "
            "their caps are taken",
        )
    return record


def check_transcript(raw: bytes) -> None:
    """Every non-blank line is one JSON object (review OR1-016-11): the run page parses the
    stored lines and the replay reads them, so a line that is not one is refused here, at intake,
    instead of becoming a 500 on ``GET /wsp/runs/{id}/transcript`` later. The content is never
    interpreted; it stays data."""
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise _invalid("transcript", "the transcript is not UTF-8") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise _invalid("transcript", f"transcript line {number} is not JSON") from exc
        if not isinstance(entry, dict):
            raise _invalid("transcript", f"transcript line {number} is not a JSON object")


def json_object(raw: bytes, check: str) -> dict[str, Any]:
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _invalid(check, f"{check} is not JSON") from exc
    if not isinstance(doc, dict):
        raise _invalid(check, f"{check} is not an object")
    return doc
