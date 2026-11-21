"""The one canonical-JSON function  (task 2.6; ADR-005).

**One function, used for hashing and for writing.** ADR-005's reasoning, stated as a rule
this project repeats: *two canonicalisers are two answers.* If the bytes that are hashed
are produced by one code path and the bytes written to disk by another, the two agree
until someone changes one of them - and then a manifest's recorded hash stops matching
the manifest, which is exactly the class of defect R-02.21's guarantee rests on.

So :func:`canonical_bytes` is the only place a Python object becomes JSON bytes in this
codebase, :func:`sha256_of` hashes *its* output, and :func:`write_canonical` writes *its*
output. The test asserts both paths call it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

#: Separators with no whitespace, keys sorted, UTF-8, no trailing newline inside the
#: hashed bytes. Fixed deliberately: every one of these is a choice that changes the
#: digest, so they live in one constant rather than at each call site.
_DUMP_KWARGS: dict[str, Any] = {
    "sort_keys": True,
    "separators": (",", ":"),
    "ensure_ascii": False,
    "allow_nan": False,  # NaN/Infinity are not JSON; refuse rather than emit `NaN`
}


def canonical_bytes(obj: Any) -> bytes:
    """Serialise ``obj`` to canonical JSON bytes. **The only serialiser in the codebase.**

    ``allow_nan=False`` matters more than it looks: ``json`` would otherwise emit the
    literal ``NaN``, which no JSON parser is required to accept, so a loss value that
    diverged would produce a manifest that validates here and fails everywhere else.
    A diverged loss should refuse loudly at the point it is recorded.
    """
    return json.dumps(obj, **_DUMP_KWARGS).encode("utf-8")


def sha256_of(obj: Any) -> str:
    """SHA-256 of ``obj``'s canonical bytes, lowercase hex (ADR-005: SHA-256 throughout)."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def sha256_of_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    """Streaming SHA-256 of a file's bytes.

    Streaming because a full parameter set is gigabytes: a 2.4 GB read into memory to hash
    it would double the peak footprint on a card this project is already budgeting to the
    megabyte (R-02.42).
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_canonical(path: Path, obj: Any, *, newline: bool = True) -> str:
    """Write ``obj`` as canonical JSON and return the digest of the bytes **as hashed**.

    The returned digest covers :func:`canonical_bytes`'s output, *not* the trailing
    newline. That distinction is the whole point of returning it from the writer: a caller
    that hashed the file afterwards would include the newline and get a different answer
    from a caller that hashed the object, and the two would disagree silently. Here the
    writer hands back the one true digest, so nothing has to re-derive it.

    The write is **atomic within the file**: a temp file in the same directory, ``fsync``,
    then ``rename``. A half-written ``manifest.json`` that still parses is worse than one
    that is absent, because the absent one is obviously absent.
    """
    payload = canonical_bytes(obj)
    digest = hashlib.sha256(payload).hexdigest()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            if newline:
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        # fsync the directory too, or the rename itself may not survive a power loss.
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return digest
