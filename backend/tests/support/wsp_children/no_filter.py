"""A WSP test child that skips ``child_seccomp.install()`` (R3A-5): it must never get its line.

It sets ``PR_SET_NO_NEW_PRIVS`` itself, as every process in the worker's container already has
(``allowPrivilegeEscalation: false``), so only the ``Seccomp_filters`` comparison can refuse it.
"""

from __future__ import annotations

import ctypes
import os
import select
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])
from wsp_worker import spec

spec.restore_fresh_dispositions()
ctypes.CDLL(None).prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
os.write(1, b"R")
ready, _, _ = select.select([0], [], [], 5)
if ready:
    raw = sys.stdin.buffer.readline()
    if raw:
        line = spec.parse_child_line(raw)
        Path(str(line["paths"]["out"]), "got_line.json").write_bytes(raw)
os._exit(0)
