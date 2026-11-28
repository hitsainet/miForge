"""The ``miwsp_engine`` test child (feature 018; FTASKS 7.5). Run under the worker's interpreter as
``python -I engine_cases.py <backend/src> <line.json> <options.json>``.

It behaves as the real child up to the session: the real preamble (``child_preamble``: signal
dispositions, the child's seccomp filter, the file-size limit, the ready byte), so the kernel
grandchild's filter check sees a real parent filter to stack on. Then it runs the engine on the
line with the test's options (a short kernel timeout, say) and prints one JSON line: the exit
code and whether the kernel it recorded still runs. ``{"probe_self": ...}`` instead runs the
security probe IN THIS PROCESS (the negative control of FR-018.14: here the token must be found).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from wsp_worker.__main__ import child_preamble

child_preamble()
sys.path.insert(1, str(Path(__file__).resolve().parent))
line = json.loads(Path(sys.argv[2]).read_text())
options = json.loads(Path(sys.argv[3]).read_text())
if "probe_self" in options:
    from engine_probe import probe

    print(json.dumps({"probe": probe(**options["probe_self"])}), flush=True)
    sys.exit(0)

from wsp_worker import miwsp_engine  # noqa: E402

code = miwsp_engine.run(line, **options)
# Whether a kernel pid the editor recorded (``tmp/kernel.pid``) still runs when ``run`` returns.
recorded = Path(line["paths"]["tmp"]) / "kernel.pid"
kernel_alive = None
if recorded.exists():
    import os

    try:
        os.kill(int(recorded.read_text()), 0)
        kernel_alive = True
    except ProcessLookupError:
        kernel_alive = False
print(
    json.dumps(
        {
            "exit": code,
            "kernel_alive": kernel_alive,
        }
    ),
    flush=True,
)
