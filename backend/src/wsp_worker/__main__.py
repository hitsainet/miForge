"""``python -m wsp_worker <mode>``: the supervisor, or one of the three child modes (FTDD §6.1).

- ``supervisor``: PID 1 of the worker pod (``wsp_worker/supervisor.py``).
- ``run``, ``rerun``, ``replay``: the child the supervisor starts for one session. Its first acts,
  **before it reads stdin or imports anything else** (FR-016.10 step 9):

  1. restore a fresh interpreter's signal dispositions (the supervisor ignored them);
  2. set ``PR_SET_NO_NEW_PRIVS`` and install its seccomp filter (``child_seccomp``); a failure
     exits ``EXIT_FILTER_FAILED`` before the ready byte, so the supervisor refuses the session;
  3. set ``RLIMIT_FSIZE``, and ``RLIMIT_NPROC`` when its parent is PID 1 (OVN-wsp-forge-15);
  4. write one ready byte ``R`` to stdout.

  Only then does it read the supervisor's one stdin line (spec, gateway, paths). ``run`` serves a
  ``session`` or ``program`` push, and ``replay`` the same over the pushed transcript with no
  gateway, through the run's own provider (``spec.request.provider.id``; feature 018, FR-018.2):
  ``miwsp_engine`` (``miwsp_engine/``) or ``weightspace`` (``adapters/weightspace_adapter.py``);
  any other id refuses ``worker_provider_unknown``. ``rerun`` runs the stored program through
  ``program.py`` alone, never importing weightspace (R1B-A9). A push whose mode is not the
  child's refuses ``worker_mode_unavailable``.
"""

from __future__ import annotations

import json
import os
import sys

#: ``RLIMIT_NPROC`` for the child: per real uid across the node, so it holds only because uid
#: 10002 is the worker's alone (a cross-repo test pins it; P60, R3A-3).
CHILD_NPROC = 256
#: ``RLIMIT_FSIZE``: one file's bound, well below ``/work``'s 64 GiB ``sizeLimit``; the backend
#: keeps this much room free beside the pushed files (``wsp.capacity.EDITOR_HEADROOM_BYTES``).
CHILD_FSIZE = 4 * 1024**3
CHILD_MODES = ("run", "rerun", "replay")


def _refuse(code: str, message: str, exit_code: int) -> None:
    os.write(2, (json.dumps({"refused": code, "message": message}) + "\n").encode())
    os._exit(exit_code)


def child_preamble() -> None:
    """Steps 1-4 above. Test children (``WSP_TEST_SCRIPT``) call this to behave as the real one."""
    from wsp_worker import spec

    spec.restore_fresh_dispositions()
    from wsp_worker import child_seccomp

    try:
        child_seccomp.install()
    except child_seccomp.SeccompUnavailableError as exc:
        _refuse(
            "worker_filter_missing", f"filter not installed: {exc.code}", spec.EXIT_FILTER_FAILED
        )
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (CHILD_FSIZE, CHILD_FSIZE))
    if os.getppid() == 1:
        # Only under the real supervisor: outside a container the developer's uid already runs
        # more tasks than this, and no containment is claimed there (FR-016.10 step 11).
        resource.setrlimit(resource.RLIMIT_NPROC, (CHILD_NPROC, CHILD_NPROC))
    os.write(1, b"R")


def read_child_line() -> dict[str, object]:
    """The supervisor's one stdin line, bounded."""
    from wsp_worker import spec

    raw = sys.stdin.buffer.readline(spec.MAX_CHILD_LINE_BYTES + 1)
    try:
        return spec.parse_child_line(raw)
    except spec.SpecError as exc:
        _refuse(exc.code, str(exc), spec.EXIT_SPEC_INVALID)
        raise AssertionError("unreachable") from exc


def _require_spec(line: dict[str, object]) -> None:
    from wsp_worker import spec

    pushed = line.get("spec")
    if not isinstance(pushed, dict) or not all(
        isinstance(pushed.get(k), dict) for k in ("manifest", "request")
    ):
        _refuse(
            "worker_spec_invalid",
            "a session needs the bundle manifest and the request in its spec",
            spec.EXIT_SPEC_INVALID,
        )


def provider_id(line: dict[str, object]) -> object:
    """``spec.request.provider.id``: frozen by the backend at ``wsp/freeze.py``, never guessed."""
    pushed = line.get("spec")
    request = pushed.get("request") if isinstance(pushed, dict) else None
    provider = request.get("provider") if isinstance(request, dict) else None
    return provider.get("id") if isinstance(provider, dict) else None


def run_provider(line: dict[str, object]) -> int:
    """A ``run`` or ``replay`` child, routed by the run's own provider (feature 018, FR-018.2).

    ``miwsp_engine`` never imports ``weightspace``; the run fails if anything did (the pattern of
    ``program.no_llm_loaded``). Any other id is refused ``worker_provider_unknown``: no provider
    runs in place of the one the run names."""
    from wsp_worker import spec

    wanted = provider_id(line)
    if wanted == "miwsp_engine":
        from wsp_worker import miwsp_engine

        code = miwsp_engine.run(line)
        if "weightspace" in sys.modules:
            raise SystemExit("weightspace was imported during an miwsp_engine run")
        return code
    if wanted == "weightspace":
        from wsp_worker.adapters import weightspace_adapter

        return weightspace_adapter.run(line)
    _refuse(
        "worker_provider_unknown",
        f"this worker runs no provider {wanted!r}",
        spec.EXIT_REFUSED,
    )
    raise AssertionError("unreachable")


def child_main(mode: str) -> None:
    child_preamble()
    line = read_child_line()
    from wsp_worker import spec
    from wsp_worker.bundle import BundleError

    wanted = {"run": ("session", "program"), "rerun": ("program_rerun",), "replay": ("replay",)}
    if line.get("mode") not in wanted.get(mode, ()):
        _refuse(
            "worker_mode_unavailable",
            f"child mode {mode} does not run a {line.get('mode')!r} push",
            spec.EXIT_MODE_UNAVAILABLE,
        )
    _require_spec(line)
    try:
        if mode == "rerun":
            # No weightspace import on this path, ever (R1B-A9): the program sandbox is ours.
            import datetime as dt

            from wsp_worker import program

            now = dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")
            code = program.rerun(line, now)
        else:
            pushed = line["spec"]
            assert isinstance(pushed, dict)
            if mode == "run" and not isinstance(pushed.get("editor"), dict):
                _refuse("worker_spec_invalid", "a session needs its editor", spec.EXIT_SPEC_INVALID)
            if mode == "run" and not isinstance(line.get("gateway"), dict):
                _refuse(
                    "worker_spec_invalid", "a session needs its gateway", spec.EXIT_SPEC_INVALID
                )
            from wsp_worker.observation_tools import ObservationToolError

            try:
                code = run_provider(line)
            except ObservationToolError as exc:
                _refuse(exc.code, str(exc), spec.EXIT_REFUSED)
                raise AssertionError("unreachable") from exc
    except BundleError as exc:
        _refuse(exc.code, str(exc), spec.EXIT_REFUSED)
        raise AssertionError("unreachable") from exc
    sys.stdout.flush()
    os._exit(code)


def main(argv: list[str]) -> None:
    mode = argv[0] if argv else ""
    if mode in CHILD_MODES:
        child_main(mode)
    elif mode == "supervisor":
        from wsp_worker import supervisor

        supervisor.main()
    else:
        _refuse(
            "worker_mode_unknown",
            f"usage: python -m wsp_worker supervisor|{'|'.join(CHILD_MODES)}",
            64,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
