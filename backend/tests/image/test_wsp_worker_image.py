"""What the WSP worker image holds and what a session's child can see in it (feature 016;
FR-016.9, FR-016.25; FTASKS 8.5; WR-12).

Image facts are read from the **prod** target (what the mirror publishes;
``MIFORGE_WSP_WORKER_PROD_IMAGE``), with ``python -I`` in a throwaway container. The environment and
filesystem dump is a fixture session on the **test** target, with the pod's flags (``harness.py``).
Skips by name outside the image job; ``MIFORGE_REQUIRE_WSP_IMAGE=1`` makes a skip a failure.

The README examples (FTASKS 8.5; R1B-A15; the operator's mapping, OVN-wsp-forge-68): the numpy
ports ``ex_linear_fix`` (``fix_quadrants.py``), ``ex_read_mlp`` (``read_which_task.py``) and
``ex_write_xor`` (the README's ``wsp.write`` on a 2-input XOR MLP) each run as a real session in the
``test`` target: the supervisor is PID 1 with the pod's flags, the child runs its preamble and
seccomp filter, then ``tests/support/wsp_children/readme_examples.py`` drives the image's own
``wsp_worker.adapters.weightspace_adapter`` with a scripted LLM into the pod's ``/out``. The tar the
supervisor builds passes the intake's reader, and the documented outcome is asserted here
(``tests/support/wsp_readme_expected.py``, shared with the local twin
``unit/wsp_worker/test_readme_examples.py``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

import pytest

from tests.image import harness as h
from tests.support import wsp_readme_expected as readme
from tests.support.wsp_expected import assert_child_env
from tests.support.wsp_supervisor import push_doc
from wsp import result_archive

#: FR-016.9: none of these may be importable in the worker image.
FORBIDDEN = ("torch", "sqlalchemy", "psycopg", "psycopg2", "redis", "celery")
#: FR-016.25: the only places a session's child may create a file. ``/dev/mqueue`` is the pod's
#: IPC namespace, which the child cannot be kept out of and the supervisor wipes as PID 1
#: (OVN-wsp-forge-18).
WRITABLE_ALLOWED = ("/work", "/out", "/tmp", "/dev/shm", "/dev/mqueue")


@pytest.fixture(scope="module")
def prod() -> str:
    return h.require_image(h.PROD_IMAGE_ENV)


@pytest.fixture(scope="module")
def test_image() -> str:
    return h.require_image()


@pytest.fixture
def pod(test_image: str) -> Iterator[h.Pod]:
    with h.pod(test_image) as p:
        yield p


def _pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in h.REQUIREMENTS.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            m = re.fullmatch(r"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==(\S+)", line)
            assert m, line
            pins[m.group(1).lower()] = m.group(2)
    return pins


@pytest.mark.parametrize("module", FORBIDDEN)
def test_forbidden_modules_cannot_be_imported(prod: str, module: str) -> None:
    r = h.run_once(prod, "-c", f"import {module}")
    assert r.returncode != 0, f"{module} is importable in the worker image"
    assert "ModuleNotFoundError" in r.stderr, r.stderr


def test_pinned_versions(prod: str) -> None:
    """weightspace is 0.3.0 (BRD-WSP-001), and the two SDKs are exactly their lock lines: the
    gateway's block allow-list is written against what these versions send (R3A-6)."""
    r = h.run_once(
        prod,
        "-c",
        "import json, weightspace, anthropic, openai\n"
        "print(json.dumps({'weightspace': weightspace.__version__,\n"
        "    'anthropic': anthropic.__version__, 'openai': openai.__version__}))\n",
    )
    assert r.returncode == 0, r.stderr
    pins = _pins()
    assert json.loads(r.stdout) == {
        "weightspace": "0.3.0",
        "anthropic": pins["anthropic"],
        "openai": pins["openai"],
    }
    assert pins["weightspace"] == "0.3.0"


def test_libseccomp_resolves(prod: str) -> None:
    r = h.run_once(prod, "-c", "import ctypes.util; print(ctypes.util.find_library('seccomp'))")
    assert r.returncode == 0 and r.stdout.strip().startswith("libseccomp.so"), r.stdout + r.stderr


def test_the_loader_refuses_pt(prod: str) -> None:
    """WR-12: a pickled format is refused by name before any read (the image has no torch)."""
    r = h.run_once(
        prod,
        "-c",
        "from pathlib import Path\n"
        "from wsp_worker import bundle\n"
        "try:\n"
        "    bundle.read_heads(Path('/work/heads.pt'))\n"
        "except bundle.BundleError as e:\n"
        "    print(e.code)\n",
    )
    assert r.stdout.strip() == "wsp_bundle_format_refused", r.stdout + r.stderr


def test_prod_has_no_pytest(prod: str) -> None:
    """P62: the published target carries no test tooling."""
    r = h.run_once(prod, "-c", "import pytest")
    assert "ModuleNotFoundError" in r.stderr, r.stdout + r.stderr


def test_environment_dump(pod: h.Pod) -> None:
    """FR-016.25: a fixture session's CHILD has exactly the allow-listed environment, no secrets
    mount, and can create files only under the pod's writable mounts (control C25: a fake
    ``DATABASE_URL`` in the child env turns this red)."""
    _, members, _, record = pod.start().run(["dump_env", "dump_writable", "write_result"])
    assert record["exit_code"] == 0
    env = json.loads(members["env.json"])
    assert_child_env(env)
    seen = json.loads(members["writable.json"])
    assert seen["secrets"] == []
    outside = [
        p
        for p in seen["writable"]
        if not any(p == root or p.startswith(root + "/") for root in WRITABLE_ALLOWED)
    ]
    assert outside == [], outside
    assert {"/work", "/out", "/tmp", "/dev/shm"} <= set(seen["writable"]), seen["writable"]


def test_in_container_suite(test_image: str, pod: h.Pod) -> None:
    """``tests/image/in_container/`` under pytest INSIDE the test target, as PID 1 of a throwaway
    container with the pod's flags (its own PID namespace, never ``--pid host``): a supervisor
    that is not PID 1 harms nothing of its uid (C10ab)."""
    suite = ("-I", "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rs")
    w = pod.start(
        entrypoint="python",
        command=(*suite, f"{h.MOUNT_ROOT}/in_container"),
        env={"WSP_IN_IMAGE_TESTS": "1"},
        restart="no",
        wait_ready=False,
    )
    code = w.wait_exit(timeout=300)
    logs = w.logs()
    assert code == 0, logs
    # Every in-container test ran and passed: the count is derived from the files (it read a
    # literal 1 until wsp-forge3 added two allowlist tests, and the job was not re-run).
    expected = sum(
        len(re.findall(r"^(?:    )?def test_", f.read_text(), flags=re.M))
        for f in h.IN_CONTAINER.glob("test_*.py")
    )
    assert expected >= 3, expected
    assert re.search(rf"\b{expected} passed\b", logs) and "skipped" not in logs, logs
    assert "failed" not in logs and "error" not in logs.lower(), logs


def test_intake_accepts_a_real_worker_tar(pod: h.Pod) -> None:
    """R3A-14: the tar the REAL supervisor builds in the image, as PID 1 with the pod's flags,
    passes the intake's hostile-archive reader (``wsp.result_archive``, the module
    ``wsp.runs.intake_result`` reads every result through; it is standard library only so this
    host, which has pytest and nothing else, can import it). Its local twin, through
    ``intake_result`` itself, is ``test_dispatch.py::test_one_pass_end_to_end_against_the_real_
    supervisor``. Control ``C-8.5-intake-members`` (``logs/stdout.txt`` dropped from
    ``RESULT_MEMBERS``) turns this red."""
    w = pod.start()
    boot = w.boot()
    run_id = "wsr_intake_image"
    actions = ["write_transcript", "print:hello", "write_result"]
    reply = w.session(push_doc(boot, actions=actions, run_id=run_id))
    assert reply.status == 202, reply
    w.wait_state("finished", timeout=90)
    raw, _digest = w.result_raw()
    members = result_archive.read_result_archive(raw)
    assert {"supervisor.json", "result.json", "transcript.jsonl", "logs/stdout.txt"} <= set(members)
    record = result_archive.check_record(members, run_id, boot)
    assert record["exit_code"] == 0 and record["deadline_hit"] is False
    result_archive.check_transcript(members["transcript.jsonl"])
    result = result_archive.json_object(members["result.json"], "result_json")
    assert result == {"stop_reason": "submitted"}
    assert b"hello" in members["logs/stdout.txt"]


# ------------------------------------------------------------------------- README examples (8.5)


@pytest.mark.parametrize("example", readme.EXAMPLES)
def test_readme_example_reproduces_in_the_worker(pod: h.Pod, example: str) -> None:
    """FTASKS 8.5 (R1B-A15): one README example port as a real session in the image. The tar the
    supervisor builds holds exactly what an adapter session writes (``result.json``,
    ``transcript.jsonl``, ``patch/`` when the weights changed, the child's stdout) and passes the
    intake's reader; the outcome, measured in the child on the patch in ``/out``, is the
    example's documented one; the port ran twice in the child and agreed with itself."""
    w = pod.start()
    boot = w.boot()
    run_id = f"wsr_{example}"
    push = push_doc(boot, actions=[f"readme_example:{example}"], run_id=run_id, time_budget_s=180)
    reply = w.session(push)
    assert reply.status == 202, reply
    w.wait_state("finished", timeout=300)
    raw, _digest = w.result_raw()
    members = result_archive.read_result_archive(raw)
    record = result_archive.check_record(members, run_id, boot)
    assert record["exit_code"] == 0 and record["deadline_hit"] is False, record
    expected = {"supervisor.json", "result.json", "transcript.jsonl", "logs/stdout.txt"}
    assert expected <= set(members), f"no adapter session ran: {sorted(members)}"
    result_archive.check_transcript(members["transcript.jsonl"])
    patched = {"patch/patch.json", "patch/edit.npz"} <= set(members)
    stdout = members["logs/stdout.txt"].decode("utf-8", "replace")
    outcome = readme.outcome_line(stdout)
    result = result_archive.json_object(members["result.json"], "result_json")
    patch = (
        result_archive.json_object(members["patch/patch.json"], "patch_json") if patched else None
    )
    assert set(members) - expected <= {"patch/patch.json", "patch/edit.npz", "logs/stderr.txt"}
    readme.assert_example(example, outcome, result, patch)
