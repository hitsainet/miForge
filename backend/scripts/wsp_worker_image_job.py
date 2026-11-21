"""The private-CI job ``wsp-worker-image`` (``.github/workflows/backend-tests.yml``), as one script
that CI and a developer run alike (feature 016; FR-016.25; FTASKS 4.3a; R1B-M4, R3A-23).

    python backend/scripts/wsp_worker_image_job.py [--prefix wsp-ci] [--tests PATH ...]

Run from the repository root. Each step is fatal, in this order:

1. ``docker build -f backend/Dockerfile.wsp-worker --target prod`` and ``--target test``, tagged
   ``<prefix>/wsp-worker:prod`` and ``:test``. Never pushed.
2. **The layer check** (R3A-23): the ``prod`` image's layer digests are a PREFIX of the ``test``
   image's, and the extra layers add only the distributions pinned in
   ``backend/requirements-wsp-worker-test.txt`` (pytest and its dependencies: every file listed in
   their ``RECORD``) and ``/opt/wsp/wsp_gateway/allowlist.py`` - which they MUST add, byte-equal,
   whenever ``backend/src/wsp_gateway/allowlist.py`` exists (FTASKS 6.5), and must not otherwise.
   No whiteout (a deletion), no other file, no new directory but those files' parents. So the live
   tests exercise the shipped bits.
3. ``pytest backend/tests/image`` with ``MIFORGE_REQUIRE_WSP_IMAGE=1`` (a skip is a failure) and the
   two tags; the tests start their containers with the pod's flags (``tests/image/harness.py``).
4. Always: remove every container, volume and network labelled with this run's id.

The exit code is the first failing step's (pytest's own code for step 3), so the workflow job is
red exactly when a step fails. ``--tests`` replaces the test paths; it exists so the gating can be
proved with a placeholder that must fail (FTASKS 4.3a).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = "backend/Dockerfile.wsp-worker"
TEST_REQUIREMENTS = REPO / "backend" / "requirements-wsp-worker-test.txt"
ALLOWLIST_SOURCE = REPO / "backend" / "src" / "wsp_gateway" / "allowlist.py"
ALLOWLIST_IN_IMAGE = "opt/wsp/wsp_gateway/allowlist.py"
LABEL = "miforge.wsp-image-test"
DEFAULT_TESTS = ("backend/tests/image",)


def run(cmd: Sequence[str], env: dict[str, str] | None = None) -> int:
    """Every external command goes through here (the unit test records the calls)."""
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(list(cmd), cwd=REPO, env=env, check=False).returncode


def output(cmd: Sequence[str]) -> str:
    return subprocess.run(
        list(cmd), cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


# ----------------------------------------------------------------------------- the layer check


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned(text: str) -> dict[str, str]:
    """``name -> version`` of an exact-pin requirements file."""
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==(\S+)", line)
        if m is None:
            raise ValueError(f"not an exact pin: {raw!r}")
        pins[canon(m.group(1))] = m.group(2)
    return pins


@dataclass
class Layer:
    """One layer's entries: ``path -> kind`` ('dir', 'file', 'link', 'other') and file bytes for
    the few files the check reads (``RECORD`` and the allow-list)."""

    entries: dict[str, str] = field(default_factory=dict)
    data: dict[str, bytes] = field(default_factory=dict)


def read_layer(tar: tarfile.TarFile, keep: Iterable[str] = ()) -> Layer:
    keep_set = set(keep)
    layer = Layer()
    for member in tar.getmembers():
        name = member.name.removeprefix("./").rstrip("/")
        if not name:
            continue
        if member.isdir():
            kind = "dir"
        elif member.isreg():
            kind = "file"
        elif member.issym() or member.islnk():
            kind = "link"
        else:
            kind = "other"
        layer.entries[name] = kind
        if kind == "file" and (name.endswith(".dist-info/RECORD") or name in keep_set):
            fh = tar.extractfile(member)
            assert fh is not None
            layer.data[name] = fh.read()
    return layer


def check_layers(
    prod: list[str],
    test: list[str],
    extra: list[Layer],
    base_dirs: set[str],
    test_pins: dict[str, str],
    allowlist: bytes | None,
) -> list[str]:
    """Every way the ``test`` target differs from ``prod`` + test tooling; empty when it does not.

    ``prod``/``test``: the images' layer diff ids. ``extra``: the layers ``test`` adds.
    ``base_dirs``: directories present in the ``prod`` layers. ``allowlist``: the source bytes of
    ``backend/src/wsp_gateway/allowlist.py``, or ``None`` when it does not exist yet."""
    problems: list[str] = []
    if test[: len(prod)] != prod:
        problems.append("the prod image's layers are not a prefix of the test image's")
    if len(extra) != len(test) - len(prod):
        problems.append(f"expected {len(test) - len(prod)} extra layers, got {len(extra)}")
    entries: dict[str, str] = {}
    data: dict[str, bytes] = {}
    for layer in extra:
        entries.update(layer.entries)
        data.update(layer.data)
    allowed: set[str] = set()
    found_dists: dict[str, str] = {}
    for path, body in data.items():
        if not path.endswith(".dist-info/RECORD"):
            continue
        dist_info = path.rsplit("/", 1)[0]
        site = dist_info.rsplit("/", 1)[0]
        name, _, version = dist_info.rsplit("/", 1)[1].removesuffix(".dist-info").partition("-")
        found_dists[canon(name)] = version
        for row in body.decode("utf-8").splitlines():
            rel = row.split(",", 1)[0]
            if rel:
                allowed.add(os.path.normpath(os.path.join(site, rel)))
    if found_dists != test_pins:
        problems.append(
            f"the extra layers install {sorted(found_dists.items())}, "
            f"not exactly {sorted(test_pins.items())}"
        )
    if allowlist is not None:
        allowed.add(ALLOWLIST_IN_IMAGE)
        if data.get(ALLOWLIST_IN_IMAGE) != allowlist:
            problems.append(
                f"{ALLOWLIST_IN_IMAGE} is missing from the test layers or differs from its source"
            )
    parents = {str(Path(p).parent) for p in allowed}
    ancestors = {a for p in parents for a in [p, *(str(x) for x in Path(p).parents)]}
    for path, kind in sorted(entries.items()):
        base = path.rsplit("/", 1)[-1]
        if base.startswith(".wh."):
            problems.append(f"a whiteout (a deletion of prod's file): {path}")
        elif kind == "dir":
            if path not in base_dirs and path not in ancestors:
                problems.append(f"a new directory with no allowed file in it: {path}")
        elif path not in allowed:
            problems.append(f"a {kind} that is neither test tooling nor the allow-list: {path}")
    return problems


def layer_problems(prod_tag: str, test_tag: str) -> list[str]:
    """Run :func:`check_layers` against the two local images (``docker save`` of the test one)."""
    prod = json.loads(
        output(["docker", "image", "inspect", prod_tag, "--format", "{{json .RootFS.Layers}}"])
    )
    test = json.loads(
        output(["docker", "image", "inspect", test_tag, "--format", "{{json .RootFS.Layers}}"])
    )
    allowlist = ALLOWLIST_SOURCE.read_bytes() if ALLOWLIST_SOURCE.exists() else None
    with tempfile.TemporaryDirectory(prefix="wsp-image-layers-") as tmp:
        saved = Path(tmp) / "test.tar"
        if run(["docker", "save", "-o", str(saved), test_tag]) != 0:
            return ["docker save failed"]
        with tarfile.open(saved) as outer:
            manifest = json.load(outer.extractfile("manifest.json"))  # type: ignore[arg-type]
            paths = manifest[0]["Layers"]
            if len(paths) != len(test):
                return [f"docker save lists {len(paths)} layers, inspect {len(test)}"]

            def layer(i: int, keep: Iterable[str] = ()) -> Layer:
                inner = outer.extractfile(paths[i])
                assert inner is not None
                with tarfile.open(fileobj=inner, mode="r:*") as t:
                    return read_layer(t, keep)

            base_dirs = {
                p for i in range(len(prod)) for p, k in layer(i).entries.items() if k == "dir"
            }
            extra = [layer(i, keep=[ALLOWLIST_IN_IMAGE]) for i in range(len(prod), len(test))]
    return check_layers(
        prod, test, extra, base_dirs, pinned(TEST_REQUIREMENTS.read_text()), allowlist
    )


# ----------------------------------------------------------------------------- the job


def build(prefix: str, target: str) -> int:
    return run(
        [
            "docker",
            "build",
            "-f",
            DOCKERFILE,
            "--target",
            target,
            "-t",
            f"{prefix}/wsp-worker:{target}",
            ".",
        ]
    )


def cleanup(run_id: str) -> None:
    for kind in ("container", "volume", "network"):
        listed = output(["docker", kind, "ls", "-q", *(["-a"] if kind == "container" else []),
                         "--filter", f"label={LABEL}={run_id}"])  # fmt: skip
        names = listed.split()
        if names:
            run(["docker", kind, "rm", *(["-f", "-v"] if kind == "container" else []), *names])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prefix", default=os.environ.get("MIFORGE_WSP_IMAGE_PREFIX", "wsp-ci"))
    parser.add_argument("--tests", nargs="+", default=list(DEFAULT_TESTS))
    args = parser.parse_args(argv)
    prefix: str = args.prefix
    prod_tag, test_tag = f"{prefix}/wsp-worker:prod", f"{prefix}/wsp-worker:test"
    run_id = secrets.token_hex(4)
    try:
        for target in ("prod", "test"):
            code = build(prefix, target)
            if code != 0:
                print(f"wsp-worker-image: the {target} build failed", flush=True)
                return code
        problems = layer_problems(prod_tag, test_tag)
        for problem in problems:
            print(f"LAYER CHECK: {problem}", flush=True)
        if problems:
            return 3
        env = {
            **os.environ,
            "MIFORGE_WSP_WORKER_IMAGE": test_tag,
            "MIFORGE_WSP_WORKER_PROD_IMAGE": prod_tag,
            "MIFORGE_REQUIRE_WSP_IMAGE": "1",
            "MIFORGE_WSP_IMAGE_PREFIX": prefix,
            "MIFORGE_WSP_IMAGE_RUN_ID": run_id,
        }
        env.pop("MIFORGE_WSP_IMAGE_SOURCE", None)  # the job runs the built image, never a mount
        return run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q",
             "-rs", *args.tests],
            env=env,
        )  # fmt: skip
    finally:
        cleanup(run_id)


if __name__ == "__main__":
    sys.exit(main())
