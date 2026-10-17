"""The backend image carries what a training step needs at RUNTIME.

Two defects shipped in `hitsai/miforge-backend` and **neither was visible to this suite**,
because until this file nothing asserted anything about the image at all. Both were found by
executing real GPU tests inside the running cluster pod, which is the only place they could
be found — the whole suite runs from the repository on a machine that has a compiler.

1. **No C compiler, and every Llama training step therefore failed.** `python:3.11-slim`
   ships none: `gcc`, `cc`, `clang`, `g++` and `ld` were all absent, verified in the pod.
   torch 2.14's `_native` registry routes `bmm_outer_product` to a **Triton** implementation,
   reached from `transformers`' Llama rotary embedding, and Triton **JIT-compiles its
   launcher stub at runtime, on the first step**. Isolated repro with no miForge code in it:
   a raw CUDA matmul is fine; a Llama forward on cuda raises `RuntimeError: Failed to find C
   compiler. Please specify via CC environment variable or set triton.knobs.build.impl`. No
   env knob disables the routing and the op ships no eager fallback.
2. **`bitsandbytes` absent while the default `optimizer_state_precision` is `8bit`.** So
   `optimizer.build()` refused `optimizer_precision_unavailable` — correctly, since "miForge
   will not downgrade silently" — which means the deployed trainer could not build its
   default optimizer. ADR-023 makes it the `[bnb]` extra because there is no pure-CPU wheel
   everywhere; the GPU image is exactly where that extra should be installed.

**The stage is the subtle half, and it is why this file parses rather than greps.** Triton
compiles at RUNTIME. A toolchain installed in a builder stage that the final image only
`COPY --from`s would read as a fix and leave the defect exactly in place. So the parser below
resolves the chain of stages the final image is actually *built from* — the last `FROM`, plus
each ancestor it inherits through a named base — and asserts into that chain only.

**It derives the extra's name from `pyproject.toml` instead of spelling `bnb` a second
time.** Every restated name in this project has eventually drifted (42 basenames cited two
ways, `full` vs `change`, `queued` vs `pending`, a self-test planting into
`Selectors_and_Graders` while the real chain is `Selectors_And_Graders`). The one spelling
that matters here is the extra's, so it is read from the file that defines it.

── WHAT THIS PROVES ────────────────────────────────────────────────────────────────────
* The Dockerfile instructs a C compiler to be installed into the filesystem of the image
  that is actually published, not into a discarded builder stage.
* The Dockerfile installs the extra that `pyproject.toml` declares `bitsandbytes` under, by
  that extra's real name, into that same filesystem.
* `bitsandbytes` is **not** a core dependency, so a cardless developer's
  `pip install -e ".[dev]"` is unaffected (ADR-023) — the other half of defect 2, which a
  "fix" that promoted it to `dependencies` would silently break.

── WHAT IT DOES NOT PROVE ──────────────────────────────────────────────────────────────
* **Not that the image builds.** No build happens here.
* **Not that `gcc` is sufficient for Triton.** That was established by reading
  `triton/runtime/build.py` *inside* the image (`_find_compiler("c")` reads `CC`, then
  `clang`, then `gcc`; the only `language="c++"` caller is the AMD/HIP backend) and by a
  hand build — not by this file. A future Triton that compiles C++ would pass here and fail
  in the pod.
* **Not that `import bitsandbytes` succeeds**, nor that it finds a usable CUDA runtime.
* **Not that a training step completes.** Only a run on the card proves that, and the GPU
  tests under `backend/tests/integration/` (marker `gpu`, deselected by default) are where
  that lives.

The runtime half is checked by hand against a built image, and the commands are written down
here so the next reader repeats them rather than reinventing them:

    docker build -f backend/Dockerfile -t miforge-backend:verify .
    docker run --rm --entrypoint bash miforge-backend:verify -lc \
        'command -v cc gcc; python -c "import bitsandbytes; print(bitsandbytes.__version__)"'

**This test never skips.** It reads two files that are always in the repository and imports
nothing but the standard library — no `yaml`, no `docker`, no GPU, no network. A test that
skips on the machine where the defect lives is worth nothing, and this project has been
bitten by exactly that twice in one day (a `pyarrow` refusal whose test skipped the moment
the dependency landed; thirteen tests that passed by taking an absent-`torch` branch).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "backend" / "Dockerfile"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Debian/Alpine packages that put a C compiler on `PATH`. Triton looks for `clang` then
#: `gcc` (`triton/runtime/build.py::_find_compiler`), so either satisfies it;
#: `build-essential` is the heavy superset and is accepted rather than preferred.
C_COMPILER_PACKAGES: tuple[str, ...] = ("gcc", "clang", "build-essential")


# ======================================================================================
# A Dockerfile parser, because a grep cannot tell one stage from another
# ======================================================================================
def _instructions(text: str) -> list[tuple[int, str]]:
    """``[(line number, one logical instruction)]`` — continuations joined, comments gone.

    Comment-only lines are dropped whether they stand alone or sit inside a `\\`
    continuation, which is what Docker itself does. Dropping them matters: this repository's
    Dockerfile explains its own defects in prose, so the words `gcc` and `[bnb]` appear in
    comments, and a parser that kept them would pass against a file that installs neither.
    """
    out: list[tuple[int, str]] = []
    buf: str | None = None
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue  # blank or comment, inside a continuation or not
        if buf is None:
            buf, start = "", number
        if stripped.endswith("\\"):
            buf += stripped[:-1].rstrip() + " "
        else:
            out.append((start, buf + stripped))
            buf = None
    if buf is not None:  # a trailing `\` with nothing after it
        out.append((start, buf.rstrip()))
    return out


def _stages(instructions: list[tuple[int, str]]) -> list[dict[str, Any]]:
    """Split instructions into build stages, one per `FROM`."""
    stages: list[dict[str, Any]] = []
    for number, instruction in instructions:
        verb, _, rest = instruction.partition(" ")
        if verb.upper() == "FROM":
            tokens = [t for t in rest.split() if not t.startswith("--")]
            alias = None
            if len(tokens) >= 3 and tokens[-2].upper() == "AS":
                alias = tokens[-1].lower()
            stages.append(
                {
                    "line": number,
                    "base": tokens[0].lower() if tokens else "",
                    "alias": alias,
                    "instructions": [],
                }
            )
        elif stages:
            stages[-1]["instructions"].append((number, instruction))
    return stages


def _runtime_chain(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The stages whose instructions end up in the published image, base-first.

    `docker build` with no `--target` builds the **last** stage, so that is the runtime
    stage. It may inherit from an earlier named stage (`FROM builder AS runtime`), in which
    case that ancestor's filesystem is part of the image too and its `RUN`s count. A stage
    that the final image merely `COPY --from`s is **not** in this chain, which is the whole
    point: Triton compiles at runtime, so a compiler installed only there is not installed.
    """
    by_alias = {s["alias"]: s for s in stages if s["alias"]}
    chain: list[dict[str, Any]] = []
    current = stages[-1]
    seen: set[int] = {id(current)}
    while True:
        chain.insert(0, current)
        parent = by_alias.get(current["base"])
        if parent is None or id(parent) in seen:
            break
        seen.add(id(parent))
        current = parent
    return chain


@pytest.fixture(scope="module")
def runtime_instructions() -> list[tuple[int, str]]:
    assert DOCKERFILE.is_file(), f"no Dockerfile at {DOCKERFILE}"
    stages = _stages(_instructions(DOCKERFILE.read_text("utf-8")))
    assert stages, f"{DOCKERFILE} declares no FROM"
    chain = _runtime_chain(stages)
    return [entry for stage in chain for entry in stage["instructions"]]


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    assert PYPROJECT.is_file(), f"no pyproject at {PYPROJECT}"
    return tomllib.loads(PYPROJECT.read_text("utf-8"))


def _runs(instructions: list[tuple[int, str]]) -> list[tuple[int, str]]:
    return [(n, i) for n, i in instructions if i.partition(" ")[0].upper() == "RUN"]


def _requirement_name(requirement: str) -> str:
    """`"bitsandbytes>=0.44,<1"` -> `"bitsandbytes"`, extras and markers discarded."""
    head = re.split(r"[;\[]", requirement, maxsplit=1)[0]
    return re.split(r"[<>=!~\s]", head, maxsplit=1)[0].strip().lower().replace("_", "-")


# ======================================================================================
# The parser has to be right, so it is checked against known inputs
# ======================================================================================
class TestTheParserItself:
    """A checker nobody has seen fail is not known to work.

    Every fresh checker in this project that disagreed with correct work turned out to be
    the wrong one — four rounds of that on the review harness — so this one is pinned
    against inputs whose answers are known before it is pointed at the real file.
    """

    def test_a_comment_mentioning_a_compiler_is_not_an_installation(self) -> None:
        text = "FROM scratch\n# gcc is deliberately not installed here\nRUN true\n"
        runtime = _runtime_chain(_stages(_instructions(text)))
        joined = " ".join(i for _, i in runtime[-1]["instructions"])
        assert "gcc" not in joined

    def test_a_comment_inside_a_continuation_is_dropped(self) -> None:
        text = "FROM scratch\nRUN true \\\n# gcc\n    && false\n"
        instructions = _instructions(text)
        assert instructions == [(1, "FROM scratch"), (2, "RUN true && false")]

    def test_a_builder_only_toolchain_is_not_in_the_runtime_chain(self) -> None:
        """The exact trap: a compiler installed in a stage the image only copies FROM."""
        text = (
            "FROM python:3.11-slim AS builder\n"
            "RUN apt-get install -y gcc\n"
            "FROM python:3.11-slim\n"
            "COPY --from=builder /app /app\n"
        )
        runtime = [
            e for s in _runtime_chain(_stages(_instructions(text))) for e in s["instructions"]
        ]
        assert not [i for _, i in _runs(runtime) if "gcc" in i]

    def test_an_inherited_stage_is_in_the_runtime_chain(self) -> None:
        """The converse, so the check above is a real discrimination and not a blanket no."""
        text = (
            "FROM python:3.11-slim AS base\n"
            "RUN apt-get install -y gcc\n"
            "FROM base AS runtime\n"
            "RUN pip install .\n"
        )
        runtime = [
            e for s in _runtime_chain(_stages(_instructions(text))) for e in s["instructions"]
        ]
        assert [i for _, i in _runs(runtime) if "gcc" in i]

    def test_requirement_names_are_parsed_off_their_specifiers(self) -> None:
        assert _requirement_name("bitsandbytes>=0.44,<1") == "bitsandbytes"
        assert _requirement_name('torch>=2.4,<3; platform_system == "Linux"') == "torch"
        assert _requirement_name("uvicorn[standard]>=0.32,<0.33") == "uvicorn"


# ======================================================================================
# D1 — a C compiler, in the stage that becomes the image
# ======================================================================================
class TestTritonHasACompilerAtRuntime:
    def test_a_c_compiler_is_installed_into_the_runtime_image(
        self, runtime_instructions: list[tuple[int, str]]
    ) -> None:
        installers = [
            (n, i)
            for n, i in _runs(runtime_instructions)
            if re.search(r"\b(apt-get|apt|apk|dnf|yum)\b", i) and re.search(r"\b(install|add)\b", i)
        ]
        found = [
            (n, package)
            for n, i in installers
            for package in C_COMPILER_PACKAGES
            if re.search(rf"(?<![\w.+-]){re.escape(package)}(?![\w.+-])", i)
        ]
        assert found, (
            "No C compiler is installed into the stage that becomes "
            "`hitsai/miforge-backend`. Triton JIT-compiles its launcher stub AT RUNTIME on "
            "the first training step — torch's `_native` registry routes "
            "`bmm_outer_product` to a Triton implementation with no eager fallback, reached "
            "from every Llama forward pass — so without one, every training step in the "
            "cluster fails with `RuntimeError: Failed to find C compiler`. Install one of "
            f"{C_COMPILER_PACKAGES} in the runtime stage; a builder stage the final image "
            "only COPY --from=s does NOT fix this, because the compile happens in the "
            "running container."
        )

    def test_torch_is_still_what_makes_that_necessary(self, pyproject: dict[str, Any]) -> None:
        """The premise, asserted rather than assumed: `torch` brings Triton in.

        If `torch` ever stops being a core dependency the test above is arguing from a cause
        that no longer exists, and a reader deserves to be told that here rather than
        deduce it. It is not a licence to drop the compiler — `transformers` and `peft` both
        pull `torch` regardless — only a marker that the reasoning needs re-deriving.
        """
        core = {_requirement_name(r) for r in pyproject["project"]["dependencies"]}
        assert "torch" in core


# ======================================================================================
# D8 — the bnb extra, installed in the image and nowhere near the core dependencies
# ======================================================================================
class TestTheDefaultOptimizerCanBeBuilt:
    @pytest.fixture(scope="class")
    def bnb_extra(self, pyproject: dict[str, Any]) -> str:
        """The name `pyproject.toml` declares `bitsandbytes` under — read, never restated."""
        extras = pyproject["project"].get("optional-dependencies", {})
        carriers = [
            name
            for name, requirements in extras.items()
            if any(_requirement_name(r) == "bitsandbytes" for r in requirements)
        ]
        assert len(carriers) == 1, (
            "Expected exactly one extra to carry `bitsandbytes`, found "
            f"{sorted(carriers)}. Two extras carrying it is two sources of truth for the "
            "version range, and zero means the GPU image has no extra to install."
        )
        return carriers[0]

    def test_the_extra_is_installed_into_the_runtime_image(
        self, runtime_instructions: list[tuple[int, str]], bnb_extra: str
    ) -> None:
        installed: set[str] = set()
        for _, instruction in _runs(runtime_instructions):
            if not re.search(r"\bpip\b[^&|;]*\binstall\b", instruction):
                continue
            for group in re.findall(r"\[([A-Za-z0-9_.,\-]+)\]", instruction):
                installed.update(part.strip().lower() for part in group.split(","))
        assert bnb_extra.lower() in installed, (
            f"The `[{bnb_extra}]` extra is not installed into the stage that becomes "
            "`hitsai/miforge-backend`. The default `optimizer_state_precision` is `8bit`, so "
            "`optimizer.build()` refuses `optimizer_precision_unavailable` rather than "
            "downgrading silently — which means the deployed trainer cannot build its "
            "default optimizer at all. An extra is a statement about who may SKIP a "
            "dependency (ADR-023: a cardless developer), never a licence for the GPU image "
            f"to ship without it. Extras seen installed: {sorted(installed) or 'none'}."
        )

    def test_bitsandbytes_is_not_a_core_dependency(self, pyproject: dict[str, Any]) -> None:
        """The other half, and the one a careless fix breaks.

        Promoting `bitsandbytes` to `dependencies` would make the test above pass and break
        ADR-023: there is no pure-CPU wheel everywhere, so a cardless developer could no
        longer install the package or run the suite. The image installs the extra; the extra
        stays an extra.
        """
        core = {_requirement_name(r) for r in pyproject["project"]["dependencies"]}
        assert "bitsandbytes" not in core

    def test_the_dockerfile_names_no_extra_pyproject_does_not_declare(
        self, runtime_instructions: list[tuple[int, str]], pyproject: dict[str, Any]
    ) -> None:
        """A `pip install .[bnb]` against an extra that was renamed installs **nothing**.

        pip does not fail on an unknown extra; it warns and carries on, so a rename in
        `pyproject.toml` would silently restore defect 2 with a green build. This is the
        assertion that makes the two files one contract rather than two spellings.
        """
        declared = {name.lower() for name in pyproject["project"].get("optional-dependencies", {})}
        for number, instruction in _runs(runtime_instructions):
            if not re.search(r"\bpip\b[^&|;]*\binstall\b", instruction):
                continue
            for group in re.findall(r"\[([A-Za-z0-9_.,\-]+)\]", instruction):
                for part in group.split(","):
                    name = part.strip().lower()
                    assert name in declared, (
                        f"{DOCKERFILE.name} line {number} installs the extra `[{name}]`, "
                        f"which `pyproject.toml` does not declare (it has {sorted(declared)}). "
                        "pip only WARNS on an unknown extra, so this installs nothing and "
                        "the build still goes green."
                    )
