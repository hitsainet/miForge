"""Every image's build context can actually reach the LICENSE it embeds.

**Why this exists: the frontend image could not be built outside CI, and nothing said so.**

Both Dockerfiles embed the repository's LICENSE — `backend/Dockerfile` as
``COPY LICENSE /app/LICENSE``, `frontend/Dockerfile` as ``COPY LICENSE /licenses/LICENSE``.
A ``COPY`` of a path outside the build context fails the build, and a glob matching nothing
fails it too, so neither is tolerant of the file's absence. That is the right strictness for
a licence.

The contexts differ, and that is where the gap was:

* the **backend** image builds from ``.``, the repository root, so the root ``LICENSE`` is in
  context and it has always worked;
* the **frontend** image builds from ``./frontend`` (deliberately — the Dockerfile's own
  ``COPY package.json`` lines are relative to that directory), and the root ``LICENSE`` is
  **not** in that context.

CI papered over it with a *"Copy LICENSE to build context"* step that runs ``cp LICENSE
./frontend/`` before the build, guarded so it does not fire when the context is already the
root. So the image built in CI and **could not be built by hand or by `docker compose`** —
`docker build ./frontend` failed on a missing ``LICENSE``, and the Dockerfile's own comment
admitted it: *"A hand-build of this directory needs `cp ../LICENSE frontend/` first."*

That stopped being acceptable on 2026-10-14, when the operator made Docker Compose a
**deployment** path for every app in the suite rather than a dev convenience. An image that
only one machine's CI can build is not a deployment path.

**The fix is the pattern this project already uses for vendored contracts:** a real, committed,
byte-identical copy plus a test that fails on any difference. ``docs/schemas/`` works exactly
this way, and for the same reason — ``cmp`` agreeing today is not enforcement. CI's ``cp``
still runs and now overwrites an identical file, which is idempotent and harmless.

**Not fixed by making the ``COPY`` tolerant**, which was the other option. A glob
(``COPY LICENSE* ...``) that silently matches nothing produces an image with no licence in
it, and nothing downstream would notice — the published artifact would simply be missing the
one file that states its terms. A build that fails loudly is the better failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Each image's build context, as ``docker-images.yml`` resolves it, paired with where the
#: Dockerfile lives. Read from this list rather than from the workflow because the workflow
#: computes the context in shell; if the two ever disagree, the assertions below are about
#: what a *developer* running ``docker build`` would get, which is the case that broke.
CONTEXTS: tuple[tuple[str, str, str], ...] = (
    # (image, build context relative to the repo root, dockerfile relative to the root)
    ("miforge-backend", ".", "backend/Dockerfile"),
    ("miforge-frontend", "frontend", "frontend/Dockerfile"),
)

_COPY_LICENSE = re.compile(r"^\s*COPY\s+(?:--[\w=:-]+\s+)*LICENSE\b", re.MULTILINE)


def _dockerfile(relative: str) -> str:
    return (REPO_ROOT / relative).read_text("utf-8")


class TestEveryContextCanReachTheLicenceItCopies:
    @pytest.mark.parametrize(("image", "context", "dockerfile"), CONTEXTS)
    def test_the_licence_is_inside_the_build_context(
        self, image: str, context: str, dockerfile: str
    ) -> None:
        """The assertion that was false for the frontend until 2026-10-14.

        ``COPY LICENSE`` resolves against the **context**, not the Dockerfile's directory, so
        a context that cannot see the file fails the build — which is what happened to every
        hand-build and would happen to every ``docker compose`` build.
        """
        source = _dockerfile(dockerfile)
        if not _COPY_LICENSE.search(source):
            pytest.skip(f"{dockerfile} does not embed the LICENSE")

        in_context = REPO_ROOT / context / "LICENSE"
        assert in_context.is_file(), (
            f"{dockerfile} does `COPY LICENSE`, but the {image} build context "
            f"({context!r}) has no LICENSE in it, so `docker build {context}` fails and the "
            f"image can only be built by CI's 'Copy LICENSE to build context' step. That is "
            f"not a deployment path."
        )

    @pytest.mark.parametrize(("image", "context", "dockerfile"), CONTEXTS)
    def test_a_vendored_copy_is_byte_identical_to_the_root(
        self, image: str, context: str, dockerfile: str
    ) -> None:
        """A second copy that drifts is worse than no second copy.

        Compared as **bytes**, for the same reason the schema sync test does: a licence that
        differs by a year or a holder is a different licence, and a diff of one character is
        exactly the kind a reader skims past.
        """
        if context == ".":
            pytest.skip("the root context uses the LICENSE itself; there is no copy to drift")
        vendored = REPO_ROOT / context / "LICENSE"
        if not vendored.is_file():
            pytest.fail(f"{context}/LICENSE is missing; see the test above")
        assert vendored.read_bytes() == (REPO_ROOT / "LICENSE").read_bytes(), (
            f"{context}/LICENSE has drifted from the repository root's. It is a vendored "
            f"byte-identical copy so the image can be built without CI; update it from the "
            f"root rather than editing it in place."
        )

    def test_the_vendored_copy_is_not_excluded_by_the_dockerignore(self) -> None:
        """A committed file that BuildKit ignores is not in the context either.

        The subtler half of the same failure: the file would be present in the working tree,
        the test above would pass, and the build would still fail — so the exclusion is
        checked rather than assumed. BuildKit resolves ``.dockerignore`` **per context**, so
        it is the frontend's own file that governs here, not the repository root's.
        """
        ignore = REPO_ROOT / "frontend" / ".dockerignore"
        if not ignore.is_file():
            return
        patterns = [
            line.strip()
            for line in ignore.read_text("utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        offending = [p for p in patterns if p.lstrip("!/").rstrip("/") in {"LICENSE", "*"}]
        assert not offending, (
            f"frontend/.dockerignore excludes the LICENSE via {offending}, so the committed "
            f"copy never reaches the build context and `COPY LICENSE` fails anyway."
        )
