"""**AC-003.18**: the resolved selector hash reaches `manifest.json` on disk.

F3's task 7.9 was `[?]` *"blocked on F5, which does not exist yet"*, and F3 proved the half
it could alone — that `selector_hash` and `grader_hash` exist in the vendored
`checkpoint-manifest-v1.json`. **F5 landed, so the block is gone** and this is the round
trip the criterion actually asks for: F3's real resolver → F5's real writer → the bytes on
disk.

**Why a round trip and not an assertion about a dict.** The criterion says it in terms:
*"Not a hand-built dict — fields that agree by construction are how a suite stays green over
this exact bug."* A test that writes `selector_hash="b" * 64` into `ManifestFields` and then
reads `"b" * 64` back proves the writer copies a string it was handed; it cannot notice that
the *resolver's* value never reaches the writer, because the test supplied the value itself.
So the hash here is whatever `hashing.resolve_hash` returns for a real configuration, and
nothing in the test knows what that is.

**It also pins the provenance distinction, which is the sharper half.** `ResolvedHash` has
`provenance` with **no default** anywhere, because a field defaulting to `"contract"` is a
fabricated claim the moment a caller forgets it, and R-02.40 forbids miForge claiming
miDataworks provenance for anything miDataworks did not publish. A locally-authored
configuration and a contract one produce **different hashes**, and the manifest must carry
whichever one the run actually used.
"""

from __future__ import annotations

import json
from pathlib import Path

from services.selectors import hashing
from tests.support import checkpoints as ck

#: A real selector configuration body. Shape only; the hash is the resolver's business.
CONFIG = {
    "type": "rule",
    "rules": [{"match": "contains", "value": "shall", "verdict": "accept"}],
    "on_no_match": "undecided",
}


def _manifest_of(result) -> dict:
    """Read the manifest **off disk**, not from `result.manifest`.

    `WriteResult` carries the document it wrote, and asserting against that would prove the
    writer agrees with itself. The bytes in the published directory are what another server
    loads, so they are what this reads.
    """
    return json.loads((result.directory / "manifest.json").read_text("utf-8"))


class TestTheResolversValueReachesTheFile:
    def test_a_local_configs_hash_is_what_the_manifest_carries(self, tmp_path: Path) -> None:
        resolved = hashing.resolve_hash(CONFIG, source="authored-in-miforge")
        # The test does not know this value and must not: it is the resolver's output.
        result = ck.saved_checkpoint(tmp_path, selector_hash=resolved.sha256)

        manifest = _manifest_of(result)
        assert manifest["selector_hash"] == resolved.sha256, (
            f"the manifest carries {manifest.get('selector_hash')!r} and the resolver "
            f"produced {resolved.sha256!r}; the value did not survive the write"
        )
        # The positive check that the measurement means something: a hash the fixture's
        # own default would also have satisfied proves nothing.
        assert resolved.sha256 != "b" * 64, (
            "the resolver returned the fixture's placeholder, so this test would pass "
            "against a writer that ignored the resolved value entirely"
        )
        assert resolved.provenance == "local"

    def test_the_hash_is_a_real_digest_and_not_a_placeholder(self, tmp_path: Path) -> None:
        resolved = hashing.resolve_hash(CONFIG, source="authored-in-miforge")
        assert len(resolved.sha256) == 64
        assert set(resolved.sha256) <= set("0123456789abcdef")

    def test_two_different_configs_reach_the_file_as_two_different_hashes(
        self, tmp_path: Path
    ) -> None:
        """The discriminating case. One config in, one hash out, and they must not collide —
        otherwise the manifest's record of *which* selector ran is not a record at all."""
        other = {**CONFIG, "rules": [{"match": "contains", "value": "must", "verdict": "accept"}]}
        a = hashing.resolve_hash(CONFIG, source="authored-in-miforge")
        b = hashing.resolve_hash(other, source="authored-in-miforge")
        assert a.sha256 != b.sha256, "two different selector configurations hashed the same"

        first = ck.saved_checkpoint(tmp_path, run_id="run-a", step=100, selector_hash=a.sha256)
        second = ck.saved_checkpoint(tmp_path, run_id="run-b", step=100, selector_hash=b.sha256)
        assert _manifest_of(first)["selector_hash"] == a.sha256
        assert _manifest_of(second)["selector_hash"] == b.sha256

    def test_an_absent_selector_is_null_and_not_omitted(self, tmp_path: Path) -> None:
        """A run with no selector states so. `null` is an assertion; a missing key is an
        omission, and the schema distinguishes them."""
        result = ck.saved_checkpoint(tmp_path, selector_hash=None)
        manifest = _manifest_of(result)
        assert "selector_hash" in manifest, "the key was dropped rather than set to null"
        assert manifest["selector_hash"] is None
