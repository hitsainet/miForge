"""FR-011.7's build check: wired at `attach.py:261`, and until now tested by nothing.

**Why this file exists.** A sweep of F11 found that its reader took four field names from
BRD-02's prose instead of from miLLM's committed FTDD, and that three of the four failed
*silently*. This one was the worst of the three: ``check_build_identity`` read
``response["build"]`` while the contract sends ``producer`` (miLLM FTDD 031 §5.3), so on
every real attach response the key was absent, the function took its
``build_identity_absent`` branch, logged a warning and **returned** — and FR-011.7's
*"refuse an incompatible build before rebuilding a single tensor"* was a no-op.

Correcting the key was not enough, and that is the transferable part. Reverting
``producer`` back to ``build`` as a negative control left **283 tests passing**, because
``check_build_identity`` had **no test at all** — grep for it across ``tests/`` found
nothing. It was wired in production and wholly unexercised, so the fix was as unprotected
as the defect had been. ADR-016 #1 in its plainest form: *a capability is not shipped until
a test fails when its wiring is removed.*

The load-bearing test here is
:meth:`TestItReadsTheNameTheContractActuallySends.test_the_real_attach_response_is_not_read_as_absent`.
The others check the comparison; only that one catches a wrong **key**, because a wrong key
and a compatible build are indistinguishable by the function's return value — both are
``None`` — and can only be told apart by the log line the absent branch emits.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import rebuild

#: The key miLLM actually sends (FTDD 031 §5.3), and the one F11 invented. Named together
#: so the distinction is in one place rather than spelled twice per test.
CONTRACT_KEY = "producer"
INVENTED_KEY = "build"

LOCAL = rebuild.BuildIdentity(torch="2.4.1", cuda="12.1", driver="580.178")


def _response(**producer: str) -> dict[str, object]:
    return {CONTRACT_KEY: dict(producer)}


class TestItReadsTheNameTheContractActuallySends:
    def test_the_real_attach_response_is_not_read_as_absent(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        """**The regression for the silent no-op.** Driven by the fake's real response.

        The fake emits miLLM's shape, so if the reader looks for any other key this check
        takes the absent branch — which is exactly what shipped. Asserting on the
        *absence of that log line* is the only thing that can tell "compatible build" from
        "I looked for the wrong field", since both return ``None``.
        """
        response = harness.attach()
        assert CONTRACT_KEY in response, (
            f"the fake no longer sends {CONTRACT_KEY!r}; miLLM FTDD 031 §5.3 says it does, "
            f"and without it every assertion in this file is vacuous"
        )
        # `local=LOCAL` matches the triple the fake states, so this test turns on WHICH KEY
        # IS READ and not on version equality. Passing no `local` compares against this
        # machine's real torch and refuses `incompatible_torch_build` — which is the
        # corrected check working end to end against the fake, and is asserted as such by
        # `test_it_fires_against_the_fake_when_the_builds_really_differ` below.
        with caplog.at_level("WARNING"):
            rebuild.check_build_identity(response, local=LOCAL)
        assert not [
            r for r in caplog.records if r.__dict__.get("code") == "build_identity_absent"
        ], (
            "check_build_identity reported the build identity as absent while reading a "
            "real attach response, which means it is looking for a field the contract does "
            "not carry. That is the defect this file was written for: FR-011.7 becomes a "
            "no-op and nothing else notices."
        )

    def test_it_fires_against_the_fake_when_the_builds_really_differ(
        self, harness: Harness
    ) -> None:
        """End to end: the fake's real response, this machine's real build, a refusal.

        This is the assertion that the no-op is gone. Before the key was corrected the
        same call returned ``None`` and logged `build_identity_absent`, whatever the two
        builds were — so a genuinely incompatible pair rebuilt tensors anyway.
        """
        response = harness.attach()
        with pytest.raises(Refusal) as caught:
            rebuild.check_build_identity(response)
        assert caught.value.code == "incompatible_torch_build"
        assert "torch" in caught.value.details["mismatched"]

    def test_the_invented_key_is_NOT_honoured(self, caplog: pytest.LogCaptureFixture) -> None:
        """A compatibility fallback would make the wrong name work again, invisibly.

        A translation table between two names for one concept is a second source of truth —
        the family behind `step-`, `full` vs `change` and `queued` vs `pending` here.
        """
        with caplog.at_level("WARNING"):
            rebuild.check_build_identity(
                {INVENTED_KEY: {"torch": "0.0.1", "cuda": "0.0", "driver": "0"}}, local=LOCAL
            )
        assert any(r.__dict__.get("code") == "build_identity_absent" for r in caplog.records), (
            f"{INVENTED_KEY!r} was honoured. It is F11's own coinage and appears in no "
            f"version of BRD-01 or of miLLM's FTDD; a build triple that wild must not be "
            f"compared, it must be reported as absent."
        )


class TestTheComparison:
    def test_a_matching_build_passes_quietly(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            rebuild.check_build_identity(
                _response(torch="2.4.1", cuda="12.1", driver="580.178"), local=LOCAL
            )
        assert not caplog.records, "a matching build must not warn about anything"

    @pytest.mark.parametrize(("field", "value"), [("torch", "2.10.0+cu128"), ("cuda", "12.8")])
    def test_a_mismatch_refuses_and_names_BOTH_triples(self, field: str, value: str) -> None:
        """Both sides, because an operator who sees one cannot tell which to change."""
        stated = {"torch": "2.4.1", "cuda": "12.1", "driver": "580.178"} | {field: value}
        with pytest.raises(Refusal) as caught:
            rebuild.check_build_identity(_response(**stated), local=LOCAL)
        assert caught.value.code == "incompatible_torch_build"
        assert field in caught.value.details["mismatched"]
        assert value in caught.value.details["serving_build"]
        assert "2.4.1" in caught.value.details["miforge_build"] or field == "torch"

    def test_the_driver_alone_does_not_refuse(self) -> None:
        """Only `torch` and `cuda` gate the handle; the driver is recorded, not compared.

        Asserted rather than left implicit: widening the comparison to the driver would
        refuse every legitimate attach after a host driver update, and narrowing it to
        `torch` alone would let a CUDA mismatch through. The pair is the decision.
        """
        rebuild.check_build_identity(
            _response(torch="2.4.1", cuda="12.1", driver="999.999"), local=LOCAL
        )

    def test_a_side_that_states_nothing_for_a_field_is_not_compared(self) -> None:
        """`None` means unstated, and an unstated field cannot mismatch."""
        rebuild.check_build_identity(_response(torch="2.4.1"), local=LOCAL)


class TestTheDriverIsReadFromTheKernelModule:
    """The NVIDIA driver in the local triple. Until 2026-10-16 it came from
    ``torch.cuda.driver_version``, which torch does not have - swallowed, so it was ``None``
    on every machine including the GPU node. Found by MyPy strict, not by a test."""

    def test_the_version_line_is_parsed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        version = tmp_path / "version"
        version.write_text(
            "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  Tue Jul  7 12:38:21 "
            "UTC 2026\nGCC version:  gcc version 13.3.0\n"
        )
        assert rebuild.nvidia_driver_version(str(version)) == "580.178.04"

    def test_no_driver_is_unknown_not_a_placeholder(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        assert rebuild.nvidia_driver_version(str(tmp_path / "absent")) is None

    def test_the_local_triple_uses_it(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setattr(rebuild, "nvidia_driver_version", lambda: "580.178.04")
        assert rebuild.local_build_identity().driver == "580.178.04"
