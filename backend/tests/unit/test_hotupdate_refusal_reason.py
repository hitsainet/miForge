"""miLLM's structured ``reason`` is carried, and the message defers to it.

**The defect.** `interpret` read the upstream code and message and **discarded the reason**,
while `_MESSAGES["resident_model_not_attachable"]` asserts a cause: *"the tensor layout must
be plain FP16 or BF16 (BRD-01 R-01.5 refuses a quantized model)."*

Against BRD-01 **as committed** that assertion is correct — R-01.5 and acceptance criterion
5 make a quantized model the only `NOT_ATTACHABLE` case — so this was not a contract
mismatch. It was the *shape* of a wrong diagnosis waiting for a second reason to exist, and
miLLM's implementation already sends one (`reason: weights_writing`, attach while writing).
Asserting a cause while throwing away the cause the upstream stated is the guess ADR-018
forbids, wearing a diagnosis's clothes — and a wrong diagnosis ends an investigation.

The fix carries `upstream_reason` always and has the message **name** it rather than replace
the mapped text: the mapped message holds the action to take, the reason holds the fact, and
an operator needs both. It also states which wins, because a build whose mapping is behind
miLLM's contract should say so rather than quietly mislead.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from training.hotupdate import refusals


def _body(code: str, *, reason: str | None = None, nest: str = "error.details") -> dict:
    """A refusal body with the reason at one of the places it could legitimately sit."""
    inner: dict = {"code": code, "message": "refused"}
    if reason is not None:
        if nest == "error.details":
            inner["details"] = {"reason": reason}
        elif nest == "error":
            inner["reason"] = reason
    body = {"error": inner}
    if reason is not None and nest == "top":
        body["reason"] = reason
    return body


class TestTheReasonIsCarried:
    @pytest.mark.parametrize("nest", ["error.details", "error", "top"])
    def test_it_is_read_from_every_plausible_nesting(self, nest: str) -> None:
        """Defensive, like `_code_of` and `_message_of`.

        miLLM puts it in the error's `details`; a build that moves it must make the reason
        *absent*, never make this raise.
        """
        caught = refusals.interpret(
            400, _body("NOT_ATTACHABLE", reason="weights_writing", nest=nest), op="attach"
        )
        assert caught.details["upstream_reason"] == "weights_writing", nest

    def test_the_message_NAMES_the_reason_and_says_which_wins(self) -> None:
        caught = refusals.interpret(
            400, _body("NOT_ATTACHABLE", reason="weights_writing"), op="attach"
        )
        assert "weights_writing" in caught.message, (
            "the refusal asserts a cause and does not mention the reason miLLM stated; an "
            "operator reading it would chase quantization while the model was being written"
        )
        assert (
            "AUTHORITATIVE" in caught.message
        ), "the message must say the upstream reason wins over this build's mapping"
        # The action to take is still there - the reason is added, not substituted.
        assert "load" in caught.message.lower()

    def test_an_absent_reason_is_absent_and_the_message_is_UNCHANGED(self) -> None:
        """The addition must not fire spuriously.

        Against BRD-01 as committed, `NOT_ATTACHABLE` with no reason means a quantized
        model, and the mapped message says exactly that. Appending a sentence about a reason
        that was never stated would be its own fabrication.
        """
        caught = refusals.interpret(400, _body("NOT_ATTACHABLE"), op="attach")
        assert caught.details["upstream_reason"] is None
        assert "AUTHORITATIVE" not in caught.message
        assert caught.message == refusals._MESSAGES["resident_model_not_attachable"]

    def test_the_unmapped_branch_carries_it_too(self) -> None:
        """The branch that matters: an unknown code stops the run, and the reason is the
        only structured clue about why."""
        caught = refusals.interpret(
            409, _body("BARRIER_UNAVAILABLE", reason="continuous_batching"), op="attach"
        )
        assert caught.code == "weights_refusal_unmapped"
        assert caught.details["upstream_reason"] == "continuous_batching"

    def test_a_non_dict_body_does_not_raise(self) -> None:
        """An unparseable refusal is itself a refusal, never a partially read object."""
        for body in (None, "gateway timeout", [1, 2], {"error": "flat"}):
            caught = refusals.interpret(502, body, op="attach")
            assert isinstance(caught, Refusal)
            assert caught.details["upstream_reason"] is None
