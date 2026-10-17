"""A refusal that resolves by itself says so — without becoming a retry.

**The cost this fixes.** Five of the six refusals miLLM added fall to
`weights_refusal_unmapped`, which is honest (verbatim, nothing fabricated) and **terminal**:
the run stops for good. Two of them — `PAUSE_IN_PROGRESS`, and `MODEL_BUSY` with reason
`weights_operation` — mean *miLLM is mid-save or mid-load and will finish*. Stopping a run
permanently where waiting is correct is a real loss.

**And the fix is deliberately NOT a retry.** `client.py`'s prohibition is one line and
absolute: *"a refusal is never retried — no backoff, no jitter, no 'maybe transient' branch
(R-02.34, FR-011.35)"*, because a blind retry of a weights call can double a write. So the
classification is carried **outward**, in the refusal's details and message, for the operator
to act on. The client's behaviour is unchanged, which `test_no_retry_branch_was_added` pins.

**`RETRY_UNSAFE_OPS` is the sharper half.** "Clears on its own" does not mean "repeat the
call": `commit-end` twice can apply a non-idempotent update twice and `save` twice can publish
two checkpoints claiming one step. For those the note says so and points at R-02.30's
recovery, because a doubled write is worse than a stalled run.
"""

from __future__ import annotations

import pytest

from training.hotupdate import client as client_module
from training.hotupdate import refusals


def _body(code: str, *, reason: str | None = None) -> dict:
    inner: dict = {"code": code, "message": "refused"}
    if reason is not None:
        inner["details"] = {"reason": reason}
    return {"error": inner}


class TestTheTwoTransientOnesAreNamed:
    @pytest.mark.parametrize(
        ("code", "reason"),
        [
            ("PAUSE_IN_PROGRESS", None),
            ("MODEL_BUSY", "weights_operation"),
            ("NOT_ATTACHABLE", "weights_writing"),
            ("NOT_QUIESCED", None),
        ],
    )
    def test_it_is_reported_as_clearing(self, code: str, reason: str | None) -> None:
        caught = refusals.interpret(409, _body(code, reason=reason), op="quiesce")
        assert caught.details[
            "clears_on_its_own"
        ], f"{code}/{reason} stops the run permanently and nothing says it will clear"
        assert "CLEARS WITHOUT INTERVENTION" in caught.message

    def test_the_message_still_says_miforge_does_not_retry(self) -> None:
        """Naming a transient condition must not read as permission to retry."""
        caught = refusals.interpret(409, _body("PAUSE_IN_PROGRESS"), op="quiesce")
        assert "never retries a refusal" in caught.message
        assert "double a write" in caught.message


class TestAnUnknownCodeIsUNKNOWNAndNotTerminal:
    def test_it_carries_None_rather_than_False(self) -> None:
        """`None` is "not classified"; `False` would assert permanence we cannot know."""
        caught = refusals.interpret(409, _body("SOMETHING_NEW"), op="attach")
        assert caught.details["clears_on_its_own"] is None
        assert "CLEARS WITHOUT INTERVENTION" not in caught.message

    def test_a_terminal_condition_is_unchanged(self) -> None:
        """The positive control: BARRIER_UNAVAILABLE is a deployment setting and must NOT be
        reported as clearing, or an operator waits forever for a config change."""
        caught = refusals.interpret(409, _body("BARRIER_UNAVAILABLE", reason="cbm"), op="quiesce")
        assert caught.details["clears_on_its_own"] is None
        assert "CLEARS WITHOUT INTERVENTION" not in caught.message


class TestTheUnsafeOpsSayDoNotRepeatTheCall:
    @pytest.mark.parametrize("op", sorted(refusals.RETRY_UNSAFE_OPS))
    def test_an_unsafe_op_points_at_recovery_instead(self, op: str) -> None:
        caught = refusals.interpret(409, _body("MODEL_BUSY", reason="weights_operation"), op=op)
        assert caught.details["retry_unsafe_op"] is True
        assert "not safe to send twice" in caught.message
        assert "R-02.30" in caught.message

    def test_a_safe_op_does_not_carry_that_warning(self) -> None:
        caught = refusals.interpret(409, _body("PAUSE_IN_PROGRESS"), op="status")
        assert caught.details["retry_unsafe_op"] is False
        assert "not safe to send twice" not in caught.message

    def test_commit_end_and_save_are_both_in_the_unsafe_set(self) -> None:
        """Named explicitly: these are the two whose double-send corrupts rather than annoys.

        `commit-end` twice can apply a non-idempotent AdamW update twice; `save` twice can
        publish two checkpoints claiming the same step.
        """
        assert {"commit-end", "save"} <= refusals.RETRY_UNSAFE_OPS


class TestTheProhibitionStillHolds:
    def test_no_retry_branch_was_added(self) -> None:
        """The client must contain no "maybe transient" branch (R-02.34, FR-011.35).

        Asserted against the module's source rather than its behaviour, because the defect
        this guards is someone later reading `clears_on_its_own` and acting on it *inside* the
        client, where the one-line prohibition lives.
        """
        import inspect

        source = inspect.getsource(client_module)
        assert "clears_on_its_own" not in source, (
            "client.py now reads the clearing classification. That table is information for "
            "the CALLER; acting on it here reintroduces the 'maybe transient' branch the "
            "retry prohibition forbids, and a blind retry of a weights call can double a write."
        )
