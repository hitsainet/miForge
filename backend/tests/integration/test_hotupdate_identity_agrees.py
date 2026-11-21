"""The attach-time identity and the status-time identity must describe one model.

**The defect this exists for.** F11 read the identity out of two different responses with
two different, invented field names: `attach.py` took
`response["resident_model"] or response["model"]` and a top-level `manifest_sha256`, while
`state.read` took the same from the status body. miLLM sends an **object** in both — `model`
on attach (FTDD 031 §5.3) and `resident_model` on status (FPRD FR-31.16.1), each
`{model_id, name, manifest_hash}` — so `str(...)` of the fallback produced a stringified
Python dict and the hash was always `None`.

Correcting one side alone is what makes this test necessary rather than nice to have. Once
`state.read` derived the real name and `attach` still produced `"{'model_id': 7, ...}"`, the
two sides disagreed about the same unchanged model, `differs_from` reported `model`, and the
**first `require_clean` of every §5.10 run would refuse `weights_identity_changed`** —
diagnosing "the serving process restarted" for a process that had not restarted. A wrong
diagnosis ends an investigation, which is exactly what FR-011's refusal discipline forbids.

No existing test could see it: every F11 test that touches attach and every test that
touches status passed, because neither compared the two. **The invariant lives between
them**, so the assertion has to as well.
"""

from __future__ import annotations

from tests.support.fixtures_hotupdate import Harness
from training.hotupdate import state

# ⚠ NO LOCAL DERIVATION HERE, AND THAT IS THE WHOLE POINT.
#
# The first version of this file defined its own `_identity_from_attach` that re-read the
# `model` block exactly as `attach.py` did, justified by "the acquirer's `acquire` needs
# torch and a card". **That made the test vacuous**: reverting the production fix as a
# negative control left it at 3 passed, because the test was asserting against its own copy
# of the logic rather than against the code that ships.
#
# The fix was to give the derivation ONE home — `state.identity_from_attach` — which the
# acquirer calls and this test calls. So the control now bites, and the duplicate reading
# that caused the original defect is gone as well. *A test that reimplements the thing it
# tests agrees with itself by construction, which is the fixture-shaped failure ADR-016
# names, arriving as a helper instead of a fixture.*


class TestTheTwoResponsesDescribeOneModel:
    def test_the_attach_identity_and_the_status_identity_do_not_differ(
        self, harness: Harness
    ) -> None:
        """**The regression.** Nothing changed between the two reads, so nothing may differ."""
        response = harness.attach()
        at_attach = state.identity_from_attach(response)
        _, at_status = state.read(harness.client)

        assert at_attach.differs_from(at_status) == [], (
            f"attach and status disagree about an unchanged model: "
            f"attach={at_attach.model!r} hash={at_attach.manifest_sha256!r} vs "
            f"status={at_status.model!r} hash={at_status.manifest_sha256!r}. "
            f"The first require_clean of every run would refuse weights_identity_changed "
            f"and blame a serving restart that did not happen."
        )

    def test_neither_side_is_a_stringified_dict(self, harness: Harness) -> None:
        """The positive check that the measurement means something.

        Both sides agreeing is necessary and not sufficient: two identical stringified dicts
        also agree. This is what distinguishes "reads the field" from "reads the object".
        """
        response = harness.attach()
        at_attach = state.identity_from_attach(response)
        _, at_status = state.read(harness.client)
        for label, value in (("attach", at_attach.model), ("status", at_status.model)):
            assert (
                "{" not in value and "'" not in value
            ), f"the {label} identity is a stringified object, not a name: {value!r}"
        assert at_attach.model, "the attach identity is empty, so it cannot discriminate"

    def test_the_hash_is_read_from_INSIDE_the_model_object(self, harness: Harness) -> None:
        """`manifest_hash` is nested, and is `null` until miLLM's Feature 32 supplies it.

        The fake states one, so a reader looking at the top level gets `None` here while the
        real field is present — the difference between "unverified" and "not read".
        """
        response = harness.attach()
        assert response["model"]["manifest_hash"] is not None, (
            "the fake must state a hash, or this test cannot tell a nested read from a "
            "top-level one"
        )
        assert response.get("manifest_sha256") is None, (
            "there is no top-level manifest_sha256 in miLLM's contract; if the fake grew "
            "one, the defect this file guards could return invisibly"
        )
        assert state.identity_from_attach(response).manifest_sha256 is not None
