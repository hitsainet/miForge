"""The judge is never the resident model  (AC-003.5, AC-003.6; **R-02.11**, E-6, E-7).

This is the feature's load-bearing guarantee. R-02.11 is the stated mitigation for PPRD
§8's recursion risk, and the failure it prevents is **silent**: a model that grades its own
outputs and trains on the winners converges on whatever it already likes - loss falls, the
test score does not move, and nothing errors.

So the guard is tested at **all three call sites**, with all three outcomes, and with the
normalisation variants it exists for. *A guard called from two of three places is a guard
that is absent in the third, and the third is the one a long run reaches.*

**Mutation control 1** for this feature is: flip the comparison in
:func:`services.selectors.identity.check_judge_identity` so a matching judge is *permitted*.
Several tests here must go red. Result recorded in the feature report.
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from providers.base import ProviderIdentity
from services.selectors.identity import (
    CAVEAT_SHARES_LINEAGE,
    check_judge_identity,
    guard_at_preflight,
    guard_at_resolution,
    guard_on_identity_change,
    normalise_base_url,
    normalise_model,
)

ALL_THREE_CALL_SITES = (guard_at_resolution, guard_at_preflight, guard_on_identity_change)

RESIDENT = ProviderIdentity(kind="in_process", model="lfm2-1.2b")
SERVED_RUN_TARGET = ProviderIdentity(
    kind="openai_compatible", model="lfm2-1.2b", base_url="http://millm:8000"
)
DIFFERENT_JUDGE = ProviderIdentity(
    kind="openai_compatible", model="qwen2.5-7b-instruct", base_url="http://judge:8000"
)


@pytest.mark.parametrize("call_site", ALL_THREE_CALL_SITES, ids=lambda f: f.__name__)
class TestAllThreeCallSites:
    """Each outcome at each site. The parametrisation IS the point of this class."""

    def test_an_in_process_judge_is_refused_unconditionally(self, call_site) -> None:
        with pytest.raises(Refusal) as exc:
            call_site(RESIDENT, SERVED_RUN_TARGET)
        assert exc.value.code == "judge_is_resident_model"

    def test_an_in_process_judge_is_refused_even_with_no_run_target(self, call_site) -> None:
        """Unconditional means unconditional: no comparison is consulted.

        If this ever passes, someone has written a comparison for the in-process case, and
        a comparison implies there is a case where in-process is an acceptable judge.
        """
        with pytest.raises(Refusal) as exc:
            call_site(RESIDENT, None)
        assert exc.value.code == "judge_is_resident_model"

    def test_an_identical_http_target_is_refused(self, call_site) -> None:
        with pytest.raises(Refusal) as exc:
            call_site(SERVED_RUN_TARGET, SERVED_RUN_TARGET)
        assert exc.value.code == "judge_is_resident_model"

    def test_a_different_model_is_permitted(self, call_site) -> None:
        assert call_site(DIFFERENT_JUDGE, SERVED_RUN_TARGET) == ()

    def test_the_same_model_elsewhere_is_permitted_with_a_caveat(self, call_site) -> None:
        """E-7, OQ-3: a different parameter set of the same lineage.

        Permitted, because the strict reading removes a cheap and genuinely useful judge -
        but the caveat is what makes the choice auditable rather than invisible.
        """
        elsewhere = ProviderIdentity(
            kind="openai_compatible", model="lfm2-1.2b", base_url="http://base-snapshot:8000"
        )
        assert call_site(elsewhere, SERVED_RUN_TARGET) == (CAVEAT_SHARES_LINEAGE,)


class TestBothIdentitiesAreInTheDetails:
    def test_the_refusal_carries_both_identities(self) -> None:
        """FR-003.17: a refusal whose details omit the figures makes the operator guess."""
        with pytest.raises(Refusal) as exc:
            check_judge_identity(SERVED_RUN_TARGET, SERVED_RUN_TARGET)
        details = exc.value.details
        assert details["judge"]["model"] == "lfm2-1.2b"
        assert details["run_target"]["base_url"] == "http://millm:8000"

    def test_the_in_process_refusal_carries_the_run_target_too(self) -> None:
        with pytest.raises(Refusal) as exc:
            check_judge_identity(RESIDENT, SERVED_RUN_TARGET)
        assert exc.value.details["judge"]["kind"] == "in_process"
        assert exc.value.details["run_target"] is not None


class TestTheNormaliserIsTheGuard:
    """An unnormalised comparison is a guard a trailing slash defeats."""

    @pytest.mark.parametrize(
        ("judge_url", "target_url"),
        [
            ("http://millm:8000/", "http://millm:8000"),
            ("HTTP://MiLLM:8000", "http://millm:8000"),
            ("http://millm", "http://millm:80"),
            ("https://millm", "https://millm:443"),
            ("http://millm:8000//", "http://millm:8000"),
            ("http://millm:8000?x=1", "http://millm:8000"),
        ],
        ids=[
            "trailing-slash",
            "case",
            "default-port-http",
            "default-port-https",
            "double-slash",
            "query",
        ],
    )
    def test_equivalent_spellings_are_still_refused(self, judge_url, target_url) -> None:
        judge = ProviderIdentity(kind="openai_compatible", model="m", base_url=judge_url)
        target = ProviderIdentity(kind="openai_compatible", model="m", base_url=target_url)
        with pytest.raises(Refusal) as exc:
            check_judge_identity(judge, target)
        assert exc.value.code == "judge_is_resident_model"

    def test_the_model_name_case_does_not_defeat_the_guard(self) -> None:
        judge = ProviderIdentity(
            kind="openai_compatible", model="LFM2-1.2B", base_url="http://millm:8000"
        )
        with pytest.raises(Refusal):
            check_judge_identity(judge, SERVED_RUN_TARGET)

    def test_a_genuinely_different_host_is_not_a_false_positive(self) -> None:
        """A guard that cried wolf would get switched off, and then it protects nothing."""
        judge = ProviderIdentity(
            kind="openai_compatible", model="lfm2-1.2b", base_url="http://millm2:8000"
        )
        assert check_judge_identity(judge, SERVED_RUN_TARGET) == (CAVEAT_SHARES_LINEAGE,)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("judge:9000", "http://judge:9000"),
        ],
        ids=["none", "empty", "blank", "bare-host-port"],
    )
    def test_normalise_base_url_edge_cases(self, raw, expected) -> None:
        assert normalise_base_url(raw) == expected

    def test_normalise_model_trims_and_folds(self) -> None:
        assert normalise_model("  LFM2-1.2B ") == "lfm2-1.2b"
        assert normalise_model(None) == ""
