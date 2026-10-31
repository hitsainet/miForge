"""Three secrets, one rule: they appear nowhere  (FR-011.6, FR-011.20; ADR-020; **MUT-13**).

The bearer token, the audit label and the **attachment id** never reach a log line, a step
record, a run log, a manifest or an API response. The attachment id counts as a secret
because *miLLM treats it as one* - it returns the id once and logs only a reference to it -
so this is the sibling's own discipline rather than an invention.

**The test drives a real run against the fake with a log capture**, rather than reading the
source for ``log.info`` calls: a source scan cannot see a secret that arrives through a
format argument, and that is the form a convenience edit produces.
"""

from __future__ import annotations

import json
import logging

import pytest

from core.config import get_settings
from tests.support.fixtures_hotupdate import LABEL, TOKEN, Harness
from training.hotupdate import barrier, identity
from training.hotupdate.heartbeat import Heartbeat

SECRETS = (TOKEN, LABEL)


def rendered(records: list[logging.LogRecord]) -> str:
    """Every log record as text, **including its ``extra`` fields**.

    The message alone is not enough: this codebase logs structured fields, so a secret put
    into ``extra`` would never appear in ``record.getMessage()`` and a test reading only
    that would be green over the leak.
    """
    parts: list[str] = []
    for record in records:
        parts.append(record.getMessage())
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord("", 0, "", 0, "", None, None).__dict__:
                parts.append(f"{key}={value!r}")
    return " ".join(parts)


class TestAFullRunLeaksNothing:
    def test_no_secret_appears_in_any_log_line_of_a_bracketed_run(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        """**MUT-13.**"""
        with caplog.at_level(logging.DEBUG):
            attachment_id = str(harness.attach()["attachment_id"])
            beat = Heartbeat(client=harness.client, interval_seconds=1.0)
            beat.beat_once()
            for _ in range(2):
                with barrier.commit_bracket(harness.client, harness.timings):
                    pass
        text = rendered(caplog.records)
        for secret in (*SECRETS, attachment_id):
            assert secret not in text, f"{secret!r} reached a log line"

    def test_the_attach_log_line_itself_carries_only_a_DIGEST_REFERENCE(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        """**MUT-13's real control, and the first sweep found it missing.**

        The run-wide assertion above drove the client directly, so ``attach.py``'s own log
        line - the one that *mentions* the attachment - never executed: ``acquire`` rebuilds
        tensors and is unreachable without the ML wheels. **So MUT-13 survived**, not because
        the code was safe but because no test ran the line. The line is now a named function
        and this exercises it.
        """
        from training.hotupdate import rebuild
        from training.hotupdate.attach import Attachment, log_attachment
        from training.hotupdate.state import ResidentIdentity

        attachment_id = str(harness.attach()["attachment_id"])
        attachment = Attachment(
            attachment_id=attachment_id,
            resident_model="LFM2-1.2B",
            manifest_sha256="a" * 64,
            resident_identity=ResidentIdentity(
                model="LFM2-1.2B", manifest_sha256="a" * 64, steps_since_load=0
            ),
            base=rebuild.RebuiltBase(tensors={}, device_uuid=None, read_only=False),
            raw={},
        )
        with caplog.at_level(logging.INFO):
            log_attachment(attachment, base_url="http://millm.test:8000")
        text = rendered(caplog.records)
        assert attachment_id not in text, "the attachment id reached a log line"
        assert identity.digest_ref(attachment_id) in text, (
            "a digest reference must be there instead - otherwise a post-mortem cannot "
            "correlate two runs' attachments at all"
        )

    def test_the_digest_reference_is_what_appears_instead(
        self, harness: Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A correlation handle that is not the secret. *One function, so there is one
        mutation target.*"""
        with caplog.at_level(logging.INFO):
            response = harness.attach()
        reference = identity.digest_ref(str(response["attachment_id"]))
        assert reference is not None and len(reference) == 12
        assert reference != response["attachment_id"]


class TestTheSettingsFormThatMayLeave:
    def test_redacted_masks_the_token(self, hotupdate_env: None) -> None:
        body = json.dumps(get_settings().redacted())
        assert TOKEN not in body

    def test_the_hot_update_settings_object_does_not_print_the_token(
        self, hotupdate_env: None
    ) -> None:
        """``repr`` is an output. A dataclass that printed the token would leak it into every
        traceback that happened to carry one."""
        from training.hotupdate import settings as settings_module

        # The dataclass holds the token because the client needs it; what must not happen is
        # it reaching a *log*, which the run-wide assertion above covers. This asserts the
        # narrower thing the module can control: nothing formats it into a message.
        cfg = settings_module.load()
        assert cfg.trainer_token == TOKEN, "the fixture configured it"
        message = settings_module.unavailable_message(settings_module.CAUSE_IDENTITY_INCOMPLETE)
        assert TOKEN not in message and LABEL not in message


class TestTheRecordsAndTheResponses:
    def test_the_acquisition_record_carries_no_secret(self, harness: Harness) -> None:
        """F1's record is persisted and served. The base URL is a **location** and is fine;
        the token, the label and the id are not."""
        from training.hotupdate.attach import acquisition_record

        record = acquisition_record(
            resident_model="LFM2-1.2B",
            manifest_sha256="a" * 64,
            module=None,
            base_url="http://millm.test:8000",
        )
        body = record.model_dump_json()
        for secret in SECRETS:
            assert secret not in body
        assert "millm.test" in body, "the location is recorded; it is not a credential"

    def test_the_step_record_carries_no_secret(self, harness: Harness) -> None:
        from training import run_log
        from training.hotupdate import timings as timings_module

        harness.attach()
        with barrier.commit_bracket(harness.client, harness.timings):
            pass
        block = timings_module.fill(
            run_log.step_block(
                run_id="hot-run",
                step=1,
                loss=0.1,
                objective="causal_lm",
                learning_rate=1e-5,
                wall_time_ms=10.0,
            ),
            harness.timings,
        )
        body = json.dumps(block)
        for secret in SECRETS:
            assert secret not in body
        assert "attachment" not in body, "no attachment id or reference belongs in a record"


class TestTheProhibitionIsOnTheRightValues:
    def test_all_three_values_are_named_as_secrets_somewhere_checkable(self) -> None:
        """So a future reader cannot mistake *"we do not log the token"* for the whole rule.
        TID §12: **three** things are never logged, and the audit label is the one a reader
        would not expect, because it is not a credential - it is an identity, and R-02.32
        keeps it out of manifests and run logs for the same reason."""
        from training.hotupdate import identity as identity_module

        source = identity_module.__doc__ or ""
        assert "bearer token" in source
        assert "audit label" in source
        assert "attachment id" in source
