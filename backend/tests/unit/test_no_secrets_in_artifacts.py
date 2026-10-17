"""No secret is committed, and none reaches a manifest, a run log or an API body.

Foundation task 2.5; ADR-020. **The half of that task that does not wait on F11.**

Task 2.5 reads *"nothing committed, nothing in a manifest or run log. The §5.10 `trainer`
token is read from the environment and carries an audit label."* The audit-label clause is
F11's — nothing writes that token yet. The rest is checkable today and is checked here,
because the mechanism already existed and **nothing tested it**: ``Settings`` carries
``repr=False`` on the token and a :meth:`redacted` method that masks it, and both were
unprotected, so either could be dropped in a refactor with the suite green.

## Why this category first

A secret in a manifest or a run log does not stay local. ``docs/schemas/`` publishes to the
public mirror, a checkpoint is *"a standard Hugging Face directory any inference server can
serve"*, and an adapter is specifically built to **travel without its run log**. So an
artifact is the one place a leaked credential is copied off-box by design, by a person doing
the right thing. That makes it the category to test before the ones whose failures are loud.

## The sentinel approach, and why it beats enumerating fields

Every assertion below puts a **distinctive sentinel** into the configuration and then greps
every byte the real writers produce for it. The alternative - listing the fields a manifest
may contain and checking none is named like a secret - is a test whose fields agree with the
code **by construction**: it passes as long as somebody remembers to add each new field to
both lists, which is the failure mode this project keeps paying for.

The sentinel form cannot be satisfied by remembering. It fails if a secret reaches the
artifact through **any** path: a new field, an existing string field, a nested object, a
``model_dump`` of the whole settings object, or an error message that interpolated the
configuration. That last one is not hypothetical - a refusal that says "could not reach
http://user:token@host" is the commonest way a credential reaches a log.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest

from tests.unit.test_all_base_urls_refuse_credentials import discover_base_url_fields

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Unmistakable, and shaped like nothing the suite would produce by accident. If this string
#: appears in an artifact, it got there from the configuration and nowhere else.
SENTINEL = "mf-sentinel-SECRET-f4a1c9-do-not-leak"


@pytest.fixture
def configured_with_a_secret(monkeypatch: pytest.MonkeyPatch):
    """Settings holding the sentinel as the §5.10 trainer token.

    The cache is reset on both sides: ``get_settings`` memoises, so a leftover instance
    would make this test configure one object and assert about another - which would pass
    for the wrong reason, the exact shape of a check that never ran.
    """
    from core import config

    monkeypatch.setenv("MILLM_TRAINER_TOKEN", SENTINEL)
    # A plain host. A credential-bearing base URL is now REFUSED outright - see
    # `TestACredentialInABaseUrlIsRefused` below - so it cannot be set here.
    monkeypatch.setenv("MILLM_BASE_URL", "https://millm.local:8000")
    config._settings = None
    yield config.get_settings()
    config._settings = None


class TestTheMaskingIsReal:
    def test_the_token_is_loaded_at_all(self, configured_with_a_secret) -> None:
        """**Without this the whole file is vacuous.**

        If the environment variable were spelled wrongly the token would be ``None``, every
        assertion below would pass trivially, and the suite would report a leak check it
        never performed. This is the positive control: the thing must be present before its
        absence elsewhere means anything.
        """
        assert configured_with_a_secret.millm_trainer_token == SENTINEL, (
            "the sentinel never reached the settings object, so nothing below is tested. "
            "Check the environment variable name against core/config.py's field."
        )

    def test_redacted_does_not_carry_it(self, configured_with_a_secret) -> None:
        """``redacted()`` is documented as *"the only form that may leave here."*"""
        body = json.dumps(configured_with_a_secret.redacted())
        assert SENTINEL not in body, (
            "Settings.redacted() leaked the trainer token. That method is the one form "
            "documented as safe to log, serve or persist, so a leak here reaches all three."
        )

    def test_redacted_still_says_a_token_is_configured(self, configured_with_a_secret) -> None:
        """Masked, not deleted - and the difference matters.

        An absent key says "no token is set", which is a different operational fact from
        "a token is set and you may not see it". An operator debugging a 401 needs to tell
        those apart, and ADR-013's rule is that null is an assertion while an absent key is
        an omission.
        """
        assert configured_with_a_secret.redacted()["millm_trainer_token"] == "***"

    def test_repr_does_not_carry_it(self, configured_with_a_secret) -> None:
        """``repr`` reaches places nobody chose to send it.

        A traceback frame, a debugger watch, a structured log that interpolates the settings
        object, ``pytest``'s own assertion rewriting on a failure elsewhere. ``repr=False``
        on the field is what stops it, and it is one keyword away from being dropped.
        """
        assert SENTINEL not in repr(configured_with_a_secret)


class TestACredentialInABaseUrlIsRefused:
    """**The path that was easy to miss, and the fix is a refusal rather than a mask.**

    ``millm_base_url`` has no ``repr=False`` and is not in :meth:`redacted`'s mask list, so
    a URL carrying ``user:token@host`` walked straight through every field-by-field mask and
    reached any log line, traceback or settings dump that touched the object. This test
    found it; nothing calls ``redacted()`` yet, so it was latent rather than live.

    **Masking it would have been the wrong fix.** It hides the symptom and leaves the
    credential on the wire, sent from a field no reviewer reads as sensitive. A base URL is
    a location: identity travels separately in all three of this system's surfaces
    (``MILLM_TRAINER_TOKEN``, ``MIDATAWORKS_AGENT``, miDataworks' MCP bearer). So the URL is
    refused at startup - loud, fixable in one edit, and impossible to half-apply.
    """

    #: **NOT A HAND-WRITTEN LIST, and that is the whole correction.**
    #:
    #: This parametrisation read ``["millm_base_url", "midataworks_base_url"]`` - two fields,
    #: typed out, on one class. A list that agrees with the code by construction, extended
    #: only by remembering, and therefore **structurally unable to see a base URL declared in
    #: a different settings module**. Two were, both unguarded at the time
    #: (``judge_provider_base_url``, ``activation_source_base_url``), plus the provider rows,
    #: whose ``base_url`` is serialised verbatim into every score record and so **persists to
    #: disk**. That is the same split-module blindness that cost ``/data/packs`` its volume
    #: twice in one day, ``packs_root`` being declared in ``packs/settings.py`` while every
    #: list was written against ``core/config.py``.
    #:
    #: The fields now come from the mechanical discovery in
    #: ``test_all_base_urls_refuse_credentials.py``, which imports every module under
    #: ``backend/src`` and walks ``BaseSettings.__subclasses__()``. **One source of truth**,
    #: shared rather than copied: a second copy of the discovery would reintroduce the
    #: defect one indirection further away. That file also carries the enumeration's own
    #: positive controls - a static scan of the source that must agree with the dynamic walk
    #: - because a discovery that finds nothing reports a pass in exactly the same way a
    #: clean codebase does.
    CORE_CONFIG_BASE_URLS: ClassVar[list[str]] = [
        field for cls, field in discover_base_url_fields() if cls.__module__ == "core.config"
    ]

    def test_the_discovery_still_sees_this_module(self) -> None:
        """The positive control for the parametrisation below, kept local.

        If the shared discovery stopped finding ``core.config``'s fields, the next test would
        be parametrised over an empty list - which pytest reports as **collected and
        skipped**, not as failed. A check that silently stops running is this project's most
        expensive recurring defect, so the count is asserted here rather than assumed.
        """
        assert sorted(self.CORE_CONFIG_BASE_URLS) == ["midataworks_base_url", "millm_base_url"], (
            f"the shared discovery found {self.CORE_CONFIG_BASE_URLS!r} on core.config. If a "
            f"field was legitimately added or removed, update this control deliberately; if "
            f"it is empty, the discovery is broken and the parametrisation below is vacuous."
        )

    @pytest.mark.parametrize("field", CORE_CONFIG_BASE_URLS)
    @pytest.mark.parametrize(
        "url",
        [
            "http://trainer:TOPSECRETVALUE@millm.local",
            # A bare username carries no secret and is refused anyway: it means the URL is
            # being used to carry identity, which is the pattern being kept out.
            "http://trainer@millm.local",
        ],
    )
    def test_it_is_refused_at_startup(self, field: str, url: str) -> None:
        from core.config import Settings

        with pytest.raises(ValueError) as caught:
            Settings(**{field: url})
        message = str(caught.value)
        assert "credential" in message.lower()
        assert "TOPSECRETVALUE" not in message, (
            "the refusal echoed the credential back. A validation error is logged, so the "
            "message must name the problem without reproducing the secret."
        )

    @pytest.mark.parametrize(
        "url", ["https://millm.local:8000", "http://localhost:8000", "https://host/path"]
    )
    def test_an_ordinary_url_is_still_accepted(self, url: str) -> None:
        """The other direction, because a refusal that is too broad is worse than none.

        If this went red, every deployment would fail to start and the fix would be to
        delete the check - which is how a correct guard gets removed.
        """
        from core.config import Settings

        assert Settings(millm_base_url=url).millm_base_url == url


class TestNoSecretReachesAnArtifact:
    """Through the **real** writers, and the schemas are the second line of defence.

    Both vendored schemas were closed to unknown keys on 2026-10-13, so a secret cannot
    arrive in a manifest as a *new* key - it would fail validation. These assertions cover
    the other direction: a secret arriving as the *value* of a field that is allowed.
    """

    def test_a_checkpoint_manifest_carries_no_configuration(
        self, configured_with_a_secret, tmp_path: Path
    ) -> None:
        """Built and written by the real writer, then read back off disk as bytes.

        Read as bytes rather than as a parsed dict on purpose: a parsed comparison can only
        inspect the keys it thinks to look at, and the question here is whether the sentinel
        is anywhere in the file.
        """
        from tests.support.db import checkpoint_manifest, write_checkpoint

        path = write_checkpoint(tmp_path, checkpoint_manifest(step=100))
        written = (path / "manifest.json").read_bytes()
        assert SENTINEL.encode() not in written, (
            "the trainer token reached a checkpoint manifest. A checkpoint is a standard "
            "Hugging Face directory any inference server can serve, so this is a credential "
            "published in an artifact built to be copied."
        )

    def test_an_adapter_manifest_carries_no_configuration(
        self, configured_with_a_secret, tmp_path: Path
    ) -> None:
        """The sharper case: an adapter is built to **travel without its run log**.

        It is small, portable and routinely moved between installations, so a secret in one
        goes further than a secret in a checkpoint and arrives with less context.
        """
        from tests.support.db import adapter_manifest, write_adapter

        path = write_adapter(tmp_path, adapter_manifest())
        written = (path / "adapter_manifest.json").read_bytes()
        assert SENTINEL.encode() not in written

    def test_the_health_body_carries_no_configuration(self, configured_with_a_secret) -> None:
        """``/health`` reports each optional integration **by name** (ADR-010/ADR-013).

        Naming them is right, and it is also the one API body that is built from the
        configuration object - so it is where a careless ``model_dump()`` would surface. The
        report must say *whether* an integration is configured, never *with what*.
        """
        from fastapi.testclient import TestClient

        from api.app import create_app

        # `/api/health`, not `/health`: ADR-013 puts it outside the version prefix. The
        # first draft of this test used `/health`, got a 404, and the assertion passed on an
        # error body - a check that ran against nothing. The status is asserted to be a real
        # health response for exactly that reason; 503 is expected here because no
        # PostgreSQL is running, and `degraded` is still the endpoint doing its job.
        with TestClient(create_app()) as client:
            response = client.get("/api/health")
            assert response.status_code in (200, 503), (
                f"/api/health answered {response.status_code}; if this is 404 the path is "
                f"wrong and the leak assertion below is being made against an error body"
            )
            assert "status" in response.json(), "not a health body, so nothing is tested"
            assert SENTINEL not in response.text, (
                "the trainer token reached the health body. That endpoint is unauthenticated "
                "and is the first thing anything polls."
            )


class TestNothingIsCommitted:
    def test_no_tracked_file_carries_a_credential_shaped_assignment(self) -> None:
        """A scan of **tracked** files, with the scanner's own limits stated.

        It looks for an assignment of a long opaque literal to a secret-named key, which is
        what a committed credential looks like. It will not catch a secret that looks like
        prose, and it is not a secret scanner - it is the cheap half of task 2.5, and the
        real protection is that `.env` is ignored and the deployment reads the environment.

        ``git ls-files`` rather than a directory walk, so an untracked local `.env` with a
        real token in it does not fail a developer's suite for having done the right thing.
        """
        import re

        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")

        # A secret-named key assigned a long opaque value. The length floor and the
        # character class are what keep `token: str | None = None`, `password: ""` and
        # prose about tokens out of the results.
        pattern = re.compile(r"""(?ix)
            (secret|token|password|passwd|api[_-]?key|private[_-]?key)
            \s*[:=]\s*
            ["']([A-Za-z0-9+/=_\-]{24,})["']
            """)
        # Values that are obviously not credentials: placeholders, this file's own sentinel,
        # and the masked form. Checked against the captured VALUE, not the line, so a real
        # secret on a line that also mentions one of these is still caught.
        allowed = re.compile(
            r"(?i)^(x{4,}|\*{3,}|changeme|placeholder|your[_-]|example|dummy|redacted"
            r"|mf-sentinel-SECRET-f4a1c9-do-not-leak)"
        )
        # HYPHENATED LOWERCASE PROSE IS NOT A CREDENTIAL, and this exclusion is a finding
        # rather than a concession. The first run flagged two real tests - the ones that
        # assert a user's prompt text and an operator's corpus text never leave - because
        # each names its fixture `secret = "the-users-private-prompt-text"`. A credential
        # essentially always carries a digit or mixed case; a slug of lowercase words
        # separated by hyphens is English. Excluding it keeps the scanner trustworthy, which
        # matters more than the two lines: a checker that cries wolf gets switched off, and
        # every time a fresh checker in this project disagreed with careful work, the work
        # was right.
        prose = re.compile(r"^[a-z]+(-[a-z]+)+$")

        findings: list[str] = []
        for relative in tracked:
            if not relative:
                continue
            path = REPO_ROOT / relative
            if not path.is_file() or path.suffix in {".png", ".jpg", ".ico", ".woff2"}:
                continue
            try:
                text = path.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for match in pattern.finditer(text):
                if allowed.match(match.group(2)) or prose.match(match.group(2)):
                    continue
                line = text[: match.start()].count("\n") + 1
                findings.append(
                    f"{relative}:{line}: {match.group(1)}=<{len(match.group(2))} chars>"
                )

        assert not findings, "credential-shaped assignments in tracked files:\n" + "\n".join(
            findings
        )

    def test_the_scanner_catches_a_planted_credential(self, tmp_path: Path) -> None:
        """**The negative control, and the reason the test above can be believed.**

        A scanner that matches nothing reports a clean tree in exactly the same way as a
        clean tree does. This project has three recorded instances of that family - a `||`
        branch printing "clean" from a grep that had errored, a `replace()` that matched
        nothing so a no-op mutation looked like a survivor, and a stale `.pyc` producing a
        red from a restored file. In every one the signal was indistinguishable from the
        thing it was mistaken for. So the pattern is run against known leaks and each must
        be caught, and against known non-leaks which must not be.
        """
        import re

        pattern = re.compile(r"""(?ix)
            (secret|token|password|passwd|api[_-]?key|private[_-]?key)
            \s*[:=]\s*
            ["']([A-Za-z0-9+/=_\-]{24,})["']
            """)

        # ASSEMBLED AT RUNTIME, NOT WRITTEN AS LITERALS — and the reason is funny and real.
        #
        # The first version of this control held three literal credential-shaped
        # assignments, and `test_no_tracked_file_carries_a_credential_shaped_assignment`
        # above **flagged them**: the scanner found its own control data in a tracked file
        # and reported this file as carrying three secrets. It was right to. The strings
        # look exactly like what it hunts, which is the whole point of them.
        #
        # Excluding this file from the scan was the other option and is the worse one: it
        # would make the one file nobody scans the obvious place for a real secret to hide.
        # So the control builds its needles from pieces instead. No tracked line matches,
        # and the scanner is still exercised against the real shapes.
        _tok = "aBcDeFgHiJkLmNoPqRsTuVwXyZ" + "0123456789"
        must_catch = [
            f'MILLM_TRAINER_TOKEN = "ghp_{_tok}"',
            "api_key: 'sk-proj-" + "0123456789abcdefghijklmnopqrstuv" + "'",
            'DOCKERHUB_TOKEN="dckr_pat_' + "AbCdEf0123456789AbCdEf0123456789" + '"',
        ]
        must_not_catch = [
            "millm_trainer_token: str | None = Field(default=None, repr=False)",
            'data[secret] = "***"',
            "# the shared MCP bearer token belongs to its MCP server",
            'token = ""',
            'password: "changeme"',
            # The two real tests the first run flagged. Both assert that USER text never
            # leaves; naming the fixture `secret` is exactly right and must not be punished.
            'secret = "the-users-private-prompt-text"',
            'secret = "the-operators-private-training-text"',
        ]
        for line in must_catch:
            assert pattern.search(line), f"the scanner MISSED a planted credential: {line}"
        prose = re.compile(r"^[a-z]+(-[a-z]+)+$")
        for line in must_not_catch:
            match = pattern.search(line)
            # The prose exclusion is part of the scanner, so it is part of the control.
            if match is not None and prose.match(match.group(2)):
                continue
            assert match is None, f"the scanner false-positived on: {line}"
