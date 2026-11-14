"""EVERY base URL in this codebase refuses an embedded credential, discovered mechanically.

Foundation task 2.5; ADR-010, ADR-013, ADR-018, ADR-020.

## Why this file exists, which is the whole point of it

``test_no_secrets_in_artifacts.py`` already asserted that a credential-bearing base URL is
refused. It did so over a **hand-written list of two fields on one class**::

    @pytest.mark.parametrize("field", ["millm_base_url", "midataworks_base_url"])

That list is a second source of truth for "which fields are base URLs", and it is the
weakest possible kind: it agrees with the code **by construction** and only as long as
somebody remembers to extend it. It therefore could not see a base URL declared in a
different settings module - and two were, both unguarded:

* ``services/selectors/settings.py``'s ``judge_provider_base_url`` - a class with **zero
  validators** and no ``hide_input_in_errors``;
* ``services/scoring/settings.py``'s ``activation_source_base_url``;

plus a third surface that is not a Pydantic field at all - ``providers/settings.py``'s
``PROVIDER_<NAME>_BASE_URL`` rows, whose ``base_url`` is serialised **verbatim** by the
method documented as *"the only form that may leave this process"*, and so reached
``GET /api/v1/providers``, the provider identity recorded on every score, and run log
provenance. **That one persisted to disk**, which a log line does not.

**This is the same split-module blindness this project has already paid for twice**, both
times with ``packs_root``: it is declared in ``packs/settings.py``, every volume list was
written against ``core/config.py``, and ``/data/packs`` lost its storage in compose and
again in the cluster - *"second time that same split module caused the same omission in one
day."* AGENT.md's generalisation of it is the rule this file implements: a check must read
the roots from **both** modules, "which is what makes a fifth root added anywhere fail in a
test rather than in production."

So nothing here is listed. The classes are **discovered** by importing every module under
``backend/src`` and walking ``BaseSettings.__subclasses__()``; the fields are discovered
from ``model_fields``. A new settings module with a new base URL is covered the moment it is
written, and an unguarded one fails here.

## Discovery that cannot pass vacuously

A discovery-driven test has one failure mode worse than the gap it closes: **finding
nothing, and reporting that as a pass.** This project has five recorded members of that
family - a ``||`` branch printing "clean" from a grep that had errored, a ``replace()`` that
matched nothing so a no-op mutation looked like a survivor, a genuine red that meant "a
control is working", a stale ``.pyc`` red from a restored file, and a ``head -1 ||`` whose
status was never the grep's. *In every one the signal was structurally indistinguishable
from the thing it was mistaken for.*

So the discovery is checked three ways before any refusal is asserted, by
:class:`TestTheDiscoveryItselfIsSound`:

1. **No module failed to import.** A module that cannot be imported contributes no classes,
   which silently shrinks coverage. Failures are reported by name.
2. **A static scan of the source agrees with it.** Every ``*_base_url`` declaration found by
   reading the files must also have been found by the dynamic walk. This is the positive
   check that the measurement happened - it is the one assertion that goes red if the import
   machinery quietly stops reaching a package.
3. **The known-good field is among the findings.** ``core.config.Settings.millm_base_url``
   is guarded and has been since foundation task 2.5, so its absence from the discovery
   means the discovery is broken, not that the codebase is clean.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import warnings
from pathlib import Path

import pytest
from pydantic_settings import BaseSettings

#: ``backend/src``, derived from this file's location rather than from the working
#: directory. A path that depends on where pytest was invoked is how a checker passes by
#: looking at nothing.
SRC = Path(__file__).resolve().parents[2] / "src"

#: Distinctive, and shaped like nothing the suite produces by accident. If this appears in a
#: refusal message, the guard moved the credential instead of stopping it.
CREDENTIAL = "CRED-SENTINEL-9f3a7b-DO-NOT-LEAK"

#: **A SHORT credential, and it is short for a reason that is itself a finding.**
#:
#: ``Pydantic ELIDES THE MIDDLE of a long input repr.`` With ``hide_input_in_errors``
#: removed from a settings class, ``str(ValidationError)`` rendered::
#:
#:     input_value='http://t:CRED-SENTINEL-9...DO-NOT-LEAK@judge.local'
#:
#: so a substring assertion against the full 32-character :data:`CREDENTIAL` came back
#: **clean** - from a message that was leaking the credential's head and tail. The test would
#: have passed with the protection deleted, for a reason that has nothing to do with the
#: protection: the sixth member of this project's false-measurement family, found by running
#: the mutation rather than by reading the test.
#:
#: Two consequences, both load-bearing:
#:
#: 1. The URL used for the error-rendering assertion is kept under Pydantic's ~50-character
#:    elision threshold, so the assertion is actually **sensitive** to the flag. A longer one
#:    is not a stricter test, it is a test that cannot fail.
#: 2. A real deployment credential is usually long, so the elided form is what a log would
#:    actually carry - a partial leak, which is a leak. That is why
#:    :class:`TestEverySettingsClassHidesRejectedInput` asserts the **mechanism** per class
#:    and does not rely on this end-to-end assertion alone.
SHORT_CREDENTIAL = "S3CR3T9f3a7b"
SHORT_CREDENTIALLED_URL = f"http://t:{SHORT_CREDENTIAL}@j.local"

#: The two forms, and **both are refused**. A bare username carries no secret, but it means
#: the URL is being used to carry identity - the pattern the guard exists to keep out, and
#: the reading ``core.config`` already took.
CREDENTIALLED_URLS = [
    f"http://trainer:{CREDENTIAL}@judge.local:8000",
    "http://trainer@judge.local:8000",
]


# A field name this file treats as a base URL. ``base_url`` itself is included because a
# class may well name the field that; only matching ``*_base_url`` would miss it.
def _is_a_base_url(field_name: str) -> bool:
    return field_name == "base_url" or field_name.endswith("_base_url")


def _import_everything_under_src() -> list[str]:
    """Import every module under ``backend/src``. Returns the failures, by name.

    Returned rather than raised so :meth:`TestTheDiscoveryItselfIsSound.
    test_no_module_failed_to_import` can report all of them at once, and so a single
    unimportable module does not stop the discovery of the rest.
    """
    failures: list[str] = []

    def _on_error(name: str) -> None:
        failures.append(f"{name}: failed while walking the package")

    with warnings.catch_warnings():
        # Importing the whole tree touches deprecation warnings that are not this test's
        # business. Errors are still collected above.
        warnings.simplefilter("ignore")
        for module in pkgutil.walk_packages([str(SRC)], onerror=_on_error):
            try:
                importlib.import_module(module.name)
            except Exception as exc:  # reported, not swallowed
                failures.append(f"{module.name}: {type(exc).__name__}: {exc}")
    return failures


def _all_settings_classes() -> list[type[BaseSettings]]:
    """Every ``BaseSettings`` subclass reachable from ``backend/src``, transitively.

    ``__subclasses__()`` is one level deep, so the walk is recursive: a class that subclasses
    a project settings class rather than ``BaseSettings`` directly is still found. Classes
    defined outside ``backend/src`` (a dependency's own settings model) are excluded - this
    file makes assertions about **this** codebase.
    """
    found: dict[str, type[BaseSettings]] = {}

    def _walk(cls: type[BaseSettings]) -> None:
        for sub in cls.__subclasses__():
            key = f"{sub.__module__}.{sub.__qualname__}"
            if key not in found:
                found[key] = sub
                _walk(sub)

    _walk(BaseSettings)
    owned = {}
    for key, cls in found.items():
        module = getattr(cls, "__module__", "")
        spec_origin = getattr(importlib.import_module(module), "__file__", None)
        if spec_origin and Path(spec_origin).resolve().is_relative_to(SRC):
            owned[key] = cls
    return [owned[key] for key in sorted(owned)]


def discover_base_url_fields() -> list[tuple[type[BaseSettings], str]]:
    """Every ``(settings class, base-URL field)`` pair in ``backend/src``.

    The public entry point of this module: ``test_no_secrets_in_artifacts.py`` imports it so
    that its own base-URL class is driven by the same discovery rather than by a second
    hand-written list. *One source of truth, or the next field is invisible to one of them.*
    """
    _import_everything_under_src()
    return [
        (cls, field)
        for cls in _all_settings_classes()
        for field in sorted(cls.model_fields)
        if _is_a_base_url(field)
    ]


def _statically_declared_base_url_fields() -> set[tuple[str, str]]:
    """``(module, field)`` pairs found by **reading** the source, for cross-check only.

    Deliberately a different mechanism from the dynamic walk. Two independent measurements
    that must agree is the only construction that detects the walk going quietly dark - the
    failure this file's docstring lists five instances of.

    Restricted to files that mention ``BaseSettings`` so an ORM column
    (``models/run.py``'s ``hotupdate_base_url``) or a dataclass field is not demanded of the
    Pydantic discovery.
    """
    declaration = re.compile(r"^\s{4,}(\w*base_url)\s*:\s*\S", re.MULTILINE)
    pairs: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        try:
            text = path.read_text("utf-8")
        except (UnicodeDecodeError, OSError):  # pragma: no cover - defensive
            continue
        if "BaseSettings" not in text:
            continue
        module = str(path.relative_to(SRC).with_suffix("")).replace("/", ".")
        for match in declaration.finditer(text):
            if _is_a_base_url(match.group(1)):
                pairs.add((module, match.group(1)))
    return pairs


#: Computed once at collection, because it drives the parametrisation below.
BASE_URL_FIELDS = discover_base_url_fields()
IDS = [f"{cls.__module__}.{field}" for cls, field in BASE_URL_FIELDS]


def discover_settings_classes() -> list[type[BaseSettings]]:
    """Every settings class in ``backend/src``, base URL or not. Shared, like the above."""
    _import_everything_under_src()
    return _all_settings_classes()


#: **Classes known to omit ``hide_input_in_errors`` and OUTSIDE this change's write scope.**
#:
#: Named individually, with the file that owns each, because the alternative constructions
#: are both worse. A blanket skip of the assertion would make the suite report a check it
#: never performed - the vacuous-pass failure ``test_schema_sync.py`` avoids by classing an
#: unvendored contract ``Internal`` **by name** rather than giving it an empty vendor list.
#: Deleting the two from the enumeration would hide them.
#:
#: These are **strict** xfails, so they are tripwires rather than exemptions: the moment
#: either class gains the flag, this test goes red with ``XPASS(strict)`` and whoever fixed
#: it is told to delete the row. *A live prohibition is the most expensive kind of stale
#: finding*, so this list is built to force its own removal.
#:
#: Neither class holds a credential today - ``PackSettings`` is paths and bounds,
#: ``SignalSettings`` is policies and numbers - so the exposure is lower than
#: ``ScoringSettings``', which was live through ``GET /api/health``. But the flag is a
#: **per-class** property of Pydantic's error rendering, not a property of the fields that
#: happen to be on the class today, and the first validator added to either one will
#: reproduce whatever it rejects. That is the class of defect, and it is the reason to assert
#: the flag per class rather than only on the classes that currently hold secrets.
#: **EMPTY, AND THAT IS THE TRIPWIRE HAVING WORKED.** It carried `PackSettings` and
#: `SignalSettings` when this test was written, both outside the write scope of the change
#: that added it. Both gained the flag on 2026-10-16 and both rows were deleted here in the
#: same edit, which is exactly what `strict=True` was for: the xfails would otherwise have
#: become `XPASS(strict)` failures. Leave the mechanism in place - a future class added
#: without the flag belongs here with its owning file named, never silently skipped.
UNOWNED_CLASSES_MISSING_THE_FLAG: dict[str, str] = {}


def _settings_class_params() -> list[pytest.param]:  # type: ignore[valid-type]
    params = []
    for cls in discover_settings_classes():
        key = f"{cls.__module__}.{cls.__qualname__}"
        owner = UNOWNED_CLASSES_MISSING_THE_FLAG.get(key)
        marks = (
            [
                pytest.mark.xfail(
                    strict=True,
                    reason=(
                        f"{key} does not set hide_input_in_errors=True. OWED, and {owner} is "
                        f"outside the write scope of the change that added this test. Strict: "
                        f"when the flag lands this xfail becomes XPASS and fails, which is "
                        f"the signal to delete this row from "
                        f"UNOWNED_CLASSES_MISSING_THE_FLAG."
                    ),
                )
            ]
            if owner
            else []
        )
        params.append(pytest.param(cls, id=key, marks=marks))
    return params


class TestEverySettingsClassHidesRejectedInput:
    """**A validation error is an output, and this is a property of the CLASS, not the field.**

    Pydantic appends ``input_value=...`` to ``str(ValidationError)`` by default. So *any*
    validator on *any* settings class reproduces the value it rejects unless
    ``hide_input_in_errors=True`` is set on that class - and a startup validation error is
    logged, which makes the error path an output. Asserting the flag per class closes the
    **class** of defect rather than the instance, which is the difference between this test
    preventing the next one and merely recording the last one.

    **It is not hypothetical and it was not latent.** ``core/config.py:50`` sets the flag and
    documents the two leaks that earned it: a credential in a sibling base URL defeating
    every field-by-field mask, and then the refusal added for it *reproducing the very
    credential it was refusing*. ``services/scoring/settings.py`` did not set it, and that
    single omission was a **live** credential leak reachable through ``GET /api/health`` -
    not latent like the first one. ``repr=False`` does not help: it governs ``repr()``, not
    Pydantic's error rendering.

    **It is also the second time the same split caused the same omission.** The first was
    ``packs_root``, declared in ``packs/settings.py`` while every volume list was written
    against ``core/config.py``, costing ``/data/packs`` its storage in compose and again in
    the cluster. A per-class assertion driven by discovery is the only shape that does not
    inherit that blindness, because nothing in it names a module.
    """

    @pytest.mark.parametrize("cls", _settings_class_params())
    def test_it_sets_hide_input_in_errors(self, cls: type[BaseSettings]) -> None:
        assert cls.model_config.get("hide_input_in_errors") is True, (
            f"{cls.__module__}.{cls.__qualname__} does not set hide_input_in_errors=True in "
            f"its model_config, so Pydantic appends the rejected value to "
            f"str(ValidationError) for EVERY field on it - including any that holds a "
            f"credential, and including a credential an operator embedded in a URL. Add it "
            f"with the comment from core/config.py:50, which records the two leaks that "
            f"earned it."
        )

    def test_the_enumeration_reached_more_than_core_config(self) -> None:
        """The positive control: this assertion exists because one module was not enough.

        A per-class check that only ever sees ``core.config`` is the hand-written list it
        replaced, and it would have passed on the day ``ScoringSettings`` was leaking.
        """
        modules = {cls.__module__ for cls in discover_settings_classes()}
        assert len(modules) >= 4, (
            f"the settings-class discovery reached only {sorted(modules)}. The omission this "
            f"test catches lives in feature-local settings modules, so a discovery that "
            f"sees only core/ cannot catch it."
        )

    def test_the_unowned_list_is_not_a_blanket_exemption(self) -> None:
        """Each name in the xfail list must be a real, reachable class.

        A stale entry is an exemption nobody notices: the class could be renamed or deleted
        and the row would sit there excusing nothing while reading as a known gap. This is
        the same reason ``UNOWNED_CLASSES_MISSING_THE_FLAG`` uses strict xfails - the list is
        built to force its own removal, in both directions.
        """
        reachable = {f"{cls.__module__}.{cls.__qualname__}" for cls in discover_settings_classes()}
        stale = sorted(set(UNOWNED_CLASSES_MISSING_THE_FLAG) - reachable)
        assert not stale, (
            f"UNOWNED_CLASSES_MISSING_THE_FLAG names classes that no longer exist: {stale}. "
            f"Delete the rows; a stale exemption excuses nothing and reads as a known gap."
        )


class TestTheDiscoveryItselfIsSound:
    """**Read this class first. Without it everything below can pass by finding nothing.**"""

    def test_no_module_failed_to_import(self) -> None:
        """An unimportable module contributes no classes and so silently shrinks coverage."""
        failures = _import_everything_under_src()
        assert not failures, (
            "modules under backend/src failed to import, so any settings class they "
            "declare was invisible to the enumeration below and its base URLs went "
            "unchecked:\n  " + "\n  ".join(failures)
        )

    def test_the_static_scan_and_the_dynamic_walk_agree(self) -> None:
        """The positive check that the measurement happened, by a second mechanism.

        Reading the files and importing them are independent. If the import walk stops
        reaching a package - a missing ``__init__.py``, a renamed directory, a path computed
        from the wrong parent - this is the assertion that says so, instead of the suite
        reporting a clean sweep over four fields it no longer looks at.
        """
        dynamic = {(cls.__module__, field) for cls, field in BASE_URL_FIELDS}
        static = _statically_declared_base_url_fields()
        missed = sorted(static - dynamic)
        assert not missed, (
            "a base URL is declared in the source but was NOT found by importing and "
            f"walking BaseSettings.__subclasses__(): {missed}. Either the module did not "
            "import, or the class is not a BaseSettings subclass, or the walk is not "
            "reaching it. Nothing below tested these fields."
        )

    def test_the_known_guarded_field_is_among_the_findings(self) -> None:
        """``core.config.Settings.millm_base_url`` has been guarded since task 2.5.

        So if the discovery does not find it, the discovery is broken - which is a different
        statement from "this codebase has no unguarded base URLs", and the two are
        indistinguishable from a green run without this assertion.
        """
        assert ("core.config", "millm_base_url") in {
            (cls.__module__, field) for cls, field in BASE_URL_FIELDS
        }, f"discovery found {IDS!r}, which does not include the known-guarded field"

    def test_more_than_one_module_was_reached(self) -> None:
        """The specific blindness being closed: a check that sees only ``core.config``.

        The hand-written list this file replaces covered exactly one module. If the
        enumeration ever narrows back to one, it has regressed to that list.
        """
        modules = {cls.__module__ for cls, _ in BASE_URL_FIELDS}
        assert len(modules) >= 2, (
            f"only {modules} was reached. The gap this file closes is precisely a base-URL "
            f"check that cannot see past one settings module."
        )


@pytest.mark.parametrize(("cls", "field"), BASE_URL_FIELDS, ids=IDS)
class TestEveryBaseUrlFieldRefusesACredential:
    """One source of truth: whatever the discovery finds must refuse, with no exemptions.

    An allowlist here would be the hand-written list coming back through the door.
    """

    def test_an_ordinary_url_is_accepted(
        self, cls: type[BaseSettings], field: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**The positive control for the two tests below, per field.**

        It proves the class is constructible with only this field set, so a refusal below is
        the guard firing on the credential rather than the environment, a required sibling
        field or a stray ``.env`` value refusing something else. Without it, a class that
        could not be built at all would look perfectly guarded.

        It is also the other direction of the guard, which matters on its own: a refusal that
        is too broad fails every deployment, and the fix for that is to delete the check.
        """
        monkeypatch.delenv(field.upper(), raising=False)
        instance = cls(**{field: "https://judge.local:8000"})
        assert getattr(instance, field) == "https://judge.local:8000"

    @pytest.mark.parametrize("url", CREDENTIALLED_URLS)
    def test_a_credential_in_it_is_refused(
        self, cls: type[BaseSettings], field: str, url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refused at validation - **loud, fixable in one edit, impossible to half-apply.**

        Not masked. ADR-018: masking hides the symptom and leaves the credential live on the
        wire, sent from a field no reviewer reads as sensitive. And on the provider surface
        the mask would have had to hold on a value that is written into a score record, so
        the failure would have persisted to disk.
        """
        monkeypatch.delenv(field.upper(), raising=False)
        with pytest.raises(ValueError) as caught:
            cls(**{field: url})
        message = str(caught.value).lower()
        assert "credential" in message, (
            f"{cls.__module__}.{field} rejected the URL but the message does not say why. "
            f"ADR-021: the message must name what failed and what to do next. Got: "
            f"{caught.value}"
        )

    def test_the_refusal_does_not_reproduce_the_credential(
        self, cls: type[BaseSettings], field: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**A validation error is an output, and this is the half that is easy to miss.**

        Both of this project's recorded credential leaks were on the path that exists to
        *report* a problem. Pydantic appends ``input_value=`` to ``str(ValidationError)`` by
        default, so a guard whose own refusal is rendered with the rejected value has moved
        the credential into the log rather than stopped it - and the refusal added for leak
        (1) reproduced the very credential it was refusing.

        ``repr=False`` does **not** close this: it governs ``repr()``, not Pydantic's error
        rendering. ``hide_input_in_errors=True`` in ``model_config`` does.

        **The URL here is deliberately SHORT.** See :data:`SHORT_CREDENTIAL`: Pydantic elides
        the middle of a long input repr, so the first version of this test - which used the
        32-character sentinel - passed with the protection deleted. The property and the
        mechanism are therefore asserted separately: this test is the property, and
        :class:`TestEverySettingsClassHidesRejectedInput` asserts the flag per class, because
        a long real credential renders elided and a substring check cannot see it.
        """
        monkeypatch.delenv(field.upper(), raising=False)
        with pytest.raises(ValueError) as caught:
            cls(**{field: SHORT_CREDENTIALLED_URL})
        assert SHORT_CREDENTIAL not in str(caught.value), (
            f"{cls.__module__}.{field}'s refusal echoed the credential back. Set "
            f"hide_input_in_errors=True in this class's model_config (see "
            f"core/config.py:50) and build the message from urlsplit(...).hostname only. "
            f"A validation error is logged, so this is a leak through the error path."
        )

    # The MECHANISM behind the assertion above - `hide_input_in_errors=True` - is asserted
    # once per CLASS in `TestEverySettingsClassHidesRejectedInput`, not once per base-URL
    # field here. It is a property of the class's error rendering and applies to every field
    # on it, so asserting it per field would be two sources for one check and would also
    # leave a settings class with no base URL unprotected - which is exactly the class of
    # defect that round closed.


class TestTheProviderRowSurfaceRefusesToo:
    """The surface the enumeration above **cannot** see, and the worst of the three.

    ``providers/settings.py`` holds no ``BaseSettings`` class at all: FR-004.3 needs *a row
    per endpoint, named by the operator* - an unbounded set of names - so rows are discovered
    by environment prefix into a frozen dataclass. A ``BaseSettings`` enumeration is blind to
    it by construction, which is why it is asserted here explicitly rather than left to look
    covered.

    It is the worst because :meth:`HttpProviderRow.redacted` includes ``base_url`` verbatim
    and is documented as *"the only form that may leave this process"*. So a credential here
    reached ``GET /api/v1/providers``, the provider identity recorded on **every score**
    (R-02.16) and run log provenance. A leaked log line is bounded by log retention; a score
    record and a run log **persist to disk**, and a checkpoint is a standard Hugging Face
    directory built to be copied off-box.
    """

    @pytest.mark.parametrize("url", CREDENTIALLED_URLS)
    def test_the_environment_scan_refuses_it(self, url: str) -> None:
        from core.errors import Refusal
        from providers.settings import http_provider_rows

        environ = {
            "PROVIDER_JUDGE_BASE_URL": url,
            "PROVIDER_JUDGE_MODEL": "qwen3-8b",
        }
        with pytest.raises(Refusal) as caught:
            http_provider_rows(environ)
        assert caught.value.code == "inference_provider_misconfigured"
        assert "PROVIDER_JUDGE_BASE_URL" in caught.value.message, (
            "the refusal must name the variable: the rows are an unbounded set, so an "
            "operator cannot act on 'a provider base URL is wrong'"
        )

    def test_the_environment_scan_refusal_carries_only_the_host(self) -> None:
        """Message **and** details, because the details reach the error envelope.

        ``Refusal.envelope()`` serialises ``details`` into the API response, so a guard that
        kept its message clean and put the URL in ``details`` for debugging would have
        published the credential through the error path it was added to protect.
        """
        from core.errors import Refusal
        from providers.settings import http_provider_rows

        url = f"http://trainer:{CREDENTIAL}@judge.local:8000"
        with pytest.raises(Refusal) as caught:
            http_provider_rows({"PROVIDER_JUDGE_BASE_URL": url, "PROVIDER_JUDGE_MODEL": "m"})
        import json

        serialised = json.dumps(caught.value.envelope())
        assert CREDENTIAL not in serialised, (
            "the provider refusal reproduced the credential somewhere in its envelope. "
            "Build the message from urlsplit(...).hostname and keep the URL out of details."
        )
        assert "judge.local" in serialised, (
            "the refusal names neither the host nor anything locating the endpoint, which "
            "makes it unactionable for an operator with several rows configured"
        )

    def test_a_credential_is_reported_before_the_missing_model(self) -> None:
        """**The only property the scan-site call has that the type guard does not.**

        Found by mutating the guard rather than by reading it: deleting the
        ``refuse_credentials_in_base_url`` call from ``http_provider_rows`` leaves every
        other test in this class **green**, because ``__post_init__`` refuses with a message
        that is byte-identical for a row named ``judge``. So that call site was unprotected
        by its own author's tests - *a guard never seen failing is not known to work*, and
        two guards whose messages coincide make one of them invisible.

        What it alone decides is **ordering**. The pair half-set is also a refusal, and it
        comes first in the function: with a credentialled URL **and** no model, the scan
        would report the missing model, the operator would set one, and the next start would
        ship the credential. The credential has to be the refusal the operator sees first,
        because the other one's remedy makes the leak live.
        """
        from core.errors import Refusal
        from providers.settings import http_provider_rows

        with pytest.raises(Refusal) as caught:
            http_provider_rows(
                {"PROVIDER_JUDGE_BASE_URL": f"http://trainer:{CREDENTIAL}@judge.local"}
            )
        assert "embedded user information" in caught.value.message, (
            "with a credential AND no model, the refusal reported the missing model. The "
            "operator's remedy for that is to set a model, which ships the credential. The "
            "credential must be refused first - call refuse_credentials_in_base_url before "
            "the model lookup in http_provider_rows."
        )
        assert CREDENTIAL not in json_dumps_safe(caught.value.envelope())

    def test_a_row_cannot_be_constructed_with_one(self) -> None:
        """The guard is on the **type**, not only on the env scan.

        A check in ``http_provider_rows`` guards the one path that exists today; a check in
        ``__post_init__`` guards every path there will ever be - another module building a
        row, a test fixture, a future configuration source. *A guard that can be bypassed by
        constructing the object another way is a convention, not a guard*, and this is the
        type whose serialised form is written into score records.
        """
        from core.errors import Refusal
        from providers.settings import HttpProviderRow

        with pytest.raises(Refusal) as caught:
            HttpProviderRow(
                name="judge",
                base_url=f"http://trainer:{CREDENTIAL}@judge.local:8000",
                model="qwen3-8b",
            )
        assert CREDENTIAL not in json_dumps_safe(caught.value.envelope())
        assert CREDENTIAL not in str(caught.value)

    def test_an_ordinary_row_is_still_built_and_serves_its_url(self) -> None:
        """The other direction, and it checks the thing that made this leak matter.

        ``base_url`` really is in ``redacted()`` verbatim - that is correct, a listing that
        will not say where a provider is is useless. It is only safe because a row carrying a
        credential cannot exist. So this asserts both halves at once: an ordinary row is
        built, and its URL is still served.
        """
        from providers.settings import http_provider_rows

        rows = http_provider_rows(
            {
                "PROVIDER_JUDGE_BASE_URL": "https://judge.local:8000/v1",
                "PROVIDER_JUDGE_MODEL": "qwen3-8b",
                "PROVIDER_JUDGE_API_KEY": CREDENTIAL,
            }
        )
        assert len(rows) == 1
        assert rows[0].redacted()["base_url"] == "https://judge.local:8000/v1"
        # And the credential that belongs in the credential field is still not serialised.
        assert CREDENTIAL not in json_dumps_safe(rows[0].redacted())
        assert rows[0].redacted()["has_api_key"] is True


def json_dumps_safe(value: object) -> str:
    """``str`` of anything, for a "the secret is nowhere in this" assertion.

    ``json.dumps`` with ``default=str`` rather than ``str(value)`` alone so a nested
    structure is traversed: an assertion that only inspects the top level of an object is the
    parsed-comparison mistake ``test_no_secrets_in_artifacts.py`` avoids by reading manifests
    as bytes.
    """
    import json

    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):  # pragma: no cover - default=str makes this unlikely
        return str(value)
