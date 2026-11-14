"""``domain`` has three states on a checkpoint and a fourth on an adapter  (task 4.2a).

**These assertions are made against the vendored schemas as PARSED, not as remembered.**
Every claim below about what ``checkpoint-manifest-v1.json`` requires or what
``adapter-manifest-v1.json`` declares is read out of the file at test time. That is the
check this project keeps finding things with: three requirements have asserted a field
their own normative schema lacked, and all three were caught by checking a claim against
its cited source rather than by reading.

**What each test pins, and what breaking it would cost:**

* that the checkpoint schema keeps ``domain`` **required while nullable** - the BRD's own
  fix, and the reason three states exist at all. Make it optional and ``absent`` stops
  being an omission;
* that the adapter schema declares **no** ``domain`` property and refuses unknown keys -
  so ``not_in_kind`` is a fact about the contract, not a convention;
* that the classifier distinguishes a missing key from a null one. ``.get()`` returns
  ``None`` for both, which is the one-character version of this defect;
* that the **database refuses** the collapsed forms. A Python-only distinction is
  unpinned the moment something writes a row by another route.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from models.adapter import Adapter
from models.checkpoint import Checkpoint
from models.domain_state import (
    ADAPTER_STATES,
    CHECKPOINT_STATES,
    ArtifactKind,
    DomainState,
    classify_domain,
    render_domain,
)
from services.reindex import schema_dir
from tests.support import db as support


def _schema(filename: str) -> dict:
    """The vendored schema, parsed.

    Reached through :func:`services.reindex.schema_dir`, which honours
    ``MIFORGE_SCHEMA_DIR``, rather than through a hard-coded path. Two reasons, the second
    of which is the one that matters: the package may be installed somewhere else in a
    container, and a **negative control for these assertions can be run against a
    perturbed temporary copy** instead of against the real contract. AGENT.md is explicit
    that a byte of a vendored schema is never perturbed in place.
    """
    return json.loads((schema_dir() / filename).read_text("utf-8"))


# ------------------------------------------------------------------ the schemas, parsed


def test_checkpoint_schema_keeps_domain_required_and_nullable() -> None:
    """``domain``: required, so absence is nonconforming; nullable, so null is sayable."""
    defs = _schema("checkpoint-manifest-v1.json")["$defs"]["CheckpointManifestV1"]
    assert "domain" in defs["required"], (
        "domain left the required list; 'absent' and 'asserted_null' are now the same "
        "fact and task 4.2a's distinction has no basis in the contract"
    )
    types = {entry.get("type") for entry in defs["properties"]["domain"]["anyOf"]}
    assert types == {"string", "null"}, types


def test_adapter_schema_has_no_domain_and_refuses_unknown_keys() -> None:
    """The fourth state is a property of the contract: an adapter manifest cannot carry one."""
    defs = _schema("adapter-manifest-v1.json")["$defs"]["AdapterManifestV1"]
    assert "domain" not in defs["properties"], (
        "the adapter manifest grew a 'domain' property; 'not_in_kind' is no longer the "
        "right state and models/adapter.py's CHECK constraints are now wrong"
    )
    assert defs["additionalProperties"] is False, (
        "the adapter manifest no longer closes its object, so a 'domain' key would "
        "validate and the fourth state would be ambiguous"
    )


def test_the_two_schemas_disagree_about_domain_on_purpose() -> None:
    """The asymmetry is the point, so it is asserted rather than left to be noticed."""
    checkpoint = _schema("checkpoint-manifest-v1.json")["$defs"]["CheckpointManifestV1"]
    adapter = _schema("adapter-manifest-v1.json")["$defs"]["AdapterManifestV1"]
    assert "domain" in checkpoint["properties"]
    assert "domain" not in adapter["properties"]


# ------------------------------------------------------------------ the classifier


def test_a_value_is_recorded() -> None:
    result = classify_domain({"domain": "support"}, kind=ArtifactKind.CHECKPOINT)
    assert (result.value, result.state, result.anomaly) == ("support", DomainState.RECORDED, None)


def test_an_explicit_null_is_an_assertion_not_an_omission() -> None:
    result = classify_domain({"domain": None}, kind=ArtifactKind.CHECKPOINT)
    assert result.state is DomainState.ASSERTED_NULL
    assert result.value is None
    assert result.anomaly is None, "a conforming null is not an anomaly; a from-scratch run is one"


def test_a_missing_key_is_an_omission_and_is_named_as_one() -> None:
    result = classify_domain({}, kind=ArtifactKind.CHECKPOINT)
    assert result.state is DomainState.ABSENT
    assert result.value is None
    assert result.anomaly == "checkpoint_manifest_missing_required_domain"


def test_null_and_absent_are_different_states_with_the_same_value() -> None:
    """The collapse, asserted directly: same value, different state.

    This is what a plain nullable column cannot express, and the single test most worth
    reading before changing the column pair.
    """
    explicit_null = classify_domain({"domain": None}, kind=ArtifactKind.CHECKPOINT)
    absent = classify_domain({}, kind=ArtifactKind.CHECKPOINT)
    assert explicit_null.value == absent.value is None
    assert explicit_null.state is not absent.state


def test_an_adapter_is_not_in_kind_and_its_domain_is_never_taken_from_its_base() -> None:
    manifest = support.adapter_manifest()
    result = classify_domain(manifest, kind=ArtifactKind.ADAPTER)
    assert result.state is DomainState.NOT_IN_KIND
    assert result.value is None
    # The base is right there in the manifest and carries a domain of its own. Nothing
    # may reach for it.
    assert manifest["base"]["name"]


def test_a_domain_key_on_an_adapter_is_reported_and_not_adopted() -> None:
    result = classify_domain({"domain": "support"}, kind=ArtifactKind.ADAPTER)
    assert result.state is DomainState.NOT_IN_KIND
    assert result.value is None, "an adapter must not take a domain from a nonconforming key"
    assert result.anomaly == "adapter_manifest_carries_domain"


def test_a_non_string_domain_is_refused_rather_than_coerced() -> None:
    """``str(42)`` here would fabricate the domain name ``'42'`` (ADR-018)."""
    result = classify_domain({"domain": 42}, kind=ArtifactKind.CHECKPOINT)
    assert result.state is DomainState.ABSENT
    assert result.value is None
    assert result.anomaly is not None and "not_a_string" in result.anomaly


def test_the_four_states_render_differently() -> None:
    """Four states, four strings. A blank for ``not_in_kind`` would read as "no domain"."""
    rendered = {
        render_domain("support", DomainState.RECORDED),
        render_domain(None, DomainState.ASSERTED_NULL),
        render_domain(None, DomainState.ABSENT),
        render_domain(None, DomainState.NOT_IN_KIND),
    }
    assert len(rendered) == 4, rendered
    assert render_domain(None, DomainState.NOT_IN_KIND) == "not recorded in this artifact kind"


def test_the_state_sets_do_not_overlap() -> None:
    assert CHECKPOINT_STATES.isdisjoint(ADAPTER_STATES)
    assert set(DomainState) == CHECKPOINT_STATES | ADAPTER_STATES


# ------------------------------------------------------------------ the database refuses


@pytest.fixture()
def engine(tmp_path: Path):
    return support.migrated_engine(tmp_path / "db")


def _checkpoint_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "LFM2-1.2B:run-1:step-100",
        "run_id": "run-1",
        "step": 100,
        "domain": "support",
        "domain_state": DomainState.RECORDED.value,
        "signal_source": "corpus",
        "parent": None,
        "base_model": "LFM2-1.2B",
        "weights_sha256": "b" * 64,
        "manifest_sha256": "e" * 64,
        "path": "run-1/step-000100",
        "created_at": support.utc_now(),
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        (
            {"domain_state": DomainState.ASSERTED_NULL.value, "domain": "support"},
            "a value with a state that says there is none",
        ),
        (
            {"domain_state": DomainState.RECORDED.value, "domain": None},
            "a state of 'recorded' with nothing recorded",
        ),
        (
            {"domain_state": DomainState.ABSENT.value, "domain": "support"},
            "a value with a state that says the key was missing",
        ),
        (
            {"domain_state": DomainState.NOT_IN_KIND.value, "domain": None},
            "an adapter-only state on a checkpoint",
        ),
    ],
)
def test_the_database_refuses_a_collapsed_checkpoint_domain(
    engine, overrides: dict[str, object], why: str
) -> None:
    """Each CHECK bites. A Python-only distinction is unpinned by any other write path."""
    with (
        pytest.raises(IntegrityError, match=r"CHECK|check"),
        support.session_for(engine) as session,
    ):
        session.add(Checkpoint(**_checkpoint_kwargs(**overrides)))  # type: ignore[arg-type]
        session.flush()


@pytest.mark.parametrize(
    "overrides",
    [
        {"domain_state": DomainState.RECORDED.value, "domain": "support"},
        {"domain_state": DomainState.ASSERTED_NULL.value},
        {"domain_state": DomainState.ABSENT.value},
        # The case the OTHER constraint does not cover, and the one that is actually the
        # forbidden inference: a correct `not_in_kind` state with a domain smuggled in
        # beside it, which is exactly the shape of "fill it in from the base". Added
        # 2026-10-14 after a mutation control removed `ck_mf_adapters_
        # domain_never_inferred_from_base` and the suite stayed GREEN - every existing
        # case tripped the state CHECK first, so the column constraint was protecting
        # nothing. A surviving mutation is a test finding, not a code finding.
        {"domain": "support"},
    ],
)
def test_the_database_refuses_a_domain_on_an_adapter(engine, overrides: dict[str, object]) -> None:
    """``mf_adapters`` cannot hold a domain at all - the inference 4.2a forbids is unwritable."""
    kwargs: dict[str, object] = {
        "name": "LFM2-1.2B:run-2:adapter-50",
        "run_id": "run-2",
        "step": 50,
        "domain": None,
        "domain_state": DomainState.NOT_IN_KIND.value,
        "base_name": "LFM2-1.2B:run-1:step-100",
        "weights_sha256": "d" * 64,
        "manifest_sha256": "f" * 64,
        "path": "run-2/adapter-000050",
        "created_at": support.utc_now(),
    }
    kwargs.update(overrides)
    with (
        pytest.raises(IntegrityError, match=r"CHECK|check"),
        support.session_for(engine) as session,
    ):
        session.add(Adapter(**kwargs))  # type: ignore[arg-type]
        session.flush()


def test_all_three_checkpoint_states_are_storable(engine) -> None:
    """The constraints refuse the collapses without refusing the states themselves."""
    rows = [
        _checkpoint_kwargs(
            name="a:r:step-1", step=1, domain="support", domain_state=DomainState.RECORDED.value
        ),
        _checkpoint_kwargs(
            name="a:r:step-2", step=2, domain=None, domain_state=DomainState.ASSERTED_NULL.value
        ),
        _checkpoint_kwargs(
            name="a:r:step-3", step=3, domain=None, domain_state=DomainState.ABSENT.value
        ),
    ]
    with support.session_for(engine) as session:
        for row in rows:
            session.add(Checkpoint(**row))  # type: ignore[arg-type]
        session.flush()
        assert session.query(Checkpoint).count() == 3


# ------------------------------------------------------------------ the fixtures refuse


def test_the_fixture_builder_refuses_a_manifest_its_schema_rejects() -> None:
    """The fixture validation is itself load-bearing, so it is itself tested.

    Added 2026-10-14 after a mutation control turned ``tests.support.db._validate`` into a
    no-op and the whole suite stayed green: the one line standing between this suite and
    "fixtures whose fields agree by construction" (ADR-016) was protecting nothing. With
    this test, deleting the validation goes red.
    """
    import jsonschema

    with pytest.raises(jsonschema.ValidationError):
        # `signal_source` is an enum in the schema. A fixture free to invent one would let
        # a row builder be tested against a manifest no writer could ever produce.
        support.checkpoint_manifest(signal_source="telepathy")

    with pytest.raises(jsonschema.ValidationError):
        # An adapter manifest closes its object, so a stray key must be refused here too -
        # which is what makes the `not_in_kind` fixtures trustworthy.
        support.adapter_manifest(domain="support")
