"""The vendored contract's identity, and the path the validator actually reads  (FR-010.10).

**This is not a second sync mechanism.** Byte-identity against miDataworks' normative copy is
asserted in exactly one place - ``backend/tests/contract/test_schema_sync.py``, foundation
task 1.4, whose ``OWNERSHIP`` table carries
``"midataworks-dataset-version-v1.json": Vendored(MIDATAWORKS, requirement="R-02.37")`` -
and :class:`TestFoundationOneFourCoversThisContract` below asserts that coverage rather than
duplicating it. ADR-014 allows one mechanism per obligation; two would be two answers.

What *is* here is the identity of the file this feature reads, and the assertion that the
runtime validator reads **that** file:

* a validator pointed at an embedded copy would pass every test while the vendored file
  drifted - the sync test routed around rather than broken, which is the harder failure to
  see;
* the four ``const``/identity facts (``$id``, ``title``, ``kind``, ``schema_version``) are
  what make a file *this* contract rather than a file with the right name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.parts import contract_validator as cv
from tests.contract import test_schema_sync as sync
from tests.support import part_fixtures as fx


@pytest.fixture(scope="module")
def schema() -> dict[str, object]:
    return dict(cv.load_schema())


class TestItIsThisContractAndNotMerelyAFileWithThatName:
    def test_the_id_names_midataworks_as_the_owner(self, schema):
        """miDataworks is normative for it (BRD-03 R-03.47). The ``$id`` says so, and if it
        ever pointed at miForge the vendoring relationship would have been inverted."""
        assert str(schema["$id"]).endswith("/docs/schemas/midataworks-dataset-version-v1.json")
        assert "miDataworks" in str(schema["$id"])

    def test_the_title(self, schema):
        assert schema["title"] == "miDataworks Dataset Version v1"

    def test_the_kind_const(self, schema):
        assert schema["properties"]["kind"]["const"] == "midataworks.dataset-version"

    def test_the_schema_version_const_is_the_string_one(self, schema):
        """``"1"``, a **string**. A reader that accepted the integer ``1`` would validate a
        document miDataworks would not have written."""
        assert schema["properties"]["schema_version"]["const"] == "1"

    def test_the_root_closes_its_object(self, schema):
        """``additionalProperties: false``, so R-02.35's *"exactly the schema's fields"* is
        enforced by the schema rather than by good intentions. Both vendored schemas were
        open until 2026-10-13, when an invented key validated clean."""
        assert schema["additionalProperties"] is False

    def test_it_declares_draft_2020_12(self, schema):
        """The validator is constructed for this draft **explicitly**; auto-detection would
        be a silent downgrade waiting for a day when the declaration moves."""
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


class TestTheRuntimeValidatorReadsTheVendoredFile:
    def test_the_path_is_under_docs_schemas(self):
        assert cv.schema_path().parent.name == "schemas"
        assert cv.schema_path().parent.parent.name == "docs"
        assert cv.schema_path().is_file()

    def test_it_is_the_same_file_the_sync_test_checks(self):
        """**The assertion that keeps the sync test from being routed around.**

        Not "the same contents" - the same ``Path``. An embedded copy with identical bytes
        today is a copy that diverges silently tomorrow.
        """
        assert (
            cv.schema_path().resolve()
            == (sync.SCHEMA_DIR / "midataworks-dataset-version-v1.json").resolve()
        )

    def test_the_validator_was_built_from_that_file(self):
        """Read off the compiled validator rather than from the loader, so a loader that
        read the file and then validated against something else is caught."""
        on_disk = json.loads(cv.schema_path().read_text("utf-8"))
        assert cv.validator().schema == on_disk

    def test_nothing_embeds_a_copy_of_the_schema(self):
        """A module carrying the contract as a literal would validate against itself.

        Searched for the contract's own discriminator rather than for the filename, because
        the filename legitimately appears in constants and docstrings while
        ``"midataworks.dataset-version"`` as a schema ``const`` does not.
        """
        src = Path(__file__).resolve().parents[2] / "src"
        offenders = [
            str(path.relative_to(src))
            for path in src.rglob("*.py")
            if '"const": "midataworks.dataset-version"' in path.read_text("utf-8")
        ]
        assert offenders == [], offenders


class TestFoundationOneFourCoversThisContract:
    """Confirm the one sync mechanism covers this contract - **do not add a second**."""

    def test_the_contract_is_in_the_ownership_table(self):
        entry = sync.OWNERSHIP["midataworks-dataset-version-v1.json"]
        assert isinstance(entry, sync.Vendored)
        assert entry.requirement == "R-02.37"

    def test_it_is_declared_as_vendored_from_midataworks_and_not_owned_here(self):
        """miForge is **not** normative for it, and the table must say so: an ``Owned``
        entry would make miForge's copy the reference and invert R-03.47."""
        entry = sync.OWNERSHIP["midataworks-dataset-version-v1.json"]
        assert entry.normative_here is False
        assert entry.source.repo == "miDataworks"
        assert entry.source.relative == "docs/schemas"

    def test_the_sibling_path_is_env_overridable(self):
        """A sync test that hard-codes ``/home/<someone>/app/...`` passes on one
        workstation. miDataworks' own sync test takes the same care."""
        entry = sync.OWNERSHIP["midataworks-dataset-version-v1.json"]
        assert entry.source.env == "MIFORGE_MIDATAWORKS_ROOT"

    def test_the_file_is_discovered_by_the_parametrisation(self):
        """Driven by scanning ``docs/schemas/``, so this contract is a case automatically -
        and if it were ever removed from the directory this would say so."""
        assert "midataworks-dataset-version-v1.json" in sync.discovered_schemas()

    def test_this_file_adds_no_byte_identity_assertion_of_its_own(self):
        """Stated as a test so the rule survives the next person adding "just one more
        check". One mechanism per obligation (ADR-014).

        **By AST, not by text** - this file's own prose is full of the words a text search
        would match, which is exactly the failure miDataworks' single-call-site test warns
        about: *"a comment or docstring naming a function neither satisfies nor trips a
        check."* The first draft of this very test was a substring search and it failed
        against its own assertion line.
        """
        import ast

        tree = ast.parse(Path(__file__).read_text("utf-8"))
        byte_comparisons = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and sum(
                1
                for operand in [node.left, *node.comparators]
                if isinstance(operand, ast.Call)
                and isinstance(operand.func, ast.Attribute)
                and operand.func.attr in {"read_bytes", "read_text"}
            )
            >= 2
        ]
        assert byte_comparisons == [], (
            f"lines {byte_comparisons} compare two files' contents - that is foundation "
            f"1.4's job, and a second mechanism is two answers to one question (ADR-014)."
        )


class TestTheRealDocumentRoundTrips:
    def test_midataworks_own_export_manifest_validates_byte_for_byte_as_exported(self):
        """**The contract round trip.** Validated from the committed bytes, not from a dict
        a test built - because a dict a test built agrees with this repository's reading of
        the contract by construction, and the only interesting question is whether
        miForge's reader agrees with miDataworks' *writer*."""
        version, digest = cv.parse_and_validate(fx.real_manifest_bytes(), source="fixture")
        assert version.content_kind == "rows"
        assert len(version.splits) == 2
        assert len(digest) == 64

    def test_the_fixture_records_where_it_came_from(self):
        """A fixture with no provenance is a fixture nobody can regenerate - and this one
        has to be regenerable, because it is only evidence while it is current."""
        note = fx.FIXTURE_DIR / "PROVENANCE.md"
        assert note.is_file()
        text = note.read_text("utf-8")
        assert "manifest_fixtures.py" in text
        assert "rows_manifest" in text
        # The commit it was taken at, so a reader can diff against that exact state.
        assert "4898e66" in text

    def test_the_instance_filename_has_no_v1(self):
        """Verified in their ``services/publishing/manifest_builder.py``:
        ``MANIFEST_FILE = "midataworks-dataset-version.json"``. The ``-v1`` suffix is the
        *schema* filename. Hard-coding the wrong one fails only at integration time."""
        assert cv.MANIFEST_FILENAME == "midataworks-dataset-version.json"
        assert cv.SCHEMA_FILENAME == "midataworks-dataset-version-v1.json"
        assert (fx.FIXTURE_DIR / cv.MANIFEST_FILENAME).is_file()
