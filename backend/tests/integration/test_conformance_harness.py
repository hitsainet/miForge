"""AC23: a checkpoint loads in software that has never heard of miForge  (AC-005.11).

**This is the only honest proof of the output contract**, and the only thing in this feature
that cannot run without the ML stack. ``transformers`` is a declared dependency (ADR-002)
and is not installed in every environment this suite runs in, so the loading tests **skip by
name**: a harness that reported success without loading anything would be the
construction-green failure this project keeps finding - AC23 would read as proved while
nothing had been loaded.

What **does** run everywhere is the harness's own structural half: the file-set assertion,
the miForge-import check and the negative controls that prove each can fail. Those are what
keep the skip honest, because a harness nobody can exercise is a harness nobody can trust.

F8 imports this module **by name** and passes a PEFT loader for the adapter half
(FR-005.22). The signature is the seam, so it is asserted here rather than left to F8 to
discover at integration.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tests.support import conformance
from tests.support.checkpoints import TinyTokenizer, saved_checkpoint


class TestTheSeamF8ImportsByName:
    def test_the_module_is_at_the_path_F8s_documents_name(self):
        """F8's own task list says it will write no second harness and will import this
        one; *"two harnesses asserting one output contract is two answers, and the wrong one
        is whichever nobody runs."*"""
        assert Path(conformance.__file__).name == "conformance.py"
        assert Path(conformance.__file__).parent.name == "support"

    def test_the_loader_is_an_argument_with_the_plain_transformers_default(self):
        """The loader argument is what keeps it **one** harness: F8 passes a PEFT loader in
        place of the plain one rather than forking the file."""
        signature = inspect.signature(conformance.assert_output_conforms)
        assert list(signature.parameters) == [
            "directory",
            "loader",
            "prompt",
            "max_new_tokens",
            "loader_name",
        ]
        assert signature.parameters["directory"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert signature.parameters["loader"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["loader"].default is conformance.TRANSFORMERS_LOADER

    def test_a_peft_loader_factory_is_published_for_the_adapter_half(self):
        """FR-005.22. F5 defines no adapter behaviour; this is the hook, and the adapter
        contract itself is F8's."""
        assert callable(conformance.peft_loader)
        assert inspect.signature(conformance.peft_loader).parameters["base"]


class TestTheStructuralHalfRunsEverywhere:
    def test_a_written_checkpoint_passes_the_flat_standard_file_set_assertion(self, tmp_path: Path):
        result = saved_checkpoint(tmp_path)
        names = conformance.standard_file_set(result.directory)
        assert "model.safetensors" in names and "manifest.json" in names

    def test_a_tokenizer_subdirectory_fails_the_assertion(self, tmp_path: Path):
        """The negative control for the one above. As first specified a ``tokenizer/``
        subdirectory could not be loaded by ``transformers``, vLLM, TGI or miLLM's own
        loader - it broke R-02.21 and criterion 2 before any code existed."""
        directory = tmp_path / "artifact"
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"x")
        (directory / "config.json").write_text("{}")
        (directory / "tokenizer").mkdir()
        with pytest.raises(AssertionError, match="FLAT"):
            conformance.standard_file_set(directory)

    def test_a_directory_with_no_safetensors_fails_the_assertion(self, tmp_path: Path):
        """``safe_serialization`` is the contract: a ``pytorch_model.bin`` is a pickle, and
        a consumer is entitled to refuse to unpickle one."""
        directory = tmp_path / "artifact"
        directory.mkdir()
        (directory / "pytorch_model.bin").write_bytes(b"x")
        (directory / "config.json").write_text("{}")
        with pytest.raises(AssertionError, match="safetensors"):
            conformance.standard_file_set(directory)

    def test_the_first_party_roots_match_the_projects_own_import_roots(self):
        """The import check is only as good as its list. Read against ``pyproject``'s
        ``known-first-party`` so a package added there and not here would be visible -
        the module cannot *import* miForge to find out, since importing miForge to prove
        miForge was not imported is a test that cannot fail."""
        import tomllib

        config = tomllib.loads(
            (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text("utf-8")
        )
        declared = set(config["tool"]["ruff"]["lint"]["isort"]["known-first-party"])
        assert (
            declared <= conformance.FIRST_PARTY_ROOTS
        ), f"not covered by the import check: {sorted(declared - conformance.FIRST_PARTY_ROOTS)}"

    def test_the_runtime_probe_reports_WHY_rather_than_only_that(self, monkeypatch):
        """ADR-013: a refusal says what is missing. A bare ``False`` would make the skip
        reason *"skipped"*, which tells the next reader nothing."""
        available, why = conformance.ml_runtime_available()
        if available:
            assert why == ""
        else:
            assert "ML runtime" in why and "ADR-023" in why


class TestTheRealLoad:
    """AC-005.11 / BRD AC23. **Skips by name** where the ML runtime is absent."""

    @pytest.fixture
    def real_checkpoint(self, tmp_path: Path):
        pytest.importorskip(
            "transformers",
            reason=(
                "AC23 is a LOAD, and transformers is not installed here. The structural half "
                "of the harness runs above; this is deliberately not faked, because a "
                "harness that reported success without loading anything would make the "
                "criterion read as proved while nothing had been loaded."
            ),
        )
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        # ⚠ THE TOKENIZER COMES FIRST, AND THE MODEL'S vocab_size IS READ FROM IT.
        #
        # This read `vocab_size=64` beside `AutoTokenizer.from_pretrained("gpt2")`, whose
        # vocabulary is **50,257** - so every prompt tokenised to ids far beyond the
        # embedding table and the forward pass raised
        # `IndexError: index out of range in self`. The checkpoint written was internally
        # inconsistent: a config claiming 64 tokens beside a tokenizer with 50,257.
        #
        # **It had never been observed, because the whole class needs `transformers` and
        # that was missing from the venv** - so `importorskip` above skipped it on every
        # local run and the mismatch sat there. Found only when the declared dependency set
        # was finally installed, which is foundation task 10.1.
        #
        # That matters more than a broken fixture: this class is the one that proves
        # **acceptance criterion 23** - *the artifact loads in software that has never heard
        # of miForge*. A headline guarantee whose only real test could not run.
        #
        # Everything else stays deliberately tiny (one layer, one head, 8-wide); only the
        # embedding grows, to ~400K parameters, so the test stays fast. `n_positions` is
        # raised above the prompt plus `max_new_tokens`, since GPT-2 has no rotary
        # embeddings and a generation past the context window is a second IndexError.
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = AutoConfig.for_model(
            "gpt2",
            n_layer=1,
            n_head=1,
            n_embd=8,
            vocab_size=len(tokenizer),
            n_positions=64,
        )
        model = AutoModelForCausalLM.from_config(config)

        from services.checkpoints.paths import checkpoint_dir
        from services.checkpoints.writer import write
        from tests.support.checkpoints import fields

        with torch.no_grad():
            return write(
                model,
                tokenizer,
                dest=checkpoint_dir("run-a", 1, root=tmp_path),
                fields=fields(step=1),
            )

    def test_the_no_miForge_import_check_ACTUALLY_BITES(self, real_checkpoint):
        """**A surviving mutation, turned into a test.**

        Making `imported = ()` unconditionally — i.e. gutting the first-party import check —
        changed **no test result**. So the assertion that carries acceptance criterion 23's
        real content (*the artifact loads in software that has never heard of miForge*) was
        itself unprotected: it could have been deleted and the suite would have stayed green
        while the criterion read as proved.

        This drives the harness with a loader that deliberately imports a miForge module and
        requires the harness to refuse, naming it. Re-running the mutation against this test
        turns it red, which is what makes the check load-bearing rather than decorative.
        """
        import importlib
        import sys

        import pytest as _pytest

        from tests.support import conformance as conf

        # The harness diffs `sys.modules` across the load, so an import of a module the
        # suite has ALREADY imported registers nothing and the test passes vacuously -
        # which is how the first draft of this test failed: `core.canonical` was resident
        # by the time it ran. Popping it makes the loader's import genuinely new, and the
        # `was_new` assertion below is the positive check that the measurement happened.
        probe = "core.canonical"
        saved = sys.modules.pop(probe, None)
        observed: dict[str, bool] = {}
        try:

            def loader_that_imports_miforge(directory):
                # A real load, plus one first-party import - the shape of the defect: a
                # loader that quietly needs this codebase to read its own artifact.
                observed["was_new"] = probe not in sys.modules
                importlib.import_module(probe)
                return conf.TRANSFORMERS_LOADER(directory)

            with _pytest.raises(AssertionError) as caught:
                conf.assert_output_conforms(
                    real_checkpoint.directory, loader=loader_that_imports_miforge
                )
        finally:
            if saved is not None:
                sys.modules[probe] = saved
            else:
                sys.modules.pop(probe, None)

        assert observed.get("was_new") is True, (
            f"{probe} was already in sys.modules when the loader ran, so its import added "
            f"nothing for the harness to notice and this test would pass against a gutted "
            f"check. Pick a module the suite has not imported, or keep popping it first."
        )
        message = str(caught.value)
        assert (
            "imported miForge modules" in message
        ), f"the harness did not notice a first-party import: {message[:300]}"
        assert probe in message, "the refusal must name what was imported"

    def test_a_checkpoint_loads_in_transformers_from_the_directory_alone(self, real_checkpoint):
        report = conformance.assert_output_conforms(real_checkpoint.directory)
        assert report.miforge_modules_imported == ()
        assert report.generated.strip()

    def test_it_loads_in_a_process_that_cannot_import_miforge_at_all(self, real_checkpoint):
        """The strong form: ``sys.path`` is rebuilt without this repository, so *"no miForge
        code was involved"* is structural rather than observed."""
        report = conformance.assert_loads_in_a_clean_process(real_checkpoint.directory)
        assert report["miforge_modules"] == []


class TestTheHarnessCanFail:
    def test_a_loader_that_never_generates_is_caught(self, tmp_path: Path):
        """The negative control for the whole harness. A loader that returned a model and
        never ran it would prove the files parse, not that the artifact works - so the
        harness has to notice."""
        pytest.importorskip("transformers", reason="the assertion under test needs the runtime")
        result = saved_checkpoint(tmp_path, tokenizer=TinyTokenizer())

        class Silent:
            def generate(self, **_):
                return [[0]]

        def loader(_directory):
            return Silent(), type("T", (), {"decode": lambda self, *_, **__: "  "})()

        with pytest.raises((AssertionError, TypeError)):
            conformance.assert_output_conforms(result.directory, loader=loader)
