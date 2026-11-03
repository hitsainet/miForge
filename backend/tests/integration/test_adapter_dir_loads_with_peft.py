"""AC15's in-repo substitute: a saved adapter loads with **standard** ``peft``.

Through **F5's** output-conformance harness with a PEFT loader passed in - F5's FR-005.22
assigns the adapter half to F8 in those words, and its loader is an argument *precisely* so
this is one harness and not two. **F8 writes no second harness**: two harnesses asserting one
output contract is two answers, and the wrong one is whichever nobody runs.

## This is a substitute, and what it does and does not prove

Acceptance criterion 15 requires a saved adapter to load into **miLLM** under BRD-01 R-01.27,
which is specced and unimplemented, and R-01.28's vendoring of ``adapter-manifest-v1.json``
into that repo is separate work. AC15 is therefore an **operator-run cross-repo step** (FPRD
D13). What this file proves is that the artifact is *standard* - loadable by the ordinary
tooling, with no miForge module involved in the load - which is the same thing AC23 proves
for checkpoints. It does **not** prove miLLM loads it.

## D7 is closed here - 2026-10-16

This file read ``MIFORGE_TEST_TOKENIZER`` and skipped all five of its cases without it,
which is foundation task 11.8's **D7**. The full argument is in the comment above
``pytestmark``; the short version is that the variable's documented contract could not be
satisfied - a real 32k tokenizer against ``TINY_CONFIG``'s 512-row embedding table raises
``IndexError`` at the first generation, and this gate generates - so the gate passed on no
machine. It now builds a loadable 512-vocab tokenizer with the suite's one real writer and
**runs everywhere ``peft`` is installed**.

## Why this SKIPS rather than passes when the ML stack is absent

``peft`` and ``transformers`` are declared dependencies (ADR-002) and are **not
installed in every environment this suite runs in** - including the one this feature was
built in. A harness that silently returned success without loading anything would be the
construction-green failure this project keeps finding: criterion 15 would read as proved
while nothing had been loaded. So the absence is a **skip, named**, and the feature report
says these cases have not been run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.bench_model import bench_prompts
from tests.support.conformance import (
    assert_output_conforms,
    ml_runtime_available,
    peft_loader,
)
from tests.support.test_adapter_doubles import TINY_CONFIG, adapter_config, optimizer_block
from tests.support.tiny_hf_model import write_tiny_tokenizer
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import paths as adapter_paths

_available, _why = ml_runtime_available()


def _peft_available() -> tuple[bool, str]:
    if not _available:
        return False, _why
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        return False, (
            f"peft is not installed in this environment ({exc}), so an adapter directory "
            f"cannot actually be loaded here. Reported as a skip rather than a pass: a "
            f"harness that returned success without loading anything would make "
            f"acceptance criterion 15 read as proved while nothing had been loaded."
        )
    return True, ""


_peft_ok, _peft_why = _peft_available()

# ``MIFORGE_TEST_TOKENIZER`` was read here until 2026-10-16, and that is foundation task
# 11.8's **D7**: *the env var's contract is unsatisfiable as documented.* The two halves
# could not both be true. ``TINY_CONFIG`` declares ``vocab_size`` 512, so a real tokenizer -
# which is what the variable asks an operator to point at, and every real one is 32k - hands
# the model an input id outside its embedding table and raises ``IndexError: index out of
# range in self`` at the first generation. The gate's own happy path generates. So an
# operator who satisfied the documented contract got a crash, and one who did not got five
# silent skips: **the gate could not pass on any machine.**
#
# **D7 is closed by using the product's own mechanism instead of reaching around it.**
# ``tests.support.tiny_hf_model.write_tiny_tokenizer`` builds a genuinely loadable tokenizer
# through the real ``PreTrainedTokenizerFast.save_pretrained`` writer, **sized from the
# config** - which is D7's recommendation applied in the direction that costs nothing (size
# the tokenizer to the model rather than the model to the tokenizer; the model here is a
# two-layer fixture whose whole purpose is to be small). It is the same writer
# ``test_create_run_gpu.py`` was reworked onto, and the same one the acquisition path's
# ``assert_vocab_agrees`` now refuses a mismatch against by name
# (``tokenizer_vocab_mismatch``) at preflight rather than at generation.
#
# **No env var, no network, and no skip.** The reason the variable existed - *"a test that
# silently reached the network would pass on a workstation and fail on the GPU node"* - is
# satisfied better by building the tokenizer locally than by asking for a path: there is
# nothing to point at and nothing to reach. All five cases now run wherever ``peft`` is
# installed, which is the only remaining skip and it names its reason.

pytestmark = [
    pytest.mark.skipif(not _peft_ok, reason=_peft_why or "peft unavailable"),
]

#: A prompt this fixture's tokenizer can actually ENCODE, and the reason it is not the
#: harness's default.
#:
#: **Found by making these cases run for the first time**, 2026-10-16. ``assert_output_conforms``
#: defaults to ``"miForge wrote this checkpoint and"``; the 512-token word-level tokenizer
#: splits on whitespace and knows ``<unk> <s> </s> <pad>`` plus ``t4``..``t511``, so every
#: word of that prompt encodes to ``<unk>``, the model emits ``</s>`` immediately and the
#: decoded text is **empty** - tripping the harness's own *"a loader that returns a model and
#: never runs it proves the files parse, not that the artifact works"* assertion. The
#: assertion was right and the input was degenerate. ``bench_prompts`` exists for exactly
#: this and says so in its docstring, so it is reused rather than respelled here.
ENCODABLE_PROMPT = bench_prompts(1)[0]


@pytest.fixture
def tiny_base(tmp_path: Path) -> Path:
    """A two-layer causal LM and its tokenizer, saved as a standard HF directory."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    directory = tmp_path / "base"
    config = AutoConfig.for_model(
        TINY_CONFIG["model_type"], **{k: v for k, v in TINY_CONFIG.items() if k != "model_type"}
    )
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    model.save_pretrained(directory, safe_serialization=True)
    # A tokenizer is *given* to miForge (BRD-02 section 8) - it never trains one. Written at
    # the ROOT, which is R-02.51's promise and what a standard loader requires, and **sized
    # from this config** (512), which is what closes D7: see the note above the pytestmark.
    write_tiny_tokenizer(directory, vocab_size=TINY_CONFIG["vocab_size"])
    # The positive check that the fixture is loadable rather than merely present. A
    # `tokenizer.json` that parses as a file and raises in the reader is the exact defect
    # `write_tiny_tokenizer`'s docstring records, and this gate's whole claim is that
    # STANDARD tooling can read what miForge writes.
    loaded = AutoTokenizer.from_pretrained(str(directory))
    assert len(loaded) <= TINY_CONFIG["vocab_size"], (
        f"the fixture tokenizer has {len(loaded)} tokens against a {TINY_CONFIG['vocab_size']}-"
        f"row embedding table; generation will raise IndexError, which is D7 reintroduced"
    )
    return directory


@pytest.fixture
def saved_adapter(tmp_path: Path, tiny_base: Path) -> tuple[Path, dict[str, Any]]:
    """One adapter, written by the **real** writer over a **real** ``peft`` model."""
    from transformers import AutoModelForCausalLM

    from training.runtypes.adapters import save as adapter_save
    from training.runtypes.adapters.build import build

    model = AutoModelForCausalLM.from_pretrained(str(tiny_base), trust_remote_code=False)
    config = adapter_config(rank=4, alpha=8.0, target_modules=("q_proj", "v_proj"))
    wrapped, validated = build(model, config)

    doc = adapter_manifest.build(
        run_id="run-peft",
        step=10,
        base={"name": str(tiny_base), "manifest_sha256": None},
        adapter=validated,
        optimizer=optimizer_block(),
    )
    dest = adapter_paths.adapter_dir("run-peft", 10, root=tmp_path / "artifacts")
    published = adapter_save.save(wrapped, doc, dest)
    return dest, published


class TestTheDirectoryIsLoadableByStandardTooling:
    def test_it_loads_and_generates_through_F5s_harness_with_a_PEFT_loader(
        self, tiny_base: Path, saved_adapter: tuple[Path, dict[str, Any]]
    ) -> None:
        directory, _ = saved_adapter
        report = assert_output_conforms(
            directory,
            loader=peft_loader(tiny_base),
            loader_name="peft",
            prompt=ENCODABLE_PROMPT,
        )
        assert report.generated.strip(), "a load that generates nothing proves only that "
        assert report.loader == "peft"
        assert adapter_paths.WEIGHTS_NAME in report.files

    def test_NO_MIFORGE_MODULE_is_imported_by_the_load(
        self, tiny_base: Path, saved_adapter: tuple[Path, dict[str, Any]]
    ) -> None:
        """What makes the claim a test rather than an assertion. The harness snapshots
        ``sys.modules`` across the load and requires that no miForge module appear."""
        directory, _ = saved_adapter
        report = assert_output_conforms(
            directory, loader=peft_loader(tiny_base), loader_name="peft", prompt=ENCODABLE_PROMPT
        )
        assert report.miforge_modules_imported == ()

    def test_the_tensor_keys_follow_the_PEFT_convention(
        self, saved_adapter: tuple[Path, dict[str, Any]]
    ) -> None:
        """FR-008.30. ``base_model.model.<module>.lora_A.weight`` / ``.lora_B.weight``, so
        an adapter is loadable by standard tooling and not only by this suite."""
        from safetensors import safe_open

        directory, _ = saved_adapter
        with safe_open(str(directory / adapter_paths.WEIGHTS_NAME), framework="pt") as handle:
            keys = list(handle.keys())
        assert keys
        assert all(key.startswith("base_model.model.") for key in keys), keys
        assert any(".lora_A." in key for key in keys)
        assert any(".lora_B." in key for key in keys)

    def test_the_PEFT_config_and_the_MANIFEST_agree_on_rank_and_alpha(
        self, saved_adapter: tuple[Path, dict[str, Any]]
    ) -> None:
        """The cross-check miLLM's loader performs, against a directory ``peft`` itself
        wrote - which is the case the doubles cannot cover."""
        import json

        directory, published = saved_adapter
        peft_config = json.loads((directory / adapter_paths.PEFT_CONFIG_NAME).read_text())
        assert peft_config["r"] == published["adapter"]["rank"]
        assert peft_config["lora_alpha"] == published["adapter"]["alpha"]

    def test_it_is_REFUSED_against_a_different_base(
        self, tmp_path: Path, saved_adapter: tuple[Path, dict[str, Any]]
    ) -> None:
        """E-10. The identity checked is ``base.name``; the shape a consumer checks is
        ``adapter.target_modules`` against the resident base."""
        from core.errors import Refusal
        from training.runtypes.adapters import load as adapter_load

        directory, _ = saved_adapter
        doc = adapter_load.load_manifest(directory / adapter_paths.MANIFEST_NAME)
        with pytest.raises(Refusal) as caught:
            adapter_load.assert_base_matches(doc, "some/other-model")
        assert caught.value.code == "adapter_base_mismatch"
