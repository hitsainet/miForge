"""THE output-conformance harness  (FR-005.21, FR-005.22; R-02.51, AC23, AC-005.11).

**One harness, two artifact kinds, and the loader is an argument.** F5 owns the checkpoint
half; F8's adapter half is *this* harness with a PEFT loader passed in. The alternative -
each feature writing its own - is two harnesses asserting one output contract, and *"the
wrong one is whichever nobody runs."* F8's documents say it will write no second harness and
will import this module **by this name**, so the name and the signature are the contract:

    assert_output_conforms(directory, *, loader=TRANSFORMERS_LOADER, ...)

## What "conforms" means here, in three parts

1. **The file set is standard.** A flat Hugging Face model directory - ``model.safetensors``,
   ``config.json``, the tokenizer's own files **at the root** - with ``manifest.json`` as the
   only non-standard file and purely additive. As first specified a ``tokenizer/``
   subdirectory could not be loaded by ``transformers``, vLLM, TGI or miLLM's own loader,
   which broke R-02.21 and criterion 2 before any code existed.
2. **It loads and generates** through the loader handed in, from the directory alone.
3. **No miForge code is involved in the load.** This is the part that makes criterion 23 a
   test rather than an assertion, and it is asserted two ways because the two have different
   strengths:

   * :func:`assert_output_conforms` snapshots ``sys.modules`` across the load and requires
     that **no miForge module be imported by it**. Works with any loader, including one F8
     passes in as a callable, which is why it is the general form.
   * :func:`assert_loads_in_a_clean_process` runs the load in a **subprocess whose
     ``sys.path`` does not contain miForge at all**, so the claim is structural rather than
     observed. The strongest statement of AC23, and it cannot take an arbitrary callable -
     a Python function does not cross a process boundary - so it covers the default loader.

## Why it skips rather than passes without the ML stack

``transformers`` is a declared dependency (ADR-002) and is **not installed in every
environment this suite runs in**. A harness that silently returned success without loading
anything would be the construction-green failure this project keeps finding: AC23 would read
as proved while nothing had been loaded. So the absence is a **skip, named**, and the report
records which loader ran.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "FIRST_PARTY_ROOTS",
    "TRANSFORMERS_LOADER",
    "ConformanceReport",
    "Loader",
    "assert_loads_in_a_clean_process",
    "assert_output_conforms",
    "ml_runtime_available",
    "peft_loader",
    "standard_file_set",
    "transformers_loader",
]

#: A loader takes a directory and returns ``(model, tokenizer)``. Anything with
#: ``generate`` and a callable tokenizer will do: the harness asserts behaviour, not types.
Loader = Callable[[Path], tuple[Any, Any]]

#: miForge's own import roots, from the one place that already lists them (``pyproject``'s
#: ``known-first-party``). Read as a literal here because this module must be able to make
#: its assertion **without importing anything of miForge's** - importing a miForge module to
#: find out whether miForge was imported is the shape of a test that cannot fail.
FIRST_PARTY_ROOTS: frozenset[str] = frozenset(
    {
        "api",
        "core",
        "db",
        "models",
        "schemas",
        "services",
        "clients",
        "providers",
        "training",
        "packs",
        "workers",
        "miforge",
    }
)

#: The non-standard file a checkpoint is allowed to carry, and the only one.
MIFORGE_ADDITIVE_FILES: frozenset[str] = frozenset({"manifest.json", "adapter_manifest.json"})


@dataclass(frozen=True)
class ConformanceReport:
    """What the harness observed. Returned so a caller can assert further, or log it."""

    directory: Path
    loader: str
    files: tuple[str, ...]
    #: Text the model produced. Non-empty is the point: a loader that returned a model and
    #: never ran it would prove the files parse, not that the artifact works.
    generated: str
    #: miForge modules imported **during the load**. Must be empty.
    miforge_modules_imported: tuple[str, ...]


def ml_runtime_available() -> tuple[bool, str]:
    """Whether ``transformers`` and ``torch`` can be imported here, and why not if not."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        return False, (
            f"the ML runtime is not installed in this environment ({exc}), so a checkpoint "
            f"cannot actually be loaded here. Everything else in miForge runs without it "
            f"(ADR-023); this harness is the one thing that cannot, and it says so rather "
            f"than reporting success for a load that did not happen."
        )
    return True, ""


def transformers_loader(directory: Path) -> tuple[Any, Any]:
    """The default: plain ``transformers``, from the directory alone.

    ``trust_remote_code`` is **off**: an artifact that needs to execute its own Python to
    load is not the portable artifact R-02.51 promises, and loading one here would run that
    code inside the test process.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(directory), trust_remote_code=False)
    return model, tokenizer


#: The default loader, named so F8 can see what it is replacing.
TRANSFORMERS_LOADER: Loader = transformers_loader


def peft_loader(base: Path) -> Loader:
    """A loader for F8's half: the base model with an adapter applied (FR-005.22).

    Provided here so the adapter half is *this* harness with a different loader rather than
    a second harness. **F8 owns the adapter contract**; this function owns nothing about it
    beyond "load the base, apply the adapter directory, return the pair".
    """

    def load(adapter_directory: Path) -> tuple[Any, Any]:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(base), trust_remote_code=False)
        return PeftModel.from_pretrained(model, str(adapter_directory)), tokenizer

    return load


def standard_file_set(directory: Path) -> tuple[str, ...]:
    """Assert the directory is a flat standard model directory. Returns its file set.

    Flat is the assertion that matters: a nested directory is how a ``tokenizer/``
    subdirectory would arrive, and that is the defect that broke the output contract before
    any code existed.
    """
    entries = sorted(directory.iterdir())
    nested = [e.name for e in entries if e.is_dir()]
    assert not nested, (
        f"{directory} contains subdirectories {nested}. A conforming artifact is a FLAT "
        f"Hugging Face directory with the tokenizer's files at the root; a 'tokenizer/' "
        f"subdirectory cannot be loaded by transformers, vLLM, TGI or miLLM's own loader."
    )
    names = tuple(e.name for e in entries)
    assert any(n.endswith(".safetensors") for n in names), (
        f"{directory} holds no .safetensors file: {names}. safe_serialization is the "
        f"contract, not a default - a pytorch_model.bin is a pickle, and a consumer is "
        f"entitled to refuse to unpickle one."
    )
    return names


def assert_output_conforms(
    directory: Path,
    *,
    loader: Loader = TRANSFORMERS_LOADER,
    prompt: str = "miForge wrote this checkpoint and",
    max_new_tokens: int = 4,
    loader_name: str | None = None,
) -> ConformanceReport:
    """Load ``directory`` with ``loader``, generate, and assert no miForge code was involved.

    Args:
        directory: the artifact. A checkpoint for F5, an adapter directory for F8.
        loader: ``(directory) -> (model, tokenizer)``. Defaults to plain ``transformers``;
            F8 passes :func:`peft_loader`'s result.
        prompt: anything; the assertion is that *something* is generated.
        max_new_tokens: kept tiny - the point is that generation happens, not what it says.
        loader_name: for the report, when the loader is a closure with no useful ``__name__``.

    Raises:
        AssertionError: on a non-standard file set, an empty generation, or any miForge
            module imported by the load.
    """
    import pytest

    available, why = ml_runtime_available()
    if not available:
        pytest.skip(why)

    files = standard_file_set(directory)

    before = set(sys.modules)
    model, tokenizer = loader(directory)
    inputs = tokenizer(prompt, return_tensors="pt")
    # **Only what a causal LM accepts** - ids and a mask, never `**inputs`.
    #
    # *This passed ``**inputs`` until 2026-10-16, and it is the same defect the clean-process
    # script above had fixed on that very day.* The fix landed in one of the two halves of
    # one harness, which is this project's named "corrected in three of four layers"
    # family: a tokenizer built on ``models.WordLevel`` carries ``token_type_ids`` in
    # ``model_input_names``, and ``generate`` refuses an unused ``model_kwargs`` key
    # outright - ``ValueError: The following model_kwargs are not used by the model:
    # ['token_type_ids']``.
    #
    # ``WordLevel`` is not an exotic choice here: it is what ``write_tiny_tokenizer`` must
    # use, because a 512-token vocabulary is the binding constraint of ``LLAMA_TINY``. So
    # the **general** form of the output-conformance harness could not pass for a
    # conforming checkpoint built from the only tokenizer small enough to use with the
    # suite's tiny config, and it failed it for a reason that is not about the checkpoint -
    # the opposite of what this harness is for. ``token_type_ids`` is an encoder-model
    # input; a decoder takes ids and a mask.
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    imported = tuple(
        sorted(
            name for name in set(sys.modules) - before if name.split(".")[0] in FIRST_PARTY_ROOTS
        )
    )

    assert imported == (), (
        f"loading {directory} imported miForge modules {list(imported)}. The output "
        f"contract is that the artifact loads in software that has never heard of miForge "
        f"(R-02.51, criterion 23); if its load needs this codebase, it is not a standard "
        f"artifact."
    )
    assert generated.strip(), (
        f"loading {directory} produced a model that generated nothing. A loader that "
        f"returns a model and never runs it proves the files parse, not that the artifact "
        f"works."
    )

    return ConformanceReport(
        directory=directory,
        loader=loader_name or getattr(loader, "__name__", repr(loader)),
        files=files,
        generated=generated,
        miforge_modules_imported=imported,
    )


#: The script the clean-process check runs. Deliberately a string: it is executed by an
#: interpreter whose ``sys.path`` has no miForge on it, so it cannot import anything of
#: this codebase's even by accident.
_CLEAN_LOAD_SCRIPT = """
import json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
directory = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=False)
inputs = tokenizer("miForge wrote this checkpoint and", return_tensors="pt")
# **Only what a causal LM accepts.** This passed ``**inputs`` until 2026-10-16, which
# hands ``generate`` every key the tokenizer happened to emit. A tokenizer built on
# ``models.WordLevel`` - which is what ``write_tiny_tokenizer`` must use, because a
# 512-token vocabulary is the binding constraint of ``LLAMA_TINY`` - carries
# ``token_type_ids`` in ``model_input_names``, and ``generate`` refuses an unused
# ``model_kwargs`` key outright: ``ValueError: The following model_kwargs are not used
# by the model: ['token_type_ids']``. That failed a **conforming** checkpoint for a
# reason that is not about the checkpoint, which is the opposite of what this check is
# for. ``token_type_ids`` is an encoder-model input; a decoder takes ids and a mask.
out = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs.get("attention_mask"),
    max_new_tokens=4,
    do_sample=False,
)
first_party = {"api","core","db","models","schemas","services","clients","providers",
               "training","packs","workers","miforge"}
print(json.dumps({
    "generated": tokenizer.decode(out[0], skip_special_tokens=True),
    "miforge_modules": sorted(m for m in sys.modules if m.split(".")[0] in first_party),
}))
"""


def assert_loads_in_a_clean_process(directory: Path, *, timeout: int = 300) -> dict[str, Any]:
    """The strong form of AC23: load in a process that **cannot** import miForge.

    ``sys.path`` is rebuilt without this repository's source root and every ``MIFORGE_*``
    variable is stripped, so *"no miForge code was involved"* is structural rather than
    observed. The in-process form exists for an arbitrary loader; this one is the statement
    of the criterion.
    """
    import pytest

    available, why = ml_runtime_available()
    if not available:
        pytest.skip(why)

    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MIFORGE_")}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != source_root
    )
    env["PYTHONNOUSERSITE"] = "1"

    # A fixed script and a path the test chose; nothing from a request reaches this.
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _CLEAN_LOAD_SCRIPT, str(directory)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )
    assert (
        completed.returncode == 0
    ), f"a clean process could not load {directory}:\n{completed.stderr[-4000:]}"
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["miforge_modules"] == [], report["miforge_modules"]
    assert report["generated"].strip()
    return dict(report)
