"""The one tiny Hugging Face model directory writer the suite uses.

**It lives here because two test files need it and there must be exactly one of it.**
``LLAMA_TINY``, :func:`write_tiny_model` and :func:`write_tiny_tokenizer` were private to
``tests/integration/test_acquisition_gpu.py`` until 2026-10-16. The cases that assert the
writer's output is *loadable* moved to an unmarked home so CI runs them
(``tests/integration/test_tiny_model_fixture_is_loadable.py``), and the ``gpu``-marked
acquisition cases still consume it - so a **copy** would have made the unmarked assertions
validate a second writer while the real one drifted away unobserved. That is this project's
named reason a suite stays green over a real bug: *a fixture whose fields agree by
construction*. One writer, imported by both.

**Tokenizer files at the root**, because that is what R-02.51 promises and what the acquirer
requires - a ``tokenizer/`` subdirectory is the defect the requirement records, and
``training.acquisition.local_dir.require_tokenizer_at_root`` refuses it by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: The architecture every tiny fixture in the acquisition path is built from. 512 tokens of
#: vocabulary is the binding constraint on the tokenizer below - see its docstring.
LLAMA_TINY: dict[str, Any] = {
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 512,
    "max_position_embeddings": 128,
}


def write_tiny_model(directory: Path) -> Path:
    """Instantiate a tiny model and save it as a standard Hugging Face directory.

    **Tokenizer files at the root**, because that is what R-02.51 promises and what the
    acquirer requires - a ``tokenizer/`` subdirectory is the defect the requirement records.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    directory.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.for_model(
        "llama", **{k: v for k, v in LLAMA_TINY.items() if k != "model_type"}
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    model.save_pretrained(directory, safe_serialization=True)
    write_tiny_tokenizer(directory)
    return directory


def write_tiny_tokenizer(directory: Path, vocab_size: int = LLAMA_TINY["vocab_size"]) -> None:
    """A **genuinely loadable** minimal tokenizer, at the ROOT.

    ``R-02.51`` puts the tokenizer files at the root, not in a ``tokenizer/`` subdirectory,
    and ``local_dir.require_tokenizer_at_root`` enforces that. What it cannot enforce is
    that the file is *valid*.

    *This wrote ``json.dumps({"version": "1.0"})`` until 2026-10-16.* That satisfied the
    acquirer's presence check - ``tokenizer.json`` is one of the accepted names - and then
    ``AutoTokenizer.from_pretrained`` raised ``KeyError: 'added_tokens'``, which blocked
    **both** ``TestALoadOnTheRealCard`` cases, i.e. the whole ``load`` half of F1 task 14.7.
    Reproduced on CPU with ``transformers`` 4.57.6, so it is not a version accident and not
    a card problem: a file that is present and unparseable is the file-shaped equivalent of
    a fixture that agrees with itself by construction.

    **Built through ``PreTrainedTokenizerFast.save_pretrained``** - the real writer - rather
    than by hand-writing the JSON, so it carries ``tokenizer.json``,
    ``tokenizer_config.json`` and ``special_tokens_map.json`` in the shapes the real reader
    expects, and so a future ``tokenizers`` format change breaks the *writer* here instead
    of silently diverging from it.

    **The vocabulary is sized from the model's config, not from a real tokenizer.**
    ``LLAMA_TINY`` has ``vocab_size`` 512, so a 32k-vocab tokenizer would hand the model an
    input id outside its embedding table - ``IndexError: index out of range in self`` - the
    moment anything generates with it.

    **That was foundation task 11.8's D7, and this function is what closed it** (2026-10-16).
    *This paragraph read "the same constraint ``MIFORGE_TEST_TOKENIZER`` runs into (D7,
    reported and not fixed here)" until then.* The env var asked an operator to point at a
    real tokenizer, and every real one is 32k, so its documented contract could not be
    satisfied against a 512-row embedding table: the two cases that read it passed on no
    machine - one crashed and the other skipped. Both now call this writer instead
    (``test_create_run_gpu.py`` and ``test_adapter_dir_loads_with_peft.py``), the env var is
    read nowhere, and the acquisition path refuses the mismatch by name at preflight
    (``tokenizer_vocab_mismatch``) rather than reaching ``IndexError`` at generation.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<unk>", "<s>", "</s>", "<pad>"]
    vocab = {token: index for index, token in enumerate(specials)}
    vocab.update({f"t{index}": index for index in range(len(specials), vocab_size)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    ).save_pretrained(directory)


def write_tokenizer_json_only(directory: Path, vocab_size: int = LLAMA_TINY["vocab_size"]) -> Path:
    """A ``tokenizer.json`` with nothing but a vocabulary. **Not loadable; on purpose.**

    :func:`write_tiny_tokenizer` is the one *real* writer and the one every test that
    actually loads a tokenizer uses. This writes the file the **gate** reads and nothing
    else, and it exists for two reasons neither of which the real writer can serve:

    * ``training.acquisition.scratch``'s preflight refusals - the presence check and
      :func:`~training.acquisition.scratch.assert_vocab_agrees` - parse this file and never
      load it, so they are testable with no ML wheels at all (ADR-023), which is where most
      of this feature's defect risk lives.
    * A **32k** vocabulary is one line here and a real 32k tokenizer to build; the
      mismatch refusal is the behaviour under test and the tokenizer is the input to it.

    *The honesty cost is stated rather than left implicit:* a hand-written fixture that
    agrees with the reader by construction is this project's named reason a suite stays
    green over a real bug. What keeps the two in step is
    ``test_acquisition_scratch.py::TestTheVocabularyGate::
    test_the_REAL_tokenizer_writers_output_reads_as_the_size_it_was_built_for``, which runs
    the gate over :func:`write_tiny_tokenizer`'s output and requires the number it
    computes to be the one the real writer was asked for. If the ceiling reader ever stops
    understanding a real tokenizer's file, that case goes red here rather than in a run.
    """
    import json

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "added_tokens": [],
                "model": {
                    "type": "WordLevel",
                    "vocab": {f"t{index}": index for index in range(vocab_size)},
                    "unk_token": "t0",
                },
            }
        ),
        encoding="utf-8",
    )
    return directory
