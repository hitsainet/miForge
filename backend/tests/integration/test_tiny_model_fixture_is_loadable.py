"""The suite's tiny Hugging Face fixture directory is genuinely loadable.  **Unmarked.**

## Why this file exists at all

``tests/support/tiny_hf_model.py``'s ``write_tiny_model`` is the input to every case in
``test_acquisition_gpu.py``. **A fixture is an input, so a broken one produces no red of its
own**: it produces an error in every case that depends on it - and on a ``gpu``-marked module
that ``addopts`` deselects, it produces *nothing at all*. That is exactly what happened.
``write_tiny_tokenizer`` wrote ``json.dumps({"version": "1.0"})`` until 2026-10-16, which
satisfied the acquirer's presence check (``tokenizer.json`` is one of the accepted names) and
then raised ``KeyError: 'added_tokens'`` inside ``AutoTokenizer.from_pretrained`` - blocking
the whole ``load`` half of F1 task 14.7 with nothing wrong in production code.

## Why it is HERE and not in the ``gpu`` module it came from

These two cases **need no card** - they write a directory and read it back - so living under
``pytestmark = pytest.mark.gpu`` meant CI never ran them, which is this project's recorded
*"a skipping test is not a passing test"* trap in its quietest form: not a reported skip, a
silent deselection. Moved out on 2026-10-16 so they run on every push.

**The writer was moved to ``tests/support/`` rather than copied here**, and that is
load-bearing. A copy would make these assertions validate a second writer while the one the
GPU cases actually consume drifted away unobserved - a fixture agreeing with itself by
construction, which is the project's named reason a suite stays green over a real bug. There
is one writer and both files import it.

## Why integration rather than unit

It is a round trip through a **real writer and a real reader** over the filesystem -
``PreTrainedTokenizerFast.save_pretrained`` out, ``AutoTokenizer.from_pretrained`` back, plus
production's own ``require_tokenizer_at_root`` over the result - which is the shape ADR-016
prefers to a hand-built dict and the shape ``tests/integration/`` holds.
"""

from __future__ import annotations

from pathlib import Path

from tests.support.tiny_hf_model import LLAMA_TINY, write_tiny_model


class TestTheFixtureItselfIsLoadable:
    """**Needs no card**, and exists because nothing checked it.

    ``write_tiny_model`` wrote a ``tokenizer.json`` of ``{"version": "1.0"}``, which
    satisfied the acquirer's presence check and then raised ``KeyError: 'added_tokens'``
    inside ``AutoTokenizer.from_pretrained`` - blocking both ``TestALoadOnTheRealCard``
    cases and so the whole ``load`` half of F1 task 14.7. Nothing was wrong with the
    production code.

    A fixture is an input to a test, so a broken one produces no red of its own: it produces
    an error in every case that depends on it, and on a suite that is deselected by default
    it produces nothing at all. These two assertions are the cheapest place for that to be
    visible, and they run wherever ``torch`` and ``transformers`` do - which, since both are
    declared dependencies (ADR-002) and CI installs them, is on every push.
    """

    def test_the_tiny_directory_loads_through_the_real_reader(self, tmp_path: Path) -> None:
        from transformers import AutoTokenizer

        directory = write_tiny_model(tmp_path / "tiny")
        tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False)
        ids = tokenizer("t10 t11 t12")["input_ids"]
        assert ids, "the tokenizer produced no token ids"
        assert max(ids) < LLAMA_TINY["vocab_size"], (
            f"token id {max(ids)} is outside the model's embedding table of "
            f"{LLAMA_TINY['vocab_size']}; a forward pass would raise IndexError (D7's "
            f"constraint, in this fixture's own terms)"
        )

    def test_the_tokenizer_files_are_at_the_ROOT(self, tmp_path: Path) -> None:
        """R-02.51 and ADR-015, asserted on the fixture rather than assumed from it: a
        ``tokenizer/`` subdirectory is the defect the requirement records, and the acquirer
        refuses it by name."""
        from training.acquisition.local_dir import require_tokenizer_at_root

        directory = write_tiny_model(tmp_path / "tiny")
        assert (directory / "tokenizer.json").is_file()
        assert not (directory / "tokenizer").exists()
        require_tokenizer_at_root(directory)
