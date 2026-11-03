"""A tokenizer directory WRITTEN BY transformers 5 loads through miForge's verify path.

**F11 window 4 (2026-10-18), FTASKS 18.4.** miLLM serves on transformers 5 and saves each
checkpoint's tokenizer with it, so ``tokenizer_config.json`` carries
``"tokenizer_class": "TokenizersBackend"`` - the 5.x name for what 4.x called
``PreTrainedTokenizerFast``. miForge then ran 4.57.6 (``transformers>=4.46,<5``), whose
``AutoTokenizer`` has no such class, so every D52 resume from a miLLM-written checkpoint died
at ``AttachAcquirer._tokenizer_loader`` with ``ValueError: Tokenizer class TokenizersBackend
does not exist`` - after ae617cd had correctly resolved the source to the checkpoint
directory.

**Why the earlier tests could not see it.** ae617cd's tests wrote their tokenizer with
miForge's OWN transformers, so the writer and the reader were the same version and agreed by
construction. Here the directory is written by ``save_pretrained`` and then **its format is
asserted** (``TokenizersBackend``), so a transformers downgrade turns this file red instead of
silently writing a 4.x directory the 4.x reader can load. A second case writes the
``tokenizer_config.json`` as a LITERAL - the keys a 5.x save emits - so the reader is pinned
against the format itself, independently of what the installed writer produces.

Both go through the REAL loader (``AttachAcquirer()._tokenizer_loader``) and the REAL
``AutoTokenizer.from_pretrained`` (observed and called through), offline, with a valid
``config.json`` beside the tokenizer as in a real checkpoint: from 5.x ``AutoTokenizer``
reads ``config.json`` first and refuses one that fails the config's own validation.

*Negative control (recorded 2026-10-18):* the same file under transformers 4.57.6 is red -
``save_pretrained`` writes ``PreTrainedTokenizerFast`` (format assertion) and the literal
case raises ``Tokenizer class TokenizersBackend does not exist``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.config import reset_settings
from services.checkpoints.paths import checkpoint_dir
from tests.support.fixtures_hotupdate import configure
from tests.support.tiny_hf_model import LLAMA_TINY, write_tiny_tokenizer

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

BASE = "LiquidAI/LFM2.5-1.2B-Instruct"
RUN = "f11w2-r4"
STEP = 100
QUALIFIED = f"{BASE}:{RUN}:step-{STEP}"

#: What transformers 5's ``save_pretrained`` writes for a tokenizers-backed tokenizer,
#: key for key (5.20.1, checked 2026-10-18). ``tokenizer_class`` is the field window 4 died on.
TRANSFORMERS5_TOKENIZER_CONFIG: dict[str, Any] = {
    "backend": "tokenizers",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "model_max_length": 1000000000000000019884624838656,
    "pad_token": "<pad>",
    "tokenizer_class": "TokenizersBackend",
    "unk_token": "<unk>",
}


def _write_checkpoint_config(directory: Path) -> None:
    """A real, valid ``config.json`` through the real writer, as every checkpoint has."""
    from transformers import AutoConfig

    config = AutoConfig.for_model(
        "llama", **{k: v for k, v in LLAMA_TINY.items() if k != "model_type"}
    )
    config.save_pretrained(directory)


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """miLLM's checkpoint of step 100 under a ``CHECKPOINT_READ_ROOTS`` entry."""
    primary = tmp_path / "checkpoints"
    millm = tmp_path / "millm-checkpoints"
    primary.mkdir()
    directory = checkpoint_dir(RUN, STEP, root=millm)
    directory.mkdir(parents=True)
    _write_checkpoint_config(directory)
    write_tiny_tokenizer(directory)
    configure(
        monkeypatch,
        CHECKPOINT_ROOT=str(primary),
        CHECKPOINT_READ_ROOTS=str(millm),
        HOTUPDATE_VERIFY_TOKENIZER="",
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    yield directory
    reset_settings()


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every source the REAL ``AutoTokenizer.from_pretrained`` receives (called through)."""
    import transformers

    real = transformers.AutoTokenizer.from_pretrained.__func__
    seen: list[str] = []

    def observed(cls: Any, source: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(str(source))
        return real(cls, source, *args, **kwargs)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(observed))
    return seen


def _loader(resident: str, directory: Path | None = None) -> Any:
    from training.hotupdate import attach as attach_module
    from training.hotupdate import settings as settings_module

    reset_settings()
    return attach_module.AttachAcquirer()._tokenizer_loader(
        resident, settings_module.load(), directory
    )


def _tokenizer_class(directory: Path) -> str:
    return str(json.loads((directory / "tokenizer_config.json").read_text())["tokenizer_class"])


class TestATransformers5TokenizerDirectoryLoads:
    def test_the_fixture_IS_in_the_transformers5_format(self, checkpoint: Path) -> None:
        """The premise, asserted: otherwise a writer on 4.x makes every case below a 4.x
        round trip, which is exactly how window 4's defect stayed invisible."""
        assert _tokenizer_class(checkpoint) == "TokenizersBackend"

    def test_a_resume_directory_written_by_5x_loads_and_encodes(
        self, checkpoint: Path, sources: list[str]
    ) -> None:
        """The window-4 path: a resume names the directory the identity check verified."""
        tokenizer = _loader(QUALIFIED, checkpoint)()
        assert sources == [str(checkpoint)], "the directory, once, never the qualified name"
        assert tokenizer("t5 t6")["input_ids"] == [5, 6]
        assert tokenizer.eos_token == "</s>"
        assert tokenizer.pad_token == "<pad>"

    def test_the_qualified_name_lookup_loads_the_same_directory(
        self, checkpoint: Path, sources: list[str]
    ) -> None:
        tokenizer = _loader(QUALIFIED)()
        assert sources == [str(checkpoint)]
        assert tokenizer("t9")["input_ids"] == [9]

    def test_a_LITERAL_transformers5_tokenizer_config_loads(
        self, checkpoint: Path, sources: list[str]
    ) -> None:
        """The reader pinned against the format, not against the installed writer."""
        (checkpoint / "tokenizer_config.json").write_text(
            json.dumps(TRANSFORMERS5_TOKENIZER_CONFIG, indent=2), encoding="utf-8"
        )
        special = checkpoint / "special_tokens_map.json"
        if special.exists():
            special.unlink()  # a 5.x save writes none; the literal stands alone
        tokenizer = _loader(QUALIFIED, checkpoint)()
        assert sources == [str(checkpoint)]
        assert tokenizer("t5 t6 t7")["input_ids"] == [5, 6, 7]
        assert tokenizer.eos_token == "</s>"
