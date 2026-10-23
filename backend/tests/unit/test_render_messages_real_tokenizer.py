"""``render_messages`` over a REAL tokenizer: the template refusal is reachable.

Found 2026-10-18 during the transformers 5 upgrade, and true of 4.x as well:
``render_messages`` refused ``chat_template_unavailable`` only when the tokenizer had no
``apply_chat_template`` attribute. Every real ``transformers`` tokenizer has that method (it
is on the base class), so for a real tokenizer with no template the refusal never fired and
the library's bare ``ValueError`` escaped instead. Doubles defined the method or not at will,
which is how a fixture agreed with the code by construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import Refusal
from tests.support.tiny_hf_model import write_tiny_tokenizer
from training.objectives.tokenize import render_messages

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

MESSAGES = [{"role": "user", "content": "t5"}, {"role": "assistant", "content": "t6"}]


@pytest.fixture
def tokenizer(tmp_path: Path):  # type: ignore[no-untyped-def]
    from transformers import AutoTokenizer

    write_tiny_tokenizer(tmp_path)
    return AutoTokenizer.from_pretrained(str(tmp_path), trust_remote_code=False)


def test_a_real_tokenizer_without_a_template_REFUSES_by_name(tokenizer) -> None:  # type: ignore[no-untyped-def]
    assert tokenizer.chat_template is None
    with pytest.raises(Refusal) as caught:
        render_messages(tokenizer, MESSAGES)
    assert caught.value.code == "chat_template_unavailable"


def test_a_real_tokenizer_with_a_template_renders_and_records_it(tokenizer) -> None:  # type: ignore[no-untyped-def]
    tokenizer.chat_template = (
        "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
    )
    text, identity = render_messages(tokenizer, MESSAGES)
    assert text == "<user>t5<assistant>t6"
    assert identity is not None and len(identity) == 64
