"""``attach.build_module`` builds a graph that RUNS, over the served tensors  (F11 window 2).

Found live 2026-10-17: miLLM exports parameters, not buffers (its FR-31.1.8), and
``build_module`` built the graph wholly on the meta device, so the rotary ``inv_freq`` stayed
meta and the first forward failed "Cannot copy out of meta tensor". Every caller in this suite
stubbed ``build_module``; this drives the real one on the CPU with a tiny real Llama (which has
a non-persistent rotary buffer, like LFM2) and the "served" model's own parameter tensors.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from training.hotupdate import attach, rebuild  # noqa: E402


@pytest.fixture
def served() -> tuple:
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    tensors = dict(model.named_parameters(remove_duplicate=False))
    base = rebuild.RebuiltBase(tensors=tensors, device_uuid=None, read_only=False)
    response = {"config": json.loads(config.to_json_string(use_diff=False))}
    return model, base, response


def test_no_buffer_is_left_without_data(served) -> None:  # type: ignore[no-untyped-def]
    _, base, response = served
    module = attach.build_module(response, base)
    assert [n for n, b in module.named_buffers() if b.is_meta] == []


def test_the_parameters_are_the_served_tensors_not_copies(served) -> None:  # type: ignore[no-untyped-def]
    _, base, response = served
    module = attach.build_module(response, base)
    for name, param in module.named_parameters(remove_duplicate=False):
        assert param.data_ptr() == base.tensors[name].data_ptr(), name


def test_its_forward_equals_the_served_models(served) -> None:  # type: ignore[no-untyped-def]
    model, base, response = served
    module = attach.build_module(response, base).eval()
    ids = torch.tensor([[1, 5, 9, 3]])
    with torch.no_grad():
        assert torch.equal(module(input_ids=ids).logits, model(input_ids=ids).logits)
