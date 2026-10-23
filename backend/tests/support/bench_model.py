"""A **benchmark-sized** model fixture, as distinct from ``tiny_hf_model``'s correctness one.

**Why a second config rather than reusing ``LLAMA_TINY``.** ``LLAMA_TINY`` is 2 layers of 64
hidden units: the right size for asserting that a path *works*, and the wrong size for
asserting that a path's *overhead* is within a budget. A measured lesson from 2026-10-16
says why, and it is the reason this module exists:

> the existing 5 % budget assertions fail at **25-28 %** because the fixture's whole step
> (~3.2 ms) is the same order as one ``fsync`` (3.09 ms). At real model scales the same
> overhead is **0.70-2.84 %**.

So the fixture was the finding, not the code. A ratio whose denominator is a toy is a
measurement of the toy. ``BENCH_LLAMA`` is sized so **one step is >= 100 ms**, which puts
the denominator two orders of magnitude above a single ``fsync`` and makes the ratio a
statement about miForge's overhead rather than about the fixture's smallness.

**The shape was measured, not assumed, and the first guess was wrong by 2.4x.** 12 layers /
768 hidden / sequence 512 / batch **4** was expected to land near 115 ms and measured
**42.5 ms** on the 3090 - because this fixture instantiates in **bfloat16**, which the
3090's tensor cores run far faster than the fp32 the estimate came from. The floor
assertion in ``test_boundary_loop_overhead.py`` is what caught it, on its first run, before
any figure was reported. ``BENCH_BATCH`` is therefore **16**: batch 4 measured 42.5 ms and
batch 12 measured 97.0 ms (sub-linear, because batch 4 is partly launch-bound), so 16, which
measures ~128 ms. *A fixture size taken from someone else's estimate is the same class of
error as a figure taken from an x16 link: plausible, specific and wrong.*

**And a ratio alone is still not enough.** A fixed cost hides behind a large denominator:
make the model big enough and *any* constant overhead passes a 5 % test. Every budget
assertion built on this fixture therefore also asserts an **absolute** millisecond bound, so
the two together pin both failure directions. That pairing is stated here because it is a
property of how this fixture must be used, not of any one benchmark file.

## The vocabulary is 512, deliberately, and it is load-bearing

``vocab_size`` matches :data:`tests.support.tiny_hf_model.LLAMA_TINY` so the **same**
``write_tiny_tokenizer`` serves both. A 32k-vocab tokenizer against a 512-row embedding
table raises ``IndexError: index out of range in self`` the moment anything generates - the
constraint that module's docstring records. One tokenizer writer, two model sizes.

## Topology, because a number without it is not a measurement

Every figure taken with this fixture is taken on **one RTX 3090 whose PCIe link negotiates
x4** (the slot reports ``width.max`` 16 and ``width.current`` 4, confirmed live on
``mcs-lnxhost02``). ADR-024: a host-to-device figure lifted from an x16 number is wrong by
3-6x. Nothing here measures host-to-device bandwidth, which is itself the reason to say so -
the fixture loads once and then stays resident, so PCIe is outside every loop below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.support.tiny_hf_model import write_tiny_tokenizer

#: A model whose forward+backward is >= 100 ms on an RTX 3090 at :data:`BENCH_BATCH` x
#: :data:`BENCH_SEQ`. About 86 M parameters. ``vocab_size`` is pinned to ``LLAMA_TINY``'s so
#: one tokenizer writer serves both fixtures - see the module docstring.
BENCH_LLAMA: dict[str, Any] = {
    "model_type": "llama",
    "hidden_size": 768,
    "intermediate_size": 2048,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_key_value_heads": 12,
    "vocab_size": 512,
    "max_position_embeddings": 1024,
}

#: The shape that makes a step cross 100 ms. Named constants rather than literals at the
#: call sites, because **every figure names its sample** (ADR-012) and the sample is these.
#: ``BENCH_BATCH`` is 16 because 4 measured 42.5 ms and 12 measured 97.0 ms - see the
#: module docstring.
BENCH_BATCH = 16
BENCH_SEQ = 512


def bench_model(*, dtype: str = "bfloat16") -> Any:
    """Instantiate :data:`BENCH_LLAMA` in memory. **Leaves it on the CPU.**

    Placement is the caller's, and deliberately so: ADR-008 pins a card **by UUID** resolved
    from the live driver, never an index, and a fixture that called ``.to("cuda")`` would be
    choosing a device by enumeration order. The caller holds the lease, so the caller places.

    Args:
        dtype: The torch dtype name to instantiate in. ``bfloat16`` matches what the
            checkpoint writer emits, so a benchmark is not secretly measuring fp32.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "llama", **{k: v for k, v in BENCH_LLAMA.items() if k != "model_type"}
    )
    return AutoModelForCausalLM.from_config(config).to(getattr(torch, dtype))


def write_bench_model(directory: Path, *, dtype: str = "bfloat16") -> Path:
    """Save :data:`BENCH_LLAMA` as a standard Hugging Face directory, tokenizer at the ROOT.

    Root placement is R-02.51's promise and what
    ``training.acquisition.local_dir.require_tokenizer_at_root`` refuses a violation of by
    name. It is the real acquirer that reads this, so the layout is not a detail.
    """
    directory.mkdir(parents=True, exist_ok=True)
    bench_model(dtype=dtype).save_pretrained(directory, safe_serialization=True)
    write_tiny_tokenizer(directory)
    return directory


def bench_prompts(count: int) -> list[str]:
    """``count`` distinct prompts the 512-token word-level tokenizer can actually encode.

    The tiny tokenizer's vocabulary is ``<unk> <s> </s> <pad>`` plus ``t4``..``t511``, split
    on whitespace. So a prompt of English words tokenizes to a run of ``<unk>`` - which
    *works* and silently measures a degenerate input. Using real vocabulary items keeps the
    sample honest and is why this helper exists rather than a literal list at each call site.
    """
    return [
        " ".join(f"t{4 + ((index * 7 + offset) % 500)}" for offset in range(8))
        for index in range(count)
    ]
