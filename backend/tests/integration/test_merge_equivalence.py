"""The merge arithmetic: merged logits match base-plus-adapter  (FR-008.41).

**The only check that catches a scale applied twice, or not at all.** Atomicity tests do
not: a merge with the wrong scale writes a directory that is absent-or-complete, validates,
loads, and answers differently. R-4 in the TDD's risk table - *"high, and silent."*

## Why ``peft``'s own merge, and not the hand-written formula

F8's TID sketches ``W + (B @ A) * (alpha / rank)`` per target module. The TID's own warning
is the argument against its sketch: applying the scale twice or not at all is the silent
failure, and a hand-written formula puts that one line in code nothing else exercises.
``peft``'s ``merge_and_unload`` is **the same code path a server applies when it serves the
adapter**, so base-plus-adapter and the merged checkpoint agree *structurally* rather than
because a hand-written scale happened to match. The scaling rule is validated before the
merge is reached - ``load_manifest`` refuses an unrecognised one - so ``peft``'s
alpha-over-rank is the rule the manifest **declares**, not an assumption.

This still deserves an equivalence test, because the question it answers is not "is the
formula right" but "did the merge happen at all, and to the right adapter."

## CPU, not GPU - and skipped by name when the ML stack is absent

A tiny two-layer model on CPU, deliberately: three of this feature's hardest correctness
claims run without a card. ``torch`` and ``peft`` are **not installed in the environment
this feature was built in**, so these cases are reported as a **named skip** and have **not
been run**. A harness that returned success without merging anything would make the merge
read as verified while nothing had been computed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.conformance import ml_runtime_available
from tests.support.test_adapter_doubles import TINY_CONFIG, adapter_config, optimizer_block
from training.runtypes.adapters import manifest as adapter_manifest
from training.runtypes.adapters import paths as adapter_paths

_available, _why = ml_runtime_available()


def _ready() -> tuple[bool, str]:
    if not _available:
        return False, _why
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        return False, f"peft is not installed in this environment ({exc})"
    return True, ""


_ok, _skip_why = _ready()
pytestmark = pytest.mark.skipif(not _ok, reason=_skip_why or "ML runtime unavailable")

PROMPT_IDS = [1, 2, 3, 4, 5]


@pytest.fixture
def rig(tmp_path: Path) -> dict[str, Any]:
    """A tiny base on disk, and an adapter over it whose B matrix is **not zero**.

    ``peft`` initialises ``lora_B`` to zeros, so a freshly built adapter is the identity
    and *every* merge would look correct. Perturbing B is what makes the comparison
    capable of failing - without it this whole file would be construction-green.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from training.runtypes.adapters import save as adapter_save
    from training.runtypes.adapters.build import build

    base_dir = tmp_path / "base"
    config = AutoConfig.for_model(
        TINY_CONFIG["model_type"], **{k: v for k, v in TINY_CONFIG.items() if k != "model_type"}
    )
    torch.manual_seed(7)
    base = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    base.save_pretrained(base_dir, safe_serialization=True)

    wrapped, validated = build(
        AutoModelForCausalLM.from_pretrained(str(base_dir), trust_remote_code=False),
        adapter_config(rank=4, alpha=8.0, target_modules=("q_proj", "v_proj")),
    )
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if "lora_B" in name:
                parameter.add_(torch.randn_like(parameter) * 0.05)

    doc = adapter_manifest.build(
        run_id="run-merge",
        step=10,
        base={"name": str(base_dir), "manifest_sha256": None},
        adapter=validated,
        optimizer=optimizer_block(),
    )
    directory = adapter_paths.adapter_dir("run-merge", 10, root=tmp_path / "artifacts")
    adapter_save.save(wrapped, doc, directory)
    return {"base_dir": base_dir, "adapter_dir": directory, "scale": validated.scale}


def _logits(model: Any) -> Any:
    import torch

    model.eval()
    with torch.no_grad():
        return model(input_ids=torch.tensor([PROMPT_IDS])).logits


class TestTheMergedModelMatchesBasePlusAdapter:
    def test_the_logits_agree_within_tolerance(self, rig: dict[str, Any]) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from training.runtypes.adapters.merge import apply_adapter

        applied = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(str(rig["base_dir"]), trust_remote_code=False),
            str(rig["adapter_dir"]),
        )
        merged = apply_adapter(
            AutoModelForCausalLM.from_pretrained(str(rig["base_dir"]), trust_remote_code=False),
            rig["adapter_dir"],
        )
        assert torch.allclose(_logits(applied), _logits(merged), atol=1e-4), (
            "the merged model answers differently from base-plus-adapter: the scale was "
            "applied twice, not at all, or to the wrong modules"
        )

    def test_the_merge_actually_CHANGED_the_base(self, rig: dict[str, Any]) -> None:
        """The test above would pass trivially if the merge were a no-op. This is what
        makes the comparison mean something."""
        import torch
        from transformers import AutoModelForCausalLM

        from training.runtypes.adapters.merge import apply_adapter

        bare = _logits(
            AutoModelForCausalLM.from_pretrained(str(rig["base_dir"]), trust_remote_code=False)
        )
        merged = _logits(
            apply_adapter(
                AutoModelForCausalLM.from_pretrained(str(rig["base_dir"]), trust_remote_code=False),
                rig["adapter_dir"],
            )
        )
        assert not torch.allclose(
            bare, merged, atol=1e-4
        ), "the merge was a no-op, so the equivalence test above proves nothing"

    def test_a_WRONG_scale_would_NOT_agree(self, rig: dict[str, Any]) -> None:
        """The negative control for the equivalence check itself: the same merge with the
        scale doubled must disagree. Without this, a tolerance too loose to notice would
        make the whole file green over the defect it exists to catch."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        applied = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(str(rig["base_dir"]), trust_remote_code=False),
            str(rig["adapter_dir"]),
        )
        doubled = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(str(rig["base_dir"]), trust_remote_code=False),
            str(rig["adapter_dir"]),
        )
        with torch.no_grad():
            for name, parameter in doubled.named_parameters():
                if "lora_B" in name:
                    parameter.mul_(2.0)
        assert not torch.allclose(_logits(applied), _logits(doubled.merge_and_unload()), atol=1e-4)

    def test_the_adapter_directory_SURVIVES_its_own_merge(self, rig: dict[str, Any]) -> None:
        """FR-008.41. A merged adapter cannot be taken off, which is exactly why the
        separate artifact stays."""
        from transformers import AutoModelForCausalLM

        from training.runtypes.adapters.merge import apply_adapter

        apply_adapter(
            AutoModelForCausalLM.from_pretrained(str(rig["base_dir"]), trust_remote_code=False),
            rig["adapter_dir"],
        )
        for name in (
            adapter_paths.WEIGHTS_NAME,
            adapter_paths.PEFT_CONFIG_NAME,
            adapter_paths.MANIFEST_NAME,
        ):
            assert (rig["adapter_dir"] / name).is_file(), name

    def test_the_scale_is_alpha_over_rank(self, rig: dict[str, Any]) -> None:
        assert rig["scale"] == 8.0 / 4
