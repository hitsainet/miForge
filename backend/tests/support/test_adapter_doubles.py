"""F8's test doubles, **and the tests that prove each double honours the real contract**.

A double that quietly diverged from the thing it stands for is how a suite stays green over
a real bug, so the doubles are not simply declared here - each is checked against the
property the production code depends on. ``FakePeftModel`` writing no ``adapter_config.json``
would make ``test_adapter_save.py``'s agreement assertion vacuous; a stub whose named
parameters all look like adapter tensors would make
``test_adapter_requires_grad.py``'s refusal unreachable.

It lives under ``backend/tests/support/`` beside F5's ``checkpoints.py`` and
``checkpoint_faults.py``, which F8 reuses rather than reimplementing.

**Why doubles at all, and what is NOT being faked.** ``torch``, ``peft``, ``transformers``
and ``safetensors`` are **not installed in this environment** - which ADR-023 makes a
supported state for everything except a training step. So the atomicity of the write, the
streamed digest, the PEFT/manifest agreement check, schema validation, the name and path
rendering, every refusal, and the whole budget arithmetic run against these doubles and are
genuinely exercised. What is *not* exercised here is anything that needs a real tensor: the
adapter build, the base-unchanged hash over real parameters, the merge arithmetic and the
``peft`` load. Those are ``gpu``-marked or skipped by name - never silently passed - and
they are stated as an operator-run gate in the feature report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from training.acquisition.base import AcquisitionMode, AcquisitionRecord, SourceKind
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters.config import AdapterConfigIn


@dataclass
class FakePeftModel:
    """Writes what ``peft``'s ``save_pretrained`` writes: the tensors **and** the config.

    ``config_overrides`` exists so a test can make the two files disagree without editing
    production code - which is how :func:`training.runtypes.adapters.save.assert_config_agrees`
    is tested from outside rather than by trusting it.
    """

    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj",)
    payload: bytes = b"lora-tensors-v1"
    #: Keys written into ``adapter_config.json`` on top of the derived ones.
    config_overrides: dict[str, Any] = field(default_factory=dict)
    #: Set to skip ``adapter_config.json`` entirely - the FPRD D5 defect, as a fixture.
    write_peft_config: bool = True

    def save_pretrained(self, directory: Any) -> None:
        target = Path(directory)
        (target / adapter_paths.WEIGHTS_NAME).write_bytes(self.payload)
        if not self.write_peft_config:
            return
        document = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": "none",
            **self.config_overrides,
        }
        (target / adapter_paths.PEFT_CONFIG_NAME).write_text(json.dumps(document), encoding="utf-8")


class FakeParameter(bytes):
    """A tensor stand-in whose bytes are readable through ``memoryview``.

    ``requires_grad`` is settable so M9's mutation - setting it on a base parameter - has a
    double to be expressed against.

    **A `bytes` subclass, not a dataclass with `__buffer__`.** The first version defined
    ``__buffer__``, which is PEP 688 and therefore **Python 3.12 or later**: on 3.11
    ``memoryview(p)`` does not consult it, so none of ``_parameter_bytes``' three routes
    worked and the digest refused ``base_digest_unreadable``. It passed on this workstation
    (3.12) and failed in CI, whose PostgreSQL job runs the project's declared floor of 3.11
    — a double that works only above the supported floor. Subclassing ``bytes`` gets the
    real buffer protocol on every supported version, and keeps ``requires_grad`` settable
    because a Python-level subclass still carries a ``__dict__``.
    """

    requires_grad: bool

    def __new__(
        cls, data: bytes = b"\x01\x02\x03\x04", requires_grad: bool = False
    ) -> FakeParameter:
        parameter = super().__new__(cls, data)
        parameter.requires_grad = requires_grad
        return parameter

    @property
    def data(self) -> bytes:
        """The payload, under the name the dataclass version exposed."""
        return bytes(self)

    def __repr__(self) -> str:
        return f"FakeParameter({bytes(self)!r}, requires_grad={self.requires_grad})"


@dataclass
class OpaqueParameter:
    """A parameter whose bytes cannot be read by any route the digest knows.

    The vacuity guard's fixture: without the refusal, this would contribute nothing to the
    hash and two different bases would compare equal.
    """

    requires_grad: bool = False


@dataclass
class FakeModule:
    """Something with ``named_parameters``. Base tensors and adapter tensors, by name."""

    parameters_by_name: dict[str, Any] = field(default_factory=dict)

    def named_parameters(self) -> list[tuple[str, Any]]:
        return list(self.parameters_by_name.items())

    def parameters(self) -> list[Any]:
        return list(self.parameters_by_name.values())


def base_module(payload: bytes = b"\x01\x02\x03\x04") -> FakeModule:
    """A module with two base parameters and one adapter pair."""
    return FakeModule(
        {
            "base_model.model.layers.0.q_proj.weight": FakeParameter(payload),
            "base_model.model.layers.0.mlp.down_proj.weight": FakeParameter(payload[::-1]),
            "base_model.model.layers.0.q_proj.lora_A.weight": FakeParameter(
                b"\xaa", requires_grad=True
            ),
            "base_model.model.layers.0.q_proj.lora_B.weight": FakeParameter(
                b"\xbb", requires_grad=True
            ),
        }
    )


def adapter_config(**overrides: Any) -> AdapterConfigIn:
    """A valid adapter configuration, with overrides for the refusal cases."""
    values: dict[str, Any] = {
        "rank": 4,
        "alpha": 8.0,
        "target_modules": ("q_proj",),
        "dropout": 0.0,
    }
    values.update(overrides)
    return AdapterConfigIn.model_validate(values)


def optimizer_block(state: str = "fresh") -> dict[str, Any]:
    """F1's optimizer identity block - the same shape in both manifests."""
    return {"name": "adamw-8bit", "state": state, "config_hash": "a" * 64}


def hub_record(repo: str = "LiquidAI/LFM2.5-1.2B-Instruct") -> AcquisitionRecord:
    """A load from a Hub repo: ``base.name`` is **the repo id**, ``manifest_sha256`` null."""
    return AcquisitionRecord(
        mode=AcquisitionMode.LOAD,
        source_kind=SourceKind.HUB,
        source_repo=repo,
        source_revision_requested="main",
        source_revision_resolved="c" * 40,
        starting_weights_sha256="b" * 64,
        base_model=repo,
        parent=None,
        param_count=1_200_000_000,
    )


def checkpoint_record(
    checkpoint_name: str = "demo-base:run-1:step-40", root_model: str = "demo-base"
) -> AcquisitionRecord:
    """A load of a miForge checkpoint: ``base.name`` is that checkpoint's manifest ``name``.

    The default is in F5's **real** rendering - ``<base_model>:<run_id>:step-<N>``, with the
    ``step-`` prefix - because ``parse_qualified_name`` returns ``None`` for anything else
    and the merge's base resolution would then refuse for the wrong reason. A fixture in a
    plausible-looking but wrong format is how a refusal's test passes while testing a
    different refusal; this one was caught that way."""
    return AcquisitionRecord(
        mode=AcquisitionMode.LOAD,
        source_kind=SourceKind.LOCAL_DIR,
        source_path="/data/checkpoints/run-1/step-000040",
        starting_weights_sha256="b" * 64,
        base_model=root_model,
        parent=checkpoint_name,
        param_count=1_200_000_000,
    )


def creation_record(
    config_identity: str = "config:llama@" + "d" * 64 + "+seed=7", seed: int | None = 7
) -> AcquisitionRecord:
    """A creation: ``base_model`` is the config identity and ``parent`` is **null**."""
    return AcquisitionRecord(
        mode=AcquisitionMode.CREATE,
        source_kind=SourceKind.CONFIG,
        config_identity="d" * 64,
        starting_weights_sha256="e" * 64,
        seed=seed,
        base_model=config_identity,
        parent=None,
        param_count=512_000,
    )


TINY_CONFIG: dict[str, Any] = {
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 512,
    "tie_word_embeddings": True,
}


# ======================================================================================
# The doubles are checked against the contracts the production code depends on
# ======================================================================================


class TestFakePeftModelHonoursThePeftLayout:
    def test_it_writes_the_tensors_and_the_peft_config(self, tmp_path: Path) -> None:
        """Both files, because the agreement assertion is vacuous without the second one."""
        FakePeftModel().save_pretrained(tmp_path)
        assert (tmp_path / adapter_paths.WEIGHTS_NAME).is_file()
        assert (tmp_path / adapter_paths.PEFT_CONFIG_NAME).is_file()

    def test_the_peft_config_uses_PEFTS_spellings_not_miforges(self, tmp_path: Path) -> None:
        """``r``/``lora_alpha``, not ``rank``/``alpha``.

        The whole reason a cross-check is needed is that the two files name the same two
        numbers differently; a double using miForge's spellings would make the check pass
        by accident.
        """
        FakePeftModel(rank=16, alpha=32.0).save_pretrained(tmp_path)
        document = json.loads((tmp_path / adapter_paths.PEFT_CONFIG_NAME).read_text())
        assert document["r"] == 16
        assert document["lora_alpha"] == 32.0
        assert "rank" not in document and "alpha" not in document

    def test_different_payloads_hash_differently(self, tmp_path: Path) -> None:
        """Otherwise the digest assertions would hold against a writer that hashed nothing."""
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        FakePeftModel(payload=b"one").save_pretrained(first)
        FakePeftModel(payload=b"two").save_pretrained(second)
        assert (first / adapter_paths.WEIGHTS_NAME).read_bytes() != (
            second / adapter_paths.WEIGHTS_NAME
        ).read_bytes()


class TestTheModuleDoubleDistinguishesBaseFromAdapter:
    def test_base_and_adapter_parameters_are_distinguishable_by_name(self) -> None:
        names = dict(base_module().named_parameters())
        assert any("lora_" in name for name in names)
        assert any("lora_" not in name for name in names)

    def test_adapter_parameters_are_the_trainable_ones(self) -> None:
        """A double where everything was trainable would make M9 unexpressible."""
        for name, parameter in base_module().named_parameters():
            assert parameter.requires_grad == ("lora_" in name), name

    def test_a_base_parameters_bytes_are_readable(self) -> None:
        assert bytes(memoryview(FakeParameter(b"\x07"))) == b"\x07"

    def test_an_opaque_parameters_bytes_are_NOT_readable(self) -> None:
        """The fixture for the vacuity guard. If this ever became readable the guard's test
        would pass for the wrong reason."""
        with pytest.raises(TypeError):
            memoryview(OpaqueParameter())  # type: ignore[arg-type]


class TestTheAcquisitionRecordDoublesCoverBothBaseCases:
    def test_a_hub_record_names_a_repo_id_and_no_parent(self) -> None:
        record = hub_record()
        assert "/" in record.base_model, "a Hub repo id contains a slash - that is the point"
        assert record.parent is None

    def test_a_checkpoint_record_names_a_parent_in_F5s_RENDERING(self) -> None:
        """Checked against F5's own renderer, not against a string that looks right."""
        from services.checkpoints.paths import parse_qualified_name

        assert parse_qualified_name(str(checkpoint_record().parent)) is not None
        assert checkpoint_record().parent == "demo-base:run-1:step-40"

    def test_a_creation_record_has_a_null_parent_and_a_config_identity(self) -> None:
        record = creation_record()
        assert record.parent is None
        assert record.base_model.startswith("config:")
        assert "+seed=" in record.base_model, (
            "F1 folds the seed into base_model, and writes '+seed=none' explicitly rather "
            "than omitting it, so two creations from one config with different seeds have "
            "different base_model values"
        )
