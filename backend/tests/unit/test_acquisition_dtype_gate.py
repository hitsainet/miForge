"""The dtype gate  (R-02.46, FR-001.5, E-1; the feature's highest-value unit file).

**Every weight fixture here is a real ``safetensors`` file with real dtypes**, written by
:func:`write_safetensors` and read back through the production header parser. A mocked
config object with a ``dtype`` attribute would prove the mock: the gate's whole job is to
read what a *file* claims about itself, and the header is the claim.

Each refusal asserts **the code and that the message names the reason**. A right code with a
vague message fails: ``model_quantized`` whose message does not say FP16/BF16 leaves the
operator with nothing to do, which is ADR-013's point.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from core.errors import Refusal
from training.acquisition.dtype_gate import (
    TRAINABLE_DTYPES,
    assert_trainable,
    collect_weight_dtypes,
    declared_config_dtype,
    quantization_block,
    safetensors_dtypes,
)

#: Bytes per element, so the fixture's `data_offsets` are real rather than plausible.
_WIDTH = {"F16": 2, "BF16": 2, "F32": 4, "I8": 1, "U8": 1, "I64": 8, "F8_E4M3": 1}


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> Path:
    """Write a real ``safetensors`` file: ``<u64 len><JSON header><payload>``.

    Round-tripped through the production reader by every test below, so a change to either
    the writer's or the reader's understanding of the format shows up here rather than in a
    GPU integration test months later.
    """
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        elements = 1
        for dim in shape:
            elements *= dim
        size = elements * _WIDTH[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)
    return path


@pytest.fixture
def fp16_shard(tmp_path: Path) -> Path:
    return write_safetensors(
        tmp_path / "model.safetensors",
        {"embed.weight": ("F16", [128, 16]), "layer.0.weight": ("F16", [16, 16])},
    )


class TestTheHeaderIsReadFromTheFileItself:
    def test_dtypes_come_back_from_a_real_file(self, fp16_shard: Path) -> None:
        assert safetensors_dtypes(fp16_shard) == {"F16"}

    def test_metadata_is_not_mistaken_for_a_tensor(self, tmp_path: Path) -> None:
        """``__metadata__`` is a header entry and not a tensor; counting it as one would
        produce an unclassifiable dtype set for an ordinary file."""
        path = tmp_path / "m.safetensors"
        write_safetensors(path, {"w": ("BF16", [4, 4])})
        blob = json.dumps(
            {
                "__metadata__": {"format": "pt"},
                "w": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]},
            }
        ).encode()
        path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 32)
        assert safetensors_dtypes(path) == {"BF16"}

    def test_an_unreadable_header_refuses_rather_than_returning_nothing(
        self, tmp_path: Path
    ) -> None:
        """An empty dtype set would sail through a gate that only looks for bad dtypes."""
        path = tmp_path / "truncated.safetensors"
        path.write_bytes(b"\x01\x02")
        with pytest.raises(Refusal) as caught:
            safetensors_dtypes(path)
        assert caught.value.code == "model_weights_unreadable"
        assert "dtypes are unknown" in caught.value.message

    def test_an_implausible_header_length_refuses_before_reading_it(self, tmp_path: Path) -> None:
        """The length comes from the file, so a hostile prefix must not drive a huge read."""
        path = tmp_path / "huge.safetensors"
        path.write_bytes(struct.pack("<Q", 2**60) + b"{}")
        with pytest.raises(Refusal) as caught:
            safetensors_dtypes(path)
        assert caught.value.code == "model_weights_unreadable"

    def test_the_union_across_shards_is_what_is_checked(self, tmp_path: Path) -> None:
        """A **union**: one int8 shard in an otherwise FP16 model must not be outvoted."""
        a = write_safetensors(tmp_path / "a.safetensors", {"w": ("F16", [8, 8])})
        b = write_safetensors(tmp_path / "b.safetensors", {"w": ("I8", [8, 8])})
        assert collect_weight_dtypes([a, b]) == {"F16", "I8"}


class TestWhatPasses:
    @pytest.mark.parametrize("dtype", sorted(TRAINABLE_DTYPES))
    def test_fp16_and_bf16_pass_and_the_accepted_dtype_is_returned(
        self, dtype: str, tmp_path: Path
    ) -> None:
        """Returned rather than discarded: the loader needs it, and re-deriving it would be
        a second decision about what this model is."""
        shard = write_safetensors(tmp_path / "m.safetensors", {"w": (dtype, [8, 8])})
        assert assert_trainable({}, collect_weight_dtypes([shard])) == dtype

    def test_integer_index_buffers_beside_fp16_parameters_pass(self, tmp_path: Path) -> None:
        """Position ids and rotary caches are legitimately I64 in an unquantized model."""
        shard = write_safetensors(
            tmp_path / "m.safetensors",
            {"w": ("BF16", [8, 8]), "position_ids": ("I64", [8])},
        )
        assert assert_trainable({}, collect_weight_dtypes([shard])) == "BF16"

    def test_a_config_declaring_the_same_dtype_agrees(self, tmp_path: Path) -> None:
        shard = write_safetensors(tmp_path / "m.safetensors", {"w": ("F16", [8, 8])})
        assert assert_trainable({"torch_dtype": "float16"}, collect_weight_dtypes([shard])) == "F16"


class TestWhatRefuses:
    """**Guessing is the failure mode, not strictness** (task 5.3)."""

    def test_a_quantization_block_refuses_even_when_the_dtypes_look_fine(
        self, fp16_shard: Path
    ) -> None:
        """A 4-bit model stores packed values in ordinary-looking tensors, so the dtype set
        can be perfectly clean while every tensor needs a dequantizer the gate lacks. The
        model's own statement about itself outranks the dtypes."""
        with pytest.raises(Refusal) as caught:
            assert_trainable(
                {"quantization_config": {"bits": 4}}, collect_weight_dtypes([fp16_shard])
            )
        assert caught.value.code == "model_quantized"
        assert "FP16 or BF16" in caught.value.message

    @pytest.mark.parametrize("key", ["quantization_config", "quantization", "quant_config"])
    def test_every_spelling_of_a_quantization_block_refuses(
        self, key: str, fp16_shard: Path
    ) -> None:
        with pytest.raises(Refusal) as caught:
            assert_trainable({key: {"bits": 8}}, collect_weight_dtypes([fp16_shard]))
        assert caught.value.code == "model_quantized"
        assert caught.value.details["quantization_key"] == key

    @pytest.mark.parametrize("dtype", ["I8", "U8", "F8_E4M3"])
    def test_a_quantized_payload_dtype_refuses_and_names_the_requirement(
        self, dtype: str, tmp_path: Path
    ) -> None:
        shard = write_safetensors(tmp_path / "m.safetensors", {"w": (dtype, [8, 8])})
        with pytest.raises(Refusal) as caught:
            assert_trainable({}, collect_weight_dtypes([shard]))
        assert caught.value.code == "model_quantized"
        assert "FP16 or BF16" in caught.value.message
        assert dtype in caught.value.details["disallowed_dtypes"]

    def test_an_empty_dtype_set_refuses_rather_than_passing(self) -> None:
        """**An empty set is not a pass.** It means nothing was identified, which is the
        case the gate most needs to refuse: downstream, an unchecked model is
        indistinguishable from a checked one."""
        with pytest.raises(Refusal) as caught:
            assert_trainable({}, set())
        assert caught.value.code == "model_dtype_unknown"
        assert "positively identify" in caught.value.message

    def test_a_mixed_fp16_and_bf16_set_refuses(self, tmp_path: Path) -> None:
        """The recorded optimizer identity names a single precision; a mixed parameter set
        would make that record a guess."""
        a = write_safetensors(tmp_path / "a.safetensors", {"w": ("F16", [8, 8])})
        b = write_safetensors(tmp_path / "b.safetensors", {"w": ("BF16", [8, 8])})
        with pytest.raises(Refusal) as caught:
            assert_trainable({}, collect_weight_dtypes([a, b]))
        assert caught.value.code == "model_dtype_mixed"
        assert "not a mixture" in caught.value.message

    def test_a_pure_float32_model_refuses(self, tmp_path: Path) -> None:
        """F32 is not quantized, but it is also not what R-02.46 says - and the gate does
        not decide such things for itself."""
        shard = write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", [8, 8])})
        with pytest.raises(Refusal) as caught:
            assert_trainable({}, collect_weight_dtypes([shard]))
        assert caught.value.code == "model_dtype_unknown"
        assert "FP16 or BF16" in caught.value.message

    def test_disagreement_between_config_and_weights_refuses_rather_than_voting(
        self, fp16_shard: Path
    ) -> None:
        """One of the two is wrong about this model and the gate cannot tell which, so
        neither answer may be used."""
        with pytest.raises(Refusal) as caught:
            assert_trainable({"torch_dtype": "bfloat16"}, collect_weight_dtypes([fp16_shard]))
        assert caught.value.code == "model_dtype_disagrees"
        assert caught.value.details == {
            "source": "the model",
            "config_dtype": "BF16",
            "weight_dtype": "F16",
        }

    def test_an_unrecognised_config_dtype_is_a_disagreement_not_a_pass(
        self, fp16_shard: Path
    ) -> None:
        """A ``torch_dtype`` this project does not know is unknown, and unknown refuses."""
        with pytest.raises(Refusal) as caught:
            assert_trainable({"torch_dtype": "int4_nf4"}, collect_weight_dtypes([fp16_shard]))
        assert caught.value.code == "model_dtype_disagrees"
        assert caught.value.details["config_dtype"] == "<unrecognised>"

    def test_the_refusal_names_the_source_it_was_given(self, fp16_shard: Path) -> None:
        """The operator needs to know *which* model was refused when several are in play."""
        with pytest.raises(Refusal) as caught:
            assert_trainable(
                {"quantization_config": {"bits": 4}},
                collect_weight_dtypes([fp16_shard]),
                source="org/repo@abc123",
            )
        assert "org/repo@abc123" in caught.value.message


class TestTheHelpersDoNotGuess:
    def test_an_absent_config_dtype_is_None_not_a_default(self) -> None:
        """``None`` means *the config did not say*, which is a different fact from *the
        config said something unknown* - and only one of them is a refusal."""
        assert declared_config_dtype({}) is None

    def test_a_non_string_config_dtype_is_unrecognised(self) -> None:
        assert declared_config_dtype({"torch_dtype": {"serialised": "torch.float16"}}) == (
            "<unrecognised>"
        )

    def test_a_falsy_quantization_block_is_not_a_block(self) -> None:
        """``quantization_config: null`` is how a config says *not quantized*; treating a
        present-but-empty key as a refusal would reject ordinary models."""
        assert quantization_block({"quantization_config": None}) is None
        assert quantization_block({"quantization_config": {}}) is None
