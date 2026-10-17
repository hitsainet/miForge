"""The dtype gate: FP16 or BF16, positively identified, or refuse  (R-02.46; FR-001.5).

**The highest-value twenty lines in this feature**, and the one the mutation sweep hits
first: flipping the refusal to an acceptance must turn the suite red.

A model's quantization can be declared three ways - a ``quantization_config`` block in
``config.json``, the config's own ``torch_dtype``, or the actual dtypes in the weight
files. A detector that reads one of the three and proceeds trains against weights it does
not understand, and nothing downstream can tell. So **all three are read, disagreement is
a refusal rather than a vote, and anything not positively identified as FP16 or BF16 is
refused** - including an unrecognised quantization scheme, a mixed set and an empty set
(ADR-018: refuse with a reason rather than guess).

## Metadata only, and why that is a design constraint rather than an optimisation

The gate runs in **preflight** - before the GPU lease, before the budget guard permits
anything, before a byte of VRAM. So it must not materialise tensors. ``safetensors`` files
carry a JSON header giving every tensor's dtype, and that header is read here with the
standard library: eight bytes of length, then the JSON. No ``safetensors`` import, no
``torch`` import, hence no card and no wheels needed - which is what makes every refusal
path in this module testable on a laptop (ADR-023).

**Dtypes are compared as the file's own strings**, not as ``torch.dtype`` objects. The file
is the thing making the claim; translating through torch's naming adds a step that can only
lose information, and it would drag torch into preflight for nothing.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

from core.errors import refuse

#: The two dtypes R-02.46 permits, spelled as ``safetensors`` spells them. Nothing else
#: passes - not ``F32`` either: a float32 checkpoint is not quantized, but it is also not
#: what R-02.46 says, and the gate's whole value is that it does not decide such things
#: for itself. An operator who wants it says so by changing the requirement.
TRAINABLE_DTYPES: frozenset[str] = frozenset({"F16", "BF16"})

#: How ``config.json``'s ``torch_dtype`` spells the same two, plus the aliases seen in the
#: wild. Mapped rather than string-matched so a value this table does not know refuses.
_CONFIG_DTYPE_TO_SAFETENSORS: Mapping[str, str] = {
    "float16": "F16",
    "fp16": "F16",
    "half": "F16",
    "bfloat16": "BF16",
    "bf16": "BF16",
    "float32": "F32",
    "fp32": "F32",
    "float": "F32",
    "float64": "F64",
    "int8": "I8",
    "uint8": "U8",
    "float8_e4m3fn": "F8_E4M3",
    "float8_e5m2": "F8_E5M2",
}

#: Keys whose presence in ``config.json`` is the model's **own statement** that it is
#: quantized. Presence is a refusal even when every dtype looks fine, because the block
#: describes how the tensors are to be interpreted and the gate cannot interpret them.
QUANTIZATION_KEYS: tuple[str, ...] = ("quantization_config", "quantization", "quant_config")

#: ``safetensors``' header is at most this large. A bound is needed because the length is
#: read from the file itself: a corrupt or hostile first eight bytes would otherwise ask
#: for a multi-gigabyte read before anything had been validated.
MAX_HEADER_BYTES = 100 * 1024 * 1024


def safetensors_header(path: Path) -> dict[str, Any]:
    """One ``.safetensors`` file's header, read with the standard library.

    The format is ``<u64 little-endian header_len><header_len bytes of JSON>``, and the
    JSON maps each tensor name to ``{"dtype", "shape", "data_offsets"}`` with an optional
    ``__metadata__`` entry that is not a tensor. **This is the only place the format is
    parsed** - the dtype gate and the parameter count both read it here, because two
    parsers of one binary layout are two chances to disagree about what a model is.

    Raises:
        Refusal: ``model_weights_unreadable`` when the header cannot be parsed. Refused
            rather than skipped: a weight file whose header is unreadable is a file whose
            dtypes are unknown, and an unknown dtype must not pass the gate.
    """
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError("file is shorter than a safetensors header length prefix")
            (length,) = struct.unpack("<Q", raw_length)
            if length == 0 or length > MAX_HEADER_BYTES:
                raise ValueError(f"implausible header length {length}")
            header = json.loads(handle.read(length).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise refuse(
            "model_weights_unreadable",
            f"Could not read the tensor header of {path.name}, so the weights' dtypes are "
            f"unknown. Refusing rather than proceeding: an unknown dtype is exactly what "
            f"this check exists to stop. ({exc})",
            file=str(path),
        ) from exc

    if not isinstance(header, dict):
        raise refuse(
            "model_weights_unreadable",
            f"{path.name}'s tensor header is not an object, so no dtype can be read from it.",
            file=str(path),
        )
    return header


def safetensors_dtypes(path: Path) -> set[str]:
    """Every tensor dtype declared in one ``.safetensors`` file's header."""
    dtypes: set[str] = set()
    for name, entry in safetensors_header(path).items():
        if name == "__metadata__":
            continue
        if isinstance(entry, dict) and isinstance(entry.get("dtype"), str):
            dtypes.add(entry["dtype"])
    return dtypes


def collect_weight_dtypes(weight_files: Iterable[Path]) -> set[str]:
    """The union of every declared dtype across a model's weight shards.

    A **union**, so a single int8 shard in an otherwise FP16 model refuses. Taking the
    first shard's answer, or a majority, is how a partially-quantized model passes.
    """
    dtypes: set[str] = set()
    for path in weight_files:
        dtypes |= safetensors_dtypes(path)
    return dtypes


def declared_config_dtype(config: Mapping[str, Any]) -> str | None:
    """``config.json``'s ``torch_dtype``, normalised to the file's spelling, or ``None``.

    ``None`` means *the config did not say*, which is not the same as *the config said
    something unknown* - the latter is a refusal and is handled in :func:`assert_trainable`.
    An absent declaration is common and acceptable, because the weight files themselves are
    the stronger signal.
    """
    value = config.get("torch_dtype")
    if value is None:
        return None
    if not isinstance(value, str):
        # e.g. a serialised torch object. Unknown, and unknown refuses.
        return "<unrecognised>"
    return _CONFIG_DTYPE_TO_SAFETENSORS.get(value.lower(), "<unrecognised>")


def quantization_block(config: Mapping[str, Any]) -> str | None:
    """Which quantization key the config carries, if any."""
    for key in QUANTIZATION_KEYS:
        if config.get(key):
            return key
    return None


def assert_trainable(
    config: Mapping[str, Any],
    weight_dtypes: Collection[str],
    *,
    source: str = "the model",
) -> str:
    """Refuse unless the weights are positively identified as FP16 or BF16 (R-02.46).

    Args:
        config: the parsed ``config.json``.
        weight_dtypes: every dtype declared by the weight files, as the files spell them.
        source: what to name in the refusal - a path or a repo id.

    Returns:
        The single accepted dtype, ``"F16"`` or ``"BF16"``. Returned rather than discarded
        because the loader needs it and re-deriving it would be a second decision.

    Raises:
        Refusal: ``model_quantized`` for a quantization block or a non-trainable dtype;
            ``model_dtype_unknown`` when nothing could be identified at all. **The message
            states the requirement** - *the model must be FP16 or BF16* - because a right
            code with a vague message leaves the operator with nothing to do.
    """
    block = quantization_block(config)
    if block is not None:
        # The model's own statement about itself outranks the dtypes. A 4-bit model stores
        # packed values in a uint8 or int32 tensor, so the dtype set can look perfectly
        # ordinary while every tensor needs a dequantizer the gate does not have.
        raise refuse(
            "model_quantized",
            f"{source} declares {block!r} in its config, so its weights are quantized. "
            f"miForge trains full-parameter updates and the model must be FP16 or BF16. "
            f"Use an unquantized copy of these weights.",
            source=source,
            quantization_key=block,
        )

    observed = set(weight_dtypes)
    if not observed:
        # An empty set is not a pass. It means nothing was identified, which is the case
        # this gate most needs to refuse: a model whose dtypes could not be read is
        # indistinguishable, to everything downstream, from one that was checked.
        raise refuse(
            "model_dtype_unknown",
            f"No tensor dtype could be identified for {source}, so there is nothing to "
            f"check against. miForge trains FP16 or BF16 weights and refuses anything it "
            f"cannot positively identify.",
            source=source,
        )

    unexpected = observed - TRAINABLE_DTYPES
    # Shapes, position ids and rotary caches are legitimately integer or float32 in an
    # otherwise FP16 model, but the gate does not know which tensor is which from the
    # header alone - so rather than carve out exceptions it requires that **some** tensor
    # be FP16/BF16 and that no tensor use a quantized or 8-bit float type. The carve-out
    # list is deliberately narrow and explicit; widening it is a decision, not an oversight.
    disallowed = {d for d in unexpected if not _is_benign_auxiliary_dtype(d)}
    if disallowed:
        raise refuse(
            "model_quantized",
            f"{source} carries tensors of dtype {sorted(disallowed)}. miForge trains "
            f"full-parameter updates and the model must be FP16 or BF16. Refusing rather "
            f"than guessing how to interpret them.",
            source=source,
            observed_dtypes=sorted(observed),
            disallowed_dtypes=sorted(disallowed),
        )

    accepted = observed & TRAINABLE_DTYPES
    if len(accepted) != 1:
        # Both F16 and BF16 present, or neither. Neither is a refusal above; both is one
        # here, because the master-weight path and the optimizer identity both record a
        # single precision and a mixed parameter set would make that record a guess.
        raise refuse(
            "model_dtype_mixed" if accepted else "model_dtype_unknown",
            f"{source}'s parameters are {sorted(accepted) or sorted(observed)}. miForge "
            f"needs one of FP16 or BF16, not a mixture: the recorded optimizer identity "
            f"names a single precision and a mixed set would make it a guess.",
            source=source,
            observed_dtypes=sorted(observed),
        )

    declared = declared_config_dtype(config)
    only = next(iter(accepted))
    if declared is not None and declared != only:
        # Disagreement is a refusal, not a vote. One of the two is wrong about this model
        # and the gate cannot tell which, so neither answer may be used.
        raise refuse(
            "model_dtype_disagrees",
            f"{source}'s config declares {declared} while its weight files carry {only}. "
            f"The two disagree about what this model is; refusing rather than choosing one. "
            f"miForge trains FP16 or BF16 weights.",
            source=source,
            config_dtype=declared,
            weight_dtype=only,
        )
    return only


def _is_benign_auxiliary_dtype(dtype: str) -> bool:
    """Whether a non-FP16/BF16 dtype belongs to a buffer rather than a parameter.

    Integer index buffers and float32 normalisation statistics appear in ordinary
    unquantized checkpoints. **8-bit floats and every integer *packing* width are not on
    this list**: those are how quantized payloads travel, and a gate that waved them
    through would be the mutation this module's test exists to catch.
    """
    return dtype in {"I64", "I32", "BOOL", "F32"}
