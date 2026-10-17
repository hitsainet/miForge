"""Load a model from a local Hugging Face directory  (R-02.46 case 1; FR-001.3).

**Tokenizer files at the ROOT, never in a ``tokenizer/`` subdirectory.** That is ADR-015
and R-02.51, and it is not a style preference: as originally specced a miForge checkpoint
put them in a subdirectory, and such a directory **cannot be loaded** by ``transformers``,
by vLLM, by TGI or by miLLM's own loader. It broke R-02.21 and acceptance criterion 2
before a line of code existed. This module is the reader half of that promise, so it looks
for them where every standard loader looks.

## Preflight reads; acquire allocates

Everything that can refuse does so in :meth:`LocalDirAcquirer.preflight` - path
confinement, the missing-file check naming the file, the dtype gate, the parameter count
for the budget guard. No torch, no card, no VRAM. :meth:`acquire` is the only method that
imports ``transformers``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.errors import refuse
from training.acquisition.base import (
    AcquiredModel,
    AcquisitionMode,
    AcquisitionRecord,
    AcquisitionSpec,
    Availability,
    SourceKind,
    lineage_for_first_load,
    ml_runtime_availability,
)
from training.acquisition.digest import weights_sha256
from training.acquisition.dtype_gate import (
    assert_trainable,
    collect_weight_dtypes,
    safetensors_header,
)
from training.acquisition.placement import place

#: What a loadable Hugging Face directory must contain. The tokenizer is satisfied by any
#: one of several files because the libraries accept several forms - ``tokenizer.json``
#: alone for a fast tokenizer, or the sentencepiece/vocab pair - so requiring one exact
#: name would refuse models that load perfectly well.
CONFIG_FILE = "config.json"
TOKENIZER_CANDIDATES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
)
#: ``safetensors`` only. ``pytorch_model.bin`` is refused by name rather than loaded,
#: because its dtypes cannot be read without unpickling it - and unpickling a weight file
#: in preflight both executes arbitrary code and defeats the no-allocation rule. ADR-002
#: names ``safetensors`` for exactly this reason.
WEIGHTS_SUFFIX = ".safetensors"
LEGACY_WEIGHTS_NAMES: tuple[str, ...] = ("pytorch_model.bin", "model.ckpt", "flax_model.msgpack")


def resolve_within_roots(path: Path) -> Path:
    """Resolve ``path`` and confine it to a configured model root (TDD section 8).

    A submitted path is an input even on a LAN. Confinement is what stops
    ``../../etc`` being read by a worker that holds the card, and it is checked against the
    **resolved** path so a symlink out of the root is caught too.

    When no roots are configured the only sanctioned location is the checkpoint root -
    miForge's own artifacts - rather than "anywhere", because a default of "anywhere" is
    the kind of permissive fallback that is never revisited.
    """
    settings = get_settings()
    roots = [settings.checkpoint_root, settings.parts_root]
    resolved = path.expanduser().resolve()

    for root in roots:
        try:
            resolved.relative_to(root.expanduser().resolve())
        except ValueError:
            continue
        return resolved

    raise refuse(
        "model_source_outside_root",
        f"{resolved} is not inside a configured model root. miForge reads models from "
        f"{[str(r) for r in roots]} only; move the model there or point CHECKPOINT_ROOT at "
        f"it. Refusing rather than reading an arbitrary path from a worker that holds the "
        f"card.",
        path=str(resolved),
        roots=[str(r) for r in roots],
    )


def read_config(directory: Path) -> dict[str, Any]:
    """Parse ``config.json``, naming the file in every failure (E-2, E-5)."""
    config_path = directory / CONFIG_FILE
    if not config_path.is_file():
        raise refuse(
            "model_source_incomplete",
            f"{directory} has no {CONFIG_FILE}, so miForge cannot tell what model it is. "
            f"A Hugging Face model directory carries {CONFIG_FILE}, its weight shards and "
            f"its tokenizer files at the root.",
            path=str(directory),
            missing=CONFIG_FILE,
        )
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise refuse(
            "architecture_config_invalid",
            f"{config_path} could not be parsed as JSON: {exc}. Refusing before any "
            f"allocation rather than loading a model miForge cannot describe.",
            path=str(config_path),
        ) from exc
    if not isinstance(parsed, dict):
        raise refuse(
            "architecture_config_invalid",
            f"{config_path} is not a JSON object, so no architecture can be read from it.",
            path=str(config_path),
        )
    return parsed


def weight_files(directory: Path) -> list[Path]:
    """Every ``.safetensors`` shard, named in the refusal when there are none (E-2)."""
    shards = sorted(p for p in directory.glob(f"*{WEIGHTS_SUFFIX}") if p.is_file())
    if shards:
        return shards

    legacy = [name for name in LEGACY_WEIGHTS_NAMES if (directory / name).is_file()]
    if legacy:
        raise refuse(
            "model_weights_format_unsupported",
            f"{directory} carries {legacy} but no {WEIGHTS_SUFFIX} shard. miForge reads "
            f"safetensors: a pickled weight file's dtypes cannot be checked without "
            f"unpickling it, which would execute arbitrary code in the training worker and "
            f"allocate before the budget guard has run. Convert the model to safetensors.",
            path=str(directory),
            found=legacy,
        )
    raise refuse(
        "model_source_incomplete",
        f"{directory} has no *{WEIGHTS_SUFFIX} weight file. Name the directory holding the "
        f"model's shards.",
        path=str(directory),
        missing=f"*{WEIGHTS_SUFFIX}",
    )


def require_tokenizer_at_root(directory: Path) -> None:
    """Refuse a directory whose tokenizer is absent **or in a subdirectory** (ADR-015).

    The subdirectory case gets its own message because it is the project's own historical
    defect and because the fix is a move rather than a download - telling the operator
    "tokenizer missing" when the files are sitting one level down sends them looking for
    the wrong thing.
    """
    if any((directory / name).is_file() for name in TOKENIZER_CANDIDATES):
        return

    nested = directory / "tokenizer"
    if nested.is_dir() and any((nested / name).is_file() for name in TOKENIZER_CANDIDATES):
        raise refuse(
            "model_tokenizer_not_at_root",
            f"{directory} keeps its tokenizer files in a tokenizer/ subdirectory. A "
            f"standard Hugging Face directory carries them at the root, and a subdirectory "
            f"layout cannot be loaded by transformers, vLLM, TGI or miLLM's loader "
            f"(R-02.51, ADR-015). Move them up one level.",
            path=str(directory),
            found_in=str(nested),
        )
    raise refuse(
        "model_source_incomplete",
        f"{directory} carries none of {list(TOKENIZER_CANDIDATES)} at its root. miForge "
        f"does not train a tokenizer; the model must come with one.",
        path=str(directory),
        missing=list(TOKENIZER_CANDIDATES),
    )


def param_count_from_headers(shards: list[Path]) -> int:
    """Total elements across every tensor, from the shard headers alone.

    The budget guard needs a parameter count **before** anything is allocated (ADR-008), so
    it is derived from the ``safetensors`` header's shapes rather than from a loaded model.
    Counting a loaded model's parameters would mean allocating first and asking the guard
    afterwards, which is the ordering R-02.42 exists to invert.
    """
    total = 0
    for path in shards:
        for name, entry in safetensors_header(path).items():
            if name == "__metadata__" or not isinstance(entry, dict):
                continue
            shape = entry.get("shape") or []
            if not shape:
                # A zero-dimensional tensor holds one value, but counting it as a parameter
                # would inflate the estimate by the number of scalar buffers a model
                # happens to carry. Skipped, and skipped explicitly so the choice is visible.
                continue
            elements = 1
            for dim in shape:
                elements *= int(dim)
            total += elements
    return total


class LocalDirAcquirer:
    """``load`` from a directory on this machine (FR-001.3)."""

    @property
    def mode(self) -> AcquisitionMode:
        return AcquisitionMode.LOAD

    def available(self) -> Availability:
        """Whether a directory can be *loaded* here. No network, no GPU.

        Reading a directory needs nothing optional - which is why every refusal below is
        testable without a card - but **loading one needs torch**, so the mode is reported
        unavailable by name when the ML runtime is absent rather than promising a capability
        it would then refuse. See :func:`training.acquisition.base.ml_runtime_availability`.
        """
        return ml_runtime_availability("load a model from a local directory", "run a training step")

    def preflight(self, spec: AcquisitionSpec) -> AcquisitionSpec:
        """Every refusal of E-1, E-2 and E-5, before the lease and before allocation."""
        if spec.path is None:
            raise refuse(
                "model_source_missing",
                "A local load needs a path. Refusing rather than defaulting to a directory "
                "the operator did not name.",
            )
        directory = resolve_within_roots(spec.path)
        if not directory.is_dir():
            raise refuse(
                "model_source_incomplete",
                f"{directory} is not a directory. Name the Hugging Face model directory "
                f"holding config.json, the weight shards and the tokenizer files.",
                path=str(directory),
            )

        config = read_config(directory)
        shards = weight_files(directory)
        require_tokenizer_at_root(directory)
        assert_trainable(config, collect_weight_dtypes(shards), source=str(directory))

        return spec.model_copy(
            update={"path": directory, "param_count": param_count_from_headers(shards)}
        )

    def acquire(self, spec: AcquisitionSpec) -> AcquiredModel:
        """Load the model into **this** process. The only method here that imports torch."""
        if spec.path is None:
            # Not reachable through the service, which preflights first. Refused rather
            # than asserted because `acquire` is a public method and a caller that skipped
            # preflight must be told so, not crash on None.
            raise refuse(
                "acquisition_preflight_skipped",
                "acquire() was called with no path. Call preflight() first: it is where "
                "the path is confined, the dtype gate runs and the budget guard gets its "
                "parameter count.",
            )
        directory = resolve_within_roots(spec.path)
        config = read_config(directory)
        shards = weight_files(directory)
        dtype_name = assert_trainable(config, collect_weight_dtypes(shards), source=str(directory))

        module, tokenizer = _load_pretrained(directory, dtype_name)
        # ONTO THE LEASED CARD, HERE, and the record below carries where it landed.
        # `from_pretrained` leaves every tensor in host memory, so without this line the run
        # holds a card's lease and computes on the CPU - silently, because `objectives/sft.py`
        # builds its batch tensors on *the model's* device. Placed in `acquire` rather than
        # inside `_load_pretrained` because `create` has no `from_pretrained` to hook and
        # the record has to be built from the realised placement, not from the request.
        placed = place(module, spec.gpu_request)
        identity = f"local:{directory.name}@{weights_sha256(shards, relative_to=directory)[:16]}"
        base_model, parent = lineage_for_first_load(model_identity=identity)

        return AcquiredModel(
            module=module,
            tokenizer=tokenizer,
            record=AcquisitionRecord(
                mode=AcquisitionMode.LOAD,
                source_kind=SourceKind.LOCAL_DIR,
                source_path=str(directory),
                starting_weights_sha256=weights_sha256(shards, relative_to=directory),
                base_model=base_model,
                parent=parent,
                param_count=spec.param_count or param_count_from_headers(shards),
                device_kind=placed.device_kind,
                device_uuid=placed.device_uuid,
                device_placement_reason=placed.reason,
            ),
            # One source for the UUID: the record's, which is the read-back. Two
            # independently-assigned copies of one device identity is the second-spelling
            # family this project has paid for four times.
            device_uuid=placed.device_uuid,
        )


def _load_pretrained(directory: Path, dtype_name: str) -> tuple[Any, Any]:
    """``from_pretrained`` with ``trust_remote_code`` **off**, imported lazily.

    ``trust_remote_code=False`` is passed explicitly rather than relied upon as the library
    default: the default has changed before, and a remote-code architecture would execute
    arbitrary Python inside the worker that holds the GPU lease. A test asserts the argument
    rather than trusting this comment.
    """
    torch, auto_model, auto_tokenizer = _ml_imports()
    dtype = {"F16": torch.float16, "BF16": torch.bfloat16}[dtype_name]
    module = auto_model.from_pretrained(directory, torch_dtype=dtype, trust_remote_code=False)
    # From the ROOT. A `tokenizer/` subdirectory is the defect R-02.51 records.
    tokenizer = auto_tokenizer.from_pretrained(directory, trust_remote_code=False)
    return module, tokenizer


def _ml_imports() -> tuple[Any, Any, Any]:
    """Import torch and ``transformers``, refusing **by name** when they are absent.

    ADR-023's layout means the ML wheels may not be installed at all on a machine that can
    still run every other part of this feature. An ``ImportError`` escaping from here would
    surface as a 500 with a traceback; a named refusal says what is missing and that only
    the training step needs it.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise refuse(
            "ml_runtime_unavailable",
            "torch and transformers are not installed in this process, so no model can be "
            "loaded. Everything else in miForge runs without them (ADR-023); install the "
            "package's full dependencies on the GPU node to run a training step.",
            detail=str(exc),
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer
