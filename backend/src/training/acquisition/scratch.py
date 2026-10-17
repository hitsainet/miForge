"""Create a model from an architecture config, in **this** process  (R-02.46 case 2).

Nothing is loaded and nothing upstream is contacted. Before the 2026-10-13 standalone
reframe this route went through another application: miForge asked miLLM to instantiate a
random-init resident model (BRD-01 R-01.26) and attached to it. That requirement is now
deferred, *because its only consumer was this path*, and the model is initialised here.

## The config is parsed before anything is instantiated, and that is the point

R-02.42's budget guard *"estimates from the architecture config, before ``instantiate``"* -
corrected on 2026-10-13 because the original wording estimated from *"the resident model's
parameter count"* and **a from-scratch run has no resident model to measure**. The guard
therefore fires too late for the run type that most needs it unless the config is parsed
and handed over first. So :meth:`ScratchAcquirer.preflight` parses, validates the
architecture against what ``transformers`` actually knows, and returns a spec carrying the
config - and never a parameter count, because counting one would mean building the model.

## The seed

R-02.47 wants *"the content hash of the weights it started from"* for both modes, and a
random initialisation has no reproducible hash without a seed - which no BRD-02 requirement
records. The operator's decision of 2026-10-14 records one, and
:func:`training.acquisition.digest.created_weights_sha256` is what makes
``(config, seed)`` the description of the initial parameters. ``seed=None`` is still
accepted and still recorded, as an assertion that the run is not reproducible.

## The tokenizer, and why its absence is a refusal at submit

BRD-02 section 8 says miForge never trains a tokenizer, so *"a creation run is given
one"* - and until 2026-10-16 **nothing in ``backend/src/`` could give it one.**
:func:`_instantiate` returned ``tokenizer=None`` with a docstring naming a caller that
did not exist, ``AcquisitionSpec`` carried no field for the reference, and
``CreateRunType.prepare`` - which does raise ``tokenizer_required`` - has no production
caller (see ``training/run_driver.py``'s own correction of 2026-10-16). So a ``create``
run trained for however long it was configured to, reached its first save point and died
on ``AttributeError: 'NoneType' object has no attribute 'save_pretrained'`` inside F5's
writer, having produced nothing. Scoring died the same way one level over, with
``TypeError: 'NoneType' object is not callable``.

**The fix is a supply mechanism, not a ``None`` guard.** Turning either crash into a tidy
named refusal would have made the missing capability tidy rather than present:
R-02.21 and ADR-015 make a checkpoint *a standard Hugging Face directory with the
tokenizer files at the root*, and a directory without them cannot be loaded by
``transformers``, vLLM, TGI or miLLM's loader - so *"create runs do not ship a tokenizer"*
is not an available option, and acceptance criterion 23's clean-process load would fail
for every created checkpoint.

**And the absence refuses at submit rather than being allowed with a consequence
recorded.** The consequence would be an unloadable artifact, which the output contract
(R-02.51) forbids outright; there is no state of the world in which recording it is
better than refusing before the card is leased. The code is ``tokenizer_required`` - F8's
own spelling in :meth:`training.runtypes.create.CreateRunType.prepare`, reused rather
than given a second name, because one fact with two codes is this project's signature
drift. :meth:`ScratchAcquirer.preflight` is where it fires, which puts it on the submit
path through ``services.acquisition.preflight`` - before the lease and before any
allocation.

## The vocabulary has to agree, and that is checkable without torch

``LLAMA_TINY`` declares ``vocab_size: 512``. Hand it a 32k-vocab tokenizer and the first
thing that generates raises ``IndexError: index out of range in self`` from the embedding
lookup - a crash about an index, hours after a submit that looked fine, which says nothing
about the tokenizer. :func:`assert_vocab_agrees` reads the tokenizer's own vocabulary
files and refuses ``tokenizer_vocab_mismatch`` at preflight; :func:`assert_loaded_vocab`
re-asserts it against the realised ``len(tokenizer)`` in :meth:`ScratchAcquirer.acquire`,
because the files are a *prediction* of that number and the loaded tokenizer is the fact.
A tokenizer whose vocabulary is **smaller** than the config is not refused: the surplus
embedding rows are unused, which is wasteful and not wrong.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.errors import Refusal, refuse
from training.acquisition.base import (
    AcquiredModel,
    AcquisitionMode,
    AcquisitionRecord,
    AcquisitionSpec,
    ArchitectureConfig,
    Availability,
    SourceKind,
    lineage_for_creation,
    ml_runtime_availability,
)
from training.acquisition.digest import config_identity, created_weights_sha256
from training.acquisition.local_dir import TOKENIZER_CANDIDATES, resolve_within_roots
from training.acquisition.placement import place

log = logging.getLogger("miforge.acquisition.scratch")


def parse_config(spec: AcquisitionSpec) -> ArchitectureConfig:
    """Parse the architecture config as **data**, naming the bad field on refusal (E-5).

    Raises:
        Refusal: ``architecture_config_invalid`` - unparseable, not an object, or silent
            about what architecture it describes. **The field is named**, because "invalid
            config" leaves the operator with a file and no clue which line is wrong.
    """
    if spec.config is not None:
        data: Any = spec.config
    elif spec.config_path is not None:
        path = Path(spec.config_path).expanduser()
        if not path.is_file():
            raise refuse(
                "architecture_config_invalid",
                f"No architecture config at {path}. Name a JSON file describing the "
                f"architecture to create.",
                path=str(path),
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise refuse(
                "architecture_config_invalid",
                f"{path} could not be parsed as JSON: {exc}. Refusing before any " f"allocation.",
                path=str(path),
            ) from exc
    else:
        raise refuse(
            "architecture_config_invalid",
            "A create needs an architecture config. Refusing rather than instantiating a "
            "default architecture the operator did not ask for.",
        )

    if not isinstance(data, dict):
        raise refuse(
            "architecture_config_invalid",
            "The architecture config is not a JSON object, so no architecture can be read "
            "from it.",
        )

    architecture = _architecture_name(data)
    if data.get("quantization_config"):
        # A config that asks for quantized weights cannot describe a full-parameter
        # trainable model (R-02.46), and refusing here is cheaper than refusing after
        # instantiation.
        raise refuse(
            "architecture_config_invalid",
            "The architecture config declares 'quantization_config'. miForge creates "
            "FP16/BF16 full-parameter models; a quantized architecture cannot be trained "
            "full-parameter.",
            field="quantization_config",
        )
    return ArchitectureConfig(data=dict(data), architecture=architecture)


def _architecture_name(data: dict[str, Any]) -> str:
    """``architectures[0]`` or ``model_type``, refusing when the config says neither.

    **No default.** ADR-018: a ``getattr`` default that always fires is a fabricated value.
    A config that does not say what it is cannot be instantiated, and guessing is how an
    unknown architecture reaches the allocator with the budget guard already satisfied.
    """
    architectures = data.get("architectures")
    if isinstance(architectures, list) and architectures and isinstance(architectures[0], str):
        return architectures[0]
    model_type = data.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type
    raise refuse(
        "architecture_config_invalid",
        "The architecture config names neither 'architectures' nor 'model_type', so "
        "miForge cannot tell what to create. Refusing rather than guessing an "
        "architecture.",
        field="architectures",
    )


def resolve_tokenizer_directory(spec: AcquisitionSpec) -> Path:
    """The directory a ``create``'s tokenizer files are read from. **No torch, no card.**

    One function for both spellings, so the refusals, the confinement and the
    at-the-root rule have one implementation each rather than two that drift. The Hub half
    uses :func:`training.acquisition.hub.resolve_revision` and
    :func:`~training.acquisition.hub.download_snapshot` unchanged - every outbound Hub
    call in miForge is in that module and a second one here would break that discipline
    for a convenience.

    Raises:
        Refusal: ``tokenizer_required`` when the spec names no tokenizer at all;
            ``tokenizer_source_outside_root`` for a path outside the configured model
            roots; ``tokenizer_source_incomplete`` for a directory with no tokenizer files
            at its root; ``tokenizer_not_at_root`` when they are one level down.
    """
    if spec.tokenizer_repo is not None:
        from training.acquisition import hub as hub_module

        commit = spec.tokenizer_revision_resolved or hub_module.resolve_revision(
            spec.tokenizer_repo, spec.tokenizer_revision or hub_module.DEFAULT_REVISION
        )
        # Downloads the whole repo, because `download_snapshot` takes no `allow_patterns`
        # and adding one is a change to a file this change does not own. Stated rather than
        # worked around: a tokenizer-only repo is cheap and a model repo is not, and the
        # cheap fix lives in `hub.py`.
        return assert_tokenizer_at_root(hub_module.download_snapshot(spec.tokenizer_repo, commit))

    if spec.tokenizer_path is None:
        raise refuse(
            "tokenizer_required",
            "This creation run names no tokenizer. miForge does not train one (BRD-02 "
            "section 8) and will not invent a default: a mismatched tokenizer produces a "
            "model that trains happily and generates nonsense, and a checkpoint with no "
            "tokenizer at its root cannot be loaded by transformers, vLLM, TGI or "
            "miLLM's loader at all (R-02.21, R-02.51, ADR-015). Set `tokenizer_path` to a "
            "model or tokenizer directory inside a configured model root, or "
            "`tokenizer_repo` (with `tokenizer_revision`) to a Hugging Face repo.",
            fields=["tokenizer_path", "tokenizer_repo"],
        )

    try:
        directory = resolve_within_roots(spec.tokenizer_path)
    except Refusal as exc:
        # The *confinement* is reused - one root policy for every path a worker reads -
        # and only the name changes. `model_source_outside_root` on a create run is
        # actively misleading: a create has no model source, so the operator would go
        # looking for a field the payload does not contain.
        raise refuse(
            "tokenizer_source_outside_root",
            str(exc.message).replace("miForge reads models from", "miForge reads tokenizers from"),
            **exc.details,
        ) from exc

    if not directory.is_dir():
        raise refuse(
            "tokenizer_source_incomplete",
            f"{directory} is not a directory. Name the directory holding the tokenizer's "
            f"own files - a model directory will do, since a standard one carries them at "
            f"its root.",
            path=str(directory),
        )
    return assert_tokenizer_at_root(directory)


def assert_tokenizer_at_root(directory: Path) -> Path:
    """Refuse a directory whose tokenizer files are absent **or one level down** (ADR-015).

    The candidate filenames come from the ``load`` half
    (:data:`training.acquisition.local_dir.TOKENIZER_CANDIDATES`) rather than being listed
    again: a second list is a second answer to *"what counts as a tokenizer"*, and the one
    that drifts is whichever is read less often.
    """
    if any((directory / name).is_file() for name in TOKENIZER_CANDIDATES):
        return directory

    nested = directory / "tokenizer"
    if nested.is_dir() and any((nested / name).is_file() for name in TOKENIZER_CANDIDATES):
        raise refuse(
            "tokenizer_not_at_root",
            f"{directory} keeps its tokenizer files in a tokenizer/ subdirectory. "
            f"`save_pretrained` copies them to the root of the checkpoint it writes, and a "
            f"subdirectory layout is the defect R-02.51 records - it cannot be loaded by "
            f"transformers, vLLM, TGI or miLLM's loader. Point at the subdirectory itself, "
            f"or move the files up one level.",
            path=str(directory),
            found_in=str(nested),
        )
    raise refuse(
        "tokenizer_source_incomplete",
        f"{directory} carries none of {list(TOKENIZER_CANDIDATES)} at its root, so there "
        f"is no tokenizer in it to give this run.",
        path=str(directory),
        missing=list(TOKENIZER_CANDIDATES),
    )


def tokenizer_vocab_ceiling(directory: Path) -> int | None:
    """How many embedding rows ``directory``'s tokenizer needs, from its files alone.

    ``None`` when it cannot be read - a SentencePiece ``tokenizer.model`` carries its
    vocabulary in a protobuf that needs the library to parse. **``None`` is reported, never
    treated as agreement**: :func:`assert_vocab_agrees` logs it by name and
    :func:`assert_loaded_vocab` makes the real check against the loaded tokenizer, so the
    unreadable case loses the *early* refusal and not the refusal.

    The ceiling is ``max(id) + 1`` over the vocabulary **and the added tokens**, because an
    added token's id is an index into the same embedding table and is routinely above the
    base vocabulary.
    """
    fast = directory / "tokenizer.json"
    if fast.is_file():
        try:
            data = json.loads(fast.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        ceiling = _ceiling_of_vocab((data.get("model") or {}).get("vocab"))
        added = [
            int(entry["id"])
            for entry in (data.get("added_tokens") or [])
            if isinstance(entry, dict) and isinstance(entry.get("id"), int)
        ]
        if added:
            ceiling = max(ceiling or 0, max(added) + 1)
        return ceiling

    plain = directory / "vocab.json"
    if plain.is_file():
        try:
            return _ceiling_of_vocab(json.loads(plain.read_text(encoding="utf-8")))
        except (OSError, ValueError, UnicodeDecodeError):
            return None
    return None


def _ceiling_of_vocab(vocab: Any) -> int | None:
    """``max(id) + 1`` for a ``{token: id}`` map, or ``len`` for a Unigram token list."""
    if isinstance(vocab, dict):
        ids = [int(v) for v in vocab.values() if isinstance(v, int)]
        return max(ids) + 1 if ids else None
    if isinstance(vocab, list) and vocab:
        # Unigram: a list of `[token, score]` pairs, positionally indexed.
        return len(vocab)
    return None


def assert_vocab_agrees(config: dict[str, Any], directory: Path) -> int | None:
    """Refuse a tokenizer whose ids do not fit the config's embedding table (E-5).

    Returns the ceiling when it could be read, so the acquisition record can carry it.

    Raises:
        Refusal: ``tokenizer_vocab_mismatch``, naming **both** numbers. The alternative is
            ``IndexError: index out of range in self`` from the embedding lookup, hours
            after a submit that looked fine, in a message that mentions neither the
            tokenizer nor the config.
    """
    configured = config.get("vocab_size")
    if not isinstance(configured, int) or isinstance(configured, bool):
        log.warning(
            "tokenizer vocabulary not checked",
            extra={
                "code": "tokenizer_vocab_not_checked",
                "reason": "config_states_no_vocab_size",
                "tokenizer_source": str(directory),
            },
        )
        return None

    ceiling = tokenizer_vocab_ceiling(directory)
    if ceiling is None:
        log.warning(
            "tokenizer vocabulary not checked",
            extra={
                "code": "tokenizer_vocab_not_checked",
                "reason": "tokenizer_vocabulary_not_readable_without_the_library",
                "tokenizer_source": str(directory),
                "config_vocab_size": configured,
            },
        )
        return None

    if ceiling > configured:
        raise refuse(
            "tokenizer_vocab_mismatch",
            f"The tokenizer in {directory} needs {ceiling} embedding rows and the "
            f"architecture config declares vocab_size {configured}. Every id at or above "
            f"{configured} would index past the embedding table, so the first thing that "
            f"generated would raise `IndexError: index out of range in self`. Raise "
            f"vocab_size to at least {ceiling}, or name a tokenizer sized for this config.",
            tokenizer_source=str(directory),
            tokenizer_vocab_size=ceiling,
            config_vocab_size=configured,
        )
    return ceiling


def assert_loaded_vocab(config: dict[str, Any], tokenizer: Any, *, source: str) -> int | None:
    """The same rule against the **loaded** tokenizer, which is the fact (E-5).

    :func:`assert_vocab_agrees` reads files and is therefore a prediction; ``len`` of a
    loaded tokenizer is the number the embedding lookup will actually be handed. Both
    exist because the early one is what refuses at submit and this one is what cannot be
    wrong - and the unreadable-vocabulary case has only this one.
    """
    configured = config.get("vocab_size")
    try:
        size = len(tokenizer)
    except TypeError:
        # A double or a tokenizer type with no `__len__`. Reported as unchecked rather
        # than as agreement: the files-based check above already ran, and claiming a
        # comparison that did not happen is the construction-green failure.
        return None
    if not isinstance(configured, int) or isinstance(configured, bool):
        return size
    if size > configured:
        raise refuse(
            "tokenizer_vocab_mismatch",
            f"The tokenizer loaded from {source} has {size} tokens and the architecture "
            f"config declares vocab_size {configured}. Refusing before the first step "
            f"rather than at the first generation, where it surfaces as `IndexError: index "
            f"out of range in self` and names neither the tokenizer nor the config.",
            tokenizer_source=source,
            tokenizer_vocab_size=size,
            config_vocab_size=configured,
        )
    return size


def assert_architecture_known(architecture: str) -> None:
    """Refuse an architecture ``transformers`` does not know (E-5), before instantiation.

    Checked against the live library rather than against a list maintained here: a hardcoded
    list is wrong the moment the library is upgraded, in the permissive direction for a
    removed architecture and the refusing direction for a new one.

    When ``transformers`` is not installed this **does not refuse** - the check is simply
    not available, and :meth:`ScratchAcquirer.available` has already reported the mode
    unavailable by name. Refusing here would conflate "this architecture is unknown" with
    "this machine has no ML runtime", which are different facts with different fixes.
    """
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except ImportError:
        return

    known_types = set(CONFIG_MAPPING_NAMES)
    # `architectures[0]` is a class name (`LlamaForCausalLM`) while CONFIG_MAPPING_NAMES is
    # keyed by model type (`llama`), so both forms are accepted rather than one being
    # declared canonical - the config files in the wild use both.
    lowered = architecture.lower()
    if architecture in known_types or any(t in lowered for t in known_types):
        return
    raise refuse(
        "architecture_config_invalid",
        f"transformers does not know the architecture {architecture!r}, so it cannot be "
        f"instantiated. Refusing before any allocation; check the 'architectures' or "
        f"'model_type' field against the installed transformers version.",
        field="architectures",
        architecture=architecture,
    )


class ScratchAcquirer:
    """``create`` from an architecture config with random initialisation (FR-001.6).

    **F1 owns the acquisition of a created parameter set; F8 owns R-02.41's from-scratch
    *run type*.** R-02.46 case 2 assigns the acquisition here and cites R-02.41 for the run
    type, and the distinction is load-bearing: F8's work is the budget, the lineage of a
    parentless checkpoint and the run's shape, none of which is below.
    """

    @property
    def mode(self) -> AcquisitionMode:
        return AcquisitionMode.CREATE

    def available(self) -> Availability:
        """Whether a model can be instantiated here at all. No network, no GPU."""
        return ml_runtime_availability("create a model from an architecture config")

    def preflight(self, spec: AcquisitionSpec) -> AcquisitionSpec:
        """Parse and validate the config, resolve the tokenizer, hand both to the guard.

        Returns a spec carrying the parsed config and **no parameter count**: R-02.42
        estimates from the config precisely because there is nothing to measure yet, and a
        count here could only come from building the model the guard exists to veto.

        **The tokenizer is resolved here and that is the whole point of the position.**
        This method is what ``services.acquisition.preflight`` calls, and that is what
        ``services/run_definition.py``'s budget step calls - so ``tokenizer_required``,
        ``tokenizer_source_incomplete``, ``tokenizer_not_at_root`` and
        ``tokenizer_vocab_mismatch`` all land **at submit**, before the lease and before a
        single tensor is allocated, instead of at the first save point of a run that has
        already trained.

        Raises:
            Refusal: ``architecture_config_invalid``; ``tokenizer_required`` when no
                tokenizer is named; the three tokenizer-source refusals; and
                ``tokenizer_vocab_mismatch`` when its ids do not fit the config.
        """
        parsed = parse_config(spec)
        assert_architecture_known(parsed.architecture)
        update: dict[str, Any] = {"config": parsed.data, "config_path": None}

        if spec.tokenizer_repo is not None:
            from training.acquisition import hub as hub_module

            # Resolved to a commit **before** anything is fetched, exactly as the `load`
            # half does it: a branch that moves between resolution and download must not
            # change which tokenizer the checkpoint ships.
            update["tokenizer_revision_resolved"] = hub_module.resolve_revision(
                spec.tokenizer_repo, spec.tokenizer_revision or hub_module.DEFAULT_REVISION
            )

        directory = resolve_tokenizer_directory(spec.model_copy(update=update))
        update["tokenizer_vocab_size"] = assert_vocab_agrees(parsed.data, directory)
        return spec.model_copy(update=update)

    def acquire(self, spec: AcquisitionSpec) -> AcquiredModel:
        """Instantiate the model, randomly initialised, in this process - **with its
        tokenizer.**

        The tokenizer is loaded here rather than left ``None`` for a caller to fill in.
        That docstring promise was kept by nobody for the whole life of this file, which is
        how a ``create`` run came to die at its first save point; the reference is on the
        spec now, so there is one place the answer comes from.

        Raises:
            Refusal: ``tokenizer_required`` and the tokenizer-source refusals, re-raised
                here because ``acquire`` is a public method and a caller that skipped
                preflight must be told so rather than handed ``tokenizer=None``;
                ``tokenizer_vocab_mismatch`` against the **loaded** tokenizer's length.
        """
        parsed = parse_config(spec)
        tokenizer_directory = resolve_tokenizer_directory(spec)
        module, tokenizer = _instantiate(parsed, spec.seed, tokenizer_directory)
        vocab_size = assert_loaded_vocab(parsed.data, tokenizer, source=str(tokenizer_directory))
        # ONTO THE LEASED CARD. `from_config` initialises in host memory, so a `create` run
        # was the same silent CPU run a `load` was: lease held, step on the CPU, nothing
        # raised. The record below carries the device read back from the module, never the
        # device the spec asked for.
        placed = place(module, spec.gpu_request)

        identity = _created_base_model(parsed, spec.seed)
        base_model, parent = lineage_for_creation(config_identity=identity)
        return AcquiredModel(
            module=module,
            tokenizer=tokenizer,
            record=AcquisitionRecord(
                mode=AcquisitionMode.CREATE,
                source_kind=SourceKind.CONFIG,
                config_identity=config_identity(parsed.data),
                starting_weights_sha256=created_weights_sha256(config=parsed.data, seed=spec.seed),
                seed=spec.seed,
                # WHICH TOKENIZER THIS CHECKPOINT SHIPS. The manifest is closed and cannot
                # carry it, so the record is the only place the vocabulary a created model
                # was trained against is written down at all.
                tokenizer_source=(
                    f"hf:{spec.tokenizer_repo}@{spec.tokenizer_revision_resolved}"
                    if spec.tokenizer_repo is not None
                    else str(tokenizer_directory)
                ),
                tokenizer_vocab_size=vocab_size,
                # parent is None and base_model is the config: exactly what R-02.47 says
                # and what `checkpoint-manifest-v1.json`'s own description of `base_model`
                # already allowed. No schema change was ever needed for this.
                base_model=base_model,
                parent=parent,
                param_count=_count_parameters(module),
                device_kind=placed.device_kind,
                device_uuid=placed.device_uuid,
                device_placement_reason=placed.reason,
            ),
            device_uuid=placed.device_uuid,
        )


def created_base_model_identity(config: dict[str, Any], seed: int | None) -> str:
    """The string a creation's ``base_model`` carries. Exposed for F5 and for its test.

    **Why the seed is inside this string.** ``checkpoint-manifest-v1.json``'s ``base_model``
    is a plain ``string`` and the manifest is closed to unknown keys, so there is no
    separate field to put a seed in and adding one would be a schema change in two repos.
    Folding the seed into the identity gets what the operator's decision was for: two
    creations from one config with different seeds have **different** ``base_model``
    values, so the required ``starting_weights_sha256`` is explicable from the manifest
    alone rather than only from the run log beside it.

    ``+seed=none`` is written out rather than omitted. An absent seed marker would read as
    an older identity format; the explicit ``none`` is the assertion that no seed was set
    and that the initialisation is therefore not reproducible.
    """
    marker = "none" if seed is None else str(seed)
    return f"config:{_architecture_name(config)}@{config_identity(config)}+seed={marker}"


def _created_base_model(parsed: ArchitectureConfig, seed: int | None) -> str:
    return created_base_model_identity(parsed.data, seed)


def _instantiate(
    parsed: ArchitectureConfig, seed: int | None, tokenizer_directory: Path
) -> tuple[Any, Any]:
    """``AutoConfig`` then ``from_config``, seeded, with ``trust_remote_code`` off - and
    the tokenizer loaded from ``tokenizer_directory``.

    No tokenizer is *created* here: miForge does not train one (BRD-02 section 8), so a
    creation run is **given** one and this is where the given one is read.

    *This returned ``(module, None)`` until 2026-10-16, with a docstring saying "the caller
    supplies it".* No caller in ``backend/src/`` did, because no field carried the
    reference - so every creation run reached F5's writer with ``tokenizer=None`` and
    raised ``AttributeError`` there. The third positional argument is what makes that
    impossible to reintroduce: there is no value of it that means "none".
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise refuse(
            "ml_runtime_unavailable",
            "torch and transformers are not installed in this process, so a model cannot "
            "be created. Everything else in miForge runs without them (ADR-023).",
            detail=str(exc),
        ) from exc

    if seed is not None:
        # Seeded *immediately before* initialisation and nowhere else, so the recorded
        # (config, seed) really does describe these weights. A seed set at process start
        # would be consumed by anything that drew from the RNG in between.
        torch.manual_seed(seed)

    model_type = parsed.data.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        # `from_config` needs a typed config object, and `AutoConfig` builds one from a
        # model type. An `architectures` entry alone does not name a config class, so this
        # refuses rather than reaching for a mapping from class name to model type - which
        # would be a second copy of a table transformers already owns.
        raise refuse(
            "architecture_config_invalid",
            "The architecture config must name 'model_type' to be instantiated: that is "
            "what selects the config class. 'architectures' alone identifies a model class "
            "and not a config, and miForge will not guess the mapping between them.",
            field="model_type",
        )
    config = AutoConfig.for_model(
        model_type, **{k: v for k, v in parsed.data.items() if k != "model_type"}
    )
    module = AutoModelForCausalLM.from_config(  # type: ignore[no-untyped-call]
        config, trust_remote_code=False
    )
    # `trust_remote_code=False` explicitly, for the same reason the `load` half passes it:
    # the library default has changed before, and a remote-code tokenizer would execute
    # arbitrary Python inside the worker that holds the GPU lease.
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        str(tokenizer_directory), trust_remote_code=False
    )
    return module, tokenizer


def _count_parameters(module: Any) -> int | None:
    """The realised parameter count, or ``None`` when it cannot be read.

    Recorded so the budget guard's **estimate** from the config can be compared with the
    realised figure. An estimator that is wrong is invisible otherwise: the run either fits
    or dies in the allocator, and neither outcome says the estimate was off.
    """
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except (AttributeError, TypeError):
        return None
