"""What an acquisition is, and what it records  (R-02.46, R-02.47; F1 FR-001.1-12).

Three modes, one registry, no branches. ``load`` and ``create`` are implemented here;
``attach`` is BRD-02 section 5.10's and F11's, and this feature **reserves the mode name
and nothing more**. That is why :class:`Availability` exists and why there is no
``if mode == "attach"`` anywhere under ``training/`` outside ``hotupdate/``: F11 adds a
fourth *registration*, so ADR-022's fence holds by construction rather than by review.

## Nothing here imports torch

``preflight()`` reads a config file and a ``safetensors`` header; ``acquire()`` is the only
half that materialises tensors and it imports its libraries **inside the function**.
ADR-023 requires that a developer without a GPU run everything except a training step, and
in this repository the ML wheels are not even installed - so a module-level
``import torch`` would make the whole acquisition surface, including every refusal path,
unimportable in CI. The refusals are where most of this feature's defect risk lives, so
that would be the wrong half to make untestable.

## The lineage rule that looks right and is not

:attr:`AcquisitionRecord.base_model` is *"the root model this lineage descends from"* -
the schema's own words. A first load sets it to what was loaded; **a continue or a branch
loads a checkpoint, and then the thing loaded is not the root.** Setting it to the loaded
checkpoint makes every lineage walk terminate one hop short of the real root, on a tree
that validates, saves and looks entirely plausible. :func:`lineage_for_loaded_checkpoint`
is the carry-forward, and it exists as a named function so there is one place to point a
test at.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.errors import refuse


class AcquisitionMode(StrEnum):
    """How the parameters miForge trains come to be in this process (R-02.46)."""

    #: From a local Hugging Face directory or a Hub repo. **The default** (R-02.46 case 1).
    LOAD = "load"
    #: From an architecture config, randomly initialised, in this process (case 2).
    CREATE = "create"
    #: To a live serving process's resident tensors. **Section 5.10 and F11 only** (case 3).
    ATTACH = "attach"


#: R-02.46 names ``load`` the default. Stated as a constant because the API edge, the
#: registry and the hand-off note all need the same answer, and three defaults are three
#: answers.
DEFAULT_MODE = AcquisitionMode.LOAD


class SourceKind(StrEnum):
    """What the parameters were taken *from*. Narrower than the mode: a ``load`` has two."""

    LOCAL_DIR = "local_dir"
    HUB = "hub"
    CONFIG = "config"
    #: Section 5.10 only.
    ATTACHED = "attached"


@dataclass(frozen=True)
class Availability:
    """Whether a mode can be used in *this build*, and if not, what is therefore missing.

    **Three fields rather than a boolean, because R-02.48 asks for three things:** that an
    unreachable integration be reported *by name*, naming the service **and** the
    capability that is consequently unavailable. A boolean can say neither, and
    ``GET /api/v1/models/sources`` has to answer with no run in hand - so the answer must
    be available without attempting an acquisition. A mode that could only report its
    absence by failing to acquire cannot satisfy "says so before it starts".
    """

    available: bool
    #: The refusal code *and* the human reason, e.g. ``hot_update_mode_not_available``.
    reason: str | None = None
    message: str | None = None
    #: What the operator cannot do because of this. Named, not implied.
    unavailable_capabilities: tuple[str, ...] = ()


class ArchitectureConfig(BaseModel):
    """A parsed architecture config, as data. **Never executed** (TDD section 8).

    ``trust_remote_code`` is off everywhere in this feature, so an architecture that ships
    its own Python is refused rather than imported: a Hub repo would otherwise run
    arbitrary code inside the training worker, which holds the card and the lease.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The raw config mapping, exactly as read. Hashed canonically for the config identity.
    data: dict[str, Any]
    #: ``architectures[0]`` or ``model_type``, whichever the config states. Refused when
    #: neither is present - a config that does not say what it is cannot be instantiated,
    #: and guessing is how an unknown architecture reaches the allocator.
    architecture: str


class ResumeFrom(BaseModel):
    """**Which saved checkpoint an attached run resumes from** (F11 FTASKS 18.0, D52).

    Named by ``(run_id, step)`` - the addressing every F5 route already uses - or by the
    checkpoint directory's ``path``, which is resolved back to ``(run_id, step)`` through
    its own manifest and must be that run's canonical step directory under a checkpoint
    root (:func:`services.checkpoints.resume.resolve_resume_from`). **Exactly one form**,
    for the reason :class:`AcquisitionSpec` takes exactly one source: naming both and
    choosing one by precedence resumes from a checkpoint the operator did not name.

    Generic on purpose: this names *a checkpoint*, and says nothing about miLLM. What makes
    resuming an ATTACHED run safe - that the served model IS this checkpoint - is checked
    inside ``training/hotupdate`` (``resume.py``), so the fence holds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str | None = Field(default=None, min_length=1, max_length=64)
    step: int | None = Field(default=None, ge=0)
    path: Path | None = None

    @model_validator(mode="after")
    def _one_address(self) -> ResumeFrom:
        by_run = self.run_id is not None or self.step is not None
        if by_run and (self.run_id is None or self.step is None):
            raise ValueError("resume_from by run names BOTH `run_id` and `step`")
        if by_run == (self.path is not None):
            raise ValueError(
                "resume_from names exactly one checkpoint: `run_id` + `step`, or `path` - "
                "never both and never neither"
            )
        return self


class AcquisitionSpec(BaseModel):
    """What to acquire. An in-process value object, not a wire type (TDD section 5.4).

    **Exactly one source.** A payload naming two is refused rather than resolved by
    precedence: picking one trains a parameter set the operator did not ask for and reports
    success. The validator below is what makes that a property of the type instead of a
    rule each call site remembers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: AcquisitionMode = DEFAULT_MODE

    # ---------------------------------------------------------------- the three sources
    #: ``load`` from a directory. Confined to a configured model root at preflight.
    path: Path | None = None
    #: ``load`` from the Hub: the repo id, with ``revision`` below.
    repo: str | None = None
    #: What was *asked for* - a branch, a tag or a commit. **Never overwritten.**
    revision: str | None = None
    #: What it resolved to: a commit sha, filled in by preflight.
    #:
    #: Two fields, deliberately, for the same reason the run log's definition block keeps
    #: ``submitted`` and ``resolved`` apart: what the operator asked for and what it
    #: resolved to are different facts, and collapsing them loses the ability to say *"you
    #: asked for `main` and got this commit"*. Overwriting ``revision`` in place would make
    #: the record claim a commit was requested when a moving branch was.
    revision_resolved: str | None = None
    #: ``create`` from an architecture config: a path to one, or the mapping itself.
    config_path: Path | None = None
    config: dict[str, Any] | None = None

    # ------------------------------------------- the tokenizer a ``create`` run ships
    #: **Which tokenizer a created checkpoint carries at its root** (R-02.21, R-02.51,
    #: ADR-015).
    #:
    #: ``load`` needs neither of these: its tokenizer comes out of the model directory it
    #: loads, and ``local_dir.require_tokenizer_at_root`` already refuses a directory that
    #: has none. A ``create`` has no such directory - BRD-02 section 8 says miForge never
    #: trains a tokenizer, so *"a creation run is given one"* - and until 2026-10-16
    #: **nothing could give it one**: ``scratch._instantiate`` returned ``tokenizer=None``
    #: with a docstring saying *"the caller supplies it"*, and no caller in
    #: ``backend/src/`` existed. The run trained, reached its first save point and died on
    #: ``AttributeError: 'NoneType' object has no attribute 'save_pretrained'`` -
    #: a traceback, hours in, having produced nothing.
    #:
    #: **The two spellings mirror the two a ``load`` already has**, deliberately rather
    #: than inventing a parallel shape: a directory confined to a configured model root,
    #: or a Hub repo with a revision that resolves to a commit. ``RunDefinition`` declares
    #: neither, because ``model_source`` is *handed whole* to this constructor
    #: (``services/run_definition.py``'s budget step and ``training/run_driver.py`` both
    #: spread it) - so a field declared here is postable immediately, and a second
    #: declaration on the definition would be the second-source-of-truth family this
    #: project has paid for five times.
    tokenizer_path: Path | None = None
    #: The Hub repo the tokenizer files come from.
    tokenizer_repo: str | None = None
    #: What was *asked for* - a branch, a tag or a commit. **Never overwritten**, for the
    #: same reason :attr:`revision` is not.
    tokenizer_revision: str | None = None
    #: What it resolved to, filled in by preflight. See :attr:`revision_resolved`.
    tokenizer_revision_resolved: str | None = None

    # ---------------------------------------------------------------- reproducibility
    #: **The seed for a ``create`` run's random initialisation.**
    #:
    #: R-02.47 requires the content hash of the weights a run started from, for *both*
    #: modes, and no requirement in BRD-02 records a seed - so the hash of a random
    #: initialisation identified *which* init happened without letting anyone reproduce
    #: it. Recorded here by operator decision (2026-10-14), following the precedent
    #: :attr:`providers.base.GenerationParams.seed` already set in this codebase:
    #: **null is an assertion that no seed was set**, which makes the starting hash
    #: irreproducible, and that is worth recording as a fact rather than hiding.
    seed: int | None = None

    # ---------------------------------------------------------------- resume (attach only)
    #: **An attached run resuming from a checkpoint it saved** (D52, F11 FTASKS 18.0).
    #: ``attach`` only: a ``load`` names its checkpoint as ``path`` and F5's checkpoint
    #: cases already apply to it, so a second spelling there would be two answers to "which
    #: checkpoint". With it, the attached run refuses unless the served model IS that
    #: checkpoint, restores that step's trainer state, and continues from ``step + 1``.
    resume_from: ResumeFrom | None = None

    # ---------------------------------------------------------------- budget inputs
    #: Set by preflight for a ``load`` once the weight metadata has been read. The budget
    #: guard takes **exactly one** of this and the architecture config (R-02.42).
    param_count: int | None = Field(default=None, ge=1)

    # ---------------------------------------------------------------- device
    #: ``auto`` or a ``GPU-...`` UUID. Resolved by ``core.gpu``; an index or a bus address
    #: is refused *there*, by name, so this feature adds no device logic of its own.
    gpu_request: str = "auto"

    @model_validator(mode="after")
    def _exactly_one_source(self) -> AcquisitionSpec:
        """Refuse a spec that names two sources, or none for a mode that needs one."""
        load_sources = [self.path is not None, self.repo is not None]
        create_sources = [self.config_path is not None, self.config is not None]
        named = sum(load_sources) + sum(create_sources)

        if self.mode is AcquisitionMode.LOAD:
            if sum(load_sources) != 1 or any(create_sources):
                raise ValueError(
                    "a load names exactly one of `path` or `repo` (with `revision`) and no "
                    "architecture config; naming two sources is refused rather than "
                    "resolved by precedence"
                )
        elif self.mode is AcquisitionMode.CREATE:
            if sum(create_sources) != 1 or any(load_sources):
                raise ValueError(
                    "a create names exactly one of `config_path` or `config` and no model "
                    "source; a created model is initialised, never loaded"
                )
        elif named:
            # `attach` takes no source: section 5.10 attaches to tensors already resident
            # in another process. A source here means the caller expects a load.
            raise ValueError("the attach mode takes no model source")

        # Stated over the two CORE modes, not as "attach only": ADR-022's fence forbids any
        # comparison with the attach mode under `training/` outside `hotupdate/`, and a load
        # or a create is exactly where `resume_from` has no meaning.
        if self.resume_from is not None and self.mode in (
            AcquisitionMode.LOAD,
            AcquisitionMode.CREATE,
        ):
            raise ValueError(
                f"`resume_from` is the attach mode's: a {self.mode.value!r} run names the "
                f"checkpoint it starts from as `path`, where the checkpoint checks verify it"
            )

        tokenizer_sources = [self.tokenizer_path is not None, self.tokenizer_repo is not None]
        if sum(tokenizer_sources) > 1:
            raise ValueError(
                "a tokenizer reference names exactly one of `tokenizer_path` or "
                "`tokenizer_repo` (with `tokenizer_revision`); naming two is refused "
                "rather than resolved by precedence, because the checkpoint ships one "
                "tokenizer and the one not chosen would silently not be it"
            )
        if self.mode is not AcquisitionMode.CREATE and any(tokenizer_sources):
            raise ValueError(
                f"the {self.mode.value!r} mode takes no tokenizer reference: a load's "
                f"tokenizer comes from the model directory it loads (and is required to be "
                f"at its root), and an attach holds another process's. Two answers about "
                f"which tokenizer one artifact carries is worse than one"
            )
        if self.tokenizer_revision is not None and self.tokenizer_repo is None:
            raise ValueError(
                "`tokenizer_revision` names a revision of nothing: it qualifies "
                "`tokenizer_repo`. Refused rather than ignored - a silently dropped "
                "revision means a different tokenizer than the one asked for"
            )
        return self

    @property
    def architecture_config(self) -> dict[str, Any] | None:
        """The config mapping when one was given inline. ``None`` for a path.

        Deliberately **not** reading the file: this property is what the budget guard
        inspects, and a property that touched the filesystem would make the guard's
        "exactly one input" rule depend on whether a file happened to be readable.
        :func:`training.acquisition.scratch.parse_config` is the reader, and preflight
        calls it before the guard.
        """
        return self.config


class AcquisitionRecord(BaseModel):
    """What was acquired, written **before the first step** (R-02.47, FR-001.9).

    A run's lineage has to exist even if step 1 fails, which is why this is persisted at
    acquisition rather than at the first save. It is also what F9's resource strip shows:
    in the default path nothing is attached, so the honest content of that strip is *what
    model miForge holds and where it came from*.

    **No prompt, corpus or secret content ever appears here** (ADR-020). A path, a repo, a
    revision and a hash are not secrets; a Hub token is, and is never recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: AcquisitionMode
    source_kind: SourceKind

    source_path: str | None = None
    source_repo: str | None = None
    #: What was asked for, and what it resolved to. **Both**, always, for a Hub load: a
    #: branch name resolves to a commit, the commit is what was trained, and dropping the
    #: resolved value makes "the same repo and revision gives the same hash" unverifiable.
    source_revision_requested: str | None = None
    source_revision_resolved: str | None = None
    #: For a ``create``: the canonical identity of the architecture config.
    config_identity: str | None = None

    #: **Which tokenizer a created checkpoint ships**, as the reference resolved - a
    #: directory, or ``hf:<repo>@<commit>``. ``None`` for a ``load`` (whose tokenizer is
    #: the model directory's own) and for an ``attach``.
    #:
    #: Recorded because the checkpoint manifest is closed to unknown keys and cannot carry
    #: it, so without this the one fact that decides whether a created model's output is
    #: meaningful - which vocabulary it was trained against - appears nowhere at all.
    tokenizer_source: str | None = None
    #: The number of embedding rows that tokenizer needs, when it could be read. Recorded
    #: beside the config's ``vocab_size`` so a *later* widening of the config is
    #: diagnosable from the record rather than from an ``IndexError``.
    tokenizer_vocab_size: int | None = None

    #: SHA-256 of the weights miForge started from (R-02.47), through ``core.canonical``.
    starting_weights_sha256: str

    #: The seed that produced a ``create``'s initialisation, or ``None``. See
    #: :attr:`AcquisitionSpec.seed`: null is an assertion, not an omission.
    seed: int | None = None

    # ------------------------------------------------------- the two lineage fields (F5)
    #: ``checkpoint-manifest-v1.json``'s ``base_model``: *"the root model this lineage
    #: descends from"*. **F1 derives it; F5 writes it into a manifest.**
    base_model: str
    #: ``parent``: the checkpoint this one descends from, or ``None`` for a checkpoint
    #: saved directly off a base model or off a creation.
    parent: str | None = None

    #: Parameter count as counted after acquisition. Recorded because the budget guard's
    #: estimate for a ``create`` was made from a config and this is the realised figure -
    #: comparing the two is the only way to find an estimator that is wrong.
    param_count: int | None = None

    # ------------------------------------------------------- where the parameters landed
    #: ``"cuda"`` or ``"cpu"`` - **read back from the module after placement**, never
    #: copied from the request (:mod:`training.acquisition.placement`).
    #:
    #: ``None`` is not "CPU"; it is *nothing placed these parameters*, and it is the
    #: default precisely so that the omission is visible. Until 2026-10-16 nothing in
    #: ``backend/src/`` moved a model onto a GPU at all: a run took a card's lease, pinned
    #: it by UUID, and then computed on the CPU without raising anything, because
    #: ``objectives/sft.py`` builds its batch tensors on *the model's* device. A field whose
    #: absent value reads as a legitimate state would have hidden the same defect again.
    device_kind: str | None = None
    #: The card the parameters are actually on, by UUID (ADR-008) - or ``None`` for the
    #: CPU, which is ADR-023's supported developer state. **Never an index.** This is the
    #: value :attr:`AcquiredModel.device_uuid` carries and therefore the value a score's
    #: provider provenance reports (R-02.16), so it must name the device that computed,
    #: not the device that was asked for.
    device_uuid: str | None = None
    #: Which rule chose the device - ``leased_uuid``, ``sole_visible_device`` or
    #: ``cuda_unavailable``. Recorded because "on the CPU" and "on the CPU *while holding a
    #: card's lease*" are different facts and only the reason distinguishes them.
    device_placement_reason: str | None = None


@dataclass
class AcquiredModel:
    """What the step needs, and nothing more.

    ``module`` and ``tokenizer`` are ``Any`` on purpose: typing them as ``torch.nn.Module``
    would require importing torch at module scope, which ADR-023's layout forbids here. The
    step never inspects them beyond handing them to the objective and the optimizer.
    """

    module: Any
    tokenizer: Any
    record: AcquisitionRecord
    #: The UUID of the card the parameters were placed on (ADR-008). Never an index.
    device_uuid: str | None = None
    #: **How this acquisition's weights are checkpointed, when it is not F5's writer.**
    #: ``None`` for every mode in this feature: ``load`` and ``create`` hold their own
    #: tensors and their own tokenizer, so F5 writes the directory. §5.10's ``attach`` holds
    #: neither - the tensors are another process's and ``tokenizer`` above is ``None`` - so
    #: it supplies a callable here and the loop dispatches on **its presence**, never on the
    #: mode (ADR-022: *"a fourth registration, not a branch"*). Typed ``Any`` for the same
    #: reason as ``module``, and because the two save paths return different outcome types
    #: that agree on ``directory``, which is all the loop reads.
    saver: Any = None
    #: **How this acquisition's step is taken, when it is not the run type's plain step.**
    #: ``None`` for ``load`` and ``create``: their tensors are miForge's own, so
    #: ``optimizer.step()`` needs no protocol around it. §5.10's ``attach`` writes another
    #: process's resident tensors, so its update must sit inside the commit bracket
    #: (R-02.52) - and it supplies a callable with :func:`training.step.run_one`'s signature
    #: that does exactly that. The loop dispatches on **its presence**, never on the mode,
    #: the same arrangement as ``saver`` above (ADR-022).
    stepper: Any = None
    #: **What ends this acquisition when the run ends**, when something must. ``None`` for
    #: ``load`` and ``create`` (nothing to give back). §5.10's ``attach`` holds an
    #: attachment and a heartbeat thread that would otherwise outlive the run in a solo
    #: worker - holding the serving process's attachment, and with it every model load,
    #: until the worker restarts. Called once, on every exit from the loop (presence
    #: dispatch, ADR-022).
    release: Any = None
    #: **What must be proved before the first step**, when something must. ``None`` for
    #: ``load`` and ``create``. §5.10's ``attach`` rebuilt its parameters from another
    #: process's handles, so R-02.2 compares the rebuild with what the serving process
    #: answers and **refuses the run on a mismatch** - before any step, test or save. Called
    #: once by ``drive_run`` (presence dispatch, ADR-022).
    verifier: Any = None
    #: **What the acquisition needs to know about the run it serves**, called once by
    #: ``drive_run`` with ``run_id`` and the run's ``session_scope`` (presence dispatch).
    #: §5.10 uses it to persist the attachment (FR-011.5) and to publish the attachment's
    #: live state into the run record the UI reads (migration 0020).
    bind_run: Any = None
    #: **What proved a resumed run's acquired weights ARE the checkpoint it resumes**, when
    #: the spec named ``resume_from`` (D52). ``None`` otherwise. ``drive_run`` refuses to
    #: restore a resume's optimizer state unless this names the same ``(run_id, step)`` the
    #: plan verified - presence dispatch again, so no mode is compared here (ADR-022).
    resume_verified: Any = None


@runtime_checkable
class Acquirer(Protocol):
    """One acquisition mode.

    Three methods, separated for three different reasons:

    * :meth:`available` answers ``GET /api/v1/models/sources`` with **no run in hand** and
      must touch neither the network nor the GPU;
    * :meth:`preflight` is the no-GPU-needed half, so every refusal of E-1 through E-6 is
      testable on a laptop (ADR-023) and happens **before the lease and before any
      allocation** (ADR-008);
    * :meth:`acquire` is the only half that allocates.
    """

    @property
    def mode(self) -> AcquisitionMode: ...

    def available(self) -> Availability: ...

    def preflight(self, spec: AcquisitionSpec) -> AcquisitionSpec:
        """Refuse an unusable spec, or return it enriched with what preflight learned.

        Returns the spec rather than ``None`` because preflight is where the parameter
        count and the resolved revision become known, and the budget guard needs the
        former *before* :meth:`acquire` runs.
        """
        ...

    def acquire(self, spec: AcquisitionSpec) -> AcquiredModel: ...


def ml_runtime_availability(*capabilities: str) -> Availability:
    """Whether this process can actually materialise a parameter set.

    **Why ``available()`` must consult this and not simply return ``True``.** ADR-023 splits
    this feature so that preflight - the path confinement, the missing-file check, the dtype
    gate - needs no torch, and that is deliberate: it is where most of the defect risk lives
    and it must be testable on a laptop. But a *mode* is usable only if it can also acquire,
    and ``acquire()`` needs torch.

    So a mode that answered ``available=True`` on a machine without the ML wheels would be
    promising a capability it then refuses - an option a reader is told they have and cannot
    use. ADR-013's rule cuts the other way too: reporting a mode available and failing later
    is worse than reporting it unavailable by name now, because the first sends the operator
    looking for a configuration problem.

    Args:
        *capabilities: what is lost, in the operator's terms, for the refusal to name.
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        return Availability(
            available=False,
            reason="ml_runtime_unavailable",
            message=(
                f"torch and transformers are not installed in this process, so no model can "
                f"be materialised ({exc}). Everything else in miForge runs without them: "
                f"the refusal paths, the API and the record writers are all "
                f"testable here, and a training step is not."
            ),
            unavailable_capabilities=capabilities,
        )
    return Availability(available=True)


def lineage_for_loaded_checkpoint(
    *, checkpoint_name: str, checkpoint_base_model: str
) -> tuple[str, str | None]:
    """The ``(base_model, parent)`` pair for a load **of a miForge checkpoint**.

    **This is the carry-forward, and it is the whole defect.** Read literally, "a load sets
    ``base_model`` to the loaded model's identity" is right for a first load and wrong for
    every continue and every branch, because the thing loaded *is* a checkpoint. The schema
    calls the field *"the root model this lineage descends from"* and BRD-01 R-01.16 walks
    *"back to ``base_model``"*, so the literal reading terminates every walk one hop short
    of the root - a long chain of continues produces a tree rooted at the previous step.

    It validates. It saves. The tree looks plausible. Only reading F5 against F1 found it.

    Args:
        checkpoint_name: the loaded checkpoint's ``name``, which becomes ``parent``.
        checkpoint_base_model: the loaded checkpoint's **own** ``base_model``, carried
            through unchanged.

    Returns:
        ``(base_model, parent)`` for the next checkpoint F5 writes.
    """
    if not checkpoint_base_model:
        raise refuse(
            "checkpoint_base_model_missing",
            "The loaded checkpoint's manifest carries no base_model, so the lineage root "
            "cannot be carried forward. Refusing rather than substituting this checkpoint: "
            "that would root the tree at the parent and terminate every lineage walk one "
            "hop short of the model the work actually started from.",
            checkpoint=checkpoint_name,
        )
    return checkpoint_base_model, checkpoint_name


def lineage_for_first_load(*, model_identity: str) -> tuple[str, str | None]:
    """``(base_model, parent)`` for a load of something that is **not** a miForge checkpoint.

    A local Hugging Face directory or a Hub repo at a revision: that *is* the root, and
    there is no parent checkpoint (R-02.47, and the schema permits a null ``parent``).
    """
    return model_identity, None


def lineage_for_creation(*, config_identity: str) -> tuple[str, str | None]:
    """``(base_model, parent)`` for a creation: the config, and **null** (R-02.47).

    No schema change was ever needed for this - ``parent`` is nullable and ``base_model``'s
    description already names the architecture config as the value for a model created from
    nothing.
    """
    return config_identity, None
