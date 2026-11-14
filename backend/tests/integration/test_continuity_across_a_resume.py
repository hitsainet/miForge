"""The loss curve is continuous across a resume  (AC-005.7 / BRD-02 acceptance criterion 4).

**Task 7.7.** Criterion 4 reads *"a run continued from a saved checkpoint with restored
optimizer state produces a loss curve continuous with the original."* That is a claim about
three things agreeing - the weights, the optimizer's moment buffers, and the data order - so
the test runs the **real** writer and the **real** reader over a **real** optimizer step and
compares two curves rather than inspecting one.

## What is real here, and what is fixed

Real: :func:`training.optimizer.build` (F1's), :func:`training.step.run_one` (F1's, R-02.12's
three stages), :func:`training.trainer_state.write` / ``read`` (F1's, R-02.15's atomic
layout), the production objective registry, and the real ``sft`` cross-entropy. The optimizer
state makes a genuine round trip through ``torch.save``/``torch.load`` on disk - *fixtures
whose fields agree by construction are how a suite stays green over a real bug* (ADR-016).

Fixed, **in the test, by construction rather than by a seed**: the initialisation
(``manual_seed`` immediately before the module is built, asserted identical), and the batch
sequence (a finite cycle of three hand-written batches with an explicit cursor). AGENT.md is
explicit that the same ``seed`` reproduces a ``create`` run's starting weights and **does
not** cover batch/data order, generation sampling, the scoring subsample or cuDNN/kernel
nondeterminism - BRD-02 §9 / OQ-2 records that **no requirement records a seed at all**. So
nothing here is left to a seed that would have to cover it: the data order is a parameter of
the source, and a resumed run is handed the cursor it left off at.

**The data order had to be resumed explicitly, and getting it wrong was instructive.** The
first draft rebuilt the batch source from scratch for the continued half, which restarted the
cycle - so the continued run trained on batch *a* where the reference had trained on batch
*c*. The curves diverged by 0.6 with a perfectly restored optimizer. That failure is
indistinguishable from a broken restore and it is **the thing the real product does not
enforce**: no source shuffles today, so order is *incidentally* reproducible and nothing
pins it (AGENT.md, "The seed, applied"). Here it is pinned by the cursor.

## The tolerance, and why it is this number

``CONTINUITY_TOLERANCE = 1e-5``, **absolute, in loss units** (ADR-012: a number names its
scale - the losses here run about 3.0 to 5.4, so this is ~2e-6 relative).

* **Measured noise floor: 0.0.** The uninterrupted curve is bit-identical across repeated
  runs in one process and across ``torch.set_num_threads`` of 8, 2 and 1
  (:meth:`TestTheToleranceIsEarned.test_the_measured_noise_floor_is_far_below_the_tolerance`
  re-measures it rather than citing this sentence). A restored resume reproduces the
  reference **bit for bit** on this workstation.
* **So why not assert ``== 0.0``?** Because bit-exactness is a property of this environment,
  not a promise of the system. ``torch`` does not guarantee bitwise reproducibility across
  versions, BLAS builds or reduction orders, and BRD-02 §9 disclaims kernel nondeterminism
  outright. Asserting equality would turn an unrelated BLAS change into a red that reads as
  a correctness regression, and - worse - it would **overclaim a guarantee miForge does not
  make**. An overclaimed tolerance is worse than none.
* **So why not looser?** Because a tolerance is only worth asserting if it can still see the
  defect it exists to catch. Starting the continued half with a **fresh** optimizer instead
  of the restored one moves the curve by **~0.40 in loss units** - about **40,000x** this
  tolerance. :class:`TestTheToleranceIsEarned` asserts that margin in the suite, so the
  number is earned by measurement rather than chosen for comfort.

## The join point alone proves nothing, which is the sharpest finding here

The **first** resumed loss is identical even with the optimizer state thrown away, because a
loss is computed from the weights *before* that step's update - and the weights are the
checkpoint's either way. A continuity test that checked only the step at the seam would have
passed against a completely broken restore.
:meth:`TestTheToleranceIsEarned.test_the_join_point_alone_cannot_see_a_broken_restore`
asserts that, so the assertion window can never be narrowed back to one step by someone
tidying up.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

# `torch` is a DECLARED dependency (`torch>=2.4,<3`) and CI installs the CPU wheel, so this
# is a plain import and NOT an `importorskip`. AGENT.md, 2026-10-14: *"a test whose outcome
# turns on whether an import succeeds is a test about the environment, not the code"* - 13
# tests passed locally by taking an absent-torch branch while CI ran the real path and went
# red. A missing declared dependency must fail loudly here, not skip.
import torch

from core.config import Settings
from core.errors import Refusal
from services.checkpoints.paths import checkpoint_dir
from training import optimizer as optimizer_module
from training import run_log, tasks, trainer_state
from training import step as step_module
from training.acquisition.base import (
    AcquiredModel,
    AcquisitionMode,
    AcquisitionRecord,
    SourceKind,
)
from training.run_config import RunConfig
from training.signals.base import BatchCounts, SupervisedExample, TrainingBatch

# Imported, not re-spelled: one definition of "a directory that is both a standard Hugging
# Face model and a miForge checkpoint", whose manifest's `weights_sha256` is computed by the
# production hasher. A second copy here would be a second definition of the fixture the
# whole verification turns on.
from .test_checkpoint_preflight_on_the_submit_path import load_spec, loadable_checkpoint

#: Absolute, in loss units. See the module docstring for how it was chosen.
CONTINUITY_TOLERANCE = 1e-5

#: The resumed curve must be at least this many multiples of the tolerance closer to the
#: reference than a fresh-optimizer resume is. Asserted, so the tolerance stays meaningful.
DISCRIMINATION_FACTOR = 100.0

#: Char-level ids, so every example's token ids are readable and no tokenizer download is
#: needed. 128 covers every character used below.
VOCAB = 128

#: Where the save happens. Three steps before and three after: more than one step on the
#: resumed side is **load-bearing**, because the step at the seam cannot see the defect.
SAVE_AT_STEP = 3
STEPS_AFTER = 3


class _Tokenizer:
    """One token per character. Copied in shape from ``test_objective_sft.py``'s double.

    Deliberately not a real Hugging Face tokenizer: this test is about the optimizer state
    round trip, and a downloaded tokenizer would make the test depend on the network.
    """

    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False, "special tokens must be off between the halves"
        return [ord(character) for character in text]


class _TinyLM(torch.nn.Module):
    """A real ``nn.Module`` with real fp32 parameters, in the shape the ``sft`` loss wants.

    ``forward`` returns logits ``[batch, tokens, vocab]`` directly rather than a Hugging Face
    output object - :func:`training.objectives.sft._logits` accepts both and tries nothing
    else, which is the documented contract.

    ``gradient_checkpointing_enable`` exists because :func:`training.optimizer.build` refuses
    a model without it (``gradient_checkpointing_unavailable``), and §6's memory budget
    assumes it is on. It **records** the call rather than silently accepting it, so a test
    asserting the real path was taken can check.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, 8)
        self.head = torch.nn.Linear(8, VOCAB)
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self, **_: Any) -> None:
        self.checkpointing_enabled = True

    def forward(self, *, input_ids: Any, attention_mask: Any = None) -> Any:
        del attention_mask  # every example here is the same length within its batch
        return self.head(self.embed(input_ids))


#: The initialisation seed. Applied **immediately before** the module is constructed, so
#: nothing between the two draws from the generator. `test_the_initialisation_is_identical`
#: asserts the two halves really do start from the same weights rather than assuming it.
INIT_SEED = 1234


def _fixed_init_model() -> _TinyLM:
    torch.manual_seed(INIT_SEED)
    return _TinyLM()


def _batch(tag: str, pairs: tuple[tuple[str, str], ...]) -> TrainingBatch:
    """One real :class:`training.signals.base.TrainingBatch` of supervised examples."""
    return TrainingBatch(
        objective="sft",
        record={"source": f"fixed-cycle:{tag}"},
        examples=tuple(
            SupervisedExample(prompt_id=f"{tag}-{index}", prompt=prompt, completion=completion)
            for index, (prompt, completion) in enumerate(pairs)
        ),
        counts=BatchCounts(admitted=len(pairs), requested=len(pairs)),
        tokenizer=_Tokenizer(),
        # Named explicitly: the loss refuses a batch that names no policy (E-5), and
        # whole-sequence training on prompt/completion data is a different objective.
        masking_policy="completion_only",
    )


#: Three distinct batches, cycled. **Three rather than one** because with a single repeated
#: batch the gradient direction never changes, AdamW's moments reconverge within one step of
#: a reset, and the fresh-optimizer defect becomes invisible (measured: 4.8e-6, i.e. *inside*
#: any honest tolerance). A changing gradient is what makes the moment buffers carry
#: information, which is what the requirement is about. The cycle is finite and fixed, so it
#: is deterministic by construction and involves no shuffling and no RNG.
_CYCLE: tuple[TrainingBatch, ...] = (
    _batch("a", (("hello", "world"), ("abc", "defg"))),
    _batch("b", (("zzz", "qqqq"), ("mn", "opq"))),
    _batch("c", (("qrst", "uv"), ("wx", "yz"))),
)


class _FixedCycleSource:
    """F1's ``BatchSource`` over a fixed cycle, with an **explicit, resumable cursor**.

    The cursor is the step index. A resumed run is constructed with the cursor it left off
    at, so the continued half trains on the same batches the uninterrupted reference saw at
    those steps. Rebuilding the source at zero instead produces a 0.6 divergence with a
    perfectly restored optimizer - see the module docstring.
    """

    def __init__(self, cursor: int = 0) -> None:
        self.cursor = cursor

    def next(self) -> TrainingBatch:
        batch = _CYCLE[self.cursor % len(_CYCLE)]
        self.cursor += 1
        return batch


def _cfg() -> RunConfig:
    """R-02.13's run configuration, with **32-bit** optimizer states.

    The default is ``8bit``, which needs ``bitsandbytes`` - an extra with no pure-CPU wheel
    everywhere, which is exactly why it is an extra. ``build()`` refuses
    ``optimizer_precision_unavailable`` on the default rather than downgrading silently, so
    the 32-bit path is the one a developer without the extra can execute (ADR-023).

    ``learning_rate`` is 5e-2 - large for a real run and chosen deliberately: the question
    being asked is whether the optimizer's *state* changes the trajectory, and at 1e-3 over a
    model this small the whole effect is 5e-6, which no honest tolerance can see.
    """
    return RunConfig(
        learning_rate=5e-2,
        warmup_steps=0,
        batch_size=2,
        optimizer_state_precision="32bit",
        seed=7,
    )


def _take_steps(model: Any, optimizer: Any, count: int, *, from_step: int) -> list[float]:
    """``count`` real steps through F1's ``run_one``, returning the losses.

    ``objectives=None`` on purpose: ``run_one`` then resolves the **production** registry
    (:func:`training.step.default_objective_registry`), so this exercises the real ``sft``
    objective as registered rather than a double. A broken registration shows up here.
    """
    source = _FixedCycleSource(cursor=from_step)
    losses: list[float] = []
    for _ in range(count):
        outcome = step_module.run_one(model, optimizer, source, objectives=None)
        assert outcome.finite, f"a non-finite step at cursor {source.cursor}: {outcome!r}"
        assert outcome.objective == "sft"
        losses.append(outcome.loss)
    return losses


def _reference_curve(total_steps: int) -> list[float]:
    """The uninterrupted run: one model, one optimizer, no save, no resume."""
    model = _fixed_init_model()
    optimizer, _identity = optimizer_module.build(model, _cfg())
    assert (
        model.checkpointing_enabled
    ), "build() did not turn gradient checkpointing on, so this is not F1's real optimizer path"
    return _take_steps(model, optimizer, total_steps, from_step=0)


def _run_until_save(states: Path, checkpoints: Path, run_id: str) -> tuple[list[float], Any]:
    """Train to :data:`SAVE_AT_STEP`, write trainer state through F1's **real** writer.

    Returns the losses and a deep copy of the model's ``state_dict`` - which stands in for
    the weights half of the checkpoint. The weights are copied rather than written through
    F5's ``safetensors`` writer because what is under test is the **optimizer state** round
    trip; F5's byte-identical weights round trip is AC-005.6's, covered in
    ``test_continue_and_branch.py``, and duplicating it here would test it twice and the
    optimizer once.
    """
    model = _fixed_init_model()
    optimizer, identity = optimizer_module.build(model, _cfg())
    losses = _take_steps(model, optimizer, SAVE_AT_STEP, from_step=0)

    published = trainer_state.write(
        run_id=run_id,
        step=SAVE_AT_STEP,
        optimizer=optimizer,
        identity=identity,
        root=states,
        checkpoint_root=checkpoints,
    )
    assert published.is_dir(), "F1's writer reported a directory it did not publish"
    return losses, copy.deepcopy(model.state_dict())


def _resume(
    weights: Any,
    states: Path,
    run_id: str,
    *,
    restore_optimizer: bool,
) -> list[float]:
    """Continue from the saved step. ``restore_optimizer=False`` is the defect, deliberately.

    The ``False`` branch exists so the suite carries its own negative control: it is what a
    dropped ``load_state_dict`` - or F1's production loop as it stands today - actually does.
    """
    model = _fixed_init_model()
    model.load_state_dict(weights)
    optimizer, identity = optimizer_module.build(
        model, _cfg(), state="restored" if restore_optimizer else "fresh"
    )

    if restore_optimizer:
        # THROUGH F1'S REAL READER, off the real directory on disk. `read` refuses an
        # incomplete directory by name, so a torn state cannot silently become a fresh one.
        restored = trainer_state.read(run_id, SAVE_AT_STEP, root=states)
        optimizer.load_state_dict(restored["optimizer"])
        assert optimizer.state_dict()["state"], (
            "the optimizer has no state after the restore, so load_state_dict did nothing "
            "and this test would be measuring a fresh optimizer while calling it restored"
        )
    assert identity.state == ("restored" if restore_optimizer else "fresh")

    return _take_steps(model, optimizer, STEPS_AFTER, from_step=SAVE_AT_STEP)


def _max_gap(left: list[float], right: list[float]) -> float:
    assert len(left) == len(right) and left, "two curves of equal, non-zero length"
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


class TestTheCurveIsContinuousAcrossAResume:
    """AC-005.7 / BRD-02 acceptance criterion 4, asserted end to end."""

    def test_a_resumed_run_rejoins_the_uninterrupted_curve(self, tmp_path: Path) -> None:
        states = tmp_path / "trainer-state"
        checkpoints = tmp_path / "checkpoints"
        states.mkdir()
        checkpoints.mkdir()

        reference = _reference_curve(SAVE_AT_STEP + STEPS_AFTER)
        before, weights = _run_until_save(states, checkpoints, "run-continuity")

        # The first half must itself be on the reference curve, or the comparison below is
        # against a run that had already diverged for an unrelated reason.
        assert _max_gap(before, reference[:SAVE_AT_STEP]) <= CONTINUITY_TOLERANCE

        after = _resume(weights, states, "run-continuity", restore_optimizer=True)

        gap = _max_gap(after, reference[SAVE_AT_STEP:])
        assert gap <= CONTINUITY_TOLERANCE, (
            f"the resumed loss curve left the uninterrupted one by {gap:.3e} in loss units, "
            f"over a tolerance of {CONTINUITY_TOLERANCE:.0e}. BRD-02 acceptance criterion 4 "
            f"requires continuity. reference={reference[SAVE_AT_STEP:]!r} resumed={after!r}"
        )

    def test_the_whole_resumed_window_is_compared_not_just_the_seam(self, tmp_path: Path) -> None:
        """The window is :data:`STEPS_AFTER` steps, and that is load-bearing.

        A one-step window cannot see a broken restore - asserted in
        :class:`TestTheToleranceIsEarned` - so this pins the window's size against a later
        simplification that would make the test vacuous while leaving it green.
        """
        del tmp_path
        assert STEPS_AFTER >= 2, (
            "a resumed window of one step is satisfied by a completely discarded optimizer "
            "state, because the loss at the seam is a function of the weights alone"
        )

    def test_the_initialisation_is_identical_by_construction(self) -> None:
        """Both halves start from the same weights, **checked** rather than assumed.

        If ``manual_seed`` stopped fixing the init - a different torch, a module that draws
        in a different order - every curve comparison in this file would be comparing two
        different runs and the continuity assertion would be meaningless.
        """
        first = _fixed_init_model().state_dict()
        second = _fixed_init_model().state_dict()
        assert first.keys() == second.keys()
        for key in first:
            assert torch.equal(first[key], second[key]), f"{key} differs between two inits"


class TestTheToleranceIsEarned:
    """The tolerance is only worth asserting if it can still see the defect it is for.

    This class is the suite's own **permanent negative control** for task 7.7: it runs the
    resume with the optimizer state discarded and requires the curve to break by a wide
    margin. A manual mutation sweep proves the control once; this proves it on every run.
    """

    def test_discarding_the_optimizer_state_breaks_the_curve(self, tmp_path: Path) -> None:
        states = tmp_path / "trainer-state"
        checkpoints = tmp_path / "checkpoints"
        states.mkdir()
        checkpoints.mkdir()

        reference = _reference_curve(SAVE_AT_STEP + STEPS_AFTER)
        _before, weights = _run_until_save(states, checkpoints, "run-fresh")

        restored = _resume(weights, states, "run-fresh", restore_optimizer=True)
        discarded = _resume(weights, states, "run-fresh", restore_optimizer=False)

        restored_gap = _max_gap(restored, reference[SAVE_AT_STEP:])
        discarded_gap = _max_gap(discarded, reference[SAVE_AT_STEP:])

        assert restored_gap <= CONTINUITY_TOLERANCE
        assert discarded_gap > CONTINUITY_TOLERANCE * DISCRIMINATION_FACTOR, (
            f"a resume that threw the optimizer state away moved the curve by only "
            f"{discarded_gap:.3e}, which is not {DISCRIMINATION_FACTOR:.0f}x the "
            f"{CONTINUITY_TOLERANCE:.0e} tolerance. The tolerance can no longer distinguish "
            f"a restored optimizer from a fresh one, so the continuity assertion above is "
            f"passing for a reason unrelated to the requirement. Widen the gradient "
            f"variation (the batch cycle) or the learning rate until it can."
        )

    def test_the_join_point_alone_cannot_see_a_broken_restore(self, tmp_path: Path) -> None:
        """The sharpest finding in this file, written down so it cannot be re-lost.

        The loss at the resumed step is computed from the weights **before** that step's
        update, and the weights are the checkpoint's whether or not the optimizer state came
        back. So the seam matches in both cases and a test that looked only there would have
        reported continuity for a resume that had discarded everything.
        """
        states = tmp_path / "trainer-state"
        checkpoints = tmp_path / "checkpoints"
        states.mkdir()
        checkpoints.mkdir()

        _before, weights = _run_until_save(states, checkpoints, "run-seam")
        restored = _resume(weights, states, "run-seam", restore_optimizer=True)
        discarded = _resume(weights, states, "run-seam", restore_optimizer=False)

        assert abs(restored[0] - discarded[0]) <= CONTINUITY_TOLERANCE, (
            "the first resumed loss now differs between a restored and a discarded "
            "optimizer. That is a stronger product than when this was written - but the "
            "assertion windows elsewhere in this file were sized on the assumption that it "
            "does NOT differ, so re-read them before changing this."
        )
        assert abs(restored[-1] - discarded[-1]) > CONTINUITY_TOLERANCE, (
            "the curves never separate at all, so this file is not measuring the optimizer "
            "state's effect on the trajectory"
        )

    def test_the_measured_noise_floor_is_far_below_the_tolerance(self) -> None:
        """Re-measure the floor rather than citing the docstring's figure.

        A number recorded in prose goes stale the moment anything under it changes - this
        project has already recorded two wrong md5s for a schema for exactly that reason. So
        the floor is measured here, across thread counts, because reduction order is the
        likeliest source of float32 drift on CPU.
        """
        original = torch.get_num_threads()
        try:
            curves: list[list[float]] = []
            for threads in (original, 2, 1):
                torch.set_num_threads(threads)
                curves.append(_reference_curve(SAVE_AT_STEP + STEPS_AFTER))
        finally:
            torch.set_num_threads(original)

        floor = max(_max_gap(curve, curves[0]) for curve in curves[1:])
        assert floor <= CONTINUITY_TOLERANCE / DISCRIMINATION_FACTOR, (
            f"the uninterrupted curve is only reproducible to {floor:.3e}, which is within "
            f"{DISCRIMINATION_FACTOR:.0f}x of the {CONTINUITY_TOLERANCE:.0e} continuity "
            f"tolerance. The tolerance is then measuring environment noise rather than the "
            f"restore, and must be re-derived from this floor rather than kept."
        )


# =======================================================================================
# THE SAME ASSERTION, THROUGH THE PRODUCTION LOOP  (gap 1, closed 2026-10-16)
# =======================================================================================
#
# What stood here until 2026-10-16 was `TestTheProductionLoopDoesNotRestoreYet`, a tripwire
# asserting that `training.tasks.drive_run` built its optimizer with the default
# `state="fresh"` and that **no** module under `backend/src/` called `load_state_dict` on an
# optimizer. It was correct when written: `resume_plan` decided `restored`, said so in the
# API response and in the manifest, and the worker then built a fresh optimizer - so
# acceptance criterion 4 was false in production while the mechanism tests above proved it
# in isolation. The tripwire fired the moment the wiring landed and said so by name.
#
# It is **replaced rather than deleted**, by the positive form it asked for: the continuity
# assertion driven through `drive_run` itself. A grep for an absence can only ever say the
# defect is gone; an assertion through production says the capability is there, and it goes
# red if the wiring is removed - which is what ADR-016 #1 asks for and what the tripwire
# could not provide.
#
# ## The one substitution, stated rather than discovered later
#
# `tasks.acquire_for_run` is doubled. Everything else is production: `verify_checkpoint_source`
# runs F5's real plan over a real checkpoint on disk, `planned_restore_step` reads that
# plan's verdict, `optimizer_module.build` builds the real optimizer,
# `restore_optimizer_state` loads through F1's real reader, `loop.run` is F6's one repeat,
# `step.run_one` takes the real steps against the real `sft` objective, and the losses are
# read back out of the **run log** rather than from a return value.
#
# The acquirer is doubled because `_TinyLM` is an `nn.Module`, not a `transformers`
# checkpoint: the real local-directory acquirer calls `from_pretrained`, which needs a model
# `transformers` recognises, which needs either a download or a 100-line config fixture -
# and either would make this test about `transformers` rather than about the restore. The
# module the double returns carries **the checkpoint's own weights** (the state dict as it
# stood at the save), so the comparison is honest: the same parameters the reference had at
# that step, plus whatever the loop did to the optimizer. The weights half of a continue -
# that F5's writer and reader round-trip bytes identically - is AC-005.6's and is covered in
# `test_continue_and_branch.py`; duplicating it here would test it twice and the restore
# once.

RESUMED_RUN = "run-drive-continuity"


def _production_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point the configured roots at ``tmp_path``. Yields ``(checkpoints, trainer_state)``.

    The settings are repointed rather than passed, because
    :func:`training.tasks.verify_checkpoint_source` resolves the checkpoint root from
    settings - that is how the production worker finds it, and a test that passed an
    override would not exercise the lookup.
    """
    checkpoints = tmp_path / "checkpoints"
    parts = tmp_path / "parts"
    states = tmp_path / "trainer-state"
    for directory in (checkpoints, parts, states):
        directory.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(
            _env_file=None,  # type: ignore[call-arg]
            checkpoint_root=checkpoints,
            parts_root=parts,
            trainer_state_root=states,
        ),
    )
    return checkpoints, states


def _planted_checkpoint(checkpoints: Path, states: Path, run_id: str) -> tuple[list[float], Any]:
    """Train to the save point, then publish **both** halves of a continuable checkpoint.

    * the trainer state, through F1's real :func:`training.trainer_state.write`;
    * the weights checkpoint directory, through
      ``test_checkpoint_preflight_on_the_submit_path.loadable_checkpoint`` - imported rather
      than re-spelled, so there is one definition of "a directory that is both a standard HF
      model and a miForge checkpoint" and its manifest's ``weights_sha256`` is computed by
      the production hasher.

    Returns the first half's losses and the model state dict at the save point.
    """
    losses, weights = _run_until_save(states, checkpoints, run_id)
    loadable_checkpoint(checkpoints, run_id=run_id, step=SAVE_AT_STEP)
    return losses, weights


def _acquired_double(weights: Any) -> Any:
    """The acquirer's result: the checkpoint's own weights in a real ``nn.Module``."""
    model = _fixed_init_model()
    model.load_state_dict(weights)
    return AcquiredModel(
        model,
        None,  # no tokenizer: `save_every=None` below, so no weights save is attempted
        AcquisitionRecord(
            mode=AcquisitionMode.LOAD,
            source_kind=SourceKind.LOCAL_DIR,
            source_path=str(checkpoint_dir(RESUMED_RUN, SAVE_AT_STEP)),
            starting_weights_sha256="b" * 64,
            base_model="local:base@abcdef0123456789",
            parent=None,
            param_count=sum(p.numel() for p in model.parameters()),
        ),
    )


def _losses_from_the_run_log(log_root: Path, run_id: str) -> list[float]:
    """The resumed curve, read back out of the **authoritative** record (ADR-004).

    Off the log rather than off ``RunResult``, because the log is what an operator and the
    tree screen read. A restore that produced a continuous curve and recorded something else
    would be a different defect, and this reads the thing that would carry it.
    """
    path = run_log.run_log_path(tasks.run_log_root(log_root), run_id)
    return [block["loss"] for block in run_log.read_blocks(path, block="step")]


def _drive_the_resume(
    *,
    tmp_path: Path,
    checkpoints: Path,
    states: Path,
    weights: Any,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str = RESUMED_RUN,
) -> Any:
    """``drive_run`` over the planted checkpoint, from ``SAVE_AT_STEP`` to the end."""
    monkeypatch.setattr(tasks, "acquire_for_run", lambda spec: _acquired_double(weights))
    return tasks.drive_run(
        run_id=run_id,
        spec=load_spec(checkpoint_dir(run_id, SAVE_AT_STEP, root=checkpoints)),
        cfg=_cfg(),
        batch_source=_FixedCycleSource(cursor=SAVE_AT_STEP),
        objectives=None,  # the PRODUCTION registry, so a broken `sft` registration shows up
        max_steps=SAVE_AT_STEP + STEPS_AFTER,  # an absolute ceiling, not a count of more
        start_step=SAVE_AT_STEP,
        save_every=None,  # no save point: this test is about the restore, not the write
        log_root=tmp_path,
        trainer_state_root=states,
    )


class TestAcceptanceCriterion4ThroughTheProductionLoop:
    """AC4 end to end through :func:`training.tasks.drive_run` - the wiring, not the parts.

    Three assertions, and the third is the one that makes the first two mean anything:
    the curve is continuous; the recorded identity says ``restored``; and with the trainer
    state absent the **same** call produces a broken curve and records ``fresh``. Without
    the third, a `drive_run` that restored unconditionally - or one that ignored the plan -
    would pass.
    """

    def test_a_continue_driven_by_drive_run_rejoins_the_uninterrupted_curve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoints, states = _production_roots(tmp_path, monkeypatch)
        reference = _reference_curve(SAVE_AT_STEP + STEPS_AFTER)
        before, weights = _planted_checkpoint(checkpoints, states, RESUMED_RUN)
        assert _max_gap(before, reference[:SAVE_AT_STEP]) <= CONTINUITY_TOLERANCE

        result = _drive_the_resume(
            tmp_path=tmp_path,
            checkpoints=checkpoints,
            states=states,
            weights=weights,
            monkeypatch=monkeypatch,
        )

        assert result.steps_completed == STEPS_AFTER
        assert result.last_step == SAVE_AT_STEP + STEPS_AFTER
        after = _losses_from_the_run_log(tmp_path, RESUMED_RUN)
        assert len(after) == STEPS_AFTER, f"the run log holds {len(after)} steps, not {STEPS_AFTER}"

        gap = _max_gap(after, reference[SAVE_AT_STEP:])
        assert gap <= CONTINUITY_TOLERANCE, (
            f"the resumed loss curve left the uninterrupted one by {gap:.3e} in loss units, "
            f"over a tolerance of {CONTINUITY_TOLERANCE:.0e} - driven through "
            f"`training.tasks.drive_run`, which is the path a run actually takes. BRD-02 "
            f"acceptance criterion 4 requires continuity. If the mechanism tests above are "
            f"green and this is red, the RESTORE IS NOT WIRED: check "
            f"`planned_restore_step` and `restore_optimizer_state` in training/tasks.py. "
            f"reference={reference[SAVE_AT_STEP:]!r} resumed={after!r}"
        )

    def test_the_manifest_identity_records_restored_for_a_restored_continue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The recorded provenance, asserted on the **block F5 writes**, not on a local.

        R-02.22 requires the distinction be recorded, because a fresh optimizer on a mid-run
        checkpoint produces a loss discontinuity that is otherwise unexplained. The identity
        is intercepted at :func:`training.trainer_state.write`, which is the one place
        ``drive_run`` hands it on at a save point.
        """
        checkpoints, states = _production_roots(tmp_path, monkeypatch)
        _before, weights = _planted_checkpoint(checkpoints, states, RESUMED_RUN)

        seen: list[dict[str, Any]] = []
        real_write = trainer_state.write

        def _spy(**kwargs: Any) -> Path:
            identity = kwargs["identity"]
            seen.append(identity.as_manifest_block())
            return real_write(**kwargs)

        monkeypatch.setattr(tasks.trainer_state, "write", _spy)
        monkeypatch.setattr(tasks, "acquire_for_run", lambda spec: _acquired_double(weights))
        tasks.drive_run(
            run_id=RESUMED_RUN,
            spec=load_spec(checkpoint_dir(RESUMED_RUN, SAVE_AT_STEP, root=checkpoints)),
            cfg=_cfg(),
            batch_source=_FixedCycleSource(cursor=SAVE_AT_STEP),
            objectives=None,
            max_steps=SAVE_AT_STEP + STEPS_AFTER,
            start_step=SAVE_AT_STEP,
            # A save point, so the identity reaches a writer. No `signal_source`, so the
            # WEIGHTS save is skipped by name (`checkpoint_save_skipped_no_provenance`) and
            # the run stays continuable - which is the existing documented behaviour and the
            # reason this needs no tokenizer.
            save_every=STEPS_AFTER,
            log_root=tmp_path,
            trainer_state_root=states,
        )

        assert len(seen) == 1, f"expected one trainer-state write, got {len(seen)}"
        assert seen[0]["state"] == "restored", (
            f"the optimizer identity recorded {seen[0]['state']!r} for a continue whose "
            f"trainer state was restored. A manifest that claims `fresh` over a restored "
            f"optimizer understates the run; one that claims `restored` over a fresh "
            f"optimizer makes a real discontinuity unexplainable. Both are defects."
        )
        assert seen[0]["name"] == _cfg().optimizer_name()

    def test_with_no_trainer_state_the_same_call_is_fresh_and_the_curve_breaks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The permanent negative control for the wiring, inside the suite.

        The plan - not this test, and not ``drive_run`` - decides. With the trainer-state
        directory removed, :func:`services.checkpoints.resume.resume_plan` returns ``fresh``
        with ``trainer_state_absent``, so a loop that restored unconditionally would be
        caught here, and so would one that ignored the verdict and always built fresh (the
        first assertion in this class would then be red instead).
        """
        import shutil

        checkpoints, states = _production_roots(tmp_path, monkeypatch)
        reference = _reference_curve(SAVE_AT_STEP + STEPS_AFTER)
        _before, weights = _planted_checkpoint(checkpoints, states, RESUMED_RUN)

        # The ONE difference from the test above. Removed after the write rather than never
        # written, so the checkpoint, the manifest and the hash are byte-for-byte the same
        # and the only variable is the optimizer state.
        shutil.rmtree(states / RESUMED_RUN)

        seen: list[str] = []
        real_build = tasks.optimizer_module.build

        def _spy(model: Any, cfg: Any, *, state: str = "fresh") -> Any:
            seen.append(state)
            return real_build(model, cfg, state=state)

        monkeypatch.setattr(tasks.optimizer_module, "build", _spy)
        _drive_the_resume(
            tmp_path=tmp_path,
            checkpoints=checkpoints,
            states=states,
            weights=weights,
            monkeypatch=monkeypatch,
        )

        assert seen == ["fresh"], (
            f"the optimizer was built with state={seen!r} for a continue whose trainer "
            f"state does not exist. `planned_restore_step` must return None when the plan "
            f"says `trainer_state_absent`, or the manifest asserts a restore that never "
            f"happened."
        )
        after = _losses_from_the_run_log(tmp_path, RESUMED_RUN)
        gap = _max_gap(after, reference[SAVE_AT_STEP:])
        assert gap > CONTINUITY_TOLERANCE * DISCRIMINATION_FACTOR, (
            f"a resume with NO optimizer state moved the production curve by only "
            f"{gap:.3e}, which is not {DISCRIMINATION_FACTOR:.0f}x the "
            f"{CONTINUITY_TOLERANCE:.0e} tolerance. The continuity assertion above is then "
            f"passing for a reason unrelated to the restore."
        )

    def test_the_restore_reads_the_planned_step_through_F1s_own_reader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The call, the payload **and** the call count (ADR-016 #1).

        *"Was called" passes against a call sending the wrong arguments* - and the wrong
        argument here is a plausible one: restoring the step the loop is **about to take**
        rather than the step the checkpoint was saved at would be off by one, would restore
        nothing (that directory does not exist), and would refuse at a point that reads like
        a missing checkpoint.
        """
        checkpoints, states = _production_roots(tmp_path, monkeypatch)
        _before, weights = _planted_checkpoint(checkpoints, states, RESUMED_RUN)

        reads: list[tuple[str, int, Path | None]] = []
        real_read = trainer_state.read

        def _spy(run_id: str, step: int, *, root: Path | None = None) -> Any:
            reads.append((run_id, step, root))
            return real_read(run_id, step, root=root)

        monkeypatch.setattr(tasks.trainer_state, "read", _spy)
        _drive_the_resume(
            tmp_path=tmp_path,
            checkpoints=checkpoints,
            states=states,
            weights=weights,
            monkeypatch=monkeypatch,
        )

        assert reads == [(RESUMED_RUN, SAVE_AT_STEP, states)], (
            f"the trainer state was read as {reads!r}. Exactly one read is expected, for "
            f"the run that owns the state and the step the plan resolved - once per run, "
            f"not once per step, and never for the step about to be taken."
        )


class TestTheWorkerVerifiesTheCheckpointBeforeReadingATensor:
    """Gap 2, from the loop's side: ``drive_run`` refuses a corrupted checkpoint.

    The submit-path half of this is
    ``test_checkpoint_preflight_on_the_submit_path.py``; this is the half that matters to a
    *resume*, because a continue is the case where the bytes on disk were written hours ago
    and the manifest is the only thing that can say they are still the bytes that were
    checked.
    """

    def test_a_corrupted_checkpoint_refuses_before_the_acquirer_is_reached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoints, states = _production_roots(tmp_path, monkeypatch)
        _before, weights = _run_until_save(states, checkpoints, RESUMED_RUN)
        loadable_checkpoint(
            checkpoints, run_id=RESUMED_RUN, step=SAVE_AT_STEP, corrupt_weights=True
        )

        acquired: list[Any] = []

        def _should_not_run(spec: Any) -> Any:
            acquired.append(spec)
            return _acquired_double(weights)

        monkeypatch.setattr(tasks, "acquire_for_run", _should_not_run)

        with pytest.raises(Refusal) as caught:
            tasks.drive_run(
                run_id=RESUMED_RUN,
                spec=load_spec(checkpoint_dir(RESUMED_RUN, SAVE_AT_STEP, root=checkpoints)),
                cfg=_cfg(),
                batch_source=_FixedCycleSource(cursor=SAVE_AT_STEP),
                objectives=None,
                max_steps=SAVE_AT_STEP + STEPS_AFTER,
                start_step=SAVE_AT_STEP,
                save_every=None,
                log_root=tmp_path,
                trainer_state_root=states,
            )

        assert caught.value.code == "checkpoint_hash_mismatch", (
            f"the loop raised {caught.value.code!r} rather than refusing the hash mismatch "
            f"by name. A bare raise here would read as protection: this asserts the CODE."
        )
        assert acquired == [], (
            "the acquirer was reached for a checkpoint whose bytes do not hash to its "
            "manifest. R-02.21 requires the verification BEFORE a single tensor is used, "
            "and `acquire_for_run` reads them all - so the check must sit above it in "
            "`drive_run`, not inside it."
        )


class TestTheRecordedProvenanceIsWhatHappened:
    """R-02.22's distinction, on the one case where plan and outcome can legally differ.

    ``resume_plan`` answers *"is there a complete trainer state for this step?"*. It cannot
    answer *"does that state contain anything?"* - only the process that loads it can. A
    state written **before the optimizer had taken a step** is complete, restorable and
    empty: the restore succeeds and the optimizer is, in every way that affects the
    trajectory, fresh.

    Recording ``restored`` there would put a true-sounding claim in a published manifest over
    an optimizer with no moment buffers, which makes a real loss discontinuity
    unexplainable - the exact failure the field exists to prevent. So the identity is
    downgraded and the reason is logged by name.
    """

    def test_an_empty_restored_state_is_recorded_as_fresh_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        states = tmp_path / "trainer-state"
        checkpoints = tmp_path / "checkpoints"
        states.mkdir()
        checkpoints.mkdir()

        model = _fixed_init_model()
        optimizer, identity = optimizer_module.build(model, _cfg(), state="restored")
        # Written with NO step taken, which is what makes the state empty. Through the real
        # writer, so the directory is complete and `read` does not refuse it - the point is
        # that a *valid* restore can still restore nothing.
        trainer_state.write(
            run_id="run-empty",
            step=0,
            optimizer=optimizer,
            identity=identity,
            root=states,
            checkpoint_root=checkpoints,
        )
        assert identity.state == "restored", "the planned identity is the input to this test"

        fresh_optimizer, planned = optimizer_module.build(model, _cfg(), state="restored")
        with caplog.at_level("WARNING"):
            recorded = tasks.restore_optimizer_state(
                fresh_optimizer, planned, run_id="run-empty", step=0, root=states
            )

        assert recorded.state == "fresh", (
            "an optimizer state with no per-parameter entries was recorded as `restored`. "
            "The manifest then asserts a restore that changed nothing, and the next "
            "unexplained discontinuity has a false explanation sitting next to it."
        )
        assert recorded.name == planned.name and recorded.config_hash == planned.config_hash, (
            "only the provenance may change here; renaming the optimizer or its config hash "
            "would make the manifest describe a different run"
        )
        assert any(
            getattr(record, "code", None) == "trainer_state_restored_empty"
            for record in caplog.records
        ), (
            "the downgrade was silent. A recorded `fresh` that an operator cannot trace to a "
            "cause is the same problem one step removed."
        )

    def test_a_non_empty_restore_keeps_the_planned_identity(self, tmp_path: Path) -> None:
        """The other half, so the assertion above is not satisfied by always downgrading."""
        states = tmp_path / "trainer-state"
        checkpoints = tmp_path / "checkpoints"
        states.mkdir()
        checkpoints.mkdir()
        _losses, weights = _run_until_save(states, checkpoints, "run-nonempty")

        model = _fixed_init_model()
        model.load_state_dict(weights)
        optimizer, planned = optimizer_module.build(model, _cfg(), state="restored")
        recorded = tasks.restore_optimizer_state(
            optimizer, planned, run_id="run-nonempty", step=SAVE_AT_STEP, root=states
        )
        assert recorded.state == "restored"
        assert optimizer.state_dict()["state"], "nothing was loaded, so this proves nothing"
