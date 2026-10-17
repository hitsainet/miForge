"""What a run will consume on disk - **reported, never enforced**  (FR-006.53; DEF-8).

## There is no checkpoint lifecycle anywhere in this product

**No requirement in BRD-02 defines pruning, deletion or archival of checkpoints.** Verified
against the BRD; it is recorded in its §9 and reported here as **DEF-8**. The number that
makes that concrete is the one this module computes: one 10,000-step run at
``save_every = 100`` writes 100 checkpoints and 100 trainer-state directories, which for an
LFM2 1.2B at FP16 is **480-960 GB** depending on the optimizer-state precision.

So the loop **reports** the projection at submit and **enforces nothing**. That division is
deliberate and is the requirement: inventing a retention policy here would be a product
decision taken in a service module, and refusing a run for want of disk would be a second
gate beside R-02.42's VRAM budget with no requirement behind it.
:func:`project` refuses no run; ``test_footprint.py`` asserts that it cannot.

## A range, with both bounds named  (ADR-012)

Trainer state is a **range**, not a figure: an 8-bit AdamW state is 2 bytes per parameter
and a 32-bit one is 8, and whether the FP32 master copy is written depends on the precision
the run was configured with. **Do not pick one bound and present it as the figure** - the
rendering names both, because *every number names its scale and sample* and a single number
here would be wrong by a factor of four in the direction that fills a volume.

The per-parameter constants are :mod:`training.budget_guard`'s, imported rather than
restated: a second table of byte widths is a second answer, and the one this module would
get wrong is the one an operator sizes a volume against.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.budget_guard import (
    BYTES_FP32_MASTER,
    BYTES_WEIGHTS,
    OPTIMIZER_STATE_BYTES,
)

#: Tokenizer files, the config and the manifest, per checkpoint directory. A flat allowance
#: rather than a measurement: it is tens of megabytes against gigabytes of tensors, and
#: measuring it would mean reading a model this function is called *before* loading.
OVERHEAD_BYTES_PER_CHECKPOINT = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Footprint:
    """A run's projected artifact footprint, beside free disk. **Both bounds, always.**"""

    checkpoints: int
    bytes_per_checkpoint: int
    trainer_state_bytes_low: int
    trainer_state_bytes_high: int
    free_disk_bytes: int | None
    root: str | None

    @property
    def total_low(self) -> int:
        return self.checkpoints * (self.bytes_per_checkpoint + self.trainer_state_bytes_low)

    @property
    def total_high(self) -> int:
        return self.checkpoints * (self.bytes_per_checkpoint + self.trainer_state_bytes_high)

    def payload(self) -> dict[str, Any]:
        """The resolved definition's ``footprint`` block and the API's.

        ``exceeds_free_disk`` is a **report**, not a verdict: nothing refuses on it. It is
        here so a form can warn, and so the number is not left for a client to recompute
        against its own idea of what a checkpoint weighs.
        """
        return {
            "checkpoints": self.checkpoints,
            "bytes_per_checkpoint": self.bytes_per_checkpoint,
            "trainer_state_bytes_per_save": {
                "low": self.trainer_state_bytes_low,
                "high": self.trainer_state_bytes_high,
            },
            "total_bytes": {"low": self.total_low, "high": self.total_high},
            "free_disk_bytes": self.free_disk_bytes,
            "root": self.root,
            "exceeds_free_disk": (
                None if self.free_disk_bytes is None else self.total_high > self.free_disk_bytes
            ),
            "note": (
                "A projection, not a limit: miForge enforces no checkpoint retention "
                "policy, because no requirement defines one (BRD-02 section 9). Both "
                "bounds are given because an 8-bit optimizer state is 2 bytes per "
                "parameter and a 32-bit one is 8."
            ),
        }

    def rendered(self) -> str:
        """One sentence naming **both** bounds and the sample (ADR-012)."""
        return (
            f"{self.checkpoints} checkpoints: "
            f"{_gib(self.total_low)}-{_gib(self.total_high)} GiB"
            + (
                ""
                if self.free_disk_bytes is None
                else f" against {_gib(self.free_disk_bytes)} GiB free on {self.root}"
            )
        )


def project(
    *,
    max_steps: int,
    save_every: int | None,
    param_count: int | None,
    optimizer_state_precision: str = "8bit",
    root: Path | None = None,
) -> Footprint:
    """Project the run's artifact footprint. **Refuses nothing, ever.**

    Args:
        max_steps: the run's step bound - its ``step_count`` condition's total, or the
            driver's ceiling. ``0`` or less projects the run-end save alone.
        save_every: the save cadence. ``None`` projects **one** checkpoint, the run-end
            save (FR-006.8), rather than zero: a run with no cadence still writes one.
        param_count: from F1's preflight. ``None`` leaves the per-checkpoint size ``0``
            and the projection states that - an unknown size reported as zero is wrong,
            so :meth:`Footprint.payload` carries the zero beside its reason rather than
            a guessed figure.
        optimizer_state_precision: which bound to use for the *low* end; the *high* end is
            always the widest this release supports, so the range brackets the truth
            whatever the run is reconfigured to.
        root: where artifacts land. Defaults to the configured checkpoint root.
    """
    cadenced = 0 if not save_every else max(0, int(max_steps)) // int(save_every)
    # The run-end save (FR-006.8) always happens, and it is a separate checkpoint unless it
    # lands on a cadence boundary - where F5's idempotence makes it a no-op. Projected as
    # one extra only when the arithmetic says it would be distinct, so the figure is not
    # inflated by a save that will collapse.
    run_end_is_distinct = not save_every or (int(max_steps) % int(save_every) != 0)
    checkpoints = max(1, cadenced + (1 if run_end_is_distinct else 0))

    params = int(param_count or 0)
    per_checkpoint = params * BYTES_WEIGHTS + (OVERHEAD_BYTES_PER_CHECKPOINT if params else 0)

    low_state = OPTIMIZER_STATE_BYTES.get(
        optimizer_state_precision, min(OPTIMIZER_STATE_BYTES.values())
    )
    high_state = max(OPTIMIZER_STATE_BYTES.values())
    # The FP32 master copy rides with the state in both bounds: F1's own estimator counts
    # it, and a projection that omitted it would understate every run by 4 bytes per
    # parameter - which at 1.2B parameters is 4.8 GB per save.
    trainer_low = params * (low_state + BYTES_FP32_MASTER)
    trainer_high = params * (high_state + BYTES_FP32_MASTER)

    free, where = _free_disk(root)
    return Footprint(
        checkpoints=checkpoints,
        bytes_per_checkpoint=per_checkpoint,
        trainer_state_bytes_low=trainer_low,
        trainer_state_bytes_high=trainer_high,
        free_disk_bytes=free,
        root=where,
    )


def _free_disk(root: Path | None) -> tuple[int | None, str | None]:
    """Free bytes on the artifact volume, or ``None`` when it cannot be read.

    ``None`` rather than ``0``: a volume that cannot be stat'ed is not a full volume, and
    reporting zero free would make every projection look fatal on a developer's machine
    where the directory simply does not exist yet (ADR-023).
    """
    from core.config import get_settings

    target = root if root is not None else get_settings().checkpoint_root
    probe = target
    for _ in range(6):
        if probe.exists():
            try:
                return shutil.disk_usage(probe).free, str(target)
            except OSError:  # pragma: no cover - an unreadable mount
                return None, str(target)
        if probe.parent == probe:
            break
        probe = probe.parent
    return None, str(target)


def _gib(value: int) -> str:
    return f"{value / (1024 ** 3):.1f}"


def steps_for_footprint(stop_conditions: Any, fallback: int) -> int:
    """The step bound the projection should use: the ``step_count`` total if there is one.

    ``fallback`` otherwise, and the caller's fallback is the driver's ceiling. Read from the
    condition's own configuration rather than from a second field, so the projection and
    the thing that actually ends the run cannot disagree about how long it is.
    """
    from training.stop.base import StopReason

    config = (stop_conditions or {}).get(StopReason.STEP_COUNT.value)
    if config and isinstance(config.get("total"), int):
        return int(config["total"])
    return int(fallback)


__all__ = [
    "OVERHEAD_BYTES_PER_CHECKPOINT",
    "Footprint",
    "project",
    "steps_for_footprint",
]
