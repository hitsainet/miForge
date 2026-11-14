"""Where an adapter directory lives, and what it is called  (FR-008.33; BRD-01 R-01.24).

## One integer, two renderings

The step is **one canonical integer**. The path renders it zero-padded to six digits; the
``name`` renders it unpadded (BRD-01 R-01.24). Both renderings come from F5's
``services.checkpoints.paths`` rather than from a second renderer here - *two renderers of
one convention is how* ``step-<N>`` *came back once already*.

## Under the checkpoint root, not beside it, and that is a correction

F8's TID specifies ``MIFORGE_ADAPTER_ROOT`` defaulting to ``/data/adapters``, a root of its
own. **A separate root would be invisible to ``reindex``**, which walks
``settings.checkpoint_root`` and finds artifacts by manifest *filename* rather than by
layout. Adapters under ``/data/adapters`` would be written, indexed at save, and then
**deleted from the index by the next reindex and never rebuilt** - and the
reindex-equivalence test would pass, because it compares an index rebuilt from one root
against API responses served from that same root. ADR-004 makes the manifest authoritative
and the database rebuildable; an artifact outside the walked tree is neither.

So an adapter lives at ``<checkpoint_root>/<run_id>/adapters/step-<NNNNNN>/``:

* inside the run's directory, so it cannot collide with a run whose id is ``adapters``;
* under its own ``adapters/`` level, so the ``latest`` symlink ``atomic_dir`` swaps is
  **this parent's** and does not overwrite the checkpoint ``latest`` one level up;
* inside the walked tree, so ``reindex`` rebuilds it from the manifest alone.

The configurable root is reported as a finding rather than implemented: it needs a line in
``core/config.py``, which is not F8's file, and it needs ``reindex`` to walk two roots.
"""

from __future__ import annotations

from pathlib import Path

from services.checkpoints.paths import run_dir, step_dir_name

#: The tensors. ``peft``'s ``save_pretrained`` writes this name.
WEIGHTS_NAME = "adapter_model.safetensors"
#: ``peft``'s own configuration. ``PeftConfig.from_pretrained`` **reads this file**, so a
#: directory without it is not loadable by the standard tooling R-02.51 promises - which is
#: why F8 writes three files and treats the normative layout's two-file enumeration as
#: incomplete rather than exclusive (FPRD D5).
PEFT_CONFIG_NAME = "adapter_config.json"
#: miForge's record, beside PEFT's and never a replacement for it.
MANIFEST_NAME = "adapter_manifest.json"

#: The level that separates adapters from checkpoints inside one run's directory.
ADAPTERS_SEGMENT = "adapters"

#: What distinguishes an adapter from a checkpoint of the same run and step in R-02.29's
#: tree. Declared once: the name is a join key, and two spellings of a join key is a tree
#: with two roots.
NAME_SEGMENT = "adapter"


def adapters_dir(run_id: str, *, root: Path | None = None) -> Path:
    """The run's adapter parent - the directory whose ``latest`` an adapter save swaps."""
    return run_dir(run_id, root=root) / ADAPTERS_SEGMENT


def adapter_dir(run_id: str, step: int, *, root: Path | None = None) -> Path:
    """One adapter's directory. The step is **zero-padded to six digits** here."""
    return adapters_dir(run_id, root=root) / step_dir_name(step)


def adapter_name(base_model: str, run_id: str, step: int) -> str:
    """``<base_model>:<run_id>:adapter-<step>``, with the step **unpadded**.

    The same display rule the checkpoint manifest's ``name`` follows, plus the
    ``adapter-`` segment. Deliberately *not* reusing F5's ``qualified_name``: that renders
    ``<base_model>:<run_id>:<step>`` and the segment is the whole difference, so wrapping it
    with a string edit would make the adapter's name depend on parsing the checkpoint's.
    """
    return f"{base_model}:{run_id}:{NAME_SEGMENT}-{step}"


def parse_adapter_name(name: str) -> tuple[str, str, int] | None:
    """``(base_model, run_id, step)`` from an adapter name, or ``None``.

    Split from the **right**, because ``base_model`` legitimately contains colons - a Hub
    repo id does not, but a checkpoint's own qualified name does, and an adapter trained on
    a checkpoint carries that name here. Splitting from the left would take the first
    segment of the base's name as the whole of it.
    """
    head, _, step_text = name.rpartition(":")
    if not head or not step_text.startswith(f"{NAME_SEGMENT}-"):
        return None
    base_model, _, run_id = head.rpartition(":")
    if not base_model or not run_id:
        return None
    try:
        step = int(step_text[len(NAME_SEGMENT) + 1 :])
    except ValueError:
        return None
    if step < 0:
        return None
    return base_model, run_id, step
