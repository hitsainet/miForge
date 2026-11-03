"""F8's one service seam: run types, the budget preview, adapters and the merge.

**The preview and the gate share one estimator** (FR-008.18): :func:`estimate_budget` calls
F1's ``training.budget_guard.estimate`` for a full-parameter run and F8's
``training.runtypes.adapter_budget.estimate`` for an adapter one. A preview computed
differently from the gate is a preview of nothing.

``GET /adapters/{name}`` answers from **the manifest on disk**, not from the index row, and
reports a disagreement rather than hiding it: the manifest is authoritative and where they
disagree the manifest wins **and it is logged** (ADR-004).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.errors import refuse
from models.adapter import Adapter
from services import budget_source
from services import reindex as reindex_module
from training import budget_guard
from training.acquisition.base import AcquisitionMode, AcquisitionSpec
from training.run_config import RunConfig
from training.runtypes import adapter_budget, known, resolve
from training.runtypes.adapters import load as adapter_load
from training.runtypes.adapters import paths as adapter_paths
from training.runtypes.adapters.config import validate as validate_adapter_config

log = logging.getLogger(__name__)

#: **There is no longer a mapping.** A registry name IS the stored value.
#:
#: F8 shipped ``DB_RUN_TYPE = {"change": "full", ...}`` because ``mf_runs.run_type``'s CHECK
#: constraint permitted ``create|full|adapter|merge`` while the registry and BRD-02 §4 both
#: say ``change``. F8 reported the divergence rather than editing a migration it does not
#: own, which was right. Migration ``0014`` settled it on ``change`` - the requirement's own
#: word - and the map is removed rather than left as an identity function, because a
#: translation table between two names for one concept is a second source of truth and this
#: project has paid for that repeatedly.
#:
#: ``test_runtype_registry.py`` still asserts every registry name is a value the **live**
#: constraint permits. That assertion is the thing worth keeping: it is what would have
#: caught the divergence in the first place, and it goes red if either side moves again.


def run_types() -> list[dict[str, Any]]:
    """The **live registry's** entries. ``GET /api/v1/run-types``'s body.

    Read through :func:`training.runtypes.known` rather than by importing the three modules:
    importing proves a module parses, resolving proves a caller can reach it. Removing a
    registration must make this list short and the reachability test red (M12).
    """
    return [
        {
            "name": name,
            "stored_as": name,
            "changes": _what_it_changes(name),
        }
        for name, _ in sorted(known().items())
    ]


def _what_it_changes(name: str) -> str:
    """One line per run type, for F9's selector. The labels F9 should use.

    The mockup's *"Attach and start"* predates the 2026-10-13 standalone reframe and is
    F9's to resolve, not F8's to rename.
    """
    return {
        "change": "every parameter of a model that already exists",
        "create": "a new model built from an architecture config, randomly initialized",
        "adapter": "a low-rank adapter trained beside the base model, which is left unchanged",
    }[name]


def estimate_budget(
    *,
    run_type: str,
    cfg: RunConfig,
    architecture_config: Mapping[str, Any] | None = None,
    param_count: int | None = None,
    adapter: Any = None,
    free_bytes: int | None = None,
    model_source: AcquisitionSpec | None = None,
    cards: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The estimate, the breakdown by term, and the free figure - **without submitting**.

    Args:
        model_source: the run definition's model source. The model's size is read from it
            on the server (:mod:`services.budget_source`), which is how the New run form
            asks: it knows the source the operator chose, not a parameter count. Sent
            instead of ``architecture_config`` / ``param_count``, never with them.
        cards: every card the run could be placed on (``uuid``, ``name``,
            ``free_memory_bytes``), so each gets its own verdict. One card or several:
            nothing here assumes a count.
        free_bytes: free VRAM for the run's resolved card, when a card is present. ``None``
            renders the verdict ``unknown`` rather than ``fits``: *"I could not read the
            limit, so it fits"* is the pass-through failure wearing the estimator's name
            (E-6), and a preview that said "fits" on a machine with no card would be the
            most misleading number on the screen.

    Raises:
        Refusal: ``run_type_unknown``; ``budget_spec_ambiguous`` / ``budget_spec_incomplete``
            from the estimator; ``adapter_config_required`` for an adapter run with none.
    """
    resolve(run_type)  # refuses by name and lists the three

    source: dict[str, Any] | None = None
    if model_source is not None:
        if architecture_config is not None or param_count is not None:
            raise refuse(
                "budget_spec_ambiguous",
                "The budget request names a model source and also gives the model's size. "
                "Send one: the size is read from the model source.",
                run_type=run_type,
            )
        sized = budget_source.size_of(model_source, need_config=run_type == "adapter")
        architecture_config, param_count = sized.architecture_config, sized.param_count
        source = sized.as_record()
    elif architecture_config is None and param_count is None:
        raise refuse(
            "budget_spec_incomplete",
            "Choose a model source to estimate GPU memory: the estimate is worked out from "
            "the model's size, which is read from the model source.",
            run_type=run_type,
        )

    if run_type == "adapter":
        if adapter is None:
            raise refuse(
                "adapter_config_required",
                "An adapter run's budget depends on its rank and its target modules, so "
                "the adapter configuration is required. Refusing rather than estimating "
                "a default adapter nobody asked for.",
                run_type=run_type,
            )
        estimate: Any = adapter_budget.estimate(
            cfg=cfg,
            adapter=validate_adapter_config(adapter),
            architecture_config=architecture_config,
            base_param_count=param_count,
        )
        record = estimate.as_record()
        high = estimate.total_high_bytes
    else:
        spec = AcquisitionSpec(
            mode=(
                AcquisitionMode.CREATE if architecture_config is not None else AcquisitionMode.LOAD
            ),
            config=dict(architecture_config) if architecture_config is not None else None,
            repo="preview/preview" if architecture_config is None else None,
            param_count=param_count,
        )
        projection = budget_guard.estimate(spec, cfg)
        record = {"run_type": run_type, **projection.as_record()}
        high = projection.total_high_bytes
        estimate = projection

    claimable = int(free_bytes * budget_guard.SAFETY_FRACTION) if free_bytes is not None else None
    verdict = _verdict(high, free_bytes)

    payload = {
        "verdict": verdict,
        "estimate": record,
        # Bytes, for the screen: the figure the start compares (the pessimistic end), the
        # optimistic end, and each term, so an operator who is over can see which to change.
        "estimate_bytes": high,
        "estimate_low_bytes": estimate.total_low_bytes,
        "terms": [
            {"name": label, "bytes": int(getattr(estimate, attribute))}
            for label, attribute in BUDGET_TERMS
        ],
        "source": source,
        "cards": [
            {
                "uuid": card.get("uuid"),
                "name": card.get("name"),
                "free_bytes": card.get("free_memory_bytes"),
                "verdict": _verdict(high, card.get("free_memory_bytes")),
            }
            for card in cards
        ],
        # Every number names its scale (ADR-012). Bytes and GiB both, because the API's
        # consumer is a screen and the comparison is arithmetic.
        "free_bytes": free_bytes,
        "free_gib": None if free_bytes is None else round(free_bytes / budget_guard.GIB, 3),
        "claimable_bytes": claimable,
        "claimable_gib": None if claimable is None else round(claimable / budget_guard.GIB, 3),
        "safety_fraction": budget_guard.SAFETY_FRACTION,
    }
    log.info(
        "budget previewed",
        extra={"code": "budget_previewed", "run_type": run_type, "verdict": verdict, **record},
    )
    return payload


#: The estimate's terms, in the order the screen lists them: a plain label and the attribute
#: both estimate classes carry. Activations are the upper end of their range, because the
#: total they add up to is the figure the start compares against.
BUDGET_TERMS: tuple[tuple[str, str], ...] = (
    ("Model weights", "weights_bytes"),
    ("Gradients", "gradient_bytes"),
    ("32-bit master copy", "master_bytes"),
    ("Optimizer states", "optimizer_state_bytes"),
    ("Activations (upper end)", "activation_high_bytes"),
)


def _verdict(high: int, free_bytes: Any) -> str:
    """``fits`` / ``exceeds`` against the claimable share of ``free_bytes``; ``unknown``
    when the free figure was not read - never ``fits``."""
    if not isinstance(free_bytes, int) or isinstance(free_bytes, bool):
        return "unknown"
    return "fits" if high <= int(free_bytes * budget_guard.SAFETY_FRACTION) else "exceeds"


def list_adapters(session: Any, *, run_id: str | None = None) -> list[dict[str, Any]]:
    """Indexed adapters. F10/F5's ``reindex`` listing, reused rather than re-queried."""
    return reindex_module.list_adapters(session, run_id=run_id)


def read_adapter(session: Any, name: str, *, root: Path | None = None) -> dict[str, Any]:
    """One adapter's manifest **as written on disk**, plus its path and a disagreement flag.

    The index row is consulted only to compare: ADR-004 makes the manifest authoritative,
    and an endpoint serving the row would hide exactly the disagreement an operator needs
    to see. A divergence is reported **and logged**.

    Raises:
        Refusal: ``adapter_manifest_missing`` when nothing is on disk under that name;
            ``adapter_manifest_invalid`` / ``adapter_scaling_unrecognised`` from the reader.
    """
    parsed = adapter_paths.parse_adapter_name(name)
    if parsed is None:
        raise refuse(
            "adapter_manifest_missing",
            f"{name!r} is not an adapter's qualified name "
            f"(<base_model>:<run_id>:adapter-<step>), so no directory can be resolved from "
            f"it. Refusing rather than composing a path out of what was supplied - a "
            f"traversal is a validation failure, not something to sanitise.",
            adapter=name,
        )
    _, run_id, step = parsed
    directory = adapter_paths.adapter_dir(run_id, step, root=root)
    doc = adapter_load.load_manifest(directory / adapter_paths.MANIFEST_NAME)

    row = session.get(Adapter, doc["name"])
    disagreements = _index_disagreements(doc, row)
    if disagreements:
        log.warning(
            "the adapter index disagrees with the manifest; the manifest wins",
            extra={
                "code": "adapter_index_divergent",
                "adapter": doc["name"],
                "disagreements": disagreements,
            },
        )

    return {
        "manifest": doc,
        "path": str(directory),
        "indexed": row is not None,
        "index_disagrees": bool(disagreements),
        "disagreements": disagreements,
    }


def preflight_merge(session: Any, name: str, *, root: Path | None = None) -> dict[str, Any]:
    """Everything a merge can refuse **before any tensor arithmetic** (AC17, E-8, E-11).

    The half of a merge that is reachable with no card: read the manifest (refusing an
    unrecognised scaling rule) and require that a checkpoint of the unmerged base exists.
    Returns what was admitted, so the refusal and the admission carry the same names.

    Running the check here rather than inside the worker is R-02.45's ordering: a merge that
    will be refused must not first compute merged tensors, because that wastes the card and
    makes a refusal look like a failure.

    Raises:
        Refusal: ``merge_refused_no_unmerged_base`` naming the missing checkpoint and what
            to do next (save one); ``adapter_scaling_unrecognised``;
            ``adapter_manifest_invalid``; ``adapter_manifest_missing``.
    """
    from training.runtypes.adapters import merge as merge_module

    read = read_adapter(session, name, root=root)
    doc = read["manifest"]
    base = merge_module.resolve_base_checkpoint(doc["base"], root=root)
    log.info(
        "merge preflight admitted",
        extra={
            "code": "merge_preflight_admitted",
            "adapter": doc["name"],
            "base_checkpoint": base.name,
            "scaling": doc["adapter"].get("scaling"),
        },
    )
    return {
        "name": doc["name"],
        "step": doc["step"],
        "path": read["path"],
        "scaling": doc["adapter"].get("scaling"),
        "base_checkpoint": base.name,
        "base_checkpoint_path": str(base.directory),
    }


def _index_disagreements(doc: Mapping[str, Any], row: Adapter | None) -> dict[str, Any]:
    """Which indexed columns differ from the manifest. Empty when the row is absent.

    An absent row is **not** a disagreement: it is a stale index, which ``reindex`` fixes
    and which ``indexed: false`` already reports. Treating it as one would make every
    adapter look divergent until the next rebuild.
    """
    if row is None:
        return {}
    expected = {
        "run_id": str(doc.get("run_id", "")),
        "step": int(doc.get("step", 0)),
        "base_name": str((doc.get("base") or {}).get("name", "")),
        "weights_sha256": str(doc.get("weights_sha256", "")),
    }
    return {
        key: {"manifest": value, "index": getattr(row, key)}
        for key, value in expected.items()
        if getattr(row, key) != value
    }
