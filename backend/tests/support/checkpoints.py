"""Fixtures that round-trip the **real** writer and the **real** reader  (ADR-016).

*"Fixtures whose fields agree by construction are how a suite stays green over a real
bug."* So nothing here hand-builds a manifest dict and asserts against it. A manifest
fixture is produced by :func:`services.checkpoints.writer.write` and read back by
:func:`services.checkpoints.manifest.read`, which means a rename in either one is red.

## The one place a literal is unavoidable, and what rides with it

:func:`fields` builds the *inputs* - the borrowed values six other features own. Those have
to come from somewhere, and in a test that somewhere is a literal. What keeps it honest is
that every one of them goes through ``manifest.build``'s explicit parameters and the
vendored schema, and that each example ships with the **negative control proving the
assertion can fail**.

## Why the model is a fake here and a real one in the conformance test

``transformers`` is not installed in every environment this suite runs in, and the paths
that carry this feature's defect risk - the manifest's provenance, the resave rule, the
refusals, the lineage walk - need no tensors at all (ADR-023). So :class:`TinyModel` writes
the real *file names* ``save_pretrained`` produces, and the output contract is proved
separately by ``tests/support/conformance.py`` against a real load, which **skips loudly**
rather than passing when the runtime is absent. The division is deliberate: the fake makes
the refusal paths testable on a laptop, and it is never the thing that proves AC23.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from services.checkpoints import writer as writer_module
from services.checkpoints.manifest import ManifestFields

__all__ = [
    "TinyModel",
    "TinyTokenizer",
    "create_feature_tables",
    "fields",
    "saved_checkpoint",
    "score",
    "write_chain",
]


@dataclass
class TinyModel:
    """Writes the files ``save_pretrained`` writes, with distinguishable bytes.

    ``payload`` is what lands in ``model.safetensors``, so two "models" with different
    payloads hash differently - which is what makes the resave rule testable without a GPU:
    the same payload is the idempotent case, a different one is the refusal.
    """

    payload: bytes = b"tensors-v1"
    config: dict[str, Any] = field(default_factory=lambda: {"model_type": "tiny", "n_layer": 1})

    def save_pretrained(self, directory: Any, *, safe_serialization: bool = False) -> None:
        assert safe_serialization, (
            "the writer must pass safe_serialization=True: a pytorch_model.bin is a pickle "
            "and R-02.51 promises an artifact a consumer can load without executing it"
        )
        target = Path(directory)
        (target / "model.safetensors").write_bytes(self.payload)
        (target / "config.json").write_text(json.dumps(self.config), encoding="utf-8")


@dataclass
class TinyTokenizer:
    """Writes tokenizer files **at the root**, which is the contract (FR-005.19)."""

    #: Set to a directory name to simulate the defect the file-set assertion exists to
    #: catch: as first specified, a ``tokenizer/`` subdirectory could not be loaded by
    #: ``transformers``, vLLM, TGI or miLLM's own loader.
    subdirectory: str | None = None

    def save_pretrained(self, directory: Any) -> None:
        target = Path(directory)
        if self.subdirectory is not None:
            target = target / self.subdirectory
            target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
        (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (target / "special_tokens_map.json").write_text("{}", encoding="utf-8")


def score(
    set_name: str = "contract-clauses-test",
    *,
    value: float = 0.81,
    n: int = 120,
    floor: float | None = None,
    provider: str | None = "in-process:lfm2-1.2b",
) -> dict[str, Any]:
    """A ``ScoreRecord`` as F4 would hand one over.

    ``provider`` is populated **by default** on purpose: the key-set assertion's whole
    subject is a provider that goes in and does not come out, and a fixture that omitted it
    by default would make the test for that pass against a writer that drops it.
    """
    record: dict[str, Any] = {"set": set_name, "score": value, "n": n}
    if floor is not None:
        record["floor"] = floor
    if provider is not None:
        record["provider"] = provider
    return record


def fields(
    *,
    run_id: str = "run-a",
    step: int = 500,
    domain: str | None = "contract-clauses",
    signal_source: str = "corpus",
    base_model: str = "LiquidAI/LFM2-1.2B",
    parent: str | None = None,
    optimizer_state: str = "restored",
    optimizer_name: str = "adamw-8bit",
    config_hash: str | None = "a" * 64,
    test: dict[str, Any] | None = None,
    retention: dict[str, Any] | None = None,
    merged_adapter: dict[str, Any] | None = None,
    created_at: str | None = "2026-10-14T09:00:00Z",
    selector_hash: str | None = "b" * 64,
    grader_hash: str | None = None,
) -> ManifestFields:
    """The borrowed values, as six other features would supply them."""
    return ManifestFields(
        run_id=run_id,
        step=step,
        domain=domain,
        signal_source=signal_source,
        base_model=base_model,
        parent=parent,
        optimizer={"name": optimizer_name, "state": optimizer_state, "config_hash": config_hash},
        # Overridable so AC-003.18 can pass a hash the REAL resolver produced. The default
        # is unchanged, so no existing caller moves - but a literal here is exactly what
        # that criterion forbids relying on: a hand-built value agrees with whatever the
        # writer does to it, which is how a suite stays green over a dropped field.
        selector_hash=selector_hash,
        grader_hash=grader_hash,
        test=test,
        retention=retention,
        merged_adapter=merged_adapter,
        created_at=created_at,
    )


def saved_checkpoint(
    root: Path,
    *,
    run_id: str = "run-a",
    step: int = 500,
    payload: bytes = b"tensors-v1",
    tokenizer: TinyTokenizer | None = None,
    **field_overrides: Any,
) -> writer_module.WriteResult:
    """Write one checkpoint **through the real writer**. Returns what it published."""
    from services.checkpoints.paths import checkpoint_dir

    return writer_module.write(
        TinyModel(payload=payload),
        tokenizer or TinyTokenizer(),
        dest=checkpoint_dir(run_id, step, root=root),
        fields=fields(run_id=run_id, step=step, **field_overrides),
    )


def write_chain(
    root: Path,
    *,
    run_id: str = "run-a",
    steps: tuple[int, ...] = (100, 200, 300),
    base_model: str = "LiquidAI/LFM2-1.2B",
    first_parent: str | None = None,
) -> list[writer_module.WriteResult]:
    """A run's successive saves, each one's ``parent`` the previous one's ``name``.

    Built by calling the real writer in sequence rather than by planting manifests, so the
    chain the lineage walk reads is the chain the writer produces.
    """
    results: list[writer_module.WriteResult] = []
    parent = first_parent
    for step in steps:
        result = saved_checkpoint(
            root,
            run_id=run_id,
            step=step,
            payload=f"tensors-{run_id}-{step}".encode(),
            base_model=base_model,
            parent=parent,
        )
        results.append(result)
        parent = result.name
    return results


def create_feature_tables(engine: Engine) -> None:
    """Create F5's own tables beside the base schema.

    **F5 no longer has a table of its own.** Its save-request flag lived on its own
    ``MetaData`` because ``backend/src/models/`` is not a feature agent's to edit; on
    2026-10-14 the coordinator took FR-005.1's straight reading and moved it onto
    ``mf_runs`` (``save_requested_at`` + ``save_request_count``, migration 0012), which
    ``Base.metadata`` already carries. So nothing of F5's is created here any more.

    F10's tables still are, because ``services.reindex.reindex`` rebuilds the part index in
    the same call: *"a caller that has to remember to run a second command for one table is
    a caller that will forget."* A test engine carrying only the base schema would make
    every reindex assertion in this feature fail on a missing table belonging to **another**
    feature - which looks like an F5 defect and is not.
    """
    try:
        from services.parts.tables import metadata as parts_metadata
    except ImportError:  # pragma: no cover - F10 is present in this tree
        return
    parts_metadata.create_all(engine)

    # F4's score tables and F2's batch index, for the same reason F10's are here:
    # `services.reindex.reindex` rebuilds all of them in one call, so an engine carrying
    # only some makes a reindex assertion fail on a table belonging to another feature -
    # which looks like this feature's defect and is not.
    for module, attr in (
        ("services.scoring.tables", "metadata"),
        ("training.signals.batch_index", "metadata"),
    ):
        try:
            mod = __import__(module, fromlist=[attr])
        except ImportError:  # pragma: no cover - both are present in this tree
            continue
        getattr(mod, attr).create_all(engine)


def rows_in(engine: Engine, table: sa.Table) -> Iterator[dict[str, Any]]:
    """Every row of ``table``, as mappings. For asserting over the index directly."""
    with engine.connect() as connection:
        for row in connection.execute(sa.select(table)).mappings():
            yield dict(row)
