"""The model's size for the budget preview, read **from the model source** on the server.

``POST /budget/estimate`` used to need the caller to supply a parameter count or an
architecture config. The New run form knows neither - it knows the model source the operator
chose - so every estimate it asked for was refused ``budget_spec_incomplete`` (found live,
2026-10-20). The form now sends the model source, the same block the run definition carries,
and this module turns it into the estimator's input.

## One source of truth: the acquirers' own readers

Each kind is read with the functions the start path uses, so the preview and the start
cannot disagree about a model's size:

* a **directory or checkpoint on this host**: the path is confined to the model roots and the
  count is read from the safetensors headers (``local_dir``), exactly as a load's pre-flight
  reads it. A miForge checkpoint is a model directory with a manifest beside its weights,
  so it is read the same way and reported as ``checkpoint``.
* a **Hub repository**: the revision is resolved to a commit, then the count is read from the
  same headers by range request (``hub.remote_weights``) - nothing is downloaded. A start
  downloads the snapshot; a preview must not.
* an **architecture config** (``create``): parsed by the create mode's own parser.
* **any other mode** (``attach``): the registered acquirer's own pre-flight, which reads the
  served model's parameter count from miLLM. Not named here: the registry decides.

An adapter run's estimate bounds the adapter from the architecture, so for it the
architecture config is returned (read from the same directory or repository) rather than a
count. Exactly one of the two is ever returned, which is the estimator's own rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.errors import refuse
from training import acquisition
from training.acquisition.base import AcquisitionMode, AcquisitionSpec


@dataclass(frozen=True)
class SizedSource:
    """What the estimator needs, and where it came from. Exactly one of the first two."""

    param_count: int | None
    architecture_config: dict[str, Any] | None
    #: ``directory``, ``checkpoint``, ``hub``, ``config`` or the mode's own name.
    kind: str
    #: Where it was read: a path, a repository and its resolved commit. No secrets.
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "param_count": self.param_count,
            "from_config": self.architecture_config is not None,
            **self.detail,
        }


def size_of(spec: AcquisitionSpec, *, need_config: bool = False) -> SizedSource:
    """Read the model's size from ``spec``.

    Args:
        need_config: return the architecture config rather than a count. An adapter run's
            estimate needs the shapes of the modules it targets, which a count does not
            describe.

    Raises:
        Refusal: the same refusals the start path raises for the same source (outside the
            model roots, no ``config.json``, no safetensors shard, an unknown revision, the
            Hub unreachable, an unavailable mode); ``budget_source_size_unknown`` when a
            mode's pre-flight reports no size at all.
    """
    if spec.mode is AcquisitionMode.CREATE:
        from training.acquisition.scratch import parse_config

        parsed = parse_config(spec)
        detail = {} if spec.config_path is None else {"config_path": str(spec.config_path)}
        return SizedSource(None, dict(parsed.data), "config", detail)

    if spec.mode is AcquisitionMode.LOAD and spec.path is not None:
        return _directory(spec, need_config=need_config)

    if spec.mode is AcquisitionMode.LOAD and spec.repo is not None:
        return _hub(spec, need_config=need_config)

    # Every other mode answers through its registered acquirer's own pre-flight. Read-only
    # for every registered mode: it reads, it does not acquire.
    enriched = acquisition.resolve(spec).preflight(spec)
    if enriched.param_count is not None:
        return SizedSource(enriched.param_count, None, spec.mode.value)
    if enriched.architecture_config is not None:
        return SizedSource(None, dict(enriched.architecture_config), spec.mode.value)
    raise refuse(
        "budget_source_size_unknown",
        f"The {spec.mode.value!r} model source did not report the model's size, so its GPU "
        f"memory cannot be estimated.",
        mode=spec.mode.value,
    )


def _directory(spec: AcquisitionSpec, *, need_config: bool) -> SizedSource:
    from services.acquisition import is_miforge_checkpoint
    from training.acquisition.dtype_gate import assert_trainable, collect_weight_dtypes
    from training.acquisition.local_dir import (
        param_count_from_headers,
        read_config,
        resolve_within_roots,
        weight_files,
    )

    assert spec.path is not None  # the caller's branch
    directory = resolve_within_roots(spec.path)
    if not directory.is_dir():
        raise refuse(
            "model_source_incomplete",
            f"{directory} is not a directory. Name a model directory holding config.json "
            f"and its safetensors weights.",
            path=str(directory),
        )
    kind = "checkpoint" if is_miforge_checkpoint(directory) else "directory"
    config = read_config(directory)
    shards = weight_files(directory)
    # The start's dtype gate: a quantized model's packed headers would understate its size,
    # and the start refuses it anyway, so the preview says so instead of a wrong number.
    assert_trainable(config, collect_weight_dtypes(shards), source=str(directory))
    detail = {"path": str(directory)}
    if need_config:
        return SizedSource(None, config, kind, detail)
    return SizedSource(param_count_from_headers(shards), None, kind, detail)


def _hub(spec: AcquisitionSpec, *, need_config: bool) -> SizedSource:
    from training.acquisition import hub
    from training.acquisition.dtype_gate import assert_trainable

    assert spec.repo is not None  # the caller's branch
    requested = spec.revision or hub.DEFAULT_REVISION
    commit = hub.resolve_revision(spec.repo, requested)
    detail = {"repo": spec.repo, "revision": requested, "revision_resolved": commit}
    config = hub.remote_config(spec.repo, commit)
    count, dtypes = hub.remote_weights(spec.repo, commit)
    assert_trainable(config, dtypes, source=f"{spec.repo}@{commit}")
    if need_config:
        return SizedSource(None, config, "hub", detail)
    return SizedSource(count, None, "hub", detail)


__all__ = ["SizedSource", "size_of"]
