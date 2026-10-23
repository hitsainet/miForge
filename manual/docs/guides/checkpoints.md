---
title: Checkpoints
description: Saved models, the checkpoint tree, continuing and branching, and pruning old checkpoints safely.
---

# Checkpoints

Every save point of a run is a **checkpoint**: a complete, standard model directory you can load
anywhere. Checkpoints are miForge's product; everything else exists to make good ones.

## What a checkpoint is

```text
<checkpoint root>/
  notary-1/                       one directory per run
    step-000000/
    step-000100/
      model.safetensors           the weights
      config.json                 the architecture
      tokenizer.json ...          the tokenizer's own files, at the top level
      manifest.json               what this checkpoint is (miForge's only extra file)
    latest -> step-000100         a relative link to the newest save
```

- The **directory name** pads the step to six digits so directories sort in order
  (`step-000100`).
- The **full name** does not pad it: `LFM2-1.2B:notary-1:step-100`, that is,
  `base-model:run-id:step-N`. The base model is the root of the lineage, so a branch of a branch
  still names the model it all started from.
- A checkpoint is written to a hidden temporary directory, flushed to disk and then renamed into
  place. **It is either absent or complete**; nothing can ever read a half-written one.
- Optimizer state is never written into a checkpoint. It lives separately, under the trainer
  state root, so a checkpoint stays a clean, portable model.

The manifest records the run and step, the base model, the parent checkpoint, the domain, the
test and retention scores (with provider and sample size), and a hash of the weights. Its format
is published as `checkpoint-manifest-v1`; see [Artifacts and schemas](../reference/artifacts.md).

**The files on disk are the truth.** miForge's database is an index of the manifests and can be
rebuilt from them alone. Where the two disagree, the manifest wins and the disagreement is logged.

## The **Checkpoints** screen

**Checkpoints** shows every checkpoint and adapter of every run as a forest: each checkpoint
under its parent, adapters under the model they were trained against, merged checkpoints under
both. For each node you can:

- **Read the manifest**, in full.
- **Merge into weights**, for an adapter. See [Run types](run-types.md#merging-an-adapter).

## Saving on demand

Besides the `save_every` cadence and the final save, you can ask a running run to save at its next
step boundary: `POST /api/v1/runs/{run_id}/checkpoints` (the `forge_request_save` MCP tool). The
save records the step that had completed; it never happens in the middle of a step.

## Verifying a checkpoint

`POST /api/v1/runs/{run_id}/checkpoints/{step}/verify` recomputes the weights hash and compares it
with the manifest. `GET /api/v1/runs/{run_id}/checkpoints/{step}/lineage` walks the parent chain
back to the root, through the manifests alone.

## Continuing a run

A run that was interrupted can carry on from its last save. The plan
(`POST /api/v1/runs/{run_id}/continue`, the `forge_plan_continue` MCP tool) tells you which
checkpoint will be used and whether the optimizer state for that step will be **restored**
(the loss curve continues smoothly) or start **fresh** (and why). A crash costs at most one save
interval.

For a run attached to miLLM in hot-update mode, **Continue from the last save** in the run view
submits the continue directly (see [Hot update](../operators/hot-update.md)). For a standalone
run, start a new run that loads the checkpoint directory.

## Branching from any checkpoint

To try a different direction without losing the first one, branch: start a **new run** (a new run
identifier) from any checkpoint. The branch's first saved manifest names the branch point as its
parent, and the tree shows it there. The branch plan
(`POST /api/v1/runs/{run_id}/checkpoints/{step}/branch`, the `forge_plan_branch` MCP tool) gives
you the new run's parent and confirms the starting point.

A branch always starts with **fresh** optimizer state, because optimizer state belongs to the run
that wrote it; the run log records that discontinuity rather than hiding it.

## Pruning old checkpoints

A long run with a frequent save cadence can fill a disk. miForge has a **checkpoint lifecycle**
policy that keeps:

- the **last N** saves of every run (`CHECKPOINT_LIFECYCLE_KEEP_LAST_N`, default 10);
- **every Mth** save, counted from the run's first save (`CHECKPOINT_LIFECYCLE_KEEP_EVERY_M`,
  default 10);
- and, unconditionally, **every checkpoint something descends from**: a branch's parent, an
  adapter's base, a merged checkpoint's adapter.

Everything else is deleted. The sweep runs every hour.

:::warning Off by default, and dry run by default
Pruning is **disabled** (`CHECKPOINT_LIFECYCLE_ENABLED=false`) and, once enabled, runs as a **dry
run** (`CHECKPOINT_LIFECYCLE_DRY_RUN=true`) that only logs what it would delete. Read that plan in
the maintenance worker's log before you set dry run to `false`, because `false` means **delete**.
These settings can be changed at runtime; see [Settings](../operators/settings.md).
:::

The lineage rule is a correctness rule, not a space rule: deleting a checkpoint that a branch
descends from would cut that branch off from its history. Ancestry is computed once over all
checkpoints before anything is deleted.
