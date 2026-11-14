---
title: Your first run
description: From an installed miForge to a trained checkpoint, step by step.
---

# Your first run

This walkthrough takes you from a fresh installation to a saved, scored checkpoint. It uses the
default path: miForge loads a model itself, trains it in its own process and writes its own
checkpoints, with no other application running.

You will:

1. put a model and some data where miForge can read them;
2. describe your domain as a **pack**;
3. define and start a **run**;
4. watch it, and find the checkpoints it saved.

Every screen is in the sidebar on the left. To give the page more room, click the round arrow on
the sidebar's right edge: it narrows to icons, and pointing at an icon shows its name. Click the
arrow again to widen it. miForge remembers the choice in this browser.

## 1. Make a model and data available

### A model

A run starts from one of three **model sources**:

| Source | What you give it |
|---|---|
| **Load from a directory** | A standard Hugging Face model directory (weights, `config.json` and tokenizer files at the top level). For safety, a load may only read under miForge's checkpoint root or parts root, so copy or mount the model there. |
| **Load from the Hugging Face Hub** | A repository name and a revision (a branch, tag or commit). miForge records the exact commit it resolved to. |
| **Create** | An architecture configuration and a tokenizer, for a model that starts from random weights. The form lists the tokenizer folders it can see, or takes a Hugging Face repository. |

The **New run** form lists the model directories it can see, so you can check a model is in the
right place before you start.

### Data (parts)

A pack is built from **parts**: a prompt set, optionally a corpus or preference pairs, a test set
and a retention set. Parts reach miForge in one of two ways:

- **From miDataworks**, if you run it and have set `MIDATAWORKS_BASE_URL`. miForge lists the
  dataset versions miDataworks offers and imports them under a versioned contract.
- **From your own files**, prepared as an export directory in the dataset-version format
  (`docs/schemas/midataworks-dataset-version-v1.json` in the repository) and imported with
  `POST /api/v1/parts/import` (or the `forge_import_part` MCP tool). A part from your own files
  is recorded as **external**, with the provenance you supply, and is never presented as if it
  came from miDataworks.

`POST /api/v1/parts/validate` checks a part's manifest without importing it and names exactly
what is wrong.

## 2. Describe your domain as a pack

Open **Onboard a domain** in the sidebar. The wizard has seven steps, and your progress is saved
as you go:

1. **Describe** the domain.
2. **Prompts**: choose the prompt set.
3. **Answers**: optionally choose a corpus and/or preference pairs.
4. **Judgment**: optionally choose a selector and/or grader configuration, and **dry-run** it on
   a sample to see each verdict with its reason before you commit to it.
5. **Improvement**: choose the test set, and look at a scored sample.
6. **Retention**: choose the retention set and its floor, and look at a scored sample.
7. **Review**: see anything still missing, by name, and commit the draft as a pack version.

A draft is not a pack until you commit it in step 7. Details are in [Domain packs](../guides/packs.md).

## 3. Define and start a run

Open **New run**. The form collects one complete run definition:

- **Model source**: load, create or (in hot-update mode only) attach.
- **Run type**: `change` for your first run (update the model's weights directly).
- **Pack**: the pack version you just committed.
- **Signal source**: `corpus` is the simplest (train on example text).
- **Optimizer**: learning rate, warm-up steps, batch size, gradient accumulation, optimizer-state
  precision (8-bit by default) and an optional seed.
- **Cadences**: how often to score the test set, score the retention set and save a checkpoint.
- **Stop conditions**: at least one. For a first run, a **step count** of a few hundred steps is
  a good choice.

Before you start, the **budget preview** estimates the graphics-card memory the run needs beside
the memory that is free. A run that will not fit is refused before anything is loaded.

Press **Check before committing** to run the same validation the start button runs, without
creating anything. Each problem appears next to the field it concerns, naming what failed. Then
start the run.

:::note Nothing is filled in for you
The form validates nothing on its own and invents no default for anything that decides when a
run ends. A field you leave empty is sent as absent, and the backend refuses it by name. This is
deliberate: a silently-filled blank is a run that does something you did not ask for.
:::

## 4. Watch it run

The run opens in the **run view**:

- **State and liveness**: running, with the age of the last heartbeat.
- **Tiles**: the current step against the step budget, the latest loss, and the latest test and
  retention scores.
- **Curves**: the loss at every step, and the test and retention scores at each scoring point,
  with save points and the retention floor marked.
- **Step log**: each step's loss, gradient norm, learning rate and wall time, beside the batch
  it trained on.
- **Manifest**: the latest checkpoint's manifest.

To see how close each stop condition is, ask the API
(`GET /api/v1/runs/{run_id}/stop-conditions`): each condition comes back with its target, its
latest value and how much headroom is left.

To stop a run early, send `POST /api/v1/runs/{run_id}/cancel` (or the `forge_cancel_run` MCP
tool). Cancellation is cooperative: the run stops at its next step boundary, records
`operator_stop` as its reason and saves a final checkpoint.

## 5. Find your checkpoints

Open **Checkpoints**. Every save point of every run is listed as a tree: each checkpoint under
its parent, and adapters under the model they were trained against. **Read the manifest** shows
exactly what a checkpoint is: its run, step, parent, scores and weights hash.

Every checkpoint is a standard model directory under the checkpoint root:

```text
checkpoints/
  notary-1/
    step-000000/
    step-000100/
    step-000200/
    latest -> step-000200
```

Copy one anywhere and load it with any software that reads Hugging Face models. See
[Checkpoints](../guides/checkpoints.md) for continuing, branching and pruning.
