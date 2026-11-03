---
title: Run types
description: Change a model's weights, create a model from nothing, or train an adapter and merge it.
---

# Run types

Three run types share one loop. The step, the scoring, the saving and the stop conditions are
the same for all three; what differs is **what is being changed**.

| Run type | Changes | Starts from | Writes |
|---|---|---|---|
| `change` | Every parameter of an existing model | A loaded model | Full checkpoints |
| `create` | Every parameter of a brand-new model | Random weights from an architecture configuration | Full checkpoints, starting with a step-0 baseline |
| `adapter` | A small low-rank adapter only; the base is untouched | A loaded model | Adapter directories |

## `change`: update a model's weights directly

The default. miForge loads a model, trains all of its parameters toward your pack, and saves
full checkpoints. Use it when you want the model itself to become better at your domain.

Memory is the constraint. A full-parameter run of a 1.2-billion-parameter model needs roughly 12
to 14 GB of graphics-card memory (gradients, 32-bit master weights, 8-bit optimizer states and
checkpointed activations). The budget estimate shows the real figure for your model before you
start.

## `create`: a model that did not exist

A creation run needs **no base model at all**: an architecture configuration (a `config.json`
describing the model's shape, given as a path or inline) and a **tokenizer** (a directory or a
Hub repository). miForge never trains a tokenizer, so a creation run is refused without one.

On the **New run** form, choosing **create** shows a **Tokenizer** list: every folder under the
model roots with tokenizer files at its top level, whether a tokenizer on its own, a model folder
or a checkpoint. Or name a Hugging Face repository and revision instead, and press **Check**:
miForge says which commit the revision points at now and whether that commit has tokenizer files,
and **Use this commit** fixes the run to those exact files. **Check before committing** then shows
the tokenizer the run will use and how many tokens it has. The tokenizer's vocabulary must fit
the `vocab_size` in the architecture configuration, or the run is refused before it starts.

The run saves a **step-0 checkpoint before any training**, so every later score has a baseline to
compare with. Its manifest has no parent, and its `domain` may be empty: a general pretraining
run biases toward nothing, and the manifest says so explicitly.

A creation run can use any signal source, but self-generation, preference pairs and reward start
from a model that produces noise. That is allowed and recorded in the run log, not refused.

The seed (yours, or one miForge generates) fixes the random starting weights, so the same seed
and definition reproduce the same step-0 model.

## `adapter`: specialise without touching the base

An adapter run trains a **low-rank adapter** (the technique usually called LoRA, low-rank
adaptation) beside the model. The base model's bytes are **unchanged** at the end, which you can
check by hashing them before and after. Gradients and optimizer state cover only the adapter, so
an adapter run fits where a full-parameter run does not.

Its settings:

| Setting | Meaning |
|---|---|
| `rank` | The adapter's rank: how many directions it can learn. At least 1. |
| `alpha` | The adapter's scale numerator. Greater than 0. The adapter's effect is scaled by alpha divided by rank. |
| `target_modules` | Which of the model's layers get an adapter. At least one. |
| `dropout` | Dropout applied inside the adapter during training. |

An adapter is saved as a standard Parameter-Efficient Fine-Tuning (PEFT) directory with its own
manifest (`adapter-manifest-v1`): its base model, rank, scaling and a hash of its weights. That
manifest lets the adapter travel without its run log: copy it to another machine and load it with
the `peft` library against the base it names.

## Merging an adapter

When you have finished exploring, you can **merge** an adapter into its base and get an ordinary
checkpoint with nothing to track at serving time. In **Checkpoints**, choose **Merge into
weights** on an adapter (or `POST /api/v1/adapters/{name}/merge`).

A merge cannot be undone by another merge, so miForge **refuses a merge unless a checkpoint of the
unmerged base still exists**. The merged checkpoint's manifest names both ancestors: its parent
and the adapter merged into it. A merge that is interrupted leaves either no new checkpoint or a
complete one; the unmerged base is untouched either way.

Every merge attempt is recorded, written, refused (with the reason) or interrupted, at
`GET /api/v1/merges`.

A merge runs on the graphics card, so an AI agent's merge request waits for your approval.
