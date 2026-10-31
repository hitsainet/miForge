---
title: Runs
description: Define a run, start it, watch it, and understand what stops it.
---

# Runs

A **run** trains one model toward one domain, step by step, until a stop condition fires. This
page covers the run definition, the screens, and the rules that decide when a run ends.

## The run definition

A run is created from one **definition**, submitted whole. The **New run** form builds it, and
the API accepts the same document at `POST /api/v1/runs/define`.

| Part | What it says | Required? |
|---|---|---|
| **Run identifier** | The name of the run, used in its checkpoints' names. Leave it empty and miForge makes one. | no |
| **Model source** | Where the starting weights come from: `load` (a directory or a Hugging Face Hub repository), `create` (an architecture configuration and a tokenizer), or `attach` (hot-update mode only). | yes |
| **Run type** | `change`, `create` or `adapter`. See [Run types](run-types.md). | yes |
| **Pack** or **parts** | A pack version, or a list of loose parts. Exactly one of the two. | yes |
| **Signal source** | Where training examples come from. See [Signal sources and judging](signals-and-judging.md). | yes |
| **Optimizer configuration** | Learning rate, warm-up steps, batch size, gradient accumulation (default 1), optimizer-state precision (`8bit`, the default, or `32bit`) and an optional seed. | yes |
| **Scoring** | Which provider answers for the test and retention sets. See [Scoring](scoring.md). | no |
| **Retention floor override** | A floor for this run, replacing the pack's. | no |
| **Cadences** | How often to score and save (below). | no |
| **Stop conditions** | At least one (below). | yes |
| **Refresh** | For self-generation: how the prompt set is renewed between rounds (below). | no |
| **Adapter** | For an `adapter` run: rank, alpha, target modules and dropout. | adapter runs |
| **Domain** | The domain the run is biased toward. Taken from the pack when there is one. | no |
| **GPU request** | `auto`, or a specific card's `GPU-…` identifier. | no |

A definition with an unknown field is refused, so a misspelled field can never be silently
ignored. A required part left out is refused by name. Nothing that decides **when a run ends**
is ever defaulted.

### The seed

If you give no seed, miForge **generates one and records it**, so every run can be repeated. The
seed fixes a `create` run's random starting weights and is part of the optimizer configuration's
identity. It does not make generated text repeat; sampling during self-generation and scoring has
its own seed.

### Check before committing

`POST /api/v1/runs/validate` (the form's **Check before committing**) runs the exact validator
the start path runs, and creates nothing. A clean check does not reserve anything: the
validator runs again at submit, because a part can stop resolving in between.

### Will it fit?

Before any memory is allocated, miForge estimates the run's graphics-card memory (weights,
gradients, optimizer states and activations) and compares it with the free memory on the chosen
card. A run that will not fit is refused with **both figures**, rather than crashing twenty
minutes later. The New run form's **Estimate GPU memory** button shows this estimate before you
start (`POST /api/v1/budget/estimate`). It reads the model's size from the model source you
chose: the weight files of a directory or checkpoint, the Hub repository's weight headers (nothing
is downloaded), an architecture config, or the size miLLM reports for the model it is serving.
With the GPU request left on `auto`, every visible GPU is listed with its free memory and its own
verdict; with a GPU named, only that one is.

## Cadences

| Cadence | Default | Meaning |
|---|---|---|
| `test_every` | 1 | Score the test set every N steps. |
| `retention_every` | 1 | Score the retention set every N steps. |
| `save_every` | none | Save a checkpoint every N steps. With none, the run still saves at the end. |
| `refresh_every` | none | For self-generation: a *round* is N steps, and the prompt set is refreshed between rounds. |

Each cadence is a positive whole number or empty. A cadence only changes **when** something
recorded happens, which is why it may have a default; the record says when it happened.

:::caution The retention floor is seen with a lag
A breach of the retention floor can only be seen when the retention set is scored. With
`retention_every` set to N, up to N - 1 more steps of training can happen between the breach and
the stop. At the default of 1 the lag is zero. The run view states the lag for every run.
:::

## Stop conditions

Every run needs **at least one** stop condition. Each condition's parameters must be given; none
has a default.

| Condition | Parameters | Stops when |
|---|---|---|
| `step_count` | `total` (a whole number, at least 1) | The number of completed steps reaches `total`. |
| `test_threshold` | `threshold` (0 to 1) | The latest test score is at or above `threshold`. |
| `retention_floor` | none (uses the floor) | The latest retention score is **below** the floor. Equal to the floor is not a breach. |
| `loss_plateau` | `metric` (`raw` or `window_mean`), `window`, `min_relative_improvement`, `min_steps` | Loss has improved by less than `min_relative_improvement` over the last `window` steps, once at least `min_steps` steps have run. The window never spans a prompt refresh. |
| `operator_stop` | none | You cancel the run. |

An example, in the definition:

```json
"stop_conditions": [
  {"condition": "step_count", "config": {"total": 2000}},
  {"condition": "retention_floor", "config": {}},
  {"condition": "test_threshold", "config": {"threshold": 0.85}}
]
```

### When two fire at once

Every condition that fired is recorded, but a run has one **stop reason**, chosen in this fixed
order: `operator_stop`, then `retention_floor`, then `test_threshold`, then `loss_plateau`, then
`step_count`. Your own stop comes first because a person asserted it; the floor comes next
because "it stopped because general ability fell" matters more than "it reached its target".

### What happens at the end

When a stop condition fires, the run **saves a final checkpoint**, including after a floor
breach or your own stop: the artifact tells the truth about where the run got to. A run that
**fails** (an error, not a stop condition) does not save again; its last good save is the
recovery point.

## Prompt refresh (self-generation)

A self-generation run can renew its prompts between rounds, with one of two modes:

- `new_prompt_set`: load a new prompt set.
- `template_seeds`: answers the selector **accepted** become seeds for new prompts through a
  templating rule. Only accepted answers seed; an undecided answer never does. The model never
  writes its own prompt set.

## The screens

**Runs** lists every run, newest first, with its state, type and creation time, a page at a
time. Select one to open its run view.

The **run view** shows the run's state and heartbeat age, tiles for the current step and the
latest loss and scores, the loss and score curves with save points and the floor marked, the step
log, and the latest checkpoint's manifest. A run's live updates arrive over a socket connection;
the run record is always the authority, and an update that disagrees with it is discarded and
counted.

## Operating a run through the API

| Action | Route | MCP tool |
|---|---|---|
| Read a run | `GET /api/v1/runs/{run_id}` | `forge_get_run` |
| Its steps | `GET /api/v1/runs/{run_id}/steps` | `forge_list_steps` |
| Its stop conditions and headroom | `GET /api/v1/runs/{run_id}/stop-conditions` | `forge_get_run_stop_conditions` |
| What it was asked to do | `GET /api/v1/runs/{run_id}/definition` | `forge_get_run_definition` |
| Stop it | `POST /api/v1/runs/{run_id}/cancel` | `forge_cancel_run` |
| Save at the next step boundary | `POST /api/v1/runs/{run_id}/checkpoints` | `forge_request_save` |
| Score it now, out of cadence | `POST /api/v1/runs/{run_id}/evaluate` | `forge_evaluate_now` |

The definition endpoint returns both what you **submitted** and what it **resolved** to (for
example, the seed miForge generated or the commit a Hub revision pointed at).

## One run per card

A run takes a **lease** on its graphics card. A second run for the same card is refused while
the first holds it. If a worker dies, its heartbeat stops; after the janitor's threshold (15
minutes by default) the run is marked failed and the lease is released.
