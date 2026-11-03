---
title: Chains
description: Run several stages one after another, each from the last saved checkpoint, as a curriculum.
---

# Chains

A **chain** runs several stages in order. When one stage stops, the next starts **from the last
checkpoint the previous stage saved**, with a different signal source or domain. This is how a
model is taught through a curriculum: for example, corpus first, then self-generation, then
preference pairs.

## Defining a chain

A chain has a name and **at least two stages**. Each stage is a full
[run definition](runs.md#the-run-definition) plus the stop reasons that let the chain advance past
it:

```json
{
  "name": "notary-curriculum",
  "stages": [
    {
      "definition": { "...": "a run definition with a corpus signal source" },
      "advance_on": ["step_count", "loss_plateau"]
    },
    {
      "definition": { "...": "a run definition with a self-generation signal source" },
      "advance_on": ["test_threshold", "step_count"]
    }
  ]
}
```

Submit it with `POST /api/v1/chains` (the `forge_create_chain` MCP tool). **Every stage is
validated when the chain is created**, so a mistake in stage four is found now, not three hours
in. The graphics card is leased for the chain as a whole, and stage 0 starts.

Only stage 0 may use a model source other than loading. Every later stage must declare
`"mode": "load"`; the chain fills in the path with the previous stage's last save, and keeps your
stage definition as you wrote it.

## When a chain advances, and when it halts

A stage may advance on `step_count`, `test_threshold` or `loss_plateau`. A chain **never**
advances past:

- **a retention-floor breach**: the next stage would start from the checkpoint whose general
  ability had just collapsed, and push further in the same direction;
- **your own stop**: advancing would restart work you had just stopped;
- **a failed stage**: the next stage would start from a checkpoint nothing verified.

A configuration naming any of those as an advance reason is refused. When a stage ends for a
reason that does not advance, the chain **halts** and records why.

A background sweep checks finished stages every two minutes and starts the next one.

## Watching and stopping a chain

| Action | Route | MCP tool |
|---|---|---|
| List chains | `GET /api/v1/chains` | `forge_list_chains` |
| One chain, with each stage's outcome and advance decision | `GET /api/v1/chains/{chain_id}` | `forge_get_chain` |
| Halt the chain and stop its running stage | `POST /api/v1/chains/{chain_id}/stop` | `forge_stop_chain` |

Each stage is an ordinary run, so its run view, scores and checkpoints are where they always are.
