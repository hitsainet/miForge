---
title: Key ideas
description: The handful of words miForge uses everywhere, in plain language.
---

# Key ideas

These are the words the rest of this manual, the admin interface and the API use. Each is
short; read them once and the other pages will make sense.

## Weights, checkpoints and steps

- **Weights** (or *parameters*) are the numbers inside a model. Training changes them.
- A **step** is one round of training: miForge takes a batch of examples, measures how wrong the
  model is on them (the *loss*), and nudges the weights to be less wrong. Steps are numbered from
  0.
- A **checkpoint** is the model saved to disk at one step. It is a complete, standard model
  directory that any compatible server can load. A checkpoint's full name is
  `base-model:run-id:step-N`, for example `LFM2-1.2B:notary-1:step-400`.
- A **manifest** (`manifest.json`) sits inside every checkpoint and says what it is: which run
  and step produced it, which model the lineage started from, its parent checkpoint, its
  scores, and a hash of its weights. See [Artifacts and schemas](../reference/artifacts.md).

## Runs

A **run** is one training job: one model, one pack (or set of parts), one signal source, one
configuration, and at least one condition that ends it. A run has a **run identifier** (for
example `notary-1`) that names its checkpoints and its log.

A run moves through these states: `pending` (queued), `running`, then one of `completed`,
`failed` or `cancelled`. While it runs it writes a **heartbeat**; if the heartbeat goes stale
for too long, a background **janitor** marks the run failed and frees its card, so a crashed
worker never holds a graphics card forever.

There are three **run types**: `change` (update an existing model's weights), `create` (build a
new model from random weights) and `adapter` (train a low-rank adapter beside a model). See
[Run types](../guides/run-types.md).

## Packs and parts

- A **part** is one piece of data: a prompt set, a corpus of example text, a set of preference
  pairs, a test set or a retention set. Parts can be your own files or come from miDataworks.
- A **pack** is a named, versioned bundle of the parts one domain needs, plus how to judge good
  output. A pack version pins the exact version of each part, so a pack version names a
  reproducible run input. See [Domain packs](../guides/packs.md).

## Signal sources

The **signal source** is where a run's training examples come from. There are four:
`corpus` (example text), `self-generation` (the model's own answers that a judge accepted),
`preference-pairs` (a better and a worse answer to the same prompt) and `reward` (several
answers to each prompt, each scored). See [Signal sources and judging](../guides/signals-and-judging.md).

## Selectors and graders

A **selector** decides whether an answer is good enough (accept or reject). A **grader** gives
an answer a score. Both are plugins: a rule, a judge model, a reference answer, human review, or
a combination. The model being trained is **never** its own judge.

## Test, retention and the floor

At a cadence you choose, miForge scores two held-out sets:

- the **test set** asks *is the domain getting better?*
- the **retention set** asks *is general ability being spent to get there?*

The retention set has a **floor**. If the retention score drops below it, the run stops: the
model is getting better at your domain by forgetting other things. See [Scoring](../guides/scoring.md).

## Providers

A **provider** is how miForge gets answers from a model, for scoring and for self-generation. The
default is **in-process**: miForge asks the model it is holding, with nothing else running. You
can also point a provider at any server that speaks the OpenAI-compatible API (such as miLLM,
vLLM or Ollama). Every score records which provider produced it, because two scores are only
comparable if they were taken the same way.

## Stop conditions

A run ends when one of its **stop conditions** fires: a step count, a test-score threshold, the
retention floor, a loss plateau, or the operator pressing stop. Every run needs at least one.
See [Runs](../guides/runs.md#stop-conditions).

## GPUs and leases

miForge runs **one run per graphics card** and takes a **lease** on the card for the run's
duration. Cards are always named by their unique identifier (a `GPU-…` UUID, the universally
unique identifier the NVIDIA driver reports), never by an index number, because index numbers
can change when a machine restarts. Before a run starts, miForge estimates the memory it needs
and **refuses** a run that will not fit, showing the estimate beside the free memory.

## Approvals

When an **AI agent** (not you) asks miForge to start work on a graphics card, attach to a live
server, delete something, or change a setting, the request waits in **Approvals** until you
approve or reject it. Your own clicks in the admin interface are never held. See
[Approvals](../operators/approvals.md).

## Hot update

**Hot update** is an optional mode where miForge trains the weights a running miLLM server is
answering from, without taking it out of service. It is off by default, needs extra setup, and
is the only feature that requires another application. See [Hot update](../operators/hot-update.md).
