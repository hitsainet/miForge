---
title: Working with the suite
description: How miForge uses miDataworks, miLLM and miStudio when they are present, and works without them when they are not.
---

# Working with the suite

miForge is **standalone**. It loads or creates a model itself, trains in its own process and writes
its own checkpoints. Every sibling application is optional, detected at runtime, and reported by
name when it is missing. `GET /api/health` lists each integration and its state.

| Application | What miForge uses it for | Configure with | Without it |
|---|---|---|---|
| **miDataworks** | Dataset parts for packs, under a versioned contract; the human-review queue for the human-review selector. | `MIDATAWORKS_BASE_URL` | Import parts from your own files; use rule, judge, reference or composite selectors. |
| **miLLM** | Serving miForge's checkpoints; scoring through its OpenAI-compatible endpoint (as one provider among many); the optional [hot-update](../operators/hot-update.md) mode. | `MILLM_BASE_URL`, and a `PROVIDER_*` entry to score through it | Score in-process; serve checkpoints with any compatible server. |
| **miStudio** | Interpretability of the models miForge produces. | | Nothing in miForge depends on it. |

## miDataworks: parts

miDataworks is where dataset parts are written, reviewed and versioned. miForge **consumes** its
dataset versions under the `midataworks.dataset-version/v1` contract (vendored, byte for byte, in
miForge's `docs/schemas/`), and refuses anything that does not conform, naming the field.

- `GET /api/v1/midataworks/status` says whether miDataworks is configured and reachable, and what is
  unavailable without it.
- `GET /api/v1/midataworks/available-parts` lists the versions it offers, by kind.
- `POST /api/v1/parts/import` with `"source": "http"` imports one.

A part miDataworks did not publish is recorded as **external** and is never presented as
miDataworks-sourced.

## miLLM: serving and scoring

A miForge checkpoint is a standard model directory, so miLLM loads it like any other model. Nothing
needs to be pushed into miLLM: point it at the checkpoint directory.

To score through miLLM, add it as a provider (see [Scoring](../guides/scoring.md#providers-who-answers)).
miLLM is one OpenAI-compatible endpoint beside vLLM, Ollama and others; nothing in miForge treats it
specially there.

## Contracts

The formats miForge produces and consumes are published as JSON Schema files in the repository's
`docs/schemas/`. See [Artifacts and schemas](../reference/artifacts.md).
