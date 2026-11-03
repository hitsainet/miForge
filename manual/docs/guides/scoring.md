---
title: Scoring
description: How miForge measures domain improvement and general-ability loss, and why every score names its provider.
---

# Scoring

A training loop that cannot tell improvement from damage is not worth running. miForge scores two
held-out sets on a cadence and records every score with enough detail to reproduce it.

## Test and retention

| Set | The question | Scored every |
|---|---|---|
| **Test set** | Is the domain getting better? | `test_every` steps (default 1) |
| **Retention set** | Is general ability being spent to buy that? | `retention_every` steps (default 1) |

The test set holds domain prompts with an expectation: a selector verdict, a grader score, or a
reference answer matched exactly or loosely. The retention set holds general prompts **outside**
the domain, scored the same way, and carries a **floor**.

Both scores are on a 0 to 1 scale, written to the run log at every scoring point, and copied into
each checkpoint's manifest when it is saved. So a checkpoint found months later still says how it
scored.

### The floor

A model pushed hard at one domain answers that domain better and everything else worse. Loss
falls, the test score rises, nothing errors. **The retention set is the only instrument that sees
it.** If a retention score falls **below** the floor (equal is not a breach), miForge records a
breach, and a run with the `retention_floor` stop condition stops there and saves a final
checkpoint, so you have the model at the point it went too far.

The floor comes from the pack's retention slot, or from the run's `retention_floor_override`. The
run records which one it used.

A set used for scoring is never also training material in the same run: held-out data stays held
out.

## Providers: who answers

To score a model, miForge has to ask it questions. It does that through a **provider**:

| Provider | What it is | Needs |
|---|---|---|
| `in_process` (default) | The model miForge is holding, asked directly. | Nothing else running. |
| An OpenAI-compatible endpoint | Any server that speaks the OpenAI-compatible API: miLLM, vLLM, Ollama, a llama.cpp server, or a hosted service. | Configuration (below). |

Configure an endpoint with environment variables, one set per provider name you choose:

```bash
PROVIDER_LOCALVLLM_BASE_URL=http://vllm.example.com:8000/v1
PROVIDER_LOCALVLLM_MODEL=my-served-model
PROVIDER_LOCALVLLM_API_KEY=            # optional; a local server usually needs none
```

A base address without a model (or the reverse) is refused, and a base address containing a
user name and password is refused: put credentials in `_API_KEY`, never in the address.
`GET /api/v1/providers` lists the providers and whether each is reachable, and
`POST /api/v1/providers/{name}/check` sends one prompt and shows the answer.

### Why every score names its provider

A number from a greedy pass on the in-process model and a number from a sampled request to a
different server are **two different measurements**. Plot them on one curve and the curve is a
fiction. So every score records the provider, the decoding parameters, the aggregation method and
the sample size, and the run view marks any point where the provider changed.

## The scoring block

A run definition's `scoring` block chooses how scores are taken. Everything is optional:

```json
"scoring": {
  "provider": "in_process",
  "aggregation": "mean",
  "decode": {"max_tokens": 128, "temperature": 0.0}
}
```

Decoding is **greedy by default** (temperature 0), which makes a score repeatable: load the
checkpoint at step N into any OpenAI-compatible server, score the same set the same way, and you
should get the score miForge recorded at step N.

## Scoring outside the cadence

`POST /api/v1/runs/{run_id}/evaluate` scores a run's model now, against its pinned provider (the
`forge_evaluate_now` MCP tool). Because it uses the graphics card, an AI agent's request waits for
your approval.

## Reading scores

| Route | MCP tool | Returns |
|---|---|---|
| `GET /api/v1/runs/{run_id}/scores` | `forge_list_scores` | Every test and retention score, with sample size and provider. |
| `GET /api/v1/runs/{run_id}/scores/{step}` | `forge_get_scores_at_step` | The scores at one step, with per-item detail. |
