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
is downloaded), an architecture config, or the size miServer reports for the model it is serving.
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

## Weight-space programming runs

A weight-space programming (WSP) run is not a training run: an editor model is given a probe's
weights and a task, reads and changes them through tools, and may submit a candidate. An agent asks
for one, and it waits in **Approvals** until you approve it. These runs have their own screens.

**WSP runs** lists every one, newest first: its state, provider and version, kind and mode, why it
stopped (the stop reason, the outcome and the recorded cause), its usage in requests and tokens,
the forward passes it spent, when it was created and finished, and the caller's reference. Select
one to open it.

The **WSP run** screen shows the same, plus what the run produced (its candidate, patch and
verdict, when present) and the editor's whole transcript. The transcript is shown as plain text;
an assistant turn written as JSON is laid out readably, and a turn that is not valid JSON is shown
as written. Both screens only read: nothing is started, stopped or approved from them.

**Budgets the request leaves out** take a default from the editor's endpoint. For a local editor
(miServer, or any local server) the defaults are safety stops that catch a runaway, not limits the
editor works against: 5,000,000 tokens, 100 steps and 2 hours. For an external editor they are
100,000 tokens, 50 steps and 1 hour, plus the cost cap (US$10 unless the request names one). Forward
passes are 0 unless the request names them, for every endpoint. A budget the request names always
wins, within the run's maxima (500 steps, 24 hours). The operator sets the defaults with
`WSP_DEFAULT_TOKENS_LOCAL`, `WSP_DEFAULT_MAX_STEPS_LOCAL` and `WSP_DEFAULT_TIME_BUDGET_S_LOCAL`, and
the same three ending `_EXTERNAL`.

**Paging and the ledger** (`miwsp_engine`, on by default). A tool result longer than 1,500 bytes,
such as a read of a whole 4,096-value weight, stays in the editor's conversation for the next three
requests and is then replaced by a short note, `[page p5: read_tensor name=weight, 12052 bytes;
recall("p5") to see it again]`; the editor's `recall` tool brings the full text back (a step, no
forward pass). A measurement result pages above 160 bytes, because its numbers are already in the
**ledger**: one line per measurement or edit (the turn, the action, a short digest of the weights
before and after, what has changed against the start, and the AUROC, recall, false-alarm rate and
bar the measurement returned), which the engine builds from the tool results, never from the
editor's words. Notes are replaced, and the ledger moved to the end of the conversation, at most once
every four turns, or sooner when the conversation passes half the context window; in between the
conversation only grows, so the server's prompt cache and the gateway's token estimate stay valid.
The task, refusals, the engine's own notices and the last three turns are never replaced. When the
conversation must still restart to fit the window, the restart carries the ledger (its newest rows,
or only the page names, when the window is small) and the list of pages. The engine judges when to
restart from the editor's own token counts, counting anything added since at a token a byte, as the
gateway does. The transcript keeps every result in full once, and shows the ledger
rows (`ledger`), each batch of notes (`page_out`) and each recall (`recall`) as readable lines; the
run's list row counts `pages_out`, `recalls`, `ledger_rows` and `peak_prompt_tokens` (the largest
prompt the engine estimated). A replay pages exactly as the run did. **Paging switches itself off for an
editor whose context window is under 16,384 tokens**, whatever the endpoint or the run asks: in a small window the
conversation restarts every few turns, before anything is old enough to page, so paging only cost room. The run's
transcript and result then say `paging: "off"` with `paging_off_reason: "window_small"`.

A run that stopped because the editor endpoint or its budget refused a request reads
`gateway_refused`, `token_budget` or `cost_budget`, and its cause names the refusal (for example
`editor_context_exceeded`, when a request did not fit the editor's context window). A training
run's screen opened with a WSP run's identifier (it starts `wsp_`) sends you to the WSP run.

Both screens **keep themselves current**; there is nothing to refresh. The list re-reads every 5 seconds
while any run on it is still live (queued, pushing, running or collecting) or a WSP request is waiting for
your approval, and every 30 seconds otherwise. A run's screen re-reads every 5 seconds until the run is
completed, failed or cancelled, then stops and reads the transcript once (the worker writes it only at
the end). Each screen says when it was last updated; if a read fails, it keeps what it last showed and
says how old that is.

**While a run is live** its row shows how long it has been running (or waiting, while queued), its place
in the queue, how many editor calls it has made, the tokens they used, how long ago the worker last
answered, and how much of each budget is spent. Its screen has a **Progress** panel with the same in
full: the worker's state, the last heartbeat, the calls (metered and refused), tokens in and out, cost,
the last call's time, and a bar per budget: tokens, cost, time (counted from when the run was pushed to
the worker), and steps. Steps are not counted while a session runs, so the steps bar shows editor calls
against the step limit and is labelled an estimate.

**Editor calls** lists one row per call the gateway saw: its time, what became of it (served, an
upstream error, no answer, refused by the gateway with its code, or the session ended), the HTTP status,
how long it took, the tokens and the finish reason. It never shows the prompt or the answer: the gateway
records counts and codes only. A call refused before anything was spent is listed but is not counted as
a request.

**Waiting for your approval.** A WSP run an agent has asked for but you have not yet approved has no run
yet; it is listed above the runs as a row marked "Waiting for your approval". Select it to open
Approvals. One caller can hold at most 10 queued WSP runs; a request past that is refused
`wsp_queue_limit_reached`, also when its approval executes, so file it again once one of its runs has
started.

| Action | Route |
|---|---|
| List WSP runs | `GET /api/v1/wsp/runs` |
| Read one | `GET /api/v1/wsp/runs/{run_id}` |
| Its live progress | `GET /api/v1/wsp/runs/{run_id}/progress` |
| Its editor calls (counts and codes) | `GET /api/v1/wsp/runs/{run_id}/calls` |
| Its transcript | `GET /api/v1/wsp/runs/{run_id}/transcript` |

## Editor endpoints

An editor endpoint is the model server a WSP session talks to as its editor: miServer in the cluster,
another server on this network, or an external provider. A session's start body names one by its id.
The **Editor endpoints** screen (in the sidebar, after WSP runs) is where you manage them.

**The list** shows each endpoint's id, model, base URL, kind, serving, locality, action protocol,
context window, thinking and its reasoning allowance, output cap field, prices, when it was registered and by whom, and its key as **key
sealed** with the key's fingerprint, or **no key**. A key is never shown: miForge cannot open what it
stored, and the screen shows only the record's public fields.

**Add** takes every field the registration accepts:

- **Id**: `a-z`, `0-9`, `_` and `-`, at most 40 characters. Empty, miForge names it. An id is never reused.
- **Model**: the name the server expects in a request.
- **Base URL**: `http` or `https`, no user name or password in it, a path that is empty or exactly
  `/v1`, no query. A local host must be in `WSP_LOCAL_HOST_ALLOWLIST`; an external host must use `https`
  and be in `WSP_EXTERNAL_HOST_ALLOWLIST`.
- **Kind** (`openai_compatible` or `anthropic`) and **Serving** (`miserver` for a miServer host, which
  requires it; otherwise `generic`).
- **Action protocol** (`tool_calls`, the default, or `structured`, for an OpenAI-compatible miServer
  endpoint only), **Output cap field** (empty: the kind's default) and **Context window** (empty: the
  window the server reports).
- **Thinking** (`off`, the default; `on`; or `default`) and **Reasoning allowance** (empty: 4,096
  tokens per turn). With `off`, `miwsp_engine` asks the editor not to think: on an OpenAI-compatible
  miServer endpoint it sends `chat_template_kwargs: {enable_thinking: false}`. With `on` (an
  OpenAI-compatible miServer endpoint only) it sends `enable_thinking: true` and gives each turn the
  allowance for reasoning beside the action's own cap, both inside the context window (at most half
  of it); in the structured protocol the allowance is sent as `max_reasoning_tokens`, and miServer
  closes the reasoning when it runs out and still writes the action. With `default` it sends nothing and the server's chat template decides. The reasoning the
  editor sends back is kept in the run's transcript, marked as reasoning, and never run as an action.
  With the output cap off, a turn whose remaining context could give the reasoning fewer than 256
  tokens is sent with thinking off, and the transcript records it (`thinking_skipped`). A turn
  that could not hold even the action is not sent: the conversation is restarted first, and if
  that still leaves no room, the run stops (`editor_context_window_too_small`). If the last turn's
  `submit` cannot be read (usually because the reply was cut off), the editor is asked once more,
  with thinking off.
  A run can override the endpoint's setting: the start body's `thinking` (`off`, `on` or `default`)
  and `thinking_tokens`; the run records the value in effect in its request, and the WSP run screen
  shows it.
- **Output cap** (`on`, the default, or `off`). With `on`, each editor request names a per-turn output
  cap (2,048 tokens, at most an eighth of the context window). With `off`, it names none, and the
  gateway sizes one to what the context window has left (the endpoint's context window, else the
  window miServer serves, else 8,192 tokens, less the prompt), so a reply can be as long as the
  window allows; the run's token and time budgets still apply. The gateway estimates the prompt from
  the editor's own token counts on the session's earlier calls, plus a 10% margin, and keeps at
  least 1,024 tokens for the reply while the prompt leaves that much. If miServer still answers that
  the request is past its context, the gateway sends it once more with a smaller cap; a second such
  answer ends the session. Both providers follow it, and a run can override it (the start body's
  `output_cap`). With thinking on and the cap off, the reasoning allowance takes at most half of the
  cap the gateway sizes.
- **Paging** (`on`, the default, or `off`), for `miwsp_engine` only (see *Paging and the ledger*
  above). With `off` every tool result stays in the conversation and no ledger is kept, as before
  paging existed. A run can override it (the start body's `paging`); the run records `paging: "off"`
  in its request when it is off, and the WSP run screen shows the value in effect. A `weightspace`
  run is refused a `paging` of its own.
- **Prices** in US dollars per million tokens: input and output are required for an external endpoint;
  a local one needs none.
- **API key**: required for an external endpoint. It is a password field, sent once with the
  registration, sealed for the gateway and cleared from the form as soon as it is sent, whether the
  registration succeeds or is refused. It is never saved with the form's other entries.

miForge decides the **locality** from the host when you register (local for a private, loopback or
link-local address, `localhost` or a cluster name ending `.svc` or `.cluster.local`; external
otherwise); the form says what it expects, and the list shows what was stored. A refused registration
shows miForge's code and message, and keeps your other entries.

**Change protocol, window or thinking** changes the action protocol, the context window, thinking, the
reasoning allowance, the output cap and paging, the only fields that can change: everything else is bound to the sealed key, so remove the endpoint and register it
again. A queued or approved run on that endpoint then fails at its start with `wsp_endpoint_changed`,
because the endpoint it was approved for changed. **Remove** asks you to confirm; a run that names a
removed endpoint can no longer start. **Run protocol test** queues a short inspect run on miForge's
built-in test bundle (it needs `miwsp_engine` enabled) and links it; its row in WSP runs shows how
many turns broke the one-action rule, and 0 is clean. It runs with the endpoint's thinking: with
thinking on, the row's `reasoning_returned` says whether the server sent the editor's reasoning
back, which is how you check a server honours it.

**An agent's proposal.** An agent can propose a local endpoint with `forge_wsp_register_endpoint`: no
API key, no prices, a local host. A proposal with a key, a price, or an external base URL or locality
is refused before anything is filed (`wsp_endpoint_agent_key_refused`,
`wsp_endpoint_agent_prices_refused`, `wsp_endpoint_agent_external_refused`). Otherwise it waits in
**Approvals** as *Register a local editor endpoint*, showing the id, model, base URL, protocol and the
rest of what will be registered; approving registers it once, recorded as created by that agent.
Changing or removing an endpoint stays yours: an agent is refused `wsp_endpoint_operator_only`. An agent
that wants a different protocol proposes a second endpoint.

| Action | Route | Who |
|---|---|---|
| List | `GET /api/v1/wsp/endpoints` | anyone |
| Register | `POST /api/v1/wsp/endpoints` | the operator; an agent's local, keyless proposal waits for approval |
| Change the protocol, window or thinking | `PATCH /api/v1/wsp/endpoints/{endpoint_id}` | the operator |
| Remove | `DELETE /api/v1/wsp/endpoints/{endpoint_id}` | the operator |
| Protocol test | `POST /api/v1/wsp/endpoints/{endpoint_id}/protocol-test` | anyone; an agent's on an external endpoint waits |

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
