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

Each text box in the transcript (a step's content, its reasoning, and each call's arguments) has a
**Copy** button that copies the whole box, including what is scrolled out of view. **Copy page**,
in the header, copies the whole screen as plain text, collapsed reasoning included; on a run that
has ended it waits until the transcript has been read. The header stays at the top while you
scroll, below the approvals banner when one is showing.

**The answer is the editor's own account.** Any figures in it are what the editor claims, and the
transcript labels it so. For runs started with version 4 of the observation tools or later, the
worker also measures the submitted weights on practice validation and shows the result beside the
answer: AUROC against the parent's with its 95% interval (**within noise** when that includes
zero), recall and false alarms. It costs the editor nothing. It is practice data, so the held-out
verdict still decides.

**Practice supplements.** A bundle may carry extra practice rows from other sources, for example
negatives written by a model the probe never saw. They are for measuring only: they are in no
practice split, no program fits on them, and no bar is cut on them. The observation tool
`supplement_metrics` reports each one on its own, for the current weights and the parent, at each
one's own bar. A bundle built before supplements existed answers that it has none.

**The editor can state its figures, and they are checked.** In those runs `submit` takes optional
`figures`: the split they were measured on, and any of AUROC, recall (a fraction) and false alarms.
After the submit, the worker compares them with its own measurement of the submitted weights. A
figure matches when the measurement rounds to it at the precision the editor wrote it (0.9904
does not match 0.990304; 0.9903 does), and false alarms must be equal. The transcript shows the
result beside the answer, and a figure that differs is shown in red with both values. Figures on
another split are not compared, and the line says so. A run that stated none, or one recorded
before this check, reads "The editor stated no figures in its submit." The run document carries
the same result as `submitted_check`. The answer's text is never read for numbers. An inspect
run that submits its answer without calling any observation tool (it spent no forward pass) carries
`submitted_check` too, with `measured: false` and nothing stated or compared: nothing its answer
reports was measured, however confident it reads.

**A write that moves entries only by rounding is flagged.** An editor that reads a tensor at the
default 4 significant digits and writes the whole tensor back from those readings moves every entry
it did not mean to change by its rounding error. When an accepted `write_tensor`, or a `python` run
that writes `W`, changes some entries by no more than a 4-digit reading rounds away, its answer
says how many and names them, so the editor can write only the entries it means, or name them in
its answer. The transcript records it as a `write_noise` line. It is a note only: the write is kept
as made, nothing is refused, and the candidate's patch is the same. Runs recorded before it replay
without it.

**An agent can check a start body before filing it.** `forge_wsp_validate_run`
(`POST /api/v1/wsp/runs/validate`) runs the same checks and refusals as a start and returns the
frozen request, its hash and the approval the start would file. It creates no run and files no
approval.

**A run can switch the editor's Python tool off.** A start request with `python: "off"` gives the
editor no `python` tool and starts no Python kernel; the frozen request and the approval show it.
Without it, the tool follows the operation (an inspect's Python view is read-only).

**The Python tool's names `R`, `W`, `notes` and `np` cannot be taken away.** Code that assigns
one of them (for example `W = np.array(R['weight'])`, meant as a copy) has the assignment undone
as soon as that code has run, and its output says so: `W` is the tensors again, so the next
`W['weight'][:] = ...` changes the weights instead of failing. A list assigned to `notes` becomes
the notes. In an editing run, an accepted `submit` answers how many weights the submitted weights
change against the bundle's, so an answer that says "no edits" over a real change reads against
that line in the transcript. Runs recorded before either behaviour replay without it.

**Budgets the request leaves out** take a default from the editor's endpoint. For a local editor
(miServer, or any local server) the defaults are safety stops that catch a runaway, not limits the
editor works against: 5,000,000 tokens, 100 steps and 2 hours. For an external editor they are
5,000,000 tokens, 50 steps and 1 hour, plus the cost cap (US$10 unless the request names one), which
is the real bound on a hosted run: a cached session sends its whole history again on every request,
and every input token counts toward the token budget, cache reads included. Forward
passes are 0 unless the request names them, for every endpoint. A budget the request names always
wins, within the run's maxima (500 steps, 24 hours). The operator sets the defaults with
`WSP_DEFAULT_TOKENS_LOCAL`, `WSP_DEFAULT_MAX_STEPS_LOCAL` and `WSP_DEFAULT_TIME_BUDGET_S_LOCAL`, and
the same three ending `_EXTERNAL`.

**Hosted editors** (an endpoint on an external host such as `api.anthropic.com`). Registering one
needs its context window (`context_window_tokens`) and, for an Anthropic endpoint, all four prices,
the two cache prices included; without them the registration is refused by name. Its history mode is
`append_only`: nothing already sent to the editor is ever changed, so the provider's thinking and
cache stay valid (an Anthropic endpoint cannot choose `paging`). A long tool result is still cut at
12,000 characters, and `recall` with an `offset` reads the rest. Thinking is an effort level
(`default`, `low`, `medium` or `high`). A refusal by the model ends the run `editor_refused`, with no
candidate. An endpoint registered before this release that lacks a window says so in the endpoint
list (`runs_refused`); set the window and runs start again. One that lacks its cache prices must be
registered again, because its prices are sealed with its key.

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
`gateway_refused`, `token_budget`, `cost_budget` or `time_budget`, and its cause names the refusal
(for example `editor_context_exceeded`, when a request did not fit the editor's context window). A
training run's screen opened with a WSP run's identifier (it starts `wsp_`) sends you to the WSP run.

**Each editor call has a time limit.** On a miServer endpoint the gateway asks miServer to end each
reply after at most 14 minutes (one minute inside the gateway's own 15-minute wait for an answer), or
sooner when less of the run's time is left, with the output cap on or off. A miServer too old to know
this ignores it, and one serving the model on its transformers engine refuses it, so the gateway sends
that call again without it; for those the gateway's wait is the limit. A reply ended there is a
`length` stop; when the call ran the whole limit and stopped short of its token cap, its row in
**Editor calls** carries the code `editor_call_deadline`. A request that would have under 30 seconds
is not sent: it is refused `wsp_budget_exhausted`, and the run stops at `time_budget` and completes.
The worker waits two minutes longer than the gateway, so the gateway always answers first. With
`miwsp_engine`, a reply that is lost (the gateway's wait ran out, or the connection dropped) costs its
step, and the editor is told once to keep its replies shorter; the session goes on. A second lost
reply in a row ends the session as its time budget does: the run reads `time_budget` with the cause
`editor_reply_lost`, completes, and keeps its accepted edits. The transcript shows each one as a
`lost_reply` line.

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
how long it took, the tokens, the tokens read from and written to the provider's cache ("cache not
recorded" for a call from before this release) and the finish reason. It never shows the prompt or the answer: the gateway
records counts and codes only. A call refused before anything was spent is listed but is not counted as
a request.

**Cost and cache.** Each row in the list, and the run screen's **Usage**, add a second line: the run's
cost in US dollars and the tokens its calls read from and wrote to the provider's cache, for example
"USD 0.4210 · 1,204,800 read from cache · 41,000 written to cache" ("cache not recorded" for a run from
before this release, never zero). The **Progress** panel shows the same cost and cache totals. A run
screen also shows the **Effort** the run froze, and, once the run has ended, its **History mode** and
its **Handoffs**, read from the transcript; a run the editor refused says so with the category the
model gave ("The editor refused (category: …)", or "no category given"). The transcript shows each
request's timing, each handoff and a refusal as readable lines.

**Waiting for your approval.** A WSP run an agent has asked for but you have not yet approved has no run
yet; it is listed above the runs as a row marked "Waiting for your approval". Select it to open
Approvals. One caller can hold at most 10 queued WSP runs; a request past that is refused
`wsp_queue_limit_reached`, also when its approval executes, so file it again once one of its runs has
started.

| Action | Route | MCP tool |
|---|---|---|
| List WSP runs | `GET /api/v1/wsp/runs` | |
| Read one | `GET /api/v1/wsp/runs/{run_id}` | |
| Read one: its list row and its document | `GET /api/v1/wsp/runs/{run_id}/record` | `forge_wsp_get_run` |
| Its live progress | `GET /api/v1/wsp/runs/{run_id}/progress` | |
| Its editor calls (counts and codes) | `GET /api/v1/wsp/runs/{run_id}/calls` | |
| Its gateway reservations and usage rows | `GET /api/v1/wsp/runs/{run_id}/ledger` | `forge_wsp_get_run_ledger` |
| Its transcript (`offset`, `limit` and `max_bytes` read a page) | `GET /api/v1/wsp/runs/{run_id}/transcript` | `forge_wsp_get_transcript` |
| Replay it from its transcript | `POST /api/v1/wsp/runs/{run_id}/replay` | `forge_wsp_replay_run` |
| How many times each held-out set has been scored | `GET /api/v1/wsp/heldout-uses` | `forge_wsp_list_heldout_uses` |

**What agents read.** `forge_wsp_get_run` gives a run's list row and its document, without prices or
keys (`forge_get_run` reads training runs only). `forge_wsp_get_transcript` reads the transcript a page
at a time: at most `limit` entries and `max_bytes` of entry JSON per page, never an empty page while an
entry remains, with `total` and `next_offset` to walk it. `forge_wsp_get_run_ledger` lists each
reservation and each usage row the gateway wrote for the run (endpoint, tokens, cost and times) with
totals over every row, so you can check that a run's spend is the sum of its own calls and that its rows
name only its own endpoint. A **replay** runs no editor and no gateway: it costs no tokens and no money
and needs no approval, but it holds one pool worker while it runs and counts toward the caller's queued
runs. `forge_host_resources` reports disk too: the free and total bytes of the checkpoint and WSP
volumes, and of each WSP worker's `/work` filesystem as that worker last reported it, beside the `/work`
size limit. `forge_wsp_list_heldout_uses` lists each held-out set's use count (owner, set name, the
hash of its row ids, uses, when it was last used and by which candidate); a use is counted inside a
verdict's transaction, so read it before and after an evaluation to see what the evaluation spent.
`forge_get_run` given a WSP run's id refuses `run_not_found` and names `forge_wsp_get_run` as the next
step.

## Editor endpoints

An editor endpoint is the model server a WSP session talks to as its editor: miServer in the cluster,
another server on this network, or an external provider. A session's start body names one by its id.
The **Editor endpoints** screen (in the sidebar, after WSP runs) is where you manage them.

**The list** shows each endpoint's id, model, base URL, kind, serving, locality, action protocol,
context window, history mode, thinking (an effort level on an external endpoint) and its reasoning allowance, output cap field, prices, when it was registered and by whom, and its key as **key
sealed** with the key's fingerprint, or **no key**. A key is never shown: miForge cannot open what it
stored, and the screen shows only the record's public fields.

**The window an endpoint's runs get.** Each run reads the server's context window when it starts,
unless the endpoint states one. An endpoint's detail shows the stated window, the read made at
registration, and the latest run's read, then two lines. **Next run expects** is the stated window,
else what the server reported to the latest run (with that run and its time), else 8,192 tokens
assumed. It never uses the read made at registration, which may describe another model. **If the
run cannot read the server** is what a run uses when its own read fails: the stated window, else
8,192 tokens. Agents read the same list with `forge_wsp_list_endpoints`, without the prices or the
key's fingerprint.

**A miServer endpoint never assumes a window.** miServer reports no window while no model is loaded,
so on a miServer endpoint that states none, a run that finds no window waits in the queue instead of
starting on 8,192 tokens: it starts, with the window miServer reports, once a model is loaded. Other
runs in the queue are not held back. If miServer still reports no window after 15 minutes
(`WSP_EDITOR_RESIDENT_WAIT_S`, 900 seconds), the run fails `editor_not_resident`; nothing was sent
and nothing was spent. Load the endpoint's model in miServer and start the run again, or, if the
model is loaded and its configuration states no window, set the endpoint's **Context window**. For
such an endpoint both lines above read **waits** (the list's `source` is `waits`, with `tokens` 0).
While a run waits, its page says **Waiting for the editor model since** the time it began, and its
list row, its progress read and `forge_wsp_get_run` carry that time as `editor_wait_since` (null on
a run that never waited; kept, as the record of the wait, once the run starts).
A replay or a program rerun calls no editor, so it never waits.

**Any other local server that does not answer is waited for too.** On an endpoint that is not
miServer and states no window, a run whose read gets no answer at all (the connection is refused,
the read times out, or the server answers with a 5xx error) waits in the queue the same way: it
starts once the server answers, and fails `editor_not_resident` after the same 15 minutes if it never
does, with nothing sent or spent. While it does not answer, miForge reads it again after 10 seconds,
then after 20, 40 and at most every 60 seconds, so a server that comes back is seen within a minute.
Start the server, or correct the endpoint's base URL, and start the
run again; if the server is up but miForge's backend cannot read it, set the endpoint's **Context
window**. A server that answers but states no window (a key it was not sent, a model listed without
a length, a llama.cpp router) still starts on 8,192 tokens assumed, with its warning. For such an
endpoint the detail shows two lines instead of one, **If the server answers without a window** and
**If the server does not answer**; the list's `window_if_read_fails.no_answer` is `waits` (or
`stated` when the endpoint states a window, `not_read` for an external endpoint, which is never read).

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
  answer ends the session. The engine follows it, and a run can override it (the start body's
  `output_cap`). With thinking on and the cap off, the reasoning allowance takes at most half of the
  cap the gateway sizes.
- **Paging** (`on`, the default, or `off`), for `miwsp_engine` only (see *Paging and the ledger*
  above). With `off` every tool result stays in the conversation and no ledger is kept, as before
  paging existed. A run can override it (the start body's `paging`); the run records `paging: "off"`
  in its request when it is off, and the WSP run screen shows the value in effect. (A run of
  `weightspace`, the provider `miwsp_engine` replaced, shows paging as not applicable.)
- **History mode** (`paging` or `append_only`; empty: `append_only` for an external host, `paging`
  for a local one). `append_only` never changes what was already sent, so a hosted model's cache can
  serve it, and Paging does not apply. An external Anthropic endpoint cannot be `paging`: the form says
  so, and the registration is refused `wsp_endpoint_history_mode_refused`. For an external host the
  form marks the **Context window** required (it is also where a session hands off) and offers the
  effort levels for **Thinking**: `default`, `low`, `medium` or `high` for Anthropic, and `off` as well
  for an OpenAI-compatible host, whose effort is sent as `reasoning_effort`.
- **Prices** in US dollars per million tokens: input and output are required for an external endpoint;
  a local one needs none. Both cache prices are required for an external Anthropic endpoint (the write
  price is the 5-minute price); they are sealed with the key, so they change only by registering again.
- **API key**: required for an external endpoint. It is a password field, sent once with the
  registration, sealed for the gateway and cleared from the form as soon as it is sent, whether the
  registration succeeds or is refused. It is never saved with the form's other entries.

miForge decides the **locality** from the host when you register (local for a private, loopback or
link-local address, `localhost` or a cluster name ending `.svc` or `.cluster.local`; external
otherwise); the form says what it expects, and the list shows what was stored. A refused registration
shows miForge's code and message, and keeps your other entries.

**When runs are refused.** An endpoint on which a run would be refused today says why, in an amber
notice under its details: **Window required: set it to run** (`endpoint_window_required`: set the
context window under Change), or **Cache prices required: register this endpoint again**
(`wsp_endpoint_cache_prices_required`: remove it and register it again with both cache prices and the
key). The notice is the same check a start and the push make.

**Change protocol, window, history mode, ceiling or thinking** changes the action protocol, the context
window (an external endpoint's cannot be cleared), the history mode, thinking, the reasoning allowance,
the output ceiling, the output cap and paging, the only fields that can change: everything else is bound to the sealed key, so remove the endpoint and register it
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
