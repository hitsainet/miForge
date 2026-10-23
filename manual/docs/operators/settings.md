---
title: Settings
description: Where miForge's configuration comes from, which settings can change at runtime, and the full list.
---

# Settings

miForge is configured from **environment variables** (or an `.env` file in development; a
ConfigMap and a Secret in Kubernetes). Names are not case-sensitive, and an unknown name is
ignored, so check spelling carefully.

A small, safe set of settings can also be **changed at runtime**, without a restart, through the
settings API. Those changes are stored in miForge's database as **overrides** and are read by the
API and by every worker at the start of each task.

## Reading the settings

`GET /api/v1/settings` (the `forge_get_settings` MCP tool) returns every setting with its
**effective value**, its **source** (`env` or `override`) and whether it is **runtime-writable**.
Each setting also carries a plain-language `label` and `description`, its `effect` (when a change
takes effect: `now`, `next_sweep`, `next_attached_run` or `restart`, with the pod to restart), its
`env_value` (the environment's value, beside the one in force), and who last changed it and when
(`updated_by`, `updated_at`, `last_change`). Secrets, and any credential inside a database or
Redis address, are shown as `<redacted>`, in the environment value too.

## The Settings screen

**Settings**, at the bottom of the sidebar beside the book icon that opens this manual, shows the
same list. Each setting you can change at runtime shows its plain name, what it does, a badge
saying **when a change takes effect**, the value in force beside the environment's value, and
when it was last changed and by whom (with the approval, when an AI agent's approved request made
the change). Each has a field, an optional **Reason**, **Save override** and, once an override is
set, **Use the environment value**. Everything else is listed read-only under **Needs a restart**,
with the pod to restart: change those in miForge's environment and restart the
`miforge-backend` pod. A restart stops running jobs. **Recent changes** at the bottom lists the
last 50 changes, newest first. The version under Settings is the miForge version you are running.

When a change takes effect:

| Badge | Settings | When |
|---|---|---|
| Takes effect now | `log_level`, `millm_base_url`, `midataworks_base_url` | At once in the API; each worker before its next task. |
| At the next pruning sweep | the four `checkpoint_lifecycle_*` settings | When the pruning sweep next starts. |
| At the next hot-update run | the five hot-update timeouts and tolerances | When the next attached run starts; a running one keeps its value. |
| Needs a restart | everything else | When the `miforge-backend` pod restarts. |

## The change history

Every change made through the settings API, from the screen or by an agent's approved request,
is recorded: the setting, the value before and after (and whether each came from the environment
or an override), who made it (`operator`, or the agent's name with the approval's id) and the
reason, if one was given. The record **cannot be edited or deleted**; the database refuses both.

```bash
# the last 50 changes, newest first; ?key= narrows it to one setting, ?limit= up to 500
curl http://127.0.0.1:8010/api/v1/settings/history?key=log_level
```

The `forge_get_settings_history` MCP tool reads the same record.

## Changing a setting at runtime

```bash
# set an override, with an optional reason that is kept in the history
curl -X PUT http://127.0.0.1:8010/api/v1/settings/checkpoint_lifecycle_enabled \
     -H 'Content-Type: application/json' -d '{"value": true, "reason": "the volume is filling"}'

# clear it: the environment value applies again
curl -X DELETE 'http://127.0.0.1:8010/api/v1/settings/checkpoint_lifecycle_enabled?reason=done'

# a stored log level is applied when the API starts, so a restart keeps it in force
```

Only these keys are runtime-writable:

| Setting | Default | What it does |
|---|---|---|
| `checkpoint_lifecycle_enabled` | `false` | Turn checkpoint pruning on. See [Checkpoints](../guides/checkpoints.md#pruning-old-checkpoints). |
| `checkpoint_lifecycle_dry_run` | `true` | When `true`, pruning only logs what it would delete. `false` deletes. |
| `checkpoint_lifecycle_keep_last_n` | `10` | Keep each run's last N saves. |
| `checkpoint_lifecycle_keep_every_m` | `10` | Keep every Mth save. |
| `millm_base_url` | empty | miLLM's address, for hot update. |
| `midataworks_base_url` | empty | miDataworks' address, for parts and human review. |
| `log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `hotupdate_request_timeout_seconds` | `10` | Hot update: timeout for an ordinary request to miLLM. |
| `hotupdate_checkpoint_timeout_seconds` | `600` | Hot update: timeout for a save or load, which writes a whole checkpoint. |
| `hotupdate_commit_end_timeout_seconds` | `30` | Hot update: how long to wait for a commit to finish. |
| `hotupdate_verification_tolerance` | `0.5` | Hot update: the tolerance for the start-of-run check in the served precision. |
| `hotupdate_verification_tolerance_float32` | `0.0005` | Hot update: the tolerance for the start-of-run check in 32-bit precision. |

Anything else is refused (`422 setting_not_runtime`). In particular, the database and Redis
addresses, every storage root, the miLLM trainer token and the environment name can **never** be
changed at runtime: a storage root needs a volume anyway, and a secret sent through an agent would
sit in an approval record.

When an AI agent sets or clears a setting, the change waits for your approval (`config_write`).
See [Approvals](approvals.md).

## Environment reference

### Core services and storage

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | local PostgreSQL | The PostgreSQL connection address. |
| `REDIS_URL` | local Redis | The Redis address (message broker and progress channel). |
| `CHECKPOINT_ROOT` | `/data/checkpoints` | Where checkpoints are written. |
| `TRAINER_STATE_ROOT` | `/data/trainer-state` | Where optimizer state is written, separately from checkpoints. |
| `PARTS_ROOT` | `/data/parts` | Where imported parts live. |
| `PACKS_ROOT` | `/data/packs` | Where committed packs live. |
| `CHECKPOINT_READ_ROOTS` | empty | Extra checkpoint directories miForge reads but never writes (comma-separated). Used by hot update. |
| `HF_HOME` | | The Hugging Face cache. Keep it outside the storage roots. |
| `HF_TOKEN` | | A Hugging Face access token, for gated Hub models. |

A model `load` from a directory may only read under `CHECKPOINT_ROOT` or `PARTS_ROOT`.

### Optional integrations

Empty means **not configured**, which is normal: miForge needs neither.

| Variable | Meaning |
|---|---|
| `MILLM_BASE_URL` | miLLM's address. Needed only for hot update. |
| `MIDATAWORKS_BASE_URL` | miDataworks' address, for importing parts and for human review. |
| `MIDATAWORKS_AGENT` | The identity miForge sends to miDataworks (default `agent:miforge`). |

### Providers and judges

| Variable | Meaning |
|---|---|
| `PROVIDER_NAME_BASE_URL`, `PROVIDER_NAME_MODEL` | An OpenAI-compatible provider called `name`. Both are required together. See [Scoring](../guides/scoring.md#providers-who-answers). |
| `PROVIDER_NAME_API_KEY` | Its key, if it needs one. |
| `JUDGE_PROVIDER_BASE_URL`, `JUDGE_PROVIDER_MODEL`, `JUDGE_PROVIDER_API_KEY` | The judge model for judge selectors. |
| `JUDGE_CONCURRENCY` | Requests sent to the judge at once (default 4). |

### Signals

| Variable | Default | Meaning |
|---|---|---|
| `SIGNAL_SFT_MASKING_POLICY` | none | How corpus and self-generation examples are masked. `completion_only` trains on completions (and on all of a plain-text row). Unset, those runs are refused. |
| `SIGNAL_DPO_BETA` | `0.1` | Strength of the preference objective. |
| `SIGNAL_REWARD_GROUP_SIZE` | `4` | Answers generated per prompt for a reward run. |
| `SIGNAL_REWARD_GROUPS_PER_BATCH` | `4` | Prompts per reward batch. |
| `SIGNAL_GENERATION_CONCURRENCY` | `4` | Generation requests at once. |

### Jobs

| Variable | Default | Meaning |
|---|---|---|
| `HEARTBEAT_MIN_INTERVAL_SECONDS` | `60` | The minimum time between a run's heartbeat writes. |
| `JANITOR_LAPSE_SECONDS` | `900` | How stale a heartbeat must be before the janitor marks the run failed and frees its card. |
| `GPU_LEASE_SECONDS` | `3600` | How long a card lease lasts before it must be renewed. |
| `APPROVAL_TTL_HOURS` | `24` | How long an agent's pending request waits before it expires. |
| `LOG_LEVEL` | `INFO` | Logging detail. |

### MCP server

| Variable | Meaning |
|---|---|
| `MIFORGE_API_URL` | Where the MCP server reaches the backend. |
| `MCP_AUTH_TOKEN` | The bearer token every MCP request must carry. Required. |
| `MCP_AGENT_IDENTITY` | The identity the MCP server sends (default `agent:forge-mcp`). |
| `MCP_TOOL_CATEGORIES` | Which tool categories are enabled (comma-separated). |

### Hot update

The `MILLM_TRAINER_TOKEN` (a secret) and the `HOTUPDATE_*` settings are described in
[Hot update](hot-update.md#settings).

The repository's `.env.example` lists every variable, with the optional ones blank.
