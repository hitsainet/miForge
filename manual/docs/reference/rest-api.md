---
title: REST API
description: The shape of miForge's HTTP API, its error envelope, and where to find every route.
---

# REST API

Everything the admin interface and the MCP server do goes through miForge's HTTP API. The admin
interface is just one client of it.

## Basics

| Item | Value |
|---|---|
| Base path | `/api/v1` (health is at `/api/health`) |
| Interactive documentation | `/api/docs` on your installation |
| Machine-readable description | `/api/openapi.json` (OpenAPI) |
| Success | Plain JSON resources |
| Errors | `{"error": {"code": "...", "message": "...", "details": {...}}}` |
| Authentication | None: miForge runs on a trusted network with no user accounts |

The OpenAPI document marks every route that waits for operator approval when an agent calls it with
`x-approval-action`.

## Errors are refusals with names

Every failure carries a stable, lower-case **code** (for example `budget_exceeded` or
`pack_incomplete`), a sentence saying what happened and what to do, and **details** naming the field,
the value and the limit involved. miForge prefers refusing by name to guessing: a missing value is
refused rather than defaulted, and an unknown field is refused rather than ignored.

An absent optional application is **not** an error: it is reported as not configured, by name, and
only the feature that needs it is refused. See [Troubleshooting](../troubleshooting.md) for the codes
you are most likely to meet.

## Agent requests

A request with the header `X-Miforge-Agent: agent:<name>` is an AI agent's. Requests that start
graphics-card work, attach to miLLM, delete or change a setting are answered `202` with a pending
approval instead of running. See [Approvals](../operators/approvals.md).

## Route families

| Family | Routes begin with | Covered in |
|---|---|---|
| Health and host | `/api/health`, `/api/v1/host`, `/api/v1/gpus` | [Installation](../getting-started/installation.md) |
| Models | `/api/v1/models` | [Your first run](../getting-started/first-run.md) |
| Runs | `/api/v1/runs`, `/api/v1/budget`, `/api/v1/run-types`, `/api/v1/stop-conditions` | [Runs](../guides/runs.md) |
| Signal sources and selectors | `/api/v1/signal-sources`, `/api/v1/selectors`, `/api/v1/providers` | [Signal sources and judging](../guides/signals-and-judging.md) |
| Scores | `/api/v1/runs/{run_id}/scores` | [Scoring](../guides/scoring.md) |
| Checkpoints and the tree | `/api/v1/checkpoints`, `/api/v1/runs/{run_id}/checkpoints`, `/api/v1/tree` | [Checkpoints](../guides/checkpoints.md) |
| Adapters and merges | `/api/v1/adapters`, `/api/v1/merges` | [Run types](../guides/run-types.md) |
| Chains | `/api/v1/chains` | [Chains](../guides/chains.md) |
| Parts and packs | `/api/v1/parts`, `/api/v1/packs`, `/api/v1/pack-drafts`, `/api/v1/midataworks` | [Domain packs](../guides/packs.md) |
| Hot update | `/api/v1/attachment`, `/api/v1/runs/{run_id}/continue` | [Hot update](../operators/hot-update.md) |
| Approvals | `/api/v1/approvals` | [Approvals](../operators/approvals.md) |
| Settings | `/api/v1/settings` | [Settings](../operators/settings.md) |

Every route also has an MCP tool; the [MCP tool reference](mcp-tools.md) maps each tool to its route.

## Live updates

A running run pushes progress to the admin interface over a Socket.IO connection at `/socket.io`.
The run record over HTTP is always the authority; the live channel is a convenience.
