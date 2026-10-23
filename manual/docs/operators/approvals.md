---
title: Approvals
description: How an AI agent's request to spend the GPU, attach to miLLM, delete, or change a setting waits for the operator.
---

# Approvals

AI agents can use everything in miForge through its [MCP server](../integrations/mcp-server.md).
They can read anything, validate and plan freely, and create new packs and drafts. But four kinds
of action **wait for you**:

| Action | What it covers | Approve button |
|---|---|---|
| `gpu_start` | Starting work on a graphics card: a run, a chain, an out-of-cadence evaluation, an adapter merge | **Approve and start run** |
| `hotupdate_attach` | Attaching to a live miLLM server and training the weights it is serving; continuing an attached run | **Approve attach and start** |
| `delete` | Deleting a pack version or a pack draft | **Approve delete** |
| `config_write` | Setting or clearing a runtime setting | **Approve setting change** |

**Your own clicks in the admin interface are never held.** Only requests that identify themselves
as coming from an agent are.

## How it works

1. An agent calls a gated tool (for example `forge_define_run`).
2. miForge **validates the request with the route's own code** first. An invalid request is refused
   at once and never queued, so you are never asked to approve something that would fail.
3. A valid request is stored and the agent receives an approval identifier with status `pending`.
   **Nothing runs and no graphics card is leased.**
4. The request appears in **Approvals** in the sidebar, which shows a badge with the number
   pending.
5. You approve or reject it. On approval, miForge runs the **stored request exactly once**, as if
   you had made it. The agent polls `forge_get_approval_status` to see the outcome.

## The Approvals screen

Pending requests come first, newest first. Each row says:

- **who asked** (the agent's identity, for example `agent:forge-mcp`);
- **what it will do**, in your terms: for a run, the run identifier, run type, model, graphics card
  and step budget; for an attach, the miLLM address and served model; for a delete, the target; for
  a setting, the key with its old and new values (secrets are shown as `<redacted>`);
- **how old** the request is and **when it expires**.

Decided requests keep their outcome on one line: an executed run start links to its run; a failed
one shows the refusal it met.

## Outcomes

| Status | Meaning |
|---|---|
| `pending` | Waiting for you. |
| `executing` | Approved and running now. |
| `executed` | Done; the result holds the route's answer (for example, the new run). |
| `failed` | Approved, but the route refused when it ran (for example, the card was leased by then). Nothing was half-started. |
| `rejected` | You rejected it; the stored request never runs. |
| `expired` | Nobody decided within 24 hours; it can never run. |

## Safety rules you can rely on

- **An agent cannot approve or reject.** Its attempt is refused (`403 agent_cannot_decide`).
- **A request runs at most once.** A second approval of the same request is refused
  (`409 approval_not_pending`).
- **What you approve is what runs.** The stored request is fingerprinted; if it changed after it
  was stored, the approval fails instead of running something else.
- **Requests expire.** The default is 24 hours (`APPROVAL_TTL_HOURS`).
- **Stopping is never gated.** An agent can cancel a run or stop a chain without approval, because
  stopping is the safe direction and an agent must be able to stop a run it sees misbehaving.

## How a request is identified as an agent's

A request carrying the header `X-Miforge-Agent: agent:<name>` is an agent's; one without it is the
operator's. The name may use lower-case letters, digits and hyphens, up to 48 characters. A
malformed header is refused on every route (`400 invalid_agent_identity`). The miForge MCP server
always sends its own identity and offers agents no way to change it.

:::note This is a guard rail, not a login
miForge has no user accounts. The approval gate keeps an agent from acting on the graphics card,
on live serving, on deletions or on configuration without a person deciding. Network access to the
MCP server is protected by its bearer token; see [The MCP server](../integrations/mcp-server.md).
:::

## Through the API

| Route | Purpose |
|---|---|
| `GET /api/v1/approvals` | List approvals, newest first. |
| `GET /api/v1/approvals/actions` | The four actions and what each means. |
| `GET /api/v1/approvals/{approval_id}` | One approval and its outcome. |
| `POST /api/v1/approvals/{approval_id}/approve` | Approve (operator only). |
| `POST /api/v1/approvals/{approval_id}/reject` | Reject (operator only). |
