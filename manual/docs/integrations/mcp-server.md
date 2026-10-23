---
title: The MCP server
description: Connect an AI agent to miForge through the Model Context Protocol, with operator approval for anything that spends the GPU.
---

# The MCP server

miForge includes a server for the **Model Context Protocol (MCP)**, the open protocol AI agents use
to call tools. Through it an agent can do everything the admin interface can: read every run,
checkpoint, pack and host reading; validate, plan and dry-run; and **request** the actions that
spend the graphics card, attach to miLLM, delete, or change a setting. Those requests wait for you
in [Approvals](../operators/approvals.md).

## What it offers

**83 tools in 10 categories**; 10 of them wait for your approval when an agent calls them. The full,
generated list is the [MCP tool reference](../reference/mcp-tools.md).

| Category | What its tools do |
|---|---|
| `core` | Health, host resources, graphics cards, model sources, providers, run types, stop conditions, selectors, approvals, and `forge_howto`. |
| `runs` | List and read runs, steps, scores, batches and stop conditions; validate, estimate and start runs; evaluate now; cancel. |
| `inputs` | Signal sources, parts, part import and validation, miDataworks status. |
| `checkpoints` | Checkpoints, manifests, verification, lineage, the tree, save requests and branch plans. |
| `hotupdate` | Attachment state, continue plans, and continuing an attached run. |
| `adapters` | Adapters, merges, and merging an adapter. |
| `packs` | Pack versions: read, validity, lineage, export, create, import, clone and delete. |
| `drafts` | Pack drafts: the seven authoring steps, samples, dry runs, commit and delete. |
| `chains` | Chains: read, create and stop. |
| `settings` | Read settings, and set or clear a runtime override. |

An agent should call **`forge_howto`** first: it explains the workflow, the three shapes a result can
take, the four approval actions, graphics cards and settings, without calling the backend.

### The three result shapes

1. **A resource**: the route's answer, as JSON.
2. **A pending approval**: `{"approval_id", "status": "pending", "action", "summary", "expires_at", ...}`.
   The agent polls `forge_get_approval_status` until the status is `executed` (with the result),
   `rejected`, `expired` or `failed` (with the refusal).
3. **Unavailable**: `{"unavailable": "backend", "reason": ...}` when the miForge backend cannot be
   reached. The MCP server never raises on a dead backend.

## How it is built

- The MCP server is a separate process (`python -m mcp_server`) that reaches miForge **only over its
  REST API**. It has no database, no Redis, no volumes and no graphics card, so every agent call goes
  through the same API, validation and approval gate as everything else.
- It sends `X-Miforge-Agent: agent:forge-mcp` (the `MCP_AGENT_IDENTITY` setting) on every request,
  which is what marks the call as an agent's. Agents cannot change it.
- Every tool call is logged as one structured line: tool, identity, a digest of the arguments
  (secrets redacted), status and duration.

## Connecting an agent

The server speaks **streamable HTTP** on port 8765 at the path `/mcp`. Every request except
`/health` must carry the bearer token:

```text
Authorization: Bearer <MCP_AUTH_TOKEN>
```

The server **refuses to start** without a token. Generate a long random one and keep it secret:
`MCP_AUTH_TOKEN` in Compose's `.env`, or `MCP_BEARER_TOKEN` in the `miforge-secrets` Secret in
Kubernetes.

A typical MCP client configuration looks like this (the exact format depends on your client):

```json
{
  "mcpServers": {
    "miforge": {
      "type": "http",
      "url": "https://mcp-forge.example.com/mcp",
      "headers": {"Authorization": "Bearer YOUR_TOKEN"}
    }
  }
}
```

With Docker Compose on the same machine, the address is `http://127.0.0.1:8765/mcp`.

`GET /health` (no token) reports whether the MCP server can reach the backend.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `MIFORGE_API_URL` | `http://localhost:8000` | Where the backend is. |
| `MCP_AUTH_TOKEN` | none (required) | The bearer token. |
| `MCP_AGENT_IDENTITY` | `agent:forge-mcp` | The identity sent to the backend. |
| `MCP_PORT` | `8765` | The listening port. |
| `MCP_TOOL_CATEGORIES` | all ten | Which categories to register, comma-separated. |

## Turning agent access off

Scale the MCP server to zero replicas (or stop the `mcp` Compose service). Nothing else is affected.

## Safety, in one paragraph

An agent can read and plan freely. It cannot start graphics-card work, attach to miLLM, delete, or
change a setting without you approving that exact request, once, within 24 hours. It cannot approve
its own requests. Stopping a run or a chain is never gated. Details are in
[Approvals](../operators/approvals.md).
