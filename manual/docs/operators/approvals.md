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
4. The request appears in **Approvals** in the sidebar, whose badge shows the number waiting
   ("3 waiting"), and in a **banner at the top of every screen**.
5. You approve or reject it, from the banner or from the Approvals screen; both are the same
   decision. On approval, miForge runs the **stored request exactly once**, as if you had made it.
   The agent polls `forge_get_approval_status` to see the outcome.

## The banner on every screen

While any request waits, an amber banner sits above each screen's content, below the resource
strip. It shows up to three requests as full cards, the soonest to expire first, with their
**Approve** and **Reject** buttons, and a line "and N more waiting — Open Approvals" beyond three.
**Collapse** reduces it to its heading for the rest of the browser session; a new request opens it
again. If the queue cannot be read, the banner says so and why, instead of disappearing. It is not
shown on the Approvals screen itself, which shows the same cards.

## The Approvals screen

The screen has two sections.

**Waiting for you (n)** lists the waiting requests, the soonest to expire first. Each card shows,
in this order:

1. **what it will do**, in plain words, and the status **Waiting**;
2. **who asked** ("asked by agent forge-mcp");
3. **when** it was asked and how long until it expires (hover for the exact instant);
4. the **facts** you decide on: for a run, the run identifier, run type, model, graphics card and
   step budget; for an attach, the miLLM address and served model; for a delete, the target; for a
   setting, the key with its old and new values (secrets are shown as `<redacted>`);
5. **All arguments the agent sent**, collapsed, listing every stored argument;
6. the **request id**, with a copy button;
7. the approve button, named for what it starts (**Approve and start run**), and **Reject**.
   Reject opens a field, **Reason (optional)**, up to 1,000 characters; the agent reads the reason.
   Left blank, miForge records "Rejected by the operator."

**History** lists the decided requests, the most recent decision first, with the columns
Outcome, Action, Asked by, Asked, Decided by, Decided, Reason, Result and Id. A request approved
with **Approve all settings changes** says so under Decided by, with the batch id it shares with
the others approved in the same action. The result links an
executed run start to its run, shows the refusal a failed one met, or says "expired without a
decision". It loads the latest 200 requests; **Show older** loads 200 more, up to 500.

## Approving all waiting settings changes at once

When **two or more** settings changes wait (set or clear a setting; the action is
`config_write`), the banner and the Approvals screen show **Approve all settings changes (N)**. It
opens a confirmation listing every one: the setting, its value now → the value asked for, and who
asked. **Approve these N settings changes** approves exactly the ones listed; a request that arrives
while the confirmation is open is not included. Each is then approved, applied and recorded on its
own, as if you had approved it from its card, and History shows each one. If any listed request is
no longer waiting (decided elsewhere, expired) or is not a settings change, **nothing is approved**
and miForge names the request; one change that fails while applying does not stop the others.
There is no "reject all": reject each request on its card. With one settings change waiting, use
its own **Approve setting change** button.

Times are shown in your browser's time zone, with its abbreviation; hovering shows the exact
instant in Coordinated Universal Time (UTC). The suite's four apps show approvals the same way.

## Outcomes

| Shown as | Status | Meaning |
|---|---|---|
| Waiting | `pending` | Waiting for you. |
| Running | `executing` | Approved and running now. |
| Approved and run | `executed` | Done; the result holds the route's answer (for example, the new run). |
| Approved, then failed | `failed` | Approved, but the route refused when it ran (for example, the card was leased by then). Nothing was half-started. |
| Rejected | `rejected` | You rejected it; the stored request never runs. |
| Expired | `expired` | Nobody decided within 24 hours; it can never run. "Decided by" reads "nobody (expired)". |

## Safety rules you can rely on

- **An agent cannot approve or reject.** Its attempt is refused (`403 agent_cannot_decide`).
- **A request runs at most once.** A second approval of the same request is refused
  (`409 approval_not_pending`), including inside an approve-all.
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
| `POST /api/v1/approvals/approve-batch` | Approve several waiting settings changes with one action, `{"approval_ids": [...]}` (operator only). Answers `{batch_id, count, results}`; `409 approval_batch_refused` names every id that is not a waiting settings change, and then nothing is approved. |
