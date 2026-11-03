---
title: Hot update
description: The optional mode in which miForge trains the weights a running miLLM server is serving, without taking it out of service.
---

# Hot update (optional)

Everywhere else in miForge, improving a model means writing a new checkpoint and then arranging
for a server to load it, which is a reload: the server stops, frees its memory, loads the new
weights and resumes. **Hot update removes the reload.** miForge attaches to the weights a running
**miLLM** server is answering from and updates them where they sit, while the server keeps
answering.

:::warning For operators, and off by default
Hot update is the **only** miForge feature that requires another application, and the only one
that needs privileged deployment settings. Everything else in this manual works without it. Turn
it on for a session, and turn it off again afterwards.
:::

## How it works

1. **Attach.** miForge asks miLLM to attach. miLLM shares its weight tensors with miForge's
   training process through the graphics driver's interprocess sharing (CUDA IPC, the
   interprocess communication feature of NVIDIA's CUDA platform). miForge rebuilds the model around
   those shared tensors. Only one trainer can hold the attachment at a time.
2. **Verify.** Before the first step, miForge checks that the model it rebuilt gives the same
   answers as the model miLLM is serving: it scores a fixed prompt on both sides and compares the
   results within a tolerance. A mismatch aborts the run before anything is written.
3. **Step, inside a commit bracket.** Each step computes the update, then asks miLLM to pause
   serving briefly, writes the new weights between a **commit-begin** and a **commit-end**, and
   lets serving resume. A write is held for tens of milliseconds. Requests that arrive during that
   hold wait; they are never answered from a half-written set of weights.
4. **Save through miLLM.** Checkpoints of an attached run are written by miLLM into its own
   checkpoint directory. miForge reads them from there (the `CHECKPOINT_READ_ROOTS` setting) and
   never writes into it.
5. **Detach** cleanly when the run ends.

An adapter run never writes to the served weights, so it never pauses serving at all.

## What protects the served model

- **Never a mix of two steps.** If a write is cut off part-way, miLLM marks its weights **torn**
  and refuses to serve (an explicit error) rather than answer from an inconsistent model. miForge
  then recovers by completing or rolling back the interrupted write. **An interrupted step is
  never retried**, because re-applying half of an update would compound the damage.
- **Never on miForge's own initiative.** miForge never forces, clears or works around another
  process's attachment. If a previous trainer died holding the attachment, miForge reports the
  stale holder and **you** clear it on the miLLM side.
- **Stop, do not retry, on identity errors.** If miLLM refuses miForge's trainer credential, the
  run stops and names the credential as the cause.
- **Refused by name.** A quantized served model, a build mismatch between the two processes, an
  unreachable miLLM, or a missing hot-update capability is each refused before anything is
  attached, with the reason.

## Preparing a session

You need:

1. **miLLM** running and serving a 16-bit model (FP16 or BF16), with its own hot-update support
   switched on for the session.
2. **`MILLM_BASE_URL`** set to miLLM's address.
3. **`MILLM_TRAINER_TOKEN`**: the trainer credential miLLM issues, stored as a **secret**
   (`miforge-secrets` in Kubernetes, `.env` in Compose), never in a manifest or a run log. An
   optional `MILLM_AUDIT_LABEL` is recorded beside it.
4. **A shared interprocess namespace** between the two processes, and the training process running
   as **the same user ID as miLLM's server**, because the files the graphics driver creates for
   sharing are readable only by their owner.
5. **No run in progress.** Changing the deployment restarts the training worker, which ends any
   run it is executing.

### In Kubernetes

The deployed manifests do **not** enable hot update. The repository's
`gitops/components/hostipc/` is an opt-in Kustomize component that, for the session:

- sets `hostIPC: true` on the backend pod (Kubernetes has no per-container interprocess namespace,
  so it is a pod-level switch);
- runs only the training worker as miLLM's user ID, keeping miForge's group, so files the session
  creates stay usable after the switch-back;
- points the session's caches at private temporary paths so nothing shared is written under the
  other user ID.

Enable it by adding the component to the deployed kustomization in one commit, and remove it in
another when the session ends:

```yaml
# gitops/pilot/kustomization.yaml
components:
  - ../components/hostipc
```

A test in the repository fails if the component is enabled without that test being updated in the
same commit, so turning it on is always a deliberate act. Preview the result first with
`kubectl kustomize gitops/overlays/hotupdate`. Do **not** point your GitOps application at the
preview overlay: the image-digest pins are kept in `gitops/pilot/`, and a moved application path
loses them.

miLLM has its own half of this switch in its repository; both must be on.

### In Docker Compose

Stop the default training worker and start the hot-update one, which shares the host's
interprocess namespace:

```bash
docker compose stop celery-train
docker compose --profile hotupdate up -d celery-train-hotupdate
```

## Running an attached run

With hot update available, the **New run** form offers **attach** as a model source. (If it is not
available, the form says why: miLLM not configured, unreachable, or missing the capability.) The
rest of the definition is the same as any run. An attached run started by an AI agent waits for
your approval as `hotupdate_attach` (one approval covers the attach and the training).

While it runs, the run view adds:

- the **attachment record**: miLLM's address, the served model, the holder and the weights state
  (`clean`, `writing` or `torn`);
- **hot-update timings**: for each step, how long serving was paused and how long the commit took,
  judged against miLLM's commit watchdog.

`GET /api/v1/attachment` (the `forge_attachment_status` MCP tool) reads miLLM's attachment state
live.

### Continuing an attached run

When an attached run has ended, **Continue from the last save** in the run view resumes it from the
checkpoint miLLM saved for it: miForge checks that the model miLLM is serving **is** that
checkpoint, verifies it, restores the optimizer state for that step, and continues the same run from
the next step.

### Known limitation

While an attached run is executing, a stop request can be held until the run ends, instead of
taking effect at the next step boundary. Bound attached runs with a `step_count` stop condition
rather than relying on a manual stop.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `MILLM_BASE_URL` | empty | miLLM's address. Empty means hot update is unavailable. |
| `MILLM_TRAINER_TOKEN` | empty | The trainer credential (secret). |
| `MILLM_AUDIT_LABEL` | empty | A label miLLM records with each call. |
| `HOTUPDATE_VERIFICATION_PRECISION` | `float32` | Compare the start-of-run check in 32-bit precision (`float32`) or in the served precision (`native`). If miLLM cannot score in 32-bit for this model, the check falls back to `native` and records why. |
| `HOTUPDATE_VERIFICATION_TOLERANCE_FLOAT32` | `0.0005` | The 32-bit check's tolerance. |
| `HOTUPDATE_VERIFICATION_TOLERANCE` | `0.5` | The served-precision check's tolerance (provisional). |
| `HOTUPDATE_VERIFY_PROMPT` | `The capital of France is` | The fixed prompt for the check. |
| `HOTUPDATE_VERIFY_FLOAT32_DEVICE` | `cpu` | Where miForge runs its 32-bit copy for the check. |
| `HOTUPDATE_REQUEST_TIMEOUT_SECONDS` | `10` | Timeout for ordinary calls to miLLM. |
| `HOTUPDATE_CHECKPOINT_TIMEOUT_SECONDS` | `600` | Timeout for a save or load. |
| `HOTUPDATE_COMMIT_END_TIMEOUT_SECONDS` | `30` | How long to wait for a commit to finish. |
| `HOTUPDATE_COMMIT_WATCHDOG_MS` | `800` | miLLM's commit watchdog, mirrored so a commit's duration can be judged against it. Keep it equal to miLLM's own setting. |
| `HOTUPDATE_PRECOMMIT_RETRIES` | `1` | Retries for a step refused **before** its commit began (nothing was written). Never applies after a commit began. |

The timeouts and tolerances can also be changed at runtime; see [Settings](settings.md).

## Turning it off

Remove the Kubernetes component (or stop the hot-update Compose worker and start the default one),
and switch off miLLM's half. Checkpoints the session wrote stay readable and stay in the
checkpoint tree.
