---
title: Troubleshooting
description: The refusals you are most likely to meet, what each means, and what to do.
---

# Troubleshooting

miForge refuses rather than guesses, and every refusal has a **code** and a sentence saying what to
do. This page collects the ones operators meet most often. The `details` of each refusal name the
field, value or limit involved.

## Starting a run

| Code | What it means | What to do |
|---|---|---|
| `budget_exceeded` | The run's estimated memory is more than the card has free. Both figures are in the details. | Use a smaller model or batch size, `8bit` optimizer state, an `adapter` run, or a card with more free memory. |
| `budget_unknown_capacity` | Free memory on the card could not be read. | Check the driver and that the worker can see the card (`GET /api/v1/gpus`). |
| `gpu_lease_held` | Another run holds this card. | Wait for it, choose another card, or let the janitor free a lease whose worker died. |
| `gpu_request_is_an_index`, `gpu_request_is_a_bus_address`, `gpu_uuid_not_present` | A card was named by index or bus address, or by an identifier this machine does not have. | Use `auto`, or a `GPU-…` identifier from `GET /api/v1/gpus`. |
| `model_source_outside_root` | A load path is outside the allowed roots. | Put the model under the checkpoint root or the parts root (the details list them). |
| `tokenizer_required`, `tokenizer_not_at_root` | A creation run has no tokenizer, or a model directory's tokenizer files are not at its top level. | Give a tokenizer path or Hub repository; move the tokenizer files up. |
| `pack_incomplete` | The pack is missing a required slot (named). | Fill it in **Onboard a domain**, or choose another pack version. |
| `pack_part_unresolved` | A pack slot points at a part this installation does not have. | Import that part version. |
| `stop_condition_config_incomplete` | A stop condition is missing a parameter (named). Stop conditions have no defaults. | Supply it. See [Runs](guides/runs.md#stop-conditions). |
| `sft_masking_policy_not_configured` | A corpus or self-generation run needs a masking policy and none is set. | Set `SIGNAL_SFT_MASKING_POLICY=completion_only`. |
| `retention_floor_missing` | The run has a retention set but no floor. | Set the floor in the pack's retention slot or in the run's override. |
| `judge_provider_not_configured`, `judge_is_resident_model` | A judge selector has no judge endpoint, or the judge would be the model being trained. | Configure `JUDGE_PROVIDER_BASE_URL` and `JUDGE_PROVIDER_MODEL` with a different model. |
| `midataworks_not_configured` | Something needs miDataworks and `MIDATAWORKS_BASE_URL` is empty. | Configure it, or use parts from your own files. |
| `hot_update_mode_not_available` | `attach` was chosen and hot update is not set up. | See [Hot update](operators/hot-update.md), or use `load`. |

## During and after a run

| What you see | What it means | What to do |
|---|---|---|
| A run marked `failed` with no recent heartbeat | The worker stopped writing heartbeats (it crashed or was restarted) and the janitor freed its card. | Continue from the last save. See [Checkpoints](guides/checkpoints.md#continuing-a-run). |
| The loss falls but the test score does not move (self-generation) | The selector is accepting nearly everything, so the run trains on raw output. | Dry-run the selector in the pack wizard; tighten it. |
| `retention_floor_breached` and a stop | The model was getting better at the domain by losing general ability. | Lower the learning rate, mix in a corpus stage, or accept a different trade-off deliberately. The final checkpoint is the model at the breach. |
| `merge_refused_no_unmerged_base` | A merge would leave no copy of the unmerged base. | Keep (or recreate) a checkpoint of the base first. |
| `pack_in_use` | A pack version that runs used cannot be deleted. | Clone it instead, or leave it. |

## Approvals and agents

| Code | What it means |
|---|---|
| `agent_cannot_decide` | An agent tried to approve or reject. Only the operator can. |
| `approval_not_pending` | The approval was already decided (or is executing). |
| `approval_expired` | Nobody decided within 24 hours. The agent must ask again. |
| `approval_tampered` | The stored request changed after it was stored, so it did not run. |
| `invalid_agent_identity` | The `X-Miforge-Agent` header was malformed. |
| `setting_not_runtime` | That setting cannot be changed at runtime. See [Settings](operators/settings.md). |

## Health

`GET /api/health` answers `200` when PostgreSQL and Redis are reachable and `503` (with the same
body) when one is not. Optional integrations are listed by name; "not configured" there is normal.

## Times

The admin interface shows times in US Eastern time, with the exact timestamp on hover. The API and
logs use Coordinated Universal Time (UTC).
