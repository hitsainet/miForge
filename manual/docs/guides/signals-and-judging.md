---
title: Signal sources and judging
description: The four places training examples come from, and the plugins that decide what is good.
---

# Signal sources and judging

A run learns from its **signal source**, and, for the sources that generate answers, a
**selector** or **grader** decides which answers are worth learning from.

## The four signal sources

| Source | Learns from | Needs | Training objective |
|---|---|---|---|
| `corpus` | Example text you supply | A corpus part with a `text`, `messages`, or `prompt` and `completion` column | Supervised fine-tuning (SFT): learn to reproduce the examples |
| `self-generation` | The model's own answers that a selector **accepted** | A prompt set and a selector | Supervised fine-tuning on the accepted answers |
| `preference-pairs` | A better and a worse answer to the same prompt | Preference pairs (`chosen` and `rejected`, usually with a `prompt`) | Direct preference optimization (DPO): prefer the chosen answer over the rejected one |
| `reward` | Several answers per prompt, each scored by a grader | A prompt set and a grader | Group-relative policy optimization (GRPO): answers scored above their group's average are reinforced |

Every batch a run trains on is recorded as a **batch record**: where its examples came from (by
content hash), which model generated them, and how many answers were accepted, rejected or are
still undecided. `GET /api/v1/runs/{run_id}/batches` lists them.

### How corpus text is masked

For text with a prompt and a completion, only the completion is trained on by default; the
prompt's tokens are masked out. A plain-text corpus row is all completion, so every token of it
trains. This is the `completion_only` masking policy (`SIGNAL_SFT_MASKING_POLICY`). miForge
refuses a corpus or self-generation run if no policy is configured, rather than choosing one
silently.

### Self-generation can drift, so watch it

A self-generation run that is not really selecting trains on the model's raw output: the loss
falls, the test score does not move, and nothing errors. That is why every batch record carries
the acceptance rate and its sample size, and the run view warns when a selector accepts or
rejects almost everything. Check a selector with a **dry run** in the pack wizard before you start
(see [Domain packs](packs.md)).

### Reward groups need disagreement

The reward objective compares each answer with the other answers to the same prompt. A group
whose answers all received the same score teaches nothing, so it is refused rather than
contributing a silent zero.

## Selectors and graders

A **selector** gives each answer a verdict: **accept**, **reject** or **undecided**. A **grader**
gives each answer a score. Both are plugins, configured in the pack:

| Plugin | How it decides | Needs |
|---|---|---|
| **Rule** | A rule written as data (for example, the answer must match a pattern). No model and no network. | Nothing. Always available. |
| **Judge model** | A second model scores the answer. | A judge provider (an OpenAI-compatible endpoint). |
| **Reference match** | Compares the answer with a reference answer, `exact` or `fuzzy`. | A reference in the data. |
| **Human review** | Sends the answer to a person through miDataworks' review queue. | miDataworks configured. |
| **Composite** | Combines other selectors with `all`, `any` or `majority`. | Its children's needs. |

### The model never judges itself

A judge must be a **different** model from the one being trained. miForge's in-process provider
answers from the model it is training, so it cannot be used as a judge; a judge always goes
through a configured endpoint (`JUDGE_PROVIDER_BASE_URL` and `JUDGE_PROVIDER_MODEL`). Grading
your own homework is not a measurement.

### Undecided means undecided

Human review is asynchronous, so its first verdict is always **undecided**. An undecided answer
is **never** trained on, and it is never counted as a rejection either. There is no timeout,
deadline or setting that turns an undecided answer into an accepted one. A composite selector
propagates undecided honestly: `all` with one undecided child is undecided, not accept.

### Trying a selector

`POST /api/v1/selectors/evaluate` (the `forge_evaluate_completions` MCP tool) runs a selector
configuration over answers you supply and returns each verdict with its reason, without a run.
The **New run** form also tells you which selector and grader types are available on this
installation, and why any are not.
