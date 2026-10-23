---
title: Artifacts and schemas
description: What miForge writes to disk, and the published JSON Schema contracts that describe it.
---

# Artifacts and schemas

miForge's outputs are ordinary files that software which has never heard of miForge can read. Their
formats are published as JSON Schema files in the repository's
[`docs/schemas/`](https://github.com/hitsainet/miForge/tree/main/docs/schemas) directory.

## A checkpoint

A standard Hugging Face model directory with one extra file:

| File | What it is |
|---|---|
| `model.safetensors` (or shards) | The weights. |
| `config.json` | The model architecture. |
| Tokenizer files | At the top level of the directory, as every Hugging Face loader expects. |
| `manifest.json` | miForge's record of what this checkpoint is. Purely additive: loaders ignore it. |

Load one with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/data/checkpoints/notary-1/step-000400"
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
```

### The checkpoint manifest (`checkpoint-manifest-v1`)

The manifest says, from the directory alone:

- its **name** (`base-model:run-id:step-N`), run and step;
- the **base model** the lineage started from, and its **parent** checkpoint (empty for a
  created model's step 0);
- for a merged checkpoint, the **adapter** merged into it;
- its **domain** (empty means "no domain", stated explicitly);
- its **test** and **retention** scores, each with provider, sample size and, for retention, the
  floor;
- a **hash of its weights**, which `POST .../verify` recomputes.

It contains no absolute path, host name or service address, so a copied checkpoint keeps its
meaning on another machine. Every object in the schema is closed: a field the schema does not name
cannot be written.

## An adapter

A standard Parameter-Efficient Fine-Tuning (PEFT) directory with its own manifest
(`adapter-manifest-v1`): the base model it was trained against, its rank, alpha, target modules and
scaling rule, and a hash of its weights. An adapter is small and is often moved on its own, so its
manifest says everything needed to use it without the run log.

## The run log

Each run writes an append-only log under the runs volume: the run's definition (submitted and
resolved), one **step record** per step (`step-record-v1`: loss, gradient norm, learning rate,
wall time, and the batch it trained on) and each **batch record** (`batch-record-v1`: where the
examples came from, by content hash, and the selection counts). **The run log never contains prompt
or completion text**, by schema: the log can leave the machine in exports and support bundles.

## The schemas

| Schema | Describes | Owned by |
|---|---|---|
| `checkpoint-manifest-v1.json` | A checkpoint's manifest | miForge |
| `adapter-manifest-v1.json` | An adapter's manifest | miForge |
| `step-record-v1.json` | One step in the run log | miForge |
| `batch-record-v1.json` | One batch in the run log | miForge |
| `selector-config-v1.json` | A selector or grader configuration | miForge |
| `pack-definition-v1.json` | A pack version | miForge |
| `midataworks-dataset-version-v1.json` | A dataset version that miForge imports as a part | miDataworks (copied byte for byte) |
| `millm-weights-api-v1.json` | miLLM's weights interface, used by hot update | miLLM (copied byte for byte) |
| `millm-scoring-api-v1.json` | miLLM's scoring interface, used by hot update's check | miLLM (copied byte for byte) |

A copy that disagrees with its owner's file is a bug, and the repository's tests compare the copies
byte for byte.
