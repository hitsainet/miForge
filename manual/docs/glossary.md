---
title: Glossary
description: Every term and acronym this manual uses, spelled out.
---

# Glossary

| Term | Meaning |
|---|---|
| **Adapter** | A small set of extra weights trained beside a model, leaving the model itself unchanged. miForge trains low-rank adapters. |
| **API** | Application programming interface: here, miForge's HTTP routes under `/api`. |
| **Base model** | The model a lineage of checkpoints started from. Part of every checkpoint's name. |
| **Batch record** | The record of one batch of training examples: where they came from, by content hash, and how many were accepted. |
| **BF16, FP16** | Brain floating point 16 and floating point 16: two 16-bit number formats models are commonly stored in. |
| **Branch** | A new run started from any checkpoint of another run. |
| **Cadence** | How often something happens during a run, in steps. |
| **Chain** | Several runs executed one after another, each starting from the last save of the previous one. |
| **Checkpoint** | A model saved at one step: a complete, standard Hugging Face model directory. |
| **CI** | Continuous integration: the automated build and test on every change. |
| **CUDA, CUDA IPC** | NVIDIA's platform for programming graphics cards; IPC (interprocess communication) is its feature for sharing memory between processes, used by hot update. |
| **DPO** | Direct preference optimization: a training objective that teaches a model to prefer a chosen answer over a rejected one. |
| **Floor** | The lowest acceptable retention score. Falling below it can stop a run. |
| **GitOps** | Deploying by committing manifests to a Git repository that a controller applies to the cluster. |
| **GPU** | Graphics processing unit: the graphics card training runs on. |
| **Grader** | A plugin that scores an answer. |
| **GRPO** | Group-relative policy optimization: a training objective that reinforces answers scored above their group's average. |
| **Heartbeat** | A periodic write by a running run that proves it is alive. |
| **Hot update** | The optional mode that trains the weights a live miLLM server is serving. |
| **Janitor** | The background job that fails runs whose heartbeat has stopped and frees their cards. |
| **Lease** | miForge's reservation of one graphics card for one run. |
| **LoRA** | Low-rank adaptation: the adapter technique miForge's `adapter` run type uses. |
| **Loss** | How wrong the model is on a batch of examples; training lowers it. |
| **Manifest** | `manifest.json` in a checkpoint or adapter: what it is and where it came from. |
| **MCP** | Model Context Protocol: an open protocol AI agents use to call tools. |
| **Pack** | A named, versioned bundle of the parts and judgment one domain needs. |
| **Part** | One piece of data: a prompt set, corpus, preference pairs, test set or retention set. |
| **PEFT** | Parameter-efficient fine-tuning: the standard library and directory format for adapters. |
| **Provider** | How miForge gets answers from a model: in-process, or an OpenAI-compatible endpoint. |
| **Retention set** | General prompts outside the domain, scored to detect lost general ability. |
| **Run** | One training job, from start to a stop condition. |
| **Selector** | A plugin that accepts, rejects or leaves undecided each answer. |
| **SFT** | Supervised fine-tuning: training a model to reproduce example text. |
| **Signal source** | Where a run's training examples come from: corpus, self-generation, preference pairs or reward. |
| **Step** | One round of training on one batch. |
| **Stop condition** | A rule that ends a run: step count, test threshold, retention floor, loss plateau or operator stop. |
| **Test set** | Held-out domain prompts, scored to measure domain improvement. |
| **UI** | User interface: here, miForge's admin web interface. |
| **UUID** | Universally unique identifier: how miForge names graphics cards (`GPU-…`). |
| **VRAM** | Video random-access memory: the graphics card's own memory, which bounds what a run can train. |
