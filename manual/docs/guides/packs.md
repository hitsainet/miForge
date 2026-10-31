---
title: Domain packs
description: Describe a domain once, as a versioned pack, and reuse it across runs.
---

# Domain packs

A **pack** is everything miForge needs to know about one domain: the prompts, example answers,
how good output is told from bad, how improvement is measured, and what general ability must not
be lost. You describe a domain once, as a pack, and every run on that domain can use it.

## What is in a pack

A pack is a named, versioned bundle of **slots**. Each slot points at a part (or a
configuration) and pins its exact version:

| Slot | Required? | What it is for |
|---|---|---|
| **Prompt set** | yes | The prompts the model answers during self-generation and reward runs. |
| **Corpus** | no | Example text to train on directly. |
| **Preference pairs** | no | A better and a worse answer to the same prompt. |
| **Selector configuration** | no | How to accept or reject an answer. |
| **Grader configuration** | no | How to score an answer. |
| **Test set** | yes | Held-out prompts that measure whether the domain is improving. |
| **Retention set** | yes | General prompts outside the domain, with a **floor**: the lowest acceptable score. |

**A pack holds references to parts, not copies of them.** Because every reference pins a version,
a pack version names a reproducible run input: two runs on `notary-md` version 3 trained on the
same material.

Which optional slots a run needs depends on its signal source: a `corpus` run needs a corpus, a
`preference-pairs` run needs preference pairs, and `self-generation` and `reward` runs need a way
to judge answers. See [Signal sources and judging](signals-and-judging.md).

## Where parts come from

miForge **assembles** packs; it does not author the parts inside them. A part comes from one of
two places, and the pack records which:

- **miDataworks**, the suite's data application, where parts are written, reviewed and versioned.
  miForge cites them by hash under a versioned contract.
- **Outside the suite**: your own files, imported as an **external** part with its own
  provenance. An external part is recorded as external forever and is never passed off as if it
  came from miDataworks.

## The **Domain packs** screen

**Domain packs** lists every pack version: its domain, whether it is **valid** (able to define a
run), and the runs that used it. From here you can:

- **Import** a pack exported from another miForge installation. A pack whose parts cannot be
  found here is still stored, and reported as unresolvable, slot by slot, so you can see exactly
  what to bring across.
- **Export** a pack version as a single portable document.
- **Clone** a pack version under a new name. The pack remembers what it was cloned from.
- **Delete** a pack version. (An AI agent's delete waits for your approval; see
  [Approvals](../operators/approvals.md).)

miForge ships **no built-in domain**. An example pack, if present, is labelled as an example and
can be deleted.

## Authoring a pack: **Onboard a domain**

The **Onboard a domain** wizard builds a **draft** in seven steps. Progress is saved as you go,
so you can leave and come back; a draft is not a pack until you commit it.

| Step | What you do | What you can try first |
|---|---|---|
| 1. Describe | Name the domain. | |
| 2. Prompts | Choose the prompt set. | A seeded **sample** of its rows. |
| 3. Answers | Optionally choose a corpus and/or preference pairs. Skippable. | A sample of each. |
| 4. Judgment | Optionally choose a selector and/or grader configuration. Skippable. | A **dry run**: the selector judges sampled answers and gives a reason for each verdict. |
| 5. Improvement | Choose the test set. | A **scored sample**. |
| 6. Retention | Choose the retention set and set its floor. | A scored sample. |
| 7. Review | See what is still missing, by name, and commit. | |

**Try before committing** is the point of the wizard. The cost of discovering a bad selector
should be one dry run, not a whole training run. A dry run that rejects everything, or accepts
everything, is a warning that the selector cannot tell answers apart.

When you commit, the draft becomes pack version `1` (or the version you choose). Committing
refuses, leaving the draft intact, if a required slot is empty, and names the slot.

## Validity

A pack version is **valid** when every required slot is filled and every reference resolves to a
part this installation has. The validity check names each missing or unresolved slot; it never
just says "invalid". A run cannot start from an invalid pack.

## Packs and runs

A run definition names **either** a pack version **or** a list of loose parts, never both. A run
can override the pack's retention floor for itself; the run records which floor it used and where
it came from.

## Through the API or an agent

Every screen action has a route and an MCP tool: `forge_list_packs`, `forge_get_pack`,
`forge_get_pack_validity`, `forge_create_pack`, `forge_import_pack`, `forge_export_pack`,
`forge_clone_pack`, `forge_delete_pack`, and the `forge_*_draft` tools for authoring. See the
[MCP tool reference](../reference/mcp-tools.md).
