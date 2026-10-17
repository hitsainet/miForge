# miForge

**miForge makes weights** — creating them from scratch, changing them in place, or
adding adapters that compose with them.

> ## ⚠ Status: specification complete, implementation just started
>
> This repository is at the **beginning** of its build. What exists today is the
> foundation: packaging and test harness, core utilities, CI workflows, container
> images and the GitOps manifests. **The training loop, the API and the admin UI are
> not implemented yet.** Nothing below describes software you can run and get a
> checkpoint out of.
>
> The section [What actually works today](#what-actually-works-today) is the honest
> inventory, and it is deliberately placed before the capability descriptions rather
> than after them.

---

## What it is

A checkpoint is a set of tensor files. Training is a process that changes tensors in
GPU memory and then writes them out. Labs do this at scale on clusters; the same
mechanism works on one consumer GPU with a small model.

miForge is that mechanism, built around one loop: **push the weights in a direction
defined by some domain material, test the result, save a checkpoint, repeat.** The
material for the next round can come from anywhere — including from the model itself.

Three things it does, not one:

| | |
|---|---|
| **Creates weights** | A model from an architecture config and random initialisation. No base model, nothing upstream at all. |
| **Changes weights** | Full-parameter updates to an existing model. |
| **Adds weights** | Low-rank adapters that compose with a base at serve time, or merge down into it. |

All three run through the same loop, and **every checkpoint in the series is an
ordinary loadable model.**

## Two properties that shape everything else

**It is standalone.** miForge requires no other application to function. It loads a
base model itself (or creates one), trains in its own process, and writes its own
checkpoints. It can consume other systems' outputs and produce outputs they consume,
but it needs none of them present. There is one optional mode that attaches to a
running inference server to update weights without a reload; it is fenced off from
everything else and is not on any critical path.

**Its outputs are standard.** A checkpoint is a **standard Hugging Face directory**
— `model.safetensors`, `config.json`, tokenizer files at the root — servable by vLLM,
TGI, `transformers`, or any other engine, with no knowledge of miForge required. An
adapter is a standard PEFT directory with the usual tensor key convention. Each carries
a manifest recording what it is, what it came from and how it scored.

That is a deliberate constraint rather than a convenience: an artifact that only
miForge can read would make the tool, not the weights, the thing you own.

## What actually works today

| Area | State |
|---|---|
| Package, dependency pins, lint and type configuration | present |
| Test harness, with the schema-sync test and the no-sibling-import and hard-coded-GPU guards | present |
| Core utilities — settings, error envelope, canonical JSON, structured logging, GPU resolution | present |
| Shared contract schemas under `docs/schemas/` | present, and these are the public ones |
| CI: backend and frontend test workflows, image builds, mirror sync | present |
| Container images and the Kubernetes/ArgoCD manifests | present, not yet deployed |
| Database schema, API surface, job machinery | in progress |
| The training loop, the run types, scoring, packs, the admin UI | **not started** |

The feature specifications are complete and detailed; they are not in this repository.

## The shared contracts

`docs/schemas/` is published deliberately. These are the formats miForge's artifacts
use, and they are here so that software which has never heard of miForge can read what
it produces:

- `checkpoint-manifest-v1.json` — a checkpoint's identity, lineage and scores.
- `adapter-manifest-v1.json` — an adapter's identity, its base, its scaling rule and
  its tensor digest. Self-describing on purpose: an adapter is small and routinely
  moved between installations, so it must be able to say what it trained on without
  anything else alongside it.
- `midataworks-dataset-version-v1.json` — the input contract miForge consumes, vendored
  byte-identical from the system that owns it.

The directory also holds miForge's internal contracts — the batch record, the step
record, the selector configuration and the pack definition — so that the shapes four
different parts of the system write are defined in one place rather than four.
`docs/schemas/README.md` is the ownership table: what miForge is normative for, what it
vendors, and from where.

Every one of them is normative for its format. A copy that disagrees with its owning
repository is a bug, and a test asserts byte-identity rather than trusting that they
match — a hash written down in prose goes stale the instant the file changes.

## Requirements

- Linux with an NVIDIA GPU for a training run — one 24 GB card is the reference target.
- Python 3.11 or newer, PostgreSQL 15, Redis 7.
- **A GPU is not required for development.** Everything except an actual training step
  runs without one; the tests that need a device are marked and skipped by default.
  That is a constraint on how the code is laid out, not an accident of it.

## Development

```bash
cp .env.example .env          # every variable, with the optional ones blank
docker compose up postgres redis
pip install -e ".[dev]"
pytest                        # GPU tests deselected by default
```

`docker-compose.yml` is **development only**. The deployment path is GitOps: the
manifests are in `gitops/pilot/`, images are built in CI, and ArgoCD syncs digest pins.

## The suite

miForge is one of four applications that work together and each run alone:

> miDataworks makes data. miLLM serves models. miStudio trains detectors and interprets
> models. **miForge makes weights** — creating them from scratch, changing them in
> place, or adding adapters that compose with them. None does another's job.

miForge runs a model **to learn**; miLLM runs one **to serve**; miStudio runs one **to
interpret**.

## Licence

MIT. See [LICENSE](LICENSE).
