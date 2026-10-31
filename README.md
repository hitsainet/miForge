# miForge

**miForge makes weights** — creating them from scratch, changing them in place, or
adding adapters that compose with them.

**The manual: <https://hitsainet.github.io/miForge/>** — installation, your first run, packs,
runs, checkpoints, chains, approvals, settings, the optional hot-update mode and the MCP server.

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

## What it does today

All three run types (`change`, `create`, `adapter`, with merge), the four signal sources (corpus,
self-generation, preference pairs, reward), selectors and graders, test and retention scoring with
a floor, atomic checkpoints with lineage, continue and branch, chains, domain packs with a guided
authoring wizard, the admin UI, the optional hot-update mode against miLLM, and an MCP server whose
graphics-card, attach, delete and configuration actions wait for operator approval. The
[manual](https://hitsainet.github.io/miForge/) describes each one.

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
- Python 3.13 or newer, PostgreSQL 15, Redis 7.
- **A GPU is not required for development.** Everything except an actual training step
  runs without one; the tests that need a device are marked and skipped by default.
  That is a constraint on how the code is laid out, not an accident of it.

## Development

```bash
cp .env.example .env          # every variable, with the optional ones blank
docker compose up postgres redis
pip install -c constraints.txt -e ".[dev]"   # the lock the image and CI install
python backend/scripts/check_installed_matches_lock.py constraints.txt
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
