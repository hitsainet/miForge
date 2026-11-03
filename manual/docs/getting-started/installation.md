---
title: Installation
description: Run miForge with Docker Compose on one machine, or deploy it to Kubernetes.
---

# Installation

miForge is a web application with four moving parts:

| Part | What it does |
|---|---|
| **Backend** (the API) | A Python web service that answers the admin interface and the MCP server. |
| **Workers** | Background processes that run training, generation and housekeeping (the janitor, chain advancement and checkpoint pruning), plus a scheduler that triggers the housekeeping. |
| **PostgreSQL** and **Redis** | The database that indexes runs, packs and approvals, and the message broker the workers listen on. |
| **Admin interface** | The web user interface (UI) you use in a browser. |

An optional fifth part, the **MCP server**, lets AI agents use miForge. See
[The MCP server](../integrations/mcp-server.md).

## What you need

- **Linux with an NVIDIA graphics card** for training. One 24 GB card is the reference target;
  a smaller card trains smaller models. Install the NVIDIA driver and the NVIDIA Container
  Toolkit so containers can see the card.
- **Docker with Compose**, or a **Kubernetes** cluster.
- Disk space for checkpoints. A 1.2-billion-parameter model's checkpoint is about 2.4 GB, and a
  long run saves many of them.

:::tip No card? You can still look around.
Everything except an actual training step runs without a graphics card: the admin interface,
pack authoring, validation and planning. A run that needs a card is refused by name rather than
failing later.
:::

## Option 1: Docker Compose (one machine)

The repository's `docker-compose.yml` runs the whole application on one machine. Clone the
repository and create your environment file:

```bash
git clone https://github.com/hitsainet/miForge.git
cd miForge
cp .env.example .env    # every setting, optional ones blank
```

Start everything, choosing the published admin interface with the `deploy` profile:

```bash
docker compose --profile deploy up -d
```

That starts PostgreSQL, Redis, a database migration, the backend, the training worker, the
maintenance worker, the admin interface and the MCP server. The ports are deliberately not the
usual defaults, so miForge can sit beside its sibling applications on one workstation:

| Service | Address |
|---|---|
| Admin interface | `http://127.0.0.1:8081` |
| Backend API | `http://127.0.0.1:8010` (interactive API documentation at `/api/docs`) |
| MCP server | `http://127.0.0.1:8765/mcp` |

Each port can be changed with an environment variable in `.env` (for example
`MIFORGE_UI_DEPLOY_PORT`).

### Give the worker the graphics card

The training worker reserves no card by default, so `docker compose up` also works on a laptop.
On a machine with a card, uncomment the `deploy.resources.reservations.devices` block on the
`celery-train` service in `docker-compose.yml`. Keep `count: all`: miForge chooses the card
itself, by its unique identifier, and handing the container one card picked by the runtime can
put a run on the wrong card when a second card is present.

### The MCP server needs a token

The MCP server refuses to start without a bearer token. Put a long random value in `.env`:

```bash
echo "MCP_AUTH_TOKEN=$(openssl rand -hex 32)" >> .env
```

Never commit that file.

### Developing from source

For working on miForge itself, run only the services and start the backend, workers and the
Vite development server from source:

```bash
docker compose up postgres redis
pip install -c constraints.txt -e ".[dev]"   # the lock the image and CI install
python backend/scripts/check_installed_matches_lock.py constraints.txt
pytest
docker compose --profile dev up frontend-dev   # hot-reloading interface on port 5174
```

## Option 2: Kubernetes with GitOps

The repository's `gitops/pilot/` directory holds the Kubernetes manifests (a Kustomize
directory) that a GitOps tool such as Argo CD can deploy. They declare:

- a `miforge` namespace;
- PostgreSQL and Redis;
- the backend pod: the API, four workers (training, generation, default and maintenance) and
  the scheduler;
- the admin interface;
- the MCP server, which runs from the backend image with its own command and never sees a
  graphics card;
- persistent volumes for checkpoints, trainer state, run logs, parts, packs and a model cache.
  They are host-path volumes pinned to one node by name, so change the node name and paths in
  `storage.yaml` (and the node selector in `postgres.yaml`) to match your cluster;
- an ingress for the interface, the API and the MCP server.

Before the first sync, create the `miforge-secrets` Secret yourself. Secrets never live in the
repository. It holds the database password and, if you use them, the MCP server's token
(`MCP_BEARER_TOKEN`) and the hot-update trainer token. Then edit the ingress host names to your
own domain. Full details are in [Deployment](../operators/deployment.md).

## Check that it is working

Open the admin interface. The **resource strip** at the top shows the host's CPU, memory and
each graphics card. Then ask the backend for its health:

```bash
curl -s http://127.0.0.1:8010/api/health
```

The answer lists PostgreSQL, Redis and each optional integration by name. An integration you
have not configured (miLLM or miDataworks) is reported as not configured, which is normal:
miForge needs neither.

Next: [Your first run](first-run.md).
