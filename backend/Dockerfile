# hitsai/miforge-backend — the API, every Celery worker and the training loop
# (foundation task 8.2; ADR-002, ADR-006, ADR-019, ADR-023).
#
# Built in CI on the public mirror only, never locally (ADR-019). Its build CONTEXT is
# the REPOSITORY ROOT, not `backend/`: `pyproject.toml` lives at the root and declares
# `backend/src` as the package root, so a `./backend` context cannot see it. The root
# `.dockerignore` is what keeps that wide context cheap.
#
# ── ONE IMAGE, SEVERAL PROCESSES ───────────────────────────────────────────────────
# The API, each Celery worker and (later) the janitor all run from this image; the
# Kubernetes manifest's `command:` picks the process, as does `docker-compose.yml`.
# Deliberately NOT a `SERVICE_TYPE` entrypoint script like miDataworks': one fewer file
# to keep in step, and `kubectl describe pod` then shows the actual argv rather than a
# variable that has to be looked up elsewhere.
#
# ── THE GPU ────────────────────────────────────────────────────────────────────────
# There is no CUDA base image and no `nvidia/cuda` FROM. The torch wheels from PyPI
# bundle their own CUDA runtime, and the DEVICE comes from the node's default nvidia
# container runtime — exactly how miLLM's and miStudio's images get it. The manifest
# sets `NVIDIA_VISIBLE_DEVICES=all` and requests no `nvidia.com/gpu`; see
# `gitops/pilot/backend.yaml` for why that is a request for every card and not a
# missing resource limit.
#
# This image therefore RUNS WITHOUT A GPU, which is ADR-023's requirement rather than a
# side effect: everything except a training step must work on a machine with no card.
#
# ⚠ `bitsandbytes` (the `[bnb]` extra) IS INSTALLED HERE, and this paragraph said the
# opposite until 2026-10-16. The old reasoning — "it has no pure-CPU wheel everywhere, and
# the 8-bit optimizer states are a training-step concern" — is why it is an EXTRA in
# `pyproject.toml`, which is where that argument belongs: a developer on a cardless laptop
# installs `miforge` and the extra stays out of the way. It is not an argument against
# installing it in the one image whose whole purpose is to run the training step on a card.
#
# The consequence of leaving it out, found by running the real thing inside the cluster pod:
# the default `optimizer_state_precision` is `8bit`, so `optimizer.build()` refused
# `optimizer_precision_unavailable` — correctly, because "miForge will not downgrade
# silently" and the recorded identity would otherwise name a precision that was not used.
# **So the deployed trainer could not build its default optimizer at all.** An extra is a
# statement about who may skip a dependency, never a licence for the image that needs it to
# ship without it. Installed below via the extra NAME rather than by pinning the version a
# second time, so `pyproject.toml` stays the single source of truth for the range.

FROM python:3.13-slim

# Python 3.13 (operator decision 2026-10-21; ADR-002 named 3.11 until then). The same
# version as `requires-python`'s floor, every job in `backend-tests.yml` and MyPy's
# `python_version`, so the published image is the interpreter the test workflow actually
# ran against. `test_python_version_is_one_version.py` refuses any of the four drifting.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# On the DATA VOLUME, not in the container. The default (`~/.cache/huggingface`) lives
# on the overlay filesystem, so every pod restart re-downloads the base model — 2.4 GB
# for LFM2 1.2B in FP16, across a link this node runs at x4, not x16.
ENV HF_HOME=/data/hf-cache

# The three artifact roots are NOT set here. `core/config.py` already defaults them to
# /data/checkpoints, /data/trainer-state and /data/parts, and a second copy in the image
# is a second source of truth for paths ADR-004 makes authoritative.

# The build-time version, passed by docker-images.yml because a subdirectory build
# cannot COPY the repo-root VERSION file. Recorded as a label, not baked into code.
ARG APP_VERSION=0.0.0
LABEL org.opencontainers.image.title="miforge-backend" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.source="https://github.com/hitsainet/miForge"

# ── THE C TOOLCHAIN, AND IT IS A RUNTIME DEPENDENCY ─────────────────────────────
# ⚠ WITHOUT THIS LINE EVERY TRAINING STEP ON A LLAMA-ARCHITECTURE MODEL FAILS, and it
# failed in the cluster until 2026-10-16. `python:3.x-slim` ships no compiler: `gcc`, `cc`,
# `clang`, `g++` and `ld` were all absent, verified inside the running pod.
#
# torch 2.14's `_native` op registry routes `bmm_outer_product` to a **Triton**
# implementation — reached from `transformers`' Llama rotary embedding — and Triton
# JIT-compiles its launcher stub with a C compiler AT RUNTIME, on the first step, inside
# this image. Isolated repro with no miForge code in it: a raw CUDA matmul is fine, and a
# Llama forward on cuda raises `RuntimeError: Failed to find C compiler. Please specify via
# CC environment variable or set triton.knobs.build.impl`. There is no env knob that turns
# the routing off and the op ships only a Triton implementation with no eager fallback, so
# a compiler is not optional and cannot be deferred to a derived image.
#
# **The stage matters and this Dockerfile has exactly one.** Triton compiles at RUNTIME, so
# a toolchain installed in a builder stage and discarded would leave the defect in place
# while looking like a fix. If this file is ever split into builder + runtime stages, the
# toolchain belongs in the stage the final image is built from — and
# `test_backend_image_can_train.py` asserts that rather than trusting it.
#
# **`gcc` and `libc6-dev`, not `build-essential`.** Checked against
# `triton/runtime/build.py` in this image rather than guessed: `_find_compiler("c")` reads
# `CC`, then `clang`, then `gcc`, and the stub is compiled as C with `-shared -fPIC`
# against the interpreter's own headers (present in the `python:3.x-slim` base) and Triton's
# bundled CUDA headers, linking `libcuda.so.1` from the node's driver. The only `language =
# "c++"` caller in Triton is the **AMD/HIP** backend, which this image never reaches, so
# `g++` buys nothing here. `build-essential` additionally drags in `make`, `dpkg-dev`,
# `g++` and the Perl toolchain for no reachable caller. (`gcc` depends on `libc6-dev`
# transitively; it is named anyway, because a transitive dependency is not a contract.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# uid 10001, matching miDataworks, and NON-ROOT from the first instruction that runs
# application code. The data volume is a hostPath-backed PV that the kubelet creates
# root-owned, so the manifest runs a small root init container to hand it to 10001
# before this image starts.
RUN useradd --create-home --uid 10001 --user-group miforge

WORKDIR /app

# ── Dependencies, in their own layer ───────────────────────────────────────────────
# Split from the source COPY on purpose. The dependency set includes the CUDA torch
# wheel and the nvidia-* runtime packages — several gigabytes. Fused into one layer with
# the source, every one-line change to a Python file would re-download all of it on
# every CI build. miLLM's test workflow has already exhausted a runner's disk that way
# ("[Errno 28] No space left on device"), which is the same cost on the build side.
#
# `_depsonly` exists because setuptools' `packages.find(where="backend/src")` must find
# at least one package for `pip install .` to resolve the project's dependencies. The
# project itself is then uninstalled, leaving only the dependency layer; the real
# package is installed with `--no-deps` after the source arrives.
#
# `.[bnb]` rather than `.`, for the reason in the header: this is the GPU image and 8-bit
# optimizer states are its default. `pip uninstall miforge` removes the project, never its
# dependencies, so `bitsandbytes` survives into the final image — and the `--no-deps`
# install at the bottom must stay `--no-deps`, since adding the extra there instead would
# fuse several gigabytes of wheels into the source layer.
# ── TORCH IS PINNED TO miLLM's SERVED BUILD (F11 GPU-block prerequisite, 2026-10-16) ──
# §5.10 shares tensors with miLLM through CUDA IPC, and a handle is a private contract
# between MATCHING builds: `training/hotupdate/rebuild.py::check_build_identity` refuses
# an attach when torch or CUDA differ. pyproject.toml's `torch>=2.4,<3` let this image take
# whatever PyPI served on the day it was built, and so does miLLM's `torch>=2.10.0,<3` - so
# the two images matched only by build-date coincidence. miLLM's hardware acceptance
# (0xcc/reviews/attachable_weights_hardware_2026-10-16.md §1) measured its pod at
# **2.14.1+cu130**, installed from PyPI exactly as below; installing the same version from
# the same index gives the same wheel. `test_torch_pin_matches_millm.py` compares this pin
# with miLLM's (its exact pin if it ever has one, else that measured record).
#
# Size and CUDA: unchanged in kind - still no CUDA base image; the PyPI wheel bundles the
# CUDA 13.0 runtime as the unpinned install already did (several GB, in this dependency
# layer only). The node driver 580.178.04 supports CUDA 13.0. ⚠ miLLM is NOT pinned: if its
# image is rebuilt after PyPI moves past 2.14.1, attach refuses `incompatible_torch_build`
# until both pins move together - pin miLLM too (its repo, its decision).
#
# ── EVERY TRANSITIVE VERSION COMES FROM `constraints.txt` (deps sync 2026-10-21) ──────
# `-c constraints.txt` is the same lock `backend-tests.yml` installs with, so the image and
# CI cannot resolve a dependency to different versions on different days (before this the
# image took filelock/fsspec/networkx from PyPI and CI from the PyTorch CPU index).
# `test_dependencies_are_locked.py` refuses an install line here that drops the `-c`.
ARG TORCH_VERSION=2.14.1
# The post-install check then REFUSES THE BUILD when anything installed is not in the lock or
# is at another version (a new transitive dependency the lock never named), so a drifted image
# is never pushed.
COPY pyproject.toml constraints.txt ./
COPY backend/scripts/check_installed_matches_lock.py backend/scripts/
RUN mkdir -p backend/src/_depsonly \
    && : > backend/src/_depsonly/__init__.py \
    && pip install --no-cache-dir -c constraints.txt "torch==${TORCH_VERSION}" ".[bnb]" \
    && pip uninstall -y --no-input miforge \
    && rm -rf backend/src/_depsonly \
    && python backend/scripts/check_installed_matches_lock.py constraints.txt

# ── Application ────────────────────────────────────────────────────────────────────
COPY --chown=miforge:miforge backend/src/ backend/src/
# The vendored contracts travel WITH the image. They are the normative shapes this
# process writes (ADR-014), and an image that validates against a schema it does not
# carry would validate against whatever happens to be on the node.
# ⚠ THE MIGRATIONS AND THEIR CONFIG, WITHOUT WHICH THIS IMAGE CAN SERVE BUT CAN NEVER
# CREATE ITS OWN SCHEMA. Until 2026-10-14 the image carried `backend/src/` and
# `docs/schemas/` and neither of these, so a fresh database stayed empty: `/api/health`
# reported postgres **ok** (it checks connectivity, not schema) while every data endpoint
# returned 500 with `relation "mf_runs" does not exist`. Health green and the whole API
# broken is the worst pair to ship, and it is what a cold `docker compose up` produced.
#
# `alembic.ini` sets `script_location = backend/alembic` and `prepend_sys_path =
# backend/src`, both relative, so they resolve against WORKDIR `/app` exactly as they do
# against the repository root — the paths below deliberately mirror the repo layout rather
# than flattening it.
#
# `env.py` takes the URL from `core.config`, never from `alembic.ini`, so there is one
# source of truth and nothing to template at build time.
COPY --chown=miforge:miforge alembic.ini ./
COPY --chown=miforge:miforge backend/alembic/ backend/alembic/

COPY --chown=miforge:miforge docs/schemas/ docs/schemas/

# ⚠ THE IMAGE CANNOT START WITHOUT THIS LINE, and it crashed on import until 2026-10-14.
#
# Eight modules resolve the vendored contract directory from `__file__` with their own
# `parents[N]` arithmetic. Every one of them is correct when the package is RUN FROM THE
# REPOSITORY and wrong the moment it is pip-installed: counting up from
# `/usr/local/lib/python3.13/site-packages/` lands on `/usr/local/lib/docs/schemas`, which
# does not exist. The COPY above puts them at `/app/docs/schemas` instead, and the depths
# happen to compensate so all eight computed the same wrong path — one failure, eight sites.
#
# `MIFORGE_SCHEMA_DIR` was already the override five of those modules honoured, so this
# adopts an existing convention rather than inventing one; the other three were taught to
# honour it in the same commit. The container died with `schema_not_vendored` naming
# `/usr/local/lib/docs/schemas` before the first request — **no test could have caught it**,
# because the whole suite runs from the repository where the relative path resolves.
#
# Found by running `docker compose --profile deploy up` and watching the backend exit, after
# the operator made Compose a deployment path on 2026-10-14. Pinned by
# `test_schema_dir_override.py`, which asserts every one of the eight honours the variable.
ENV MIFORGE_SCHEMA_DIR=/app/docs/schemas
# `backend/tests/` is NOT copied. The suite runs in CI against the source tree
# (backend-tests.yml), and a test directory in a production image is reachable code
# nobody reviews as such.
COPY LICENSE /app/LICENSE

RUN pip install --no-cache-dir --no-deps .

USER 10001:10001

EXPOSE 8000

# No curl or wget in the image: one fewer package, and Python is already here.
#
# This probe is for the API only. A WORKER container serves no HTTP, so it overrides or
# disables this — `docker-compose.yml` sets `healthcheck: disable: true` on the worker,
# and Kubernetes probes are declared per container in `gitops/pilot/backend.yaml`.
#
# ⚠ A worker's liveness is the DATABASE HEARTBEAT (ADR-007), never an HTTP probe.
# Progress reported only over the socket is invisible to the janitor, which is how
# miStudio reaped a live 5.8-hour job as "worker lost".
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# ⚠ `create_asgi_app`, NOT `create_app`, AND THE DIFFERENCE IS NOT COSMETIC.
# `api/app.py` defines both and says which is which: `create_asgi_app` is "the object
# uvicorn is pointed at" — FastAPI WRAPPED by Socket.IO rather than mounting it, because
# engine.io's HTTP polling transport does not survive a Starlette `Mount` intact. Serve
# `create_app` instead and the REST surface works perfectly while the run view's live
# updates silently do not, which is the hard failure to diagnose.
#
# `--factory` because both are functions, not module-level app objects — deliberately,
# so the test suite can exercise the REST surface with `TestClient` without requiring a
# Redis message queue just to construct the app.
#
# Checked against `backend/src/api/app.py`, not assumed. If that module is refactored,
# this line and `docker-compose.yml`'s backend command change with it.
CMD ["python", "-m", "uvicorn", "api.app:create_asgi_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
