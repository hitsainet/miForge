# hitsai/miforge-frontend (foundation task 8.2; ADR-011, ADR-019).
#
# Built in CI on the public mirror only, never locally. Build context is `./frontend`,
# so this file's paths are relative to that directory and `frontend/.dockerignore` is
# the ignore file BuildKit uses — not the repository root's.
#
# NO GPU. The frontend is static files and an nginx proxy (ADR-019: "the backend pod
# needs the GPU; the frontend does not").
#
# Adapted from miDataworks' `frontend/Dockerfile`, which is miStudio's with the dev
# server stage dropped. Same reasoning here: local development runs Vite from source
# (`npm run dev`), so a dev stage in the published image would be a second, untested way
# to run the UI.

# Node 22 LTS (suite decision D34), the same major `frontend-tests.yml` runs, so the
# image is built by the toolchain CI tested rather than by whatever is newest on the
# runner.
FROM node:22-alpine AS builder

WORKDIR /app

# Manifests first, for layer caching: a source change must not re-resolve the
# dependency tree.
#
# ⚠ BOTH SOURCES ON ONE `COPY`, AND THE ORDER MATTERS. `package-lock.json` does not
# exist yet (foundation task 7.1 owes it), and a `COPY package-lock.json* ./` on its own
# would FAIL THE BUILD — BuildKit errors with "no source files were specified" when a
# wildcard matches nothing. Pairing it with `package.json`, which always matches, makes
# the source set non-empty and the missing lockfile merely absent.
COPY package.json package-lock.json* ./

RUN if [ -f package-lock.json ]; then \
      npm ci; \
    else \
      echo "WARN: no package-lock.json — resolving the floating ranges in package.json fresh." \
           "This image's dependency set is NOT the one CI or any developer tested." \
           "Commit a lockfile (foundation task 7.1)." >&2; \
      npm install --no-audit --no-fund; \
    fi

COPY . .

# `npm run build` is `tsc -b && vite build`, so a type error fails the IMAGE BUILD, not
# just CI. That is deliberate: ADR-011 makes TypeScript strict a standard, and an image
# that can be published over a type error makes the standard advisory.
RUN npm run build

# ── Serving stage ──────────────────────────────────────────────────────────────────
# nginx-unprivileged: uid 101, port 8080, no root at any point. miDataworks' choice
# rather than miLLM's root-on-80 nginx.
FROM nginxinc/nginx-unprivileged:alpine AS production

ARG APP_VERSION=0.0.0
LABEL org.opencontainers.image.title="miforge-frontend" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.source="https://github.com/hitsainet/miForge"

# Alpine CVE catch-up, as both siblings do.
USER 0
RUN apk upgrade --no-cache
USER 101

COPY --chown=nginx:nginx nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder --chown=nginx:nginx /app/dist /usr/share/nginx/html
# The repo-root LICENSE. docker-images.yml's "Copy LICENSE to build context" step puts
# it here before the build, which is why it is a hard requirement and not a glob — a
# glob matching nothing fails the build anyway (see the package-lock note above), so the
# tolerant version does not exist. A hand-build of this directory needs
# `cp ../LICENSE frontend/` first.
COPY LICENSE /licenses/LICENSE

EXPOSE 8080

# Hits nginx's own `/health`, which answers while the backend is down. A frontend
# restarted because the API is unhealthy is restarted for someone else's fault.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:8080/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
