# miForge Manual

The miForge user manual: a [Docusaurus 3](https://docusaurus.io/) site published at
**<https://hitsainet.github.io/miForge/>**.

## Editing content

Pages live under `docs/` as Markdown, parsed as MDX: a bare `{`, `}` or `<` in prose is a parse
error, so put code, paths and placeholders in backticks. The sidebar is `sidebars.ts`; the site
configuration is `docusaurus.config.ts`. Spell out every acronym at its first use on a page.

`docs/reference/mcp-tools.md` is **generated** at build time from `../docs/mcp-contract.md` by
`scripts/sync-mcp-contract.mjs` and is git-ignored. Do not edit it; regenerate the contract instead.

## What may be published

This directory is published to the public mirror and to GitHub Pages. Never put an internal host
name, an address on a private network, a credential, a graphics card's identifier, or a reference to
internal planning documents here. `backend/tests/unit/test_manual_publish_allowlist.py` enforces the
file allow-list and those content rules.

## Local development

Requires Node.js 22 or newer.

```bash
npm ci
npm run start     # live-reloading server at http://localhost:3000/miForge/
```

## Build (also the link check)

```bash
npm run build
```

Broken links, broken anchors and broken Markdown links all fail the build. CI runs this on every
push and pull request (`.github/workflows/manual-ci.yml`).

## Deployment (automatic; do not run `npm run deploy`)

1. A push to `main` syncs a filtered snapshot of this repository to the public `hitsainet/miForge`.
2. The sync dispatches `.github/workflows/deploy-manual.yml` on the mirror, which builds the site and
   deploys it to GitHub Pages.
