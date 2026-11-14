// Builds docs/reference/mcp-tools.md from the repository's generated MCP contract.
//
// docs/mcp-contract.md is GENERATED from the MCP server's live tool registry, and a backend
// test fails when it drifts. Copying it here at build time (rather than pasting it into the
// manual) means the published tool list is that same file, so it cannot drift either.
//
// The contract opens with an HTML comment, which MDX cannot parse; it is dropped, and the
// page gets front matter of its own. The output is git-ignored: it is a build product.
import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const source = resolve(here, '..', '..', 'docs', 'mcp-contract.md');
const target = resolve(here, '..', 'docs', 'reference', 'mcp-tools.md');

let text;
try {
  text = readFileSync(source, 'utf8');
} catch (error) {
  // Refuse rather than publish a page that silently lost its tool list.
  console.error(`sync-mcp-contract: cannot read ${source}: ${error.message}`);
  process.exit(1);
}

const body = text
  .replace(/^<!--[\s\S]*?-->\s*/, '')
  // The contract's own H1 is replaced by the page title below.
  .replace(/^# .*\n+/, '')
  // Tool descriptions cite internal requirement, decision and feature numbers, which mean
  // nothing to a reader. Drop them here; scripts/check-published.mjs fails the build if one
  // of a shape not handled below ever reaches the site.
  .replace(/,? with the R-\d\d\.\d+ configuration/g, ', with its optimizer configuration')
  .replace(/\s*\((?:R-\d\d\.\d+|D\d+|feature \d+)\)/g, '')
  .replace(/\bF\d{1,2}'s /g, 'the ');

if (!body.includes('## `core`')) {
  console.error('sync-mcp-contract: the contract has no `core` category; refusing to publish it');
  process.exit(1);
}

const page = `---
title: MCP tool reference
sidebar_label: MCP tools
description: Every tool the miForge MCP server registers, the route it calls and the approval it waits for.
---

# MCP tool reference

This page is generated from the miForge MCP server's live tool registry every time the manual
is built, so it always matches the server in the same release. For how to connect an agent
and how approvals work, read [the MCP server guide](../integrations/mcp-server.md) and
[Approvals](../operators/approvals.md).

${body}`;

mkdirSync(dirname(target), {recursive: true});
writeFileSync(target, page);
console.log(`sync-mcp-contract: wrote ${target}`);
