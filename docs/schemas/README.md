# Shared contracts

This directory holds the schemas miForge publishes. **It is the one `docs/` path the public
mirror keeps**, so everything here is a public contract — intended, but worth knowing before
you add a file.

The ownership table below is prose for a human reader. The machine-checked copy lives in
`test_schema_sync.py` under the backend's contract tests, and **a schema here with no entry
in that table fails the suite** — a shared contract nobody declared an owner for is the
drift the table exists to catch.

## Ownership

| Schema | Normative in | Vendored into | Requirement |
|---|---|---|---|
| `checkpoint-manifest-v1.json` | **miForge** | miLLM | R-02.20 · BRD-01 R-01.12 |
| `adapter-manifest-v1.json` | **miForge** | miLLM *(owed — R-01.28)* | R-02.44 · BRD-01 R-01.28 |
| `midataworks-dataset-version-v1.json` | miDataworks | **miForge** | R-02.37 · BRD-03 R-03.47 |
| `millm-weights-api-v1.json` | miLLM *(generated from its routes)* | **miForge** | BRD-01 R-01.1/R-01.20 · suite D39 — F11's fake weights surface is validated against it |
| `millm-scoring-api-v1.json` | miLLM *(generated from its `/v1/completions` route)* | **miForge** | miLLM FR-25.15 · suite D61 — F11 14.3b's float32 verification; the fake scoring endpoint is validated against it |
| `batch-record-v1.json` | **miForge** | — | R-02.8 |
| `step-record-v1.json` | **miForge** | — | R-02.14 |
| `selector-config-v1.json` | **miForge** | — | R-02.9 |
| `pack-definition-v1.json` | **miForge** | — | R-02.38 |

**Normative** means: change it here, then copy it byte-identical to every vendor. Never the
other way round. For `midataworks-dataset-version-v1.json` that direction is reversed —
miDataworks owns it, and their `docs/schemas/` copy is the normative one (their
`backend/src/schemas/data/` copy is a packaged duplicate, so do not vendor from it).

## The byte-identity test is the authority, not a recorded hash

A vendored schema must be **byte-identical** across repos — not merely semantically
equivalent, because two JSON documents that parse to the same object can still disagree on
key order and formatting, and the point is to make a drifted copy impossible to miss.

**Do not record an md5 here and check against it.** The project tried that: the figure for
`checkpoint-manifest-v1.json` moved three times inside one hour, and once a wrong value was
recorded that matched no file in either repo. A reader comparing a *correct* pair of files
against a wrong recorded value concludes the contract has been tampered with, and then
"fixes" the file that was right. `cmp` and `md5sum` are how you check your own work in the
moment; the test is what keeps it true afterwards.

## Four properties every schema here has, and why

1. **Every object sets `additionalProperties: false`.** R-02.35 says a manifest's fields
   are *exactly* the schema's. Before 2026-10-13 neither vendored schema closed its
   objects, so that rule was enforced by nothing and an invented key validated clean —
   confirmed independently by five features. The **misspelled** key is the dangerous case:
   it validates, it writes, and the field the writer meant to set is simply absent.
2. **Nullable-but-required wherever a writer could stay silent.** `null` is an *assertion*;
   an **absent key** is an *omission*. They are different facts. `domain` is the worked
   example — nullable because general pretraining has no domain, still required because
   "nobody said" must not look like "there is none".
3. **No prompt or completion text in the run log** (ADR-020). The log leaves this machine in
   exports and support bundles, so user text in it is a privacy breach. The closed objects
   make that a *validation* failure rather than a matter of discipline — a sibling shipped
   exactly this bug by flipping one `include_context` argument, with its suite staying green.
4. **One definition, referenced.** `step-record-v1.json`'s `batch` block is a relative
   `$ref` into `batch-record-v1.json` rather than a second copy of the shape. A test
   asserts the reference actually resolves, because a silently unresolved `$ref` leaves the
   payload unconstrained and every malformed record would pass.

## Adding a schema

1. Write it with `$schema` set to draft 2020-12, 2-space indent, sorted keys, and
   `additionalProperties: false` on every object.
2. Add it to the ownership table above **and** to the machine-checked table in the contract
   test — the suite fails until you do.
3. If any sibling will vendor it, say so in both places and copy it there in the same
   change.
4. Remember that it publishes.
