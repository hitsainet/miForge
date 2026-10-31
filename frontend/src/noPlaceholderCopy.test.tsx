/**
 * **No screen tells the operator the product is unfinished** (task 14.0's inventory; operator,
 * 2026-10-18).
 *
 * The Runs screen read *"This build serves no run listing — GET /v1/runs is F6's and is not
 * mounted… Nothing here is fabricated in its place."* The operator read that, correctly, as
 * proof the product was unfinished. The Checkpoints screen said `GET /api/v1/tree` was "not
 * mounted in this build" when it was; the run view said `stop_reason` was "not carried in this
 * build" when it was; a hint called the run's floor override "F7's open question" when it was
 * wired end to end. Each was true once, or never, and each outlived its truth.
 *
 * A missing capability is now either **built**, or - when it genuinely depends on something
 * outside miForge (a sibling app, a card, a driver) - a **runtime state** naming what is
 * missing and what to do. Never a statement about the build or about which feature owes it.
 *
 * ## Two scans, because either alone has a blind spot
 *
 * 1. **Every rendered string in the source** - string literals, template literals and JSX
 *    text, with comments stripped (comments are where the history belongs). This reaches the
 *    branches a test backend never drives, like a terminal run with no stop reason.
 * 2. **The text every `ROUTES` screen renders** - and every Onboard step, from `DRAFT_STEPS` -
 *    against a stub backend **and** against one that answers nothing, since the failure
 *    copy is where a "not mounted" most likely lives. This reaches copy assembled at runtime.
 *
 * {@link ALLOWED} is for the second kind only: a phrase that matches and is a runtime fact
 * about something outside miForge, each with its reason. It is empty, and a new entry needs
 * one.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { act, cleanup, render, screen } from '@testing-library/react';
import { App } from './App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { ROUTES } from '@/config/routes';
import type { ScreenId, ScreenParams } from '@/contexts/NavigationContext';
import { DRAFT_STEPS } from '@/types/packs';
import { DRAFT_ID, answer } from '@/testing/stubBackend';

/** Copy that describes the build or a feature's ownership, not the running system. */
export const PLACEHOLDER_PATTERNS: readonly RegExp[] = [
  /not mounted/i,
  /not served\b/i,
  /\bserves no\b/i,
  /\bthis build\b/i,
  /\bnot (?:yet )?(?:built|implemented|wired)\b/i,
  /\bno (?:endpoint|route) (?:publishes|serves|exists|calls)/i,
  /\bF\d{1,2}(?:'|’|&apos;)s\b/,
  /\bbelongs to F\d/i,
  /\bopen question\b/i,
  /\bnot decided here\b/i,
  /\bFR-0\d\d\.\d+/,
  /\b(?:is|are) owed\b/i,
  // Requirement, decision and planning ids (copy review of /#/runs/new, 2026-10-20). The New
  // run screen read "R-02.53's create path", "R-02.13.", "R-02.25's first condition." and
  // "(ADR-010)": references into documents the operator does not have open. Comments keep
  // them; rendered copy says what the reference meant.
  /\bR-\d{2}\.\d+/,
  /\b(?:ADR|BRD|PRD|PPRD|PADR|TDD|TID|DEF|AMB)-\d+/,
  /\b(?:FTASKS|FTID|FTDD|FPRD)\b/,
  /§\s?\d/,
  /\bfeature \d{3}\b/i,
];

/** `{phrase: reason}` - a match that is a runtime fact about something outside miForge. */
const ALLOWED: Readonly<Record<string, string>> = {};

function placeholderIn(text: string): string | null {
  for (const pattern of PLACEHOLDER_PATTERNS) {
    const match = pattern.exec(text);
    if (match !== null && !Object.keys(ALLOWED).some((phrase) => text.includes(phrase)))
      return match[0];
  }
  return null;
}

/* ------------------------------------------------------------------------------------ *
 * 1. The source's rendered strings
 * ------------------------------------------------------------------------------------ */

const SRC = path.resolve(__dirname);

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) {
      if (entry !== 'testing') out.push(...sourceFiles(full));
    } else if (
      /\.tsx?$/.test(entry) &&
      !/\.test\.tsx?$/.test(entry) &&
      entry !== 'pickerTestKit.ts'
    )
      out.push(full);
  }
  return out;
}

/** Comments replaced by blank lines, so line numbers still point at the source. */
export function withoutComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, (block) => block.replace(/[^\n]/g, ''))
    .replace(/\{\s*\/\*[\s\S]*?\*\/\s*\}/g, '')
    .replace(/(^|[^:'"`])\/\/[^\n]*/g, '$1');
}

describe('[no placeholder copy] in the source', () => {
  it('should hold no rendered string that describes the build or who owes a feature', () => {
    const findings: string[] = [];
    for (const file of sourceFiles(SRC)) {
      const lines = withoutComments(readFileSync(file, 'utf8')).split('\n');
      lines.forEach((line, index) => {
        const hit = placeholderIn(line);
        if (hit !== null)
          findings.push(`${path.relative(SRC, file)}:${index + 1}: "${hit}" in ${line.trim()}`);
      });
    }
    expect(findings, findings.join('\n')).toEqual([]);
  });

  it('should find the placeholder that started this, if it came back (the scan is not blind)', () => {
    const shipped =
      '<Callout title="This build serves no run listing">GET /v1/runs is F6&apos;s and is not mounted</Callout>';
    expect(placeholderIn(withoutComments(shipped))).not.toBeNull();
    // ...and a comment saying the same is history, not copy.
    expect(
      placeholderIn(withoutComments('/* GET /v1/runs is not mounted in this build */')),
    ).toBeNull();
  });

  it('should find the requirement ids the New run screen shipped with (the jargon scan is not blind)', () => {
    const shipped = [
      'It collects the definition and submits it whole, which is R-02.53&apos;s create path',
      "hint: 'R-02.13. No initial value: a learning rate the operator did not choose'",
      'entryReason="the Hub cannot be listed from here (ADR-010)."',
      'The attach acquisition mode (BRD-02 §5.10, the optional hot-update mode)',
    ];
    for (const line of shipped) expect(placeholderIn(withoutComments(line)), line).not.toBeNull();
  });
});

/* ------------------------------------------------------------------------------------ *
 * 2. What every screen renders
 * ------------------------------------------------------------------------------------ */

let originalFetch: typeof globalThis.fetch | undefined;

afterEach(() => {
  if (originalFetch !== undefined) globalThis.fetch = originalFetch;
  cleanup();
});

function backend(kind: 'stub' | 'dead'): void {
  originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    if (kind === 'dead') return Promise.reject(new TypeError('fetch failed'));
    const url = new URL(typeof input === 'string' ? input : input.toString(), 'http://local');
    const { status, body } = answer((init?.method ?? 'GET').toUpperCase(), url.pathname);
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  }) as typeof globalThis.fetch;
}

interface Target {
  readonly id: ScreenId;
  readonly params: ScreenParams;
  readonly label: string;
}

function screens(): Target[] {
  return ROUTES.flatMap((route): Target[] =>
    route.id === 'onboard'
      ? DRAFT_STEPS.map((step) => ({
          id: route.id,
          params: { draftId: DRAFT_ID, step: String(step.number) },
          label: `onboard step ${step.number}`,
        }))
      : [{ id: route.id, params: route.id === 'run' ? { runId: 'run-a' } : {}, label: route.id }],
  );
}

describe('[no placeholder copy] on every screen', () => {
  for (const kind of ['stub', 'dead'] as const) {
    it(`should render none, with a ${kind} backend`, async () => {
      const findings: string[] = [];
      for (const target of screens()) {
        window.localStorage.clear();
        window.history.replaceState(null, '', '/');
        backend(kind);
        render(
          <ThemeProvider initialTheme="dark">
            <App initialScreen={target.id} initialParams={target.params} />
          </ThemeProvider>,
        );
        for (let tick = 0; tick < 5; tick += 1) {
          await act(async () => {
            await new Promise((resolve) => setTimeout(resolve, 0));
          });
        }
        const text = screen.getByTestId('shell-root').textContent ?? '';
        const hit = placeholderIn(text);
        if (hit !== null) findings.push(`${target.label}: "${hit}"`);
        cleanup();
        if (originalFetch !== undefined) globalThis.fetch = originalFetch;
      }
      expect(findings, findings.join('\n')).toEqual([]);
    });
  }
});
