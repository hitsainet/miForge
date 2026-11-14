/**
 * **Every field on every screen says what it holds** (task 14.0, D-14c; AC-009.44).
 *
 * The operator's request was that a field naming something which exists be a list of what
 * exists. A test that checked a hand-written list of field ids would stay green the day a new
 * screen added a text box for a pack id. So this walks **every row of `ROUTES`** - and every
 * step of the Onboard flow, from `DRAFT_STEPS` - renders it against a backend stubbed at
 * `fetch`, reveals what a choice reveals (it adds a stop condition and chooses an option in
 * each picker), and requires every control it finds to be one of:
 *
 * - a **`Picker`** (`role="combobox"`, `data-picker`) - and a `free` one must carry its reason;
 * - an **`Input` that declares its entry kind** - `number`, `new-name`, `prose`, `address`
 *   (a sibling's base URL on Settings), or `id-fallback` with its reason;
 * - a **native select declared a closed enum**, with the reason it is not listed by a route.
 *
 * A control that declares nothing fails, naming the screen and the control. So does a typed
 * identifier outside the audited escape hatches: those are the four the 009 FTASKS table
 * gives a reason for, and adding a fifth means adding a row there first.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { App } from './App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { ROUTES } from '@/config/routes';
import type { ScreenId, ScreenParams } from '@/contexts/NavigationContext';
import { DRAFT_STEPS } from '@/types/packs';
import { DRAFT_ID, answer } from '@/testing/stubBackend';

const ENTRY_KINDS = new Set(['number', 'new-name', 'prose', 'id-fallback', 'address']);

/**
 * The typed identifiers the audit kept, each with its reason in the 009 FTASKS table (rows 7,
 * 8, 18). Row 2 - open a run by typing its id - became a dropdown on 2026-10-18 (operator:
 * "I don't want to have to know the id from memory"); pasting an id stays inside that picker. Typed because no list here can be complete - not because nobody built one.
 */
const AUDITED_ID_FALLBACKS = new Set([
  'repo-input',
  'revision-input',
  // A create run's tokenizer from the Hub (2026-10-20, 009 FTASKS 14.0 rows 29-30): the Hub
  // cannot be listed from here, as for the model's repository and revision.
  'tokenizer-repo-input',
  'tokenizer-revision-input',
  'adapter-target-modules',
]);

let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === 'string' ? input : input.toString(), 'http://local');
    const { status, body } = answer((init?.method ?? 'GET').toUpperCase(), url.pathname);
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  }) as typeof globalThis.fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

async function settle(): Promise<void> {
  for (let index = 0; index < 5; index += 1) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
}

type Control = HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;

function controls(): Control[] {
  return [
    ...screen.getByTestId('shell-content').querySelectorAll<Control>('input, select, textarea'),
  ].filter(
    (node) =>
      !(
        node instanceof HTMLInputElement &&
        ['file', 'hidden', 'checkbox', 'radio'].includes(node.type)
      ),
  );
}

/** Reveal what a choice reveals: add a stop condition, and choose in every empty picker. */
async function reveal(): Promise<void> {
  for (const button of screen.queryAllByRole('button', { name: /^Add a stop condition$/ })) {
    fireEvent.click(button);
  }
  await settle();
  for (let pass = 0; pass < 3; pass += 1) {
    for (const node of controls()) {
      if (node.getAttribute('data-picker') !== 'strict' || node.disabled) continue;
      if ((node.getAttribute('data-value') ?? '') !== '') continue;
      fireEvent.click(node);
      const listbox = document.getElementById(node.getAttribute('aria-controls') ?? '');
      const row = [...(listbox?.querySelectorAll<HTMLElement>('[role="option"]') ?? [])]
        .reverse()
        .find((option) => option.getAttribute('aria-disabled') !== 'true');
      if (row === undefined) fireEvent.keyDown(node, { key: 'Escape' });
      else fireEvent.mouseDown(row);
    }
    await settle();
  }
}

interface Finding {
  readonly where: string;
  readonly control: string;
  readonly problem: string;
}

/** Every rendered screen, including each Onboard step. Derived from the two registries. */
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
      : [
          {
            id: route.id,
            params: route.id === 'run' ? { runId: 'run-a' } : {},
            label: route.id,
          },
        ],
  );
}

describe('[field declarations] every control on every ROUTES screen says what it holds', () => {
  it('should find no undeclared control, no unreasoned escape hatch, and only audited ones', async () => {
    const findings: Finding[] = [];
    const pickers: Record<string, number> = {};
    const fallbacks = new Set<string>();

    for (const target of screens()) {
      window.localStorage.clear();
      window.history.replaceState(null, '', '/');
      render(
        <ThemeProvider initialTheme="dark">
          <App initialScreen={target.id} initialParams={target.params} />
        </ThemeProvider>,
      );
      await settle();
      await reveal();

      for (const node of controls()) {
        const name = node.getAttribute('data-testid') ?? node.id;
        const add = (problem: string) =>
          findings.push({ where: target.label, control: name, problem });
        if (node.getAttribute('role') === 'combobox') {
          const kind = node.getAttribute('data-picker');
          if (kind !== 'strict' && kind !== 'free') add('a combobox that is not the shared Picker');
          if (kind === 'free' && (node.getAttribute('data-free-text-reason') ?? '').length === 0)
            add('a typed escape hatch with no reason');
          pickers[target.label] = (pickers[target.label] ?? 0) + 1;
          continue;
        }
        if (node instanceof HTMLSelectElement) {
          if ((node.getAttribute('data-closed-enum') ?? '').length === 0)
            add('a native select that is neither a Picker nor a declared closed enum');
          continue;
        }
        const entry = node.getAttribute('data-entry');
        if (entry === null || !ENTRY_KINDS.has(entry)) {
          add(
            'a text field that declares nothing - if it names something that exists, it is a Picker',
          );
          continue;
        }
        if (entry === 'id-fallback') {
          if ((node.getAttribute('data-entry-reason') ?? '').length === 0)
            add('an id-fallback with no reason');
          fallbacks.add(name);
        }
      }
      cleanup();
    }

    expect(findings, JSON.stringify(findings, null, 2)).toEqual([]);
    // Every typed identifier is one the audit kept, and each audited one is still there.
    expect([...fallbacks].sort()).toEqual([...AUDITED_ID_FALLBACKS].sort());
    // Not vacuous: the screens the audit converted render their pickers.
    expect(pickers['new-run'] ?? 0).toBeGreaterThanOrEqual(8);
    // The two list filters and the "Open a run" dropdown (operator, 2026-10-18).
    expect(pickers['runs'] ?? 0).toBeGreaterThanOrEqual(3);
    for (const step of DRAFT_STEPS.filter((row) => row.slots.length > 0)) {
      expect(pickers[`onboard step ${step.number}`] ?? 0, `onboard step ${step.number}`).toBe(
        step.slots.length,
      );
    }
  });
});
