/**
 * A reload keeps the screen and the entries (F9 task 13.0; AC-009.31-.39). **M23, M24, M26.**
 *
 * ## The wiring is derived, not listed
 *
 * The first test walks **every row of `ROUTES`** - the live registry - renders the screen,
 * fills **every editable control it renders**, unmounts, and mounts a fresh `<App />` with no
 * props, so the screen comes back **from the URL alone** and each control's value **from
 * storage alone**. A form that keeps its entries some other way, or not at all, fails here
 * without anybody adding it to a list - and a new screen with a new form is covered the day
 * its row is added. `localStorage` and the URL are reset between tests (`tests/setup.ts`).
 *
 * The network is stubbed at `fetch` - the HTTP boundary, the same rule the form's own spec
 * follows - with answers that are **identical on both mounts**, so a value can only come back
 * from what this browser kept.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { App } from './App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { ROUTES } from '@/config/routes';
import type { ScreenId } from '@/contexts/NavigationContext';
import { DRAFT_KEYS, DRAFT_PREFIX } from '@/stores/draftStorage';
import { DRAFT_ID, answer, draftRecord, type Call } from '@/testing/stubBackend';

let calls: Call[] = [];
let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  calls = [];
  originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === 'string' ? input : input.toString(), 'http://local');
    const method = (init?.method ?? 'GET').toUpperCase();
    calls.push({ method, path: url.pathname });
    const { status, body } = answer(method, url.pathname);
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

const inTheme = (element: React.ReactElement) => (
  <ThemeProvider initialTheme="dark">{element}</ThemeProvider>
);

type Control = HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;

/** Every control an operator can type into or choose from. A file input cannot be restored. */
function editableControls(): Control[] {
  const root = screen.getByTestId('shell-content');
  return [...root.querySelectorAll<Control>('input, select, textarea')].filter(
    (node) =>
      !node.disabled &&
      !(node instanceof HTMLInputElement && ['file', 'hidden'].includes(node.type)) &&
      !(node instanceof HTMLInputElement && node.readOnly),
  );
}

function keyOf(node: Control): string {
  return node.getAttribute('data-testid') ?? node.id;
}

/**
 * What a control holds. For a **strict** picker that is the value it would submit
 * (`data-value`), not the text in its input - which is the label, or a search in progress.
 * Typing into one chooses nothing, so the derived test chooses a row instead (task 14.0).
 */
function heldValue(node: Control): string {
  // A checkbox holds its checked state, not its `value` (which is "on" either way) - so a
  // toggle such as "Show archived runs" is checked like every other entry (2026-10-20).
  if (node instanceof HTMLInputElement && node.type === 'checkbox') return String(node.checked);
  return node.getAttribute('data-picker') === 'strict'
    ? (node.getAttribute('data-value') ?? '')
    : node.value;
}

/** Choose the last enabled row a strict picker lists, as an operator's click does. */
function chooseFromPicker(node: Control): string | undefined {
  fireEvent.click(node);
  const listbox = document.getElementById(node.getAttribute('aria-controls') ?? '');
  const row = [...(listbox?.querySelectorAll<HTMLElement>('[role="option"]') ?? [])]
    .reverse()
    .find(
      (option) =>
        option.getAttribute('aria-disabled') !== 'true' &&
        option.getAttribute('data-value') !== heldValue(node),
    );
  if (row === undefined) {
    fireEvent.keyDown(node, { key: 'Escape' });
    return undefined;
  }
  fireEvent.mouseDown(row);
  return row.getAttribute('data-value') ?? undefined;
}

async function settle(): Promise<void> {
  // Several fetch round trips, each resolving on a microtask.
  for (let index = 0; index < 5; index += 1) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
}

/** Fill every control, including the ones a choice reveals. Returns key -> value entered. */
async function fillAll(): Promise<Map<string, string>> {
  const entered = new Map<string, string>();
  for (let pass = 0; pass < 4; pass += 1) {
    let changed = false;
    for (const node of editableControls()) {
      const key = keyOf(node);
      if (entered.has(key)) continue;
      if (node.getAttribute('data-picker') === 'strict') {
        const chosen = chooseFromPicker(node);
        if (chosen === undefined) continue;
        entered.set(key, chosen);
      } else if (node instanceof HTMLInputElement && node.type === 'checkbox') {
        fireEvent.click(node);
        entered.set(key, String(node.checked));
      } else if (node instanceof HTMLSelectElement) {
        const option = [...node.options]
          .map((row) => row.value)
          .reverse()
          .find((value) => value.length > 0 && value !== node.value);
        if (option === undefined) continue;
        fireEvent.change(node, { target: { value: option } });
        entered.set(key, option);
      } else {
        const value = `7${entered.size + 1}`;
        fireEvent.change(node, { target: { value } });
        entered.set(key, value);
      }
      changed = true;
    }
    await settle();
    if (!changed) break;
  }
  return entered;
}

const PARAMS: Partial<Record<ScreenId, Readonly<Record<string, string>>>> = {
  run: { runId: 'run-a' },
};

describe('[persistence] every registered screen, derived from ROUTES', () => {
  it('should come back from the URL with every entry it was left with (AC-009.31, .34)', async () => {
    const restoredPerScreen: Record<string, number> = {};
    const pickersPerScreen: Record<string, number> = {};
    for (const route of ROUTES) {
      window.localStorage.clear();
      window.history.replaceState(null, '', '/');
      const first = render(
        inTheme(<App initialScreen={route.id} initialParams={PARAMS[route.id] ?? {}} />),
      );
      await settle();
      const entered = await fillAll();
      // Each value is on screen before the reload - otherwise the test would be checking a
      // control that never took the input (the Onboard step-1 defect, task 13.5).
      for (const node of editableControls()) {
        const expected = entered.get(keyOf(node));
        if (expected !== undefined)
          expect(heldValue(node), `${route.id}/${keyOf(node)}`).toBe(expected);
      }
      first.unmount();
      cleanup();

      // The reload: no props. The screen comes from the URL, the values from storage.
      render(inTheme(<App />));
      await settle();
      expect(
        screen.getByTestId('shell-page-head').querySelector('h1')?.textContent,
        `${route.id} did not come back from the URL`,
      ).toBe(route.id === 'run' ? 'run-a' : route.title);
      const after = new Map(editableControls().map((node) => [keyOf(node), heldValue(node)]));
      for (const [key, value] of entered) {
        expect(after.get(key), `${route.id}: "${key}" lost its entry on reload`).toBe(value);
      }
      restoredPerScreen[route.id] = entered.size;
      pickersPerScreen[route.id] = [...entered.keys()].filter(
        (key) => after.has(key) && document.querySelector(`[data-testid="${key}"][data-picker]`),
      ).length;
      cleanup();
    }
    // Not vacuous: the screens with forms rendered and restored their controls. The floor is
    // deliberately loose - a new field raises the count, a lost form drops it to zero.
    expect(restoredPerScreen['new-run'] ?? 0).toBeGreaterThanOrEqual(20);
    // The Runs screen: the two list filters (pickers) and the open-by-id box.
    expect(restoredPerScreen['runs'] ?? 0).toBeGreaterThanOrEqual(3);
    expect(restoredPerScreen['onboard'] ?? 0).toBeGreaterThanOrEqual(2);
    // Not vacuous for the pickers either (AC-009.46): New run's mode, path, config, run type,
    // signal source and GPU, and the Runs screen's two filters, each chosen and restored.
    expect(pickersPerScreen['new-run'] ?? 0).toBeGreaterThanOrEqual(6);
    expect(pickersPerScreen['runs'] ?? 0).toBeGreaterThanOrEqual(2);
  });
});

describe('[persistence] the URL', () => {
  it('should write the screen into the hash on every navigation, as a history entry (AC-009.32)', async () => {
    render(inTheme(<App />));
    await settle();
    expect(window.location.hash).toBe('#/runs');
    const before = window.history.length;
    fireEvent.click(screen.getByRole('button', { name: 'Checkpoints' }));
    await settle();
    expect(window.location.hash).toBe('#/checkpoints');
    expect(window.history.length).toBe(before + 1);
    fireEvent.click(screen.getByRole('button', { name: 'New run' }));
    expect(window.location.hash).toBe('#/runs/new');
  });

  it('should follow the URL when it changes under the app: back, forward, a typed hash', async () => {
    render(inTheme(<App />));
    await settle();
    act(() => {
      window.location.hash = '#/packs';
      window.dispatchEvent(new HashChangeEvent('hashchange'));
    });
    await settle();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Domain packs');
    act(() => {
      window.history.pushState(null, '', '#/runs/run-a');
      window.dispatchEvent(new PopStateEvent('popstate'));
    });
    await settle();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('run-a');
  });

  it('should fall back to Runs for an unknown or malformed URL and rewrite it (AC-009.33)', async () => {
    for (const hash of ['#/nowhere', '#/runs/%E0%A4%A', '#/runs/a/b/c', '#garbage']) {
      window.history.replaceState(null, '', `/${hash}`);
      render(inTheme(<App />));
      await settle();
      expect(screen.getByRole('heading', { level: 1 }), hash).toHaveTextContent('Runs');
      expect(screen.queryByText(/is not in the route registry/)).toBeNull();
      expect(window.location.hash, hash).toBe('#/runs');
      cleanup();
    }
  });

  it('should reopen the manifest the URL names once the forest is read', async () => {
    window.history.replaceState(null, '', '/#/checkpoints?node=run-a%2Fstep-40');
    render(inTheme(<App />));
    await waitFor(() => expect(screen.getByTestId('tree-manifest')).toBeInTheDocument());
    expect(calls).toContainEqual({ method: 'GET', path: '/api/v1/runs/run-a/checkpoints/40' });
  });

  it('should put an opened manifest into the URL without adding a history entry', async () => {
    window.history.replaceState(null, '', '/#/checkpoints');
    render(inTheme(<App />));
    await waitFor(() => expect(screen.getByTestId('tree')).toBeInTheDocument());
    const before = window.history.length;
    fireEvent.click(screen.getByRole('button', { name: 'Read the manifest' }));
    await settle();
    expect(window.location.hash).toBe('#/checkpoints?node=run-a%2Fstep-40');
    expect(window.history.length).toBe(before);
  });
});

describe('[persistence] Onboard a domain', () => {
  it('should re-read the same draft on reload and create none (AC-009.39)', async () => {
    render(inTheme(<App initialScreen="onboard" />));
    await waitFor(() => expect(screen.getByTestId('onboard-panel')).toBeInTheDocument());
    await settle();
    expect(window.location.hash).toBe(`#/packs/onboard?draftId=${DRAFT_ID}&step=1`);
    expect(calls.filter((call) => call.method === 'POST')).toHaveLength(1);
    cleanup();

    calls = [];
    render(inTheme(<App />));
    await waitFor(() => expect(screen.getByTestId('onboard-panel')).toBeInTheDocument());
    expect(calls.filter((call) => call.path === '/api/v1/pack-drafts')).toEqual([]);
    expect(calls).toContainEqual({ method: 'GET', path: `/api/v1/pack-drafts/${DRAFT_ID}` });
  });

  it('should open the step the URL names, and the server step for one it does not have', async () => {
    window.history.replaceState(null, '', `/#/packs/onboard?draftId=${DRAFT_ID}&step=6`);
    render(inTheme(<App />));
    await waitFor(() => expect(screen.getByTestId('onboard-panel')).toHaveTextContent('6. '));
    fireEvent.change(screen.getByTestId('draft-floor'), { target: { value: '0.78' } });
    expect((screen.getByTestId('draft-floor') as HTMLInputElement).value).toBe('0.78');
    await settle();
    cleanup();

    render(inTheme(<App />));
    await waitFor(() => expect(screen.getByTestId('onboard-panel')).toHaveTextContent('6. '));
    expect((screen.getByTestId('draft-floor') as HTMLInputElement).value).toBe('0.78');
    expect(screen.getByTestId('draft-restored')).toBeInTheDocument();
    cleanup();

    window.history.replaceState(null, '', `/#/packs/onboard?draftId=${DRAFT_ID}&step=99`);
    render(inTheme(<App />));
    await waitFor(() => expect(screen.getByTestId('onboard-panel')).toHaveTextContent('1. '));
  });

  it('should drop the kept step answers once the step is committed', async () => {
    window.history.replaceState(null, '', `/#/packs/onboard?draftId=${DRAFT_ID}&step=1`);
    render(inTheme(<App />));
    await waitFor(() => expect(screen.getByTestId('draft-name')).toBeInTheDocument());
    fireEvent.change(screen.getByTestId('draft-name'), { target: { value: 'notary-md' } });
    await settle();
    const key = DRAFT_PREFIX + DRAFT_KEYS.onboard(DRAFT_ID);
    expect(window.localStorage.getItem(key)).toContain('notary-md');
    // The PUT is not stubbed as a success here, so commit through a stub that accepts it.
    const accept = globalThis.fetch;
    globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
      if ((init?.method ?? 'GET') === 'PUT')
        return Promise.resolve(
          new Response(JSON.stringify({ ...draftRecord(), current_step: 2 }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
        );
      return accept(input, init);
    }) as typeof globalThis.fetch;
    fireEvent.click(screen.getByRole('button', { name: 'Save and continue' }));
    await settle();
    expect(window.localStorage.getItem(key)).toBeNull();
  });
});

describe('[persistence] Runs', () => {
  it('should keep the runs opened here and RE-READ them on reload, never replay them', async () => {
    render(inTheme(<App />));
    fireEvent.change(screen.getByTestId('run-id-input'), { target: { value: 'run-a' } });
    fireEvent.submit(screen.getByTestId('run-id-input').closest('form') as HTMLFormElement);
    await waitFor(() => expect(screen.getByTestId('run-index-list')).toHaveTextContent('run-a'));
    // Submitted: the box is empty, and the stored entry holds the id, not the record.
    expect((screen.getByTestId('run-id-input') as HTMLInputElement).value).toBe('');
    // The entry is written by an effect after the list renders: wait for it rather than racing it (CI lost
    // this race once the approvals banner added a render on every page, 2026-10-28).
    await waitFor(() =>
      expect(window.localStorage.getItem(DRAFT_PREFIX + DRAFT_KEYS.runsIndex) ?? '').toContain(
        '"opened":["run-a"]',
      ),
    );
    const stored = window.localStorage.getItem(DRAFT_PREFIX + DRAFT_KEYS.runsIndex) ?? '';
    expect(stored).not.toContain('stopped');
    cleanup();

    calls = [];
    render(inTheme(<App />));
    await waitFor(() => expect(screen.getByTestId('run-index-list')).toHaveTextContent('run-a'));
    expect(calls.filter((call) => call.path === '/api/v1/runs/run-a')).toHaveLength(1);
  });

  it('should drop a kept run the backend says is gone, and keep one it could not answer for', async () => {
    window.localStorage.setItem(
      DRAFT_PREFIX + DRAFT_KEYS.runsIndex,
      JSON.stringify({
        v: 1,
        savedAt: Date.now(),
        data: { query: '', opened: ['run-gone', 'run-down'] },
      }),
    );
    render(inTheme(<App />));
    await settle();
    const stored = window.localStorage.getItem(DRAFT_PREFIX + DRAFT_KEYS.runsIndex) ?? '';
    expect(stored).not.toContain('run-gone');
    expect(stored).toContain('run-down');
  });
});
