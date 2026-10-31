/**
 * A pack draft's slots are filled by choosing parts (task 14.0; AC-009.48).
 *
 * Until 2026-10-18 no control on the Onboard screen could fill a slot, and step 6 sent its
 * floor as a top-level `retention_floor` that `StepRequest` (`extra="forbid"`) refuses. The
 * assertions here are on **the PUT body and how many PUTs were sent**: the chosen part goes in
 * as the slot document `GET /pack-part-refs/{id}` returned (never the part id, which is this
 * installation's index key), and the floor travels inside the retention slot.
 *
 * Stubbed at `fetch`, the HTTP boundary, as every screen test here is.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { OnboardPage } from './OnboardPage';
import { stepBody, slotKey, FLOOR_KEY } from './stepBody';
import { partOptions } from './OnboardSteps';
import { NavigationProvider } from '@/contexts/NavigationContext';
import { choose, openPicker, pickerValues } from '@/components/common/pickerTestKit';
import type { DraftRecord } from '@/types/packs';
import type { PartRow, PinnableReference } from '@/api/packs';

const DRAFT_ID = 'draft-9';

const CONTRACT_DOC = {
  contract_reference: { type: 'prompt-set', name: 'support-tickets', version: '3' },
  external: null,
  floor: null,
};

function draft(slots: Readonly<Record<string, unknown>> = {}): DraftRecord {
  return {
    draft_id: DRAFT_ID,
    name: 'notary',
    version: '1',
    domain: null,
    current_step: 2,
    step_count: 7,
    step_states: { '1': 'done', '2': 'current' },
    slots: {
      prompt_set: null,
      corpus: null,
      preference_pairs: null,
      selector_config: null,
      grader_config: null,
      test_set: null,
      retention_set: null,
      ...slots,
    },
    created_at: '2026-10-17T00:00:00Z',
    updated_at: '2026-10-17T00:00:00Z',
    last_dry_run: null,
  };
}

function part(id: string, overrides: Partial<PartRow> = {}): PartRow {
  return {
    id,
    part_kind: 'prompt_set',
    // The INDEX's spelling (`PART_SOURCE_CONTRACT`), not a pack reference's `midataworks`.
    source: 'midataworks_contract',
    contract_reference: { type: 'prompt-set', name: 'support-tickets', version: '3' },
    rendered: 'prompt-set/support-tickets@3',
    dataset: 'support-tickets',
    version_number: 3,
    row_count: 120,
    caveats: [],
    licence_classes: ['permissive'],
    imported_at: '2026-10-17T00:00:00Z',
    ...overrides,
  };
}

const PARTS = [
  part('part-1'),
  part('part-2', {
    contract_reference: null,
    rendered: null,
    dataset: 'unpinned',
    version_number: 1,
  }),
];

const ARCHIVED_PART = part('part-3', {
  rendered: 'prompt-set/old-tickets@1',
  archived_at: '2026-10-20T12:00:00+00:00',
});

let puts: { path: string; body: unknown }[] = [];
let partQueries: string[] = [];
/** `include_archived` on each part read: `null` when not sent (operator 2026-10-20). */
let archivedFlags: (string | null)[] = [];
let refReads: string[] = [];
let originalFetch: typeof globalThis.fetch;

function respond(body: unknown, status = 200): Promise<Response> {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  );
}

beforeEach(() => {
  puts = [];
  partQueries = [];
  archivedFlags = [];
  refReads = [];
  originalFetch = globalThis.fetch;
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(typeof input === 'string' ? input : input.toString(), 'http://local');
    const method = (init?.method ?? 'GET').toUpperCase();
    if (url.pathname === `/api/v1/pack-drafts/${DRAFT_ID}` && method === 'GET')
      return respond(draft());
    if (url.pathname === '/api/v1/parts') {
      partQueries.push(url.searchParams.get('part_kind') ?? '');
      archivedFlags.push(url.searchParams.get('include_archived'));
      return respond(
        url.searchParams.get('include_archived') === 'true'
          ? { items: [...PARTS, ARCHIVED_PART], total: PARTS.length + 1 }
          : { items: PARTS, total: PARTS.length },
      );
    }
    if (url.pathname.startsWith('/api/v1/pack-part-refs/')) {
      const id = decodeURIComponent(url.pathname.split('/').pop() ?? '');
      refReads.push(id);
      const ref: PinnableReference = {
        part_id: id,
        ref: { source: 'midataworks', type: 'prompt-set', name: 'support-tickets', version: '3' },
        source: 'midataworks',
        inner_form: 'prompt-set/support-tickets@3',
        slot_document: CONTRACT_DOC,
      };
      return respond(ref);
    }
    if (method === 'PUT') {
      puts.push({ path: url.pathname, body: JSON.parse(String(init?.body ?? '{}')) });
      return respond({ ...draft({ prompt_set: CONTRACT_DOC }), current_step: 3 });
    }
    return respond({ error: { code: 'not_stubbed', message: url.pathname, details: {} } }, 404);
  }) as typeof globalThis.fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('[Onboard] a slot is filled by choosing a part', () => {
  it('should list the parts of the slots own kind, an unpinnable one disabled', async () => {
    render(
      <NavigationProvider initialScreen="onboard" initialParams={{ draftId: DRAFT_ID, step: '2' }}>
        <OnboardPage />
      </NavigationProvider>,
    );
    expect(await pickerValues('slot-picker-prompt_set')).toEqual(['part-1', 'part-2']);
    // Filtered by the slot's own name - the contract spelling, with no translation table.
    expect(partQueries).toEqual(['prompt_set']);
    const listbox = await openPicker('slot-picker-prompt_set');
    const unpinnable = listbox.querySelector('[data-value="part-2"]');
    expect(unpinnable?.getAttribute('aria-disabled')).toBe('true');
    expect(unpinnable).toHaveTextContent('part_unpinnable');
  });

  it('should leave archived parts out until "Show archived parts" is ticked', async () => {
    window.localStorage.clear();
    render(
      <NavigationProvider initialScreen="onboard" initialParams={{ draftId: DRAFT_ID, step: '2' }}>
        <OnboardPage />
      </NavigationProvider>,
    );
    expect(await pickerValues('slot-picker-prompt_set')).toEqual(['part-1', 'part-2']);
    // Not sent at all by default: the route's default (archived hidden) applies.
    expect(archivedFlags).toEqual([null]);

    fireEvent.click(screen.getByTestId('slot-show-archived-2'));

    await waitFor(() => expect(archivedFlags.at(-1)).toBe('true'));
    expect(partQueries.at(-1)).toBe('prompt_set');
    await waitFor(async () =>
      expect(await pickerValues('slot-picker-prompt_set')).toEqual(['part-1', 'part-2', 'part-3']),
    );
    const listbox = await openPicker('slot-picker-prompt_set');
    expect(listbox.querySelector('[data-value="part-3"]')).toHaveTextContent('archived');
    window.localStorage.clear();
  });

  it('should send the resolved slot document, not the part id, in ONE step PUT', async () => {
    render(
      <NavigationProvider initialScreen="onboard" initialParams={{ draftId: DRAFT_ID, step: '2' }}>
        <OnboardPage />
      </NavigationProvider>,
    );
    await choose('slot-picker-prompt_set', 'part-1');
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Save and continue' }));
    });
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(refReads).toEqual(['part-1']);
    expect(puts[0]).toEqual({
      path: `/api/v1/pack-drafts/${DRAFT_ID}/steps/2`,
      body: { slots: { prompt_set: CONTRACT_DOC } },
    });
    expect(JSON.stringify(puts[0]?.body)).not.toContain('part-1');
  });
});

describe('[stepBody]', () => {
  const resolve = (id: string) =>
    Promise.resolve({
      ok: true as const,
      value: {
        part_id: id,
        ref: { source: 'midataworks' as const, type: 't', name: 'n', version: '1' },
        source: 'midataworks',
        inner_form: 't/n@1',
        slot_document: CONTRACT_DOC,
      },
    });

  it('should put the floor INSIDE the retention slot, never as a top-level field', async () => {
    const held = { ...CONTRACT_DOC, floor: 0.5 };
    const built = await stepBody(draft({ retention_set: held }), { [FLOOR_KEY]: '0.78' }, resolve);
    expect(built).toEqual({
      ok: true,
      body: { slots: { retention_set: { ...CONTRACT_DOC, floor: 0.78 } } },
    });
  });

  it('should put a floor on a newly chosen retention set', async () => {
    const built = await stepBody(
      draft(),
      { [slotKey('retention_set')]: 'part-r', [FLOOR_KEY]: '0.7' },
      resolve,
    );
    expect(built).toEqual({
      ok: true,
      body: { slots: { retention_set: { ...CONTRACT_DOC, floor: 0.7 } } },
    });
  });

  it('should send a floor with no set as a slot naming neither source, for the backend to refuse', async () => {
    // No client-side check: `pack_document_invalid` is the backend's answer.
    const built = await stepBody(draft(), { [FLOOR_KEY]: '0.7' }, resolve);
    expect(built).toEqual({
      ok: true,
      body: { slots: { retention_set: { contract_reference: null, external: null, floor: 0.7 } } },
    });
  });

  it('should pass the scalar answers through untouched and send no slots when none was chosen', async () => {
    const built = await stepBody(draft(), { name: 'notary', domain_given: true }, resolve);
    expect(built).toEqual({ ok: true, body: { name: 'notary', domain_given: true } });
  });

  it('should stop and name the failure when a chosen part cannot be resolved', async () => {
    const built = await stepBody(draft(), { [slotKey('test_set')]: 'part-x' }, () =>
      Promise.resolve({
        ok: false as const,
        failure: {
          kind: 'refusal' as const,
          status: 422,
          code: 'part_unpinnable',
          message: 'no reference',
          details: {},
        },
      }),
    );
    expect(built.ok).toBe(false);
  });
});

describe('[partOptions]', () => {
  it('should label a part by its contract reference and submit its id', () => {
    const [first] = partOptions({ items: [part('part-1')] });
    expect(first?.label).toBe('prompt-set/support-tickets@3');
    expect(first?.value).toBe('part-1');
    expect(first?.detail).toContain('120 rows');
  });

  it('should label an external part by its dataset, marked external', () => {
    const [first] = partOptions({
      items: [
        part('part-e', { source: 'external_files', contract_reference: null, rendered: null }),
      ],
    });
    expect(first?.label).toBe('support-tickets v3');
    expect(first?.detail).toContain('external part');
    expect(first?.disabledReason).toBeUndefined();
  });
});
