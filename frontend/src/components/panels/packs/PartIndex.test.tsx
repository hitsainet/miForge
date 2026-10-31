/**
 * The part index on the Packs screen, with Archive / Unarchive (operator decision 2026-10-20).
 * Asserted on the reads and writes each action makes, through injected functions.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { PartIndex } from './PartIndex';
import { fetchParts, type PartRow } from '@/api/packs';

function part(id: string, overrides: Partial<PartRow> = {}): PartRow {
  return {
    id,
    part_kind: 'corpus',
    source: 'midataworks_contract',
    contract_reference: { type: 'corpus', name: id, version: '1' },
    rendered: `corpus/${id}@1`,
    dataset: id,
    version_number: 1,
    row_count: 400,
    caveats: [],
    licence_classes: [],
    imported_at: '2026-10-17T19:07:59+00:00',
    archived_at: null,
    ...overrides,
  };
}

const LIVE = part('live');
const OLD = part('old', { archived_at: '2026-10-20T12:00:00+00:00' });

function loader() {
  return vi.fn((options: { readonly includeArchived?: boolean } = {}) =>
    Promise.resolve({
      ok: true as const,
      value:
        options.includeArchived === true
          ? { items: [LIVE, OLD], total: 2 }
          : { items: [LIVE], total: 1 },
    }),
  );
}

const outcome = (id: string, archived: boolean) =>
  Promise.resolve({
    ok: true as const,
    value: { part_id: id, archived, archived_at: archived ? 'x' : null, changed: true },
  });

beforeEach(() => window.localStorage.clear());
afterEach(() => window.localStorage.clear());

describe('[PartIndex]', () => {
  it('should hide archived parts by default and show them, marked, when ticked', async () => {
    const load = loader();
    render(<PartIndex load={load as never} />);
    await screen.findByTestId('part-row-live');
    expect(load).toHaveBeenLastCalledWith({ includeArchived: false });
    expect(screen.queryByTestId('part-row-old')).toBeNull();

    fireEvent.click(screen.getByTestId('part-show-archived'));

    await screen.findByTestId('part-row-old');
    expect(load).toHaveBeenLastCalledWith({ includeArchived: true });
    expect(screen.getByTestId('part-row-old')).toHaveTextContent('Archived');
    expect(screen.getByTestId('part-archive-action-old')).toHaveTextContent('Unarchive');
    expect(screen.getByTestId('part-archive-action-live')).toHaveTextContent('Archive');
  });

  it('should POST archive for that part once and re-read the index', async () => {
    const load = loader();
    const archive = vi.fn((id: string) => outcome(id, true));
    const unarchive = vi.fn((id: string) => outcome(id, false));
    render(<PartIndex load={load as never} archive={archive} unarchive={unarchive} />);
    await screen.findByTestId('part-row-live');
    const reads = load.mock.calls.length;

    fireEvent.click(screen.getByTestId('part-archive-action-live'));

    await waitFor(() => expect(load.mock.calls.length).toBe(reads + 1));
    expect(archive).toHaveBeenCalledTimes(1);
    expect(archive).toHaveBeenCalledWith('live');
    expect(unarchive).not.toHaveBeenCalled();
  });

  it('should say how to see archived parts when none is listed', async () => {
    const load = vi.fn(() =>
      Promise.resolve({ ok: true as const, value: { items: [], total: 0 } }),
    );
    render(<PartIndex load={load as never} />);
    expect(await screen.findByTestId('part-index-empty')).toHaveTextContent('Show archived parts');
  });
});

describe('[fetchParts] include_archived', () => {
  let original: typeof globalThis.fetch;
  const seen: string[] = [];
  beforeEach(() => {
    original = globalThis.fetch;
    globalThis.fetch = ((input: RequestInfo | URL) => {
      seen.push(typeof input === 'string' ? input : input.toString());
      return Promise.resolve(
        new Response(JSON.stringify({ items: [], total: 0 }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    }) as typeof globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = original;
  });

  it('should send include_archived=true only when asked', async () => {
    seen.length = 0;
    await fetchParts({ partKind: 'corpus' });
    await fetchParts({ partKind: 'corpus', includeArchived: true });
    expect(seen[0]).not.toContain('include_archived');
    expect(seen[1]).toContain('include_archived=true');
  });
});
