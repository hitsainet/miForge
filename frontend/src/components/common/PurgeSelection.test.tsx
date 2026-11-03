/**
 * Select archived items and delete them permanently (operator request 2026-10-20).
 *
 * Driven through the part index (the smaller host) with the backend injected: the header
 * checkbox selects and deselects every selectable item and is indeterminate in between, the
 * count is live, a blocked item cannot be selected and says why, and the confirmation sends
 * **exactly** the selected ids that are still purgeable.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { PartIndex } from '@/components/panels/packs/PartIndex';
import type { PartRow } from '@/api/packs';
import type { PurgeBatch, PurgeKind, PurgePlan } from '@/api/purge';
import { formatBytes } from '@/api/purge';

function part(id: string, archived: boolean): PartRow {
  return {
    id,
    part_kind: 'corpus',
    source: 'midataworks_contract',
    contract_reference: { type: 'corpus', name: id, version: '1' },
    rendered: `corpus/${id}@1`,
    dataset: id,
    version_number: 1,
    row_count: 10,
    caveats: [],
    licence_classes: [],
    imported_at: '2026-10-17T19:07:59+00:00',
    archived_at: archived ? '2026-10-20T12:00:00+00:00' : null,
  };
}

const ROWS = [
  part('live', false),
  part('a', true),
  part('b', true),
  part('c', true),
  part('x', true),
];

function plan(id: string, blocker: string | null = null): PurgePlan {
  return {
    kind: 'part',
    id,
    found: true,
    name: `part corpus/${id}@1`,
    archived: true,
    purgeable: blocker === null,
    not_archived: null,
    blockers: blocker === null ? [] : [{ code: 'pack_pins_it', message: blocker }],
    removes: {
      rows: { mf_contract_parts: 1 },
      row_total: 3,
      files: 2,
      bytes: 2048,
      directories: [`/data/parts/${id}`],
    },
    kept: [],
  };
}

function backend(blocked: Readonly<Record<string, string>> = { x: 'Pack ops@1 pins this part.' }) {
  const load = vi.fn(() =>
    Promise.resolve({ ok: true as const, value: { items: ROWS, total: 5 } }),
  );
  const preview = vi.fn((_kind: PurgeKind, ids: readonly string[]) =>
    Promise.resolve({
      ok: true as const,
      value: ids.map((id) => plan(id, blocked[id] ?? null)),
    }),
  );
  const purge = vi.fn((kind: PurgeKind, ids: readonly string[]) =>
    Promise.resolve({
      ok: true as const,
      value: {
        kind,
        purged: ids.length,
        blocked: 0,
        bytes_removed: 2048 * ids.length,
        results: ids.map((id) => ({
          outcome: 'purged' as const,
          id,
          name: `part corpus/${id}@1`,
          tombstone_id: `purge-${id}`,
          bytes_removed: 2048,
          files_removed: 2,
          rows_removed: {},
          files_left: [],
        })),
      } satisfies PurgeBatch,
    }),
  );
  return { load, preview, purge };
}

const box = (id: string) => screen.getByTestId(`part-purge-check-${id}`) as HTMLInputElement;
const header = () => screen.getByTestId('part-purge-select-all') as HTMLInputElement;

async function shown(api: ReturnType<typeof backend>) {
  window.localStorage.clear();
  render(<PartIndex load={api.load as never} preview={api.preview} purge={api.purge} />);
  await screen.findByTestId('part-row-live');
  fireEvent.click(screen.getByTestId('part-show-archived'));
  await waitFor(() => expect(box('a')).not.toBeDisabled());
}

beforeEach(() => window.localStorage.clear());
afterEach(() => window.localStorage.clear());

describe('[PurgeSelection]', () => {
  it('should offer a checkbox on archived items only, and no purge UI without "Show archived"', async () => {
    const api = backend();
    render(<PartIndex load={api.load as never} preview={api.preview} purge={api.purge} />);
    await screen.findByTestId('part-row-live');
    expect(screen.queryByTestId('part-purge-header')).toBeNull();
    expect(api.preview).not.toHaveBeenCalled();

    fireEvent.click(screen.getByTestId('part-show-archived'));
    await waitFor(() => expect(box('a')).not.toBeDisabled());

    expect(screen.queryByTestId('part-purge-check-live')).toBeNull();
    expect(api.preview).toHaveBeenLastCalledWith('part', ['a', 'b', 'c', 'x']);
  });

  it('should show a blocked item its blocker inline and never let it be selected', async () => {
    const api = backend();
    await shown(api);

    expect(box('x')).toBeDisabled();
    expect(screen.getByTestId('part-purge-blocked-x')).toHaveTextContent(
      'Pack ops@1 pins this part.',
    );
    fireEvent.click(header());
    expect(box('x')).not.toBeChecked();
    expect(screen.getByTestId('part-purge-count')).toHaveTextContent('3 selected');
  });

  it('should select all, deselect all, and go indeterminate when partly selected', async () => {
    const api = backend();
    await shown(api);
    expect(header().checked).toBe(false);
    expect(screen.getByTestId('part-purge-select-all-label')).toHaveTextContent('Select all');
    expect(screen.getByTestId('part-purge-delete')).toBeDisabled();

    fireEvent.click(header());
    expect(['a', 'b', 'c'].map((id) => box(id).checked)).toEqual([true, true, true]);
    expect(header().checked).toBe(true);
    expect(header().indeterminate).toBe(false);
    expect(screen.getByTestId('part-purge-select-all-label')).toHaveTextContent('Deselect all');

    fireEvent.click(box('b'));
    expect(screen.getByTestId('part-purge-count')).toHaveTextContent('2 selected');
    expect(header().indeterminate).toBe(true);
    expect(header().checked).toBe(false);

    fireEvent.click(header()); // partial -> deselect all
    expect(['a', 'b', 'c'].map((id) => box(id).checked)).toEqual([false, false, false]);
    expect(header().indeterminate).toBe(false);
    expect(screen.getByTestId('part-purge-count')).toHaveTextContent('0 selected');

    fireEvent.click(box('c'));
    expect(screen.getByTestId('part-purge-count')).toHaveTextContent('1 selected');
    fireEvent.click(box('c'));
    expect(screen.getByTestId('part-purge-count')).toHaveTextContent('0 selected');
  });

  it('should confirm with every item listed and send exactly the selected ids', async () => {
    const api = backend();
    await shown(api);
    fireEvent.click(header());
    fireEvent.click(box('b'));

    fireEvent.click(screen.getByTestId('part-purge-delete'));
    const dialog = await screen.findByTestId('part-purge-dialog');
    await within(dialog).findByTestId('part-purge-going-a');
    expect(within(dialog).getByTestId('part-purge-going-a')).toHaveTextContent(
      `3 database rows, 2 files, ${formatBytes(2048)}`,
    );
    expect(within(dialog).getByTestId('part-purge-going-c')).toBeInTheDocument();
    expect(within(dialog).queryByTestId('part-purge-going-b')).toBeNull();
    expect(api.purge).not.toHaveBeenCalled();

    fireEvent.change(within(dialog).getByTestId('part-purge-reason'), {
      target: { value: 'test clutter' },
    });
    fireEvent.click(within(dialog).getByTestId('part-purge-confirm'));

    await within(dialog).findByTestId('part-purge-outcome');
    expect(api.purge).toHaveBeenCalledTimes(1);
    expect(api.purge).toHaveBeenCalledWith('part', ['a', 'c'], 'test clutter');
    expect(within(dialog).getByTestId('part-purge-result-a')).toHaveTextContent('deleted');
  });

  it('should skip, and name, an item that became blocked after it was selected', async () => {
    const api = backend({});
    await shown(api);
    fireEvent.click(header()); // a, b, c, x all selectable now
    api.preview.mockImplementation((_kind, ids) =>
      Promise.resolve({
        ok: true as const,
        value: ids.map((id) =>
          plan(id, id === 'c' ? 'Run r9 (not archived) used this part.' : null),
        ),
      }),
    );

    fireEvent.click(screen.getByTestId('part-purge-delete'));
    const dialog = await screen.findByTestId('part-purge-dialog');
    await within(dialog).findByTestId('part-purge-skipped-c');
    expect(within(dialog).getByTestId('part-purge-skipped-c')).toHaveTextContent('r9');
    fireEvent.click(within(dialog).getByTestId('part-purge-confirm'));

    await waitFor(() => expect(api.purge).toHaveBeenCalledTimes(1));
    expect(api.purge).toHaveBeenCalledWith('part', ['a', 'b', 'x'], '');
  });

  it('should close on Escape and on a click outside, sending nothing', async () => {
    // Operator, 2026-10-31: every popup closes the way it opened.
    const api = backend();
    await shown(api);
    fireEvent.click(box('a'));
    fireEvent.click(screen.getByTestId('part-purge-delete'));
    const dialog = await screen.findByTestId('part-purge-dialog');
    await within(dialog).findByTestId('part-purge-going-a');
    // A click inside the card does not close it.
    fireEvent.click(dialog);
    expect(screen.getByTestId('part-purge-dialog')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('part-purge-dialog-backdrop'));
    expect(screen.queryByTestId('part-purge-dialog')).toBeNull();

    fireEvent.click(screen.getByTestId('part-purge-delete'));
    await screen.findByTestId('part-purge-dialog');
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByTestId('part-purge-dialog')).toBeNull();
    expect(api.purge).not.toHaveBeenCalled();
  });

  it('should send nothing when the confirmation is cancelled', async () => {
    const api = backend();
    await shown(api);
    fireEvent.click(box('a'));
    fireEvent.click(screen.getByTestId('part-purge-delete'));
    const dialog = await screen.findByTestId('part-purge-dialog');
    await within(dialog).findByTestId('part-purge-going-a');

    fireEvent.click(within(dialog).getByTestId('part-purge-cancel'));

    expect(screen.queryByTestId('part-purge-dialog')).toBeNull();
    expect(api.purge).not.toHaveBeenCalled();
    expect(box('a')).toBeChecked();
  });
});
