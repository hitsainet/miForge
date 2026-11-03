/**
 * The import panel closes the way it opened (operator, 2026-10-31: "clicking … a toggle open,
 * then closed … for ALL pop up detailed modals"). The "Import pack" button toggles it, its own
 * "Close this panel" closes it, and so does Escape.
 */
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import type { PackListingRow } from '@/types/packs';
import { NavigationProvider } from '@/contexts/NavigationContext';
import { PacksPage } from './PacksPage';

const PACK = vi.hoisted((): PackListingRow => ({
  pack_id: 'pk-notary',
  version: '3',
  name: 'notary',
  domain: 'notary',
  parts_filled: 7,
  parts_total: 7,
  prompt_count: 640,
  runs_using: [],
  validity: {
    state: 'valid',
    missing_slots: [],
    unresolved: [],
    service_unavailable: [],
    retention_floor_missing: false,
  },
  derived_from: null,
  is_example: false,
  definition_sha256: 'd'.repeat(64),
  created_at: '2026-10-14T00:00:00Z',
}));

vi.mock('@/api/packs', async (importOriginal) => {
  const original = await importOriginal<typeof import('@/api/packs')>();
  return {
    ...original,
    fetchPacks: vi.fn(async () => ({ ok: true, value: { packs: [PACK] } })),
    fetchPackLineage: vi.fn(async () => ({ ok: true, value: { slots: [] } })),
  };
});

function renderPage() {
  render(
    <NavigationProvider initialScreen="packs">
      <PacksPage />
    </NavigationProvider>,
  );
  return screen.getByTestId('import-pack-toggle');
}

describe('[PacksPage] the import panel toggles', () => {
  it('should open on the first click and close on the second', () => {
    const toggle = renderPage();
    expect(screen.queryByTestId('import-pack')).toBeNull();
    fireEvent.click(toggle);
    expect(screen.getByTestId('import-pack')).toBeInTheDocument();
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    fireEvent.click(toggle);
    expect(screen.queryByTestId('import-pack')).toBeNull();
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
  });

  it('should close from its own close button', () => {
    fireEvent.click(renderPage());
    fireEvent.click(screen.getByRole('button', { name: 'Close this panel' }));
    expect(screen.queryByTestId('import-pack')).toBeNull();
  });

  it('should close on Escape', () => {
    fireEvent.click(renderPage());
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByTestId('import-pack')).toBeNull();
  });
});

describe('[PacksPage] the lineage list toggles', () => {
  it('should show the list on the first click and hide it on the second', async () => {
    renderPage();
    const toggle = await screen.findByTestId('lineage-toggle-pk-notary');
    expect(screen.queryByTestId('lineage-pk-notary')).toBeNull();
    fireEvent.click(toggle);
    expect(await screen.findByTestId('lineage-pk-notary')).toBeInTheDocument();
    expect(toggle).toHaveTextContent('Hide where every part came from');
    fireEvent.click(toggle);
    expect(screen.queryByTestId('lineage-pk-notary')).toBeNull();
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
  });
});
