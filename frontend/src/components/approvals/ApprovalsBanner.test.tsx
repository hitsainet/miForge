/**
 * The approvals banner (ENH-008 D5, §2.3, A1): at most three cards, the soonest to expire first,
 * "and N more", Collapse kept for the session and undone by a new request, the region label,
 * the read failure shown rather than swallowed - and an approve from the banner is the same one
 * call the Approvals screen makes.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { NavigationProvider } from '@/contexts/NavigationContext';
import { INITIAL_APPROVALS_STATE, useApprovalsStore } from '@/stores/approvalsStore';
import type { Approval } from '@/api/approvals';
import { ApprovalsBanner, BANNER_COLLAPSE_KEY, BANNER_REGION_LABEL } from './ApprovalsBanner';

function row(id: string, expiresAt: string): Approval {
  return {
    approval_id: id,
    action: 'config_write',
    status: 'pending',
    summary: { key: 'log_level', old: 'INFO', new: 'DEBUG' },
    payload: { key: 'log_level', body: { value: 'DEBUG' } },
    requested_by: 'agent:forge-mcp',
    decided_by: null,
    reason: null,
    result: null,
    error: null,
    created_at: '2026-10-28T12:00:00+00:00',
    decided_at: null,
    expires_at: expiresAt,
  };
}

const FIVE = [
  row('e', '2026-10-29T16:00:00+00:00'),
  row('a', '2026-10-29T12:00:00+00:00'),
  row('d', '2026-10-29T15:00:00+00:00'),
  row('b', '2026-10-29T13:00:00+00:00'),
  row('c', '2026-10-29T14:00:00+00:00'),
];

let pending: Approval[] = [];
let posts: { path: string; headers: Record<string, string> }[] = [];

beforeEach(() => {
  pending = [];
  posts = [];
  window.sessionStorage.clear();
  useApprovalsStore.setState({ ...INITIAL_APPROVALS_STATE, inFlight: new Set<string>() });
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://local');
    if ((init?.method ?? 'GET') === 'POST') {
      posts.push({ path: url.pathname, headers: (init?.headers ?? {}) as Record<string, string> });
      const id = url.pathname.split('/')[4];
      const decided = { ...pending.find((r) => r.approval_id === id), status: 'executed' };
      pending = pending.filter((r) => r.approval_id !== id);
      return Promise.resolve(new Response(JSON.stringify(decided), { status: 200 }));
    }
    const listed = url.searchParams.get('status') === 'pending' ? pending : [];
    return Promise.resolve(
      new Response(
        JSON.stringify({ approvals: listed, count: listed.length, pending: 0, limit: 200 }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

async function loadWith(rows: Approval[]): Promise<void> {
  pending = rows;
  await act(async () => {
    await useApprovalsStore.getState().load();
  });
}

function renderBanner(): void {
  render(
    <NavigationProvider>
      <ApprovalsBanner />
    </NavigationProvider>,
  );
}

async function settle(): Promise<void> {
  for (let i = 0; i < 6; i += 1) {
    await act(async () => {
      await Promise.resolve();
    });
  }
}

describe('[ApprovalsBanner]', () => {
  it('should draw nothing when nothing waits and the read succeeded', async () => {
    await loadWith([]);
    renderBanner();
    expect(screen.queryByTestId('approvals-banner')).toBeNull();
  });

  it('should be a labelled region showing at most three cards, the soonest to expire first', async () => {
    await loadWith(FIVE);
    renderBanner();
    const banner = screen.getByRole('region', { name: BANNER_REGION_LABEL });
    expect(banner).toHaveAttribute('data-testid', 'approvals-banner');
    expect(within(banner).getByTestId('approvals-banner-heading')).toHaveTextContent(
      '5 agent requests waiting for your approval',
    );
    expect(within(banner).getByTestId('approvals-banner-heading')).toHaveAttribute(
      'aria-live',
      'polite',
    );
    const cards = within(screen.getByTestId('approvals-banner-cards')).getAllByRole('listitem');
    expect(cards.map((card) => card.getAttribute('data-testid'))).toEqual([
      'approval-a',
      'approval-b',
      'approval-c',
    ]);
    expect(screen.getByTestId('approvals-banner-more')).toHaveTextContent(
      'and 2 more waiting — Open Approvals',
    );
    expect(within(banner).getAllByRole('link', { name: 'Open Approvals' })).toHaveLength(2);
  });

  it('should float: a sticky dock at the top of the scroll area, height-capped with its own scroll', async () => {
    await loadWith(FIVE);
    renderBanner();
    const dock = screen.getByTestId('approvals-banner-dock');
    const banner = screen.getByTestId('approvals-banner');
    expect(banner.parentElement).toBe(dock);
    expect(dock).toHaveClass('sticky', 'top-0', 'z-30', 'bg-bg');
    expect(banner).toHaveClass('max-h-[40vh]', 'overflow-y-auto');
    // Collapsed it is still the same sticky dock, now one line.
    fireEvent.click(screen.getByRole('button', { name: 'Collapse' }));
    expect(screen.getByTestId('approvals-banner-dock')).toHaveClass('sticky', 'top-0');
    expect(screen.queryByTestId('approvals-banner-cards')).toBeNull();
  });

  it("should keep every card's Approve and Reject in reach: a sticky, opaque row inside the capped scroll", async () => {
    // Operator 2026-10-28: a long card (a setting's table, the reason) put Approve below the
    // banner's 40vh cap. The row sticks to the bottom of the banner's own scroller.
    await loadWith(FIVE);
    renderBanner();
    const banner = screen.getByTestId('approvals-banner');
    // The cards' own buttons; the banner's "Approve all settings changes" (D79) is not a card's.
    const approves = within(screen.getByTestId('approvals-banner-cards')).getAllByRole('button', {
      name: /^Approve/,
    });
    expect(approves).toHaveLength(3);
    for (const approve of approves) {
      const row = approve.closest('[data-testid^="approval-actions-"]');
      expect(row, 'the Approve button sits in the sticky action row').not.toBeNull();
      expect(row).toHaveClass('sticky', 'bottom-0', 'bg-card');
      expect(row).toContainElement(
        within(row as HTMLElement).getByRole('button', { name: 'Reject' }),
      );
      // The nearest scroller between the row and the banner is the banner itself.
      for (let el = row!.parentElement; el && el !== banner; el = el.parentElement) {
        expect(el.className, el.tagName).not.toMatch(
          /overflow-(auto|scroll|hidden|y-auto|y-scroll)/,
        );
      }
    }
  });

  it('should say "request" for one and show no "more" line', async () => {
    await loadWith([FIVE[0]!]);
    renderBanner();
    expect(screen.getByTestId('approvals-banner-heading')).toHaveTextContent(
      '1 agent request waiting for your approval',
    );
    expect(screen.queryByTestId('approvals-banner-more')).toBeNull();
  });

  it('should collapse to the heading, keep that for the session, and reopen for a new request', async () => {
    await loadWith(FIVE.slice(0, 2));
    renderBanner();
    fireEvent.click(screen.getByRole('button', { name: 'Collapse' }));
    expect(screen.queryByTestId('approvals-banner-cards')).toBeNull();
    expect(screen.getByTestId('approvals-banner-heading')).toBeInTheDocument();
    expect(window.sessionStorage.getItem(BANNER_COLLAPSE_KEY)).not.toBeNull();

    // A remount (another screen) keeps it collapsed.
    cleanup();
    renderBanner();
    expect(screen.queryByTestId('approvals-banner-cards')).toBeNull();
    expect(screen.getByRole('button', { name: 'Expand' })).toBeInTheDocument();

    // A request that was not there when it was collapsed opens it again.
    await loadWith(FIVE.slice(0, 3));
    expect(screen.getByTestId('approvals-banner-cards')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Collapse' })).toBeInTheDocument();
  });

  it('should still work when session storage throws', async () => {
    const spy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    const set = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    try {
      await loadWith(FIVE.slice(0, 1));
      renderBanner();
      fireEvent.click(screen.getByRole('button', { name: 'Collapse' }));
      expect(screen.queryByTestId('approvals-banner-cards')).toBeNull();
    } finally {
      spy.mockRestore();
      set.mockRestore();
    }
  });

  it('should say why when the queue cannot be read, never go quietly dark', async () => {
    vi.stubGlobal('fetch', () =>
      Promise.resolve(
        new Response(
          JSON.stringify({ error: { code: 'db_down', message: 'no database', details: {} } }),
          { status: 503, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );
    await act(async () => {
      await useApprovalsStore.getState().load();
    });
    renderBanner();
    expect(screen.getByTestId('approvals-banner-failure')).toHaveTextContent(
      'Could not read the approval queue: db_down: no database',
    );
  });

  it('should approve from the banner with ONE post and no agent header, then drop the card', async () => {
    await loadWith([FIVE[1]!, FIVE[3]!]);
    renderBanner();
    const button = screen.getByTestId('approve-a');
    fireEvent.click(button);
    fireEvent.click(button);
    await settle();
    expect(posts).toEqual([{ path: '/api/v1/approvals/a/approve', headers: {} }]);
    expect(screen.queryByTestId('approval-a')).toBeNull();
    expect(screen.getByTestId('approvals-banner-heading')).toHaveTextContent(
      '1 agent request waiting',
    );
    // §2.7: focus moves on to the next card.
    expect(screen.getByTestId('approval-b').contains(document.activeElement)).toBe(true);
  });

  it('should send the reason typed in the banner card', async () => {
    await loadWith(FIVE.slice(0, 1));
    renderBanner();
    const id = FIVE[0]!.approval_id;
    const sent: unknown[] = [];
    const original = globalThis.fetch;
    vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
      if ((init?.method ?? 'GET') === 'POST') sent.push(JSON.parse(String(init?.body)));
      return original(input, init);
    });
    fireEvent.click(screen.getByTestId(`reject-${id}`));
    fireEvent.change(screen.getByTestId(`reject-reason-${id}`), { target: { value: 'not now' } });
    fireEvent.click(screen.getByTestId(`confirm-reject-${id}`));
    await settle();
    expect(sent).toEqual([{ reason: 'not now' }]);
  });
});
