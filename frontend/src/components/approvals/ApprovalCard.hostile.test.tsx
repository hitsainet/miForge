/**
 * Hostile text renders as text (ENH-008 §2.7, A6).
 *
 * Every field of a request - action, requested_by, summary keys and values, payload, reason,
 * error, result - carries markup. The banner, the waiting card and the History row are rendered
 * from it, and the test counts elements: an `img` or `script` anywhere means some field reached
 * the DOM as markup. The markup must still be readable, as text.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen } from '@testing-library/react';
import { NavigationProvider } from '@/contexts/NavigationContext';
import { INITIAL_APPROVALS_STATE, useApprovalsStore } from '@/stores/approvalsStore';
import type { Approval } from '@/api/approvals';
import { ApprovalsBanner } from './ApprovalsBanner';
import { ApprovalHistory } from './ApprovalHistory';
import { ApprovalCard } from './ApprovalCard';

const IMG = '<img src=x onerror=alert(1)>';
const SCRIPT = '<script>alert(2)</script>';
const BOTH = `${IMG}${SCRIPT}`;

function hostile(status: Approval['status']): Approval {
  return {
    approval_id: `apr_${status}`,
    action: BOTH as Approval['action'],
    status,
    summary: { [IMG]: SCRIPT, [SCRIPT]: { [IMG]: [IMG, SCRIPT] }, target: BOTH },
    payload: { [IMG]: BOTH, nested: { [SCRIPT]: IMG } },
    requested_by: `agent:${BOTH}`,
    decided_by: status === 'pending' ? null : BOTH,
    reason: status === 'pending' ? null : BOTH,
    result: status === 'executed' ? { run_id: BOTH } : null,
    error: status === 'failed' ? { code: BOTH, message: BOTH } : null,
    created_at: '2026-10-28T12:00:00+00:00',
    decided_at: status === 'pending' ? null : '2026-10-28T12:05:00+00:00',
    expires_at: '2026-10-29T12:00:00+00:00',
  };
}

beforeEach(() => {
  useApprovalsStore.setState({ ...INITIAL_APPROVALS_STATE, inFlight: new Set<string>() });
  vi.stubGlobal('fetch', () =>
    Promise.resolve(
      new Response(JSON.stringify({ approvals: [], count: 0, pending: 0, limit: 200 })),
    ),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function assertInert(container: HTMLElement): void {
  expect(container.querySelectorAll('img')).toHaveLength(0);
  expect(container.querySelectorAll('script')).toHaveLength(0);
  expect(container.textContent).toContain(IMG);
  expect(container.textContent).toContain(SCRIPT);
}

describe('[approvals: hostile text]', () => {
  it('should render every field of a waiting card as text', () => {
    const { container } = render(
      <NavigationProvider>
        <ul>
          <ApprovalCard approval={hostile('pending')} />
        </ul>
      </NavigationProvider>,
    );
    assertInert(container);
  });

  it('should render a refused decision message as text', () => {
    useApprovalsStore.setState({
      decisionFailures: {
        apr_pending: { kind: 'refusal', status: 409, code: BOTH, message: BOTH, details: {} },
      },
    });
    const { container } = render(
      <NavigationProvider>
        <ul>
          <ApprovalCard approval={hostile('pending')} />
        </ul>
      </NavigationProvider>,
    );
    expect(screen.getByTestId('approval-error-apr_pending')).toHaveTextContent(IMG);
    assertInert(container);
  });

  it('should render the banner from a hostile queue as text', () => {
    act(() => useApprovalsStore.setState({ waiting: [hostile('pending')] }));
    const { container } = render(
      <NavigationProvider>
        <ApprovalsBanner />
      </NavigationProvider>,
    );
    assertInert(container);
  });

  it('should render every History column as text, for every outcome', () => {
    const rows = (['executed', 'failed', 'rejected', 'expired', 'executing'] as const).map(hostile);
    const { container } = render(
      <NavigationProvider>
        <ApprovalHistory rows={rows} />
      </NavigationProvider>,
    );
    assertInert(container);
  });
});
