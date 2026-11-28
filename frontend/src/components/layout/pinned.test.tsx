/**
 * The pinned-stack registry (Tab-focus fix, 2026-11-26): every element pinned at the top of
 * `shell-main` counts in its scroll padding, so a control reached by Tab lands below them all.
 *
 * jsdom has no layout, so the wiring tests give each pinned element a height by its test id; a
 * pinned element that stops registering drops its height from the total, and goes red here. The
 * real scroll is proven in `e2e/tab-focus-under-pinned.spec.ts`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { App } from '@/App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { WSP_RUN_ID, answer } from '@/testing/stubBackend';
import { INITIAL_APPROVALS_STATE, useApprovalsStore } from '@/stores/approvalsStore';
import {
  SHELL_PINNED_BOTTOM_HEIGHT_VAR,
  SHELL_PINNED_HEIGHT_VAR,
  trackPinned,
  trackPinnedBottom,
} from './pinned';

const total = (): string =>
  document.documentElement.style.getPropertyValue(SHELL_PINNED_HEIGHT_VAR);
const tallestBottom = (): string =>
  document.documentElement.style.getPropertyValue(SHELL_PINNED_BOTTOM_HEIGHT_VAR);

function sized(height: number): HTMLDivElement {
  const element = document.createElement('div');
  element.getBoundingClientRect = () => ({ height }) as DOMRect;
  return element;
}

describe('trackPinned', () => {
  it('should publish the sum of every registered height, and remove it with the last', () => {
    expect(total()).toBe('');
    const untrackDock = trackPinned(sized(40));
    expect(total()).toBe('40px');
    const heard: number[] = [];
    const untrackHeader = trackPinned(sized(100), (height) => heard.push(height));
    expect(total()).toBe('140px');
    expect(heard).toEqual([100]);
    untrackDock();
    expect(total()).toBe('100px');
    untrackHeader();
    expect(total()).toBe('');
  });
});

describe('trackPinnedBottom', () => {
  it('should publish the tallest registered height, not the sum, and remove it with the last', () => {
    expect(tallestBottom()).toBe('');
    const untrackShort = trackPinnedBottom(sized(40));
    const untrackTall = trackPinnedBottom(sized(60));
    expect(tallestBottom()).toBe('60px');
    // The top stack is untouched: a decision row is not part of it.
    expect(total()).toBe('');
    untrackTall();
    expect(tallestBottom()).toBe('40px');
    untrackShort();
    expect(tallestBottom()).toBe('');
  });
});

/** Heights by test id, for the elements that pin themselves. Everything else measures 0. */
const HEIGHTS: Readonly<Record<string, number>> = {
  'approvals-banner-dock': 40,
  'wsp-runs-pinned': 100,
  'wsp-run-header': 70,
  'approval-actions-apr_pinned': 50,
};

const WAITING = {
  approval_id: 'apr_pinned',
  action: 'config_write',
  status: 'pending',
  summary: { key: 'log_level', old: 'INFO', new: 'DEBUG' },
  payload: { key: 'log_level' },
  requested_by: 'agent:forge-mcp',
  decided_by: null,
  reason: null,
  result: null,
  error: null,
  created_at: '2026-11-26T12:00:00+00:00',
  decided_at: null,
  expires_at: '2099-11-27T12:00:00+00:00',
};

describe('[shell] every pinned element counts in the scroll padding', () => {
  let original: typeof globalThis.fetch;
  let waiting = false;

  beforeEach(() => {
    waiting = false;
    original = globalThis.fetch;
    useApprovalsStore.setState({ ...INITIAL_APPROVALS_STATE, inFlight: new Set<string>() });
    window.history.replaceState(null, '', '/');
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (
      this: HTMLElement,
    ) {
      return { height: HEIGHTS[this.dataset['testid'] ?? ''] ?? 0 } as DOMRect;
    });
    globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(typeof input === 'string' ? input : input.toString(), 'http://local');
      const method = (init?.method ?? 'GET').toUpperCase();
      const reply =
        url.pathname === '/api/v1/approvals' && method === 'GET' && waiting
          ? { status: 200, body: { approvals: [WAITING], count: 1, pending: 1, limit: 200 } }
          : answer(method, url.pathname);
      return Promise.resolve(
        new Response(JSON.stringify(reply.body), {
          status: reply.status,
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    }) as typeof globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = original;
    vi.restoreAllMocks();
    window.history.replaceState(null, '', '/');
  });

  const inTheme = (element: React.ReactElement) => (
    <ThemeProvider initialTheme="dark">{element}</ThemeProvider>
  );

  it('should give shell-main the pinned total, plus 0.5rem, as its scroll padding', async () => {
    render(inTheme(<App initialScreen="runs" />));
    expect(await screen.findByTestId('shell-main')).toHaveClass(
      'scroll-pt-[calc(var(--shell-pinned-height,0px)_+_0.5rem)]',
    );
  });

  it('should give shell-main the tallest bottom-pinned row, plus 0.5rem, as its bottom padding', async () => {
    render(inTheme(<App initialScreen="runs" />));
    expect(await screen.findByTestId('shell-main')).toHaveClass(
      'scroll-pb-[calc(var(--shell-pinned-bottom-height,0px)_+_0.5rem)]',
    );
  });

  it("should count a waiting card's decision row as pinned at the bottom, in the dock and on Approvals", async () => {
    waiting = true;
    const docked = render(inTheme(<App initialScreen="runs" />));
    await screen.findByTestId('approval-actions-apr_pinned');
    expect(tallestBottom()).toBe('50px');
    // The dock's region scrolls inside: its padding is the row's plus the region's own padding.
    expect(screen.getByTestId('approvals-banner')).toHaveClass(
      'scroll-pb-[calc(var(--shell-pinned-bottom-height,0px)_+_var(--spacing-card)_+_0.5rem)]',
    );
    docked.unmount();
    expect(tallestBottom()).toBe('');

    const page = render(inTheme(<App initialScreen="approvals" />));
    await screen.findByTestId('approval-actions-apr_pinned');
    expect(tallestBottom()).toBe('50px');
    page.unmount();
    expect(tallestBottom()).toBe('');
  });

  it('should count the approvals dock on a screen with nothing else pinned', async () => {
    waiting = true;
    const { unmount } = render(inTheme(<App initialScreen="runs" />));
    await screen.findByTestId('approvals-banner-dock');
    expect(total()).toBe('40px');
    // The dock still publishes its own height, for the blocks that sit below it.
    expect(document.documentElement.style.getPropertyValue('--approvals-dock-height')).toBe('40px');
    unmount();
    expect(total()).toBe('');
  });

  it("should count the WSP runs list's pinned block, alone and under the dock", async () => {
    const alone = render(inTheme(<App initialScreen="wsp-runs" />));
    await screen.findByTestId('wsp-runs-pinned');
    expect(total()).toBe('100px');
    alone.unmount();
    expect(total()).toBe('');

    waiting = true;
    const docked = render(inTheme(<App initialScreen="wsp-runs" />));
    await screen.findByTestId('approvals-banner-dock');
    expect(total()).toBe('140px');
    docked.unmount();
    expect(total()).toBe('');
  });

  it("should count the WSP run page's floating header, alone and under the dock", async () => {
    window.history.replaceState(null, '', `/#/wsp/runs/${WSP_RUN_ID}`);
    const alone = render(inTheme(<App />));
    await screen.findByTestId('wsp-run-header');
    expect(total()).toBe('70px');
    alone.unmount();
    expect(total()).toBe('');

    waiting = true;
    const docked = render(inTheme(<App />));
    await screen.findByTestId('approvals-banner-dock');
    await screen.findByTestId('wsp-run-header');
    expect(total()).toBe('110px');
    docked.unmount();
    expect(total()).toBe('');
  });
});
