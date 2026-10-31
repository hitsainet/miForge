/**
 * The shell, rendered from the registry (FR-009.1, .2) — **M1 and M2's unit half.**
 *
 * Deleting a row from `ROUTES` must break something. `routes.test.ts` asserts the registry;
 * this asserts the *rendering*: every nav row reaches its screen, and the page head's title
 * and subtitle come from the same row rather than from the screen component - so a screen
 * cannot be navigable with a head that disagrees with the table.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DRAFT_ID, answer } from '@/testing/stubBackend';
import { App, APP_VERSION } from './App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { ROUTES, navEntries } from '@/config/routes';
import { MANUAL_URL } from '@/config/manual';
import { INITIAL_APPROVALS_STATE, useApprovalsStore } from '@/stores/approvalsStore';

/**
 * The shell's theme toggle reads `useTheme`, which **throws** outside a provider rather than
 * returning a default - because "a default here would render the wrong theme with no signal".
 * So every render here supplies one, which is also what `main.tsx` does.
 */
const inTheme = (element: React.ReactElement) => (
  <ThemeProvider initialTheme="dark">{element}</ThemeProvider>
);

describe('[App]', () => {
  it('should render the sidebar nav from the registry and from nothing else', () => {
    render(inTheme(<App />));
    const nav = screen.getByRole('navigation', { name: 'Screens' });
    const labels = [...nav.querySelectorAll('button')].map((node) => node.textContent?.trim());
    expect(labels).toEqual(
      navEntries()
        .filter((route) => route.nav?.placement !== 'footer')
        .map((route) => route.nav?.label),
    );
  });

  it('should end the sidebar with Settings beside the Manual link, then the version', async () => {
    // The suite's footer (operator, 2026-10-20), asserted through the real App so removing
    // the registry row, the projection in App.tsx or the footer in Sidebar.tsx is red here.
    render(inTheme(<App />));
    const footer = screen.getByTestId('sidebar-footer');
    const settings = within(footer).getByRole('button', { name: 'Settings' });
    const manual = within(footer).getByRole('link', { name: /^Manual/ });
    // One row: Settings first, the book icon to its right.
    expect(settings.compareDocumentPosition(manual) & Node.DOCUMENT_POSITION_FOLLOWING).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
    expect(within(footer).getByTestId('sidebar-version')).toHaveTextContent(APP_VERSION);
    // The footer sits after the list of screens, and outside it.
    const nav = screen.getByRole('navigation', { name: 'Screens' });
    expect(nav.contains(settings)).toBe(false);
    expect(nav.compareDocumentPosition(footer) & Node.DOCUMENT_POSITION_FOLLOWING).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );

    await userEvent.click(settings);
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Settings');
    expect(settings).toHaveAttribute('aria-current', 'page');
  });

  it('should link the sidebar to the published manual, in a new tab, outside the screen nav', () => {
    // Asserted through the real App, so removing the link from Sidebar.tsx, or Sidebar from the
    // shell, is red here - not only in a component test that renders Sidebar on its own.
    render(inTheme(<App />));
    const links = screen.getAllByRole('link', { name: /^Manual/ });
    expect(links).toHaveLength(1);
    const link = links[0]!;
    expect(link).toHaveAttribute('href', MANUAL_URL);
    expect(link).toHaveAttribute('target', '_blank');
    expect(link.getAttribute('rel')).toContain('noopener');
    // An icon-only link names itself, and says it leaves the app.
    expect(link).toHaveAttribute('aria-label', 'Manual (opens in a new tab)');
    expect(link).toHaveAttribute('title', 'User Manual');
    // Not a screen: the "Screens" nav stays exactly the route registry.
    const nav = screen.getByRole('navigation', { name: 'Screens' });
    expect(nav.contains(link)).toBe(false);
  });

  it('should render each registered screen when it is the initial one', () => {
    // M1/M2: a deleted row makes one of these render the "not in the route registry" notice.
    for (const route of ROUTES) {
      const { unmount } = render(
        inTheme(<App initialScreen={route.id} initialParams={{ runId: 'run-a' }} />),
      );
      expect(
        screen.queryByText(/is not in the route registry/),
        `${route.id} did not resolve to a screen`,
      ).toBeNull();
      unmount();
    }
  });

  it('should keep Runs current while the run view is open, since the view has no nav row', () => {
    render(inTheme(<App initialScreen="run" initialParams={{ runId: 'run-a' }} />));
    const nav = screen.getByRole('navigation', { name: 'Screens' });
    expect(nav.querySelector('[aria-current="page"]')?.textContent?.trim()).toBe('Runs');
  });

  it('should refuse by name for a screen the registry does not hold', () => {
    render(inTheme(<App initialScreen={'nowhere' as never} />));
    expect(screen.getByText(/No screen is registered as "nowhere"/)).toBeInTheDocument();
    // And it names what IS registered, so the reader can see the gap rather than guess.
    expect(screen.getByText(/runs, run, new-run, checkpoints, packs, onboard/)).toBeInTheDocument();
  });

  it('should render exactly one h1, from the screens own page head', () => {
    render(inTheme(<App initialScreen="checkpoints" />));
    expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1);
  });

  it('should show the hosts live readings in the strip, from GET /api/v1/host/resources', async () => {
    const original = globalThis.fetch;
    let reads = 0;
    globalThis.fetch = ((input: RequestInfo | URL) => {
      const url = new URL(typeof input === 'string' ? input : input.toString(), 'http://x');
      if (url.pathname === '/api/v1/host/resources') reads += 1;
      const { status, body } = answer('GET', url.pathname);
      return Promise.resolve(
        new Response(JSON.stringify(body), {
          status,
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    }) as typeof globalThis.fetch;
    try {
      render(inTheme(<App initialScreen="checkpoints" />));
      await waitFor(() => expect(screen.getByTestId('resource-cpu')).toHaveTextContent('CPU: 12%'));
      expect(screen.getByTestId('resource-ram')).toHaveTextContent('RAM: 8.0/64.0 GB');
      expect(screen.getByTestId('resource-gpu')).toHaveTextContent('RTX 3090: 0%');
      expect(screen.getByTestId('resource-gpu')).toHaveAttribute(
        'title',
        'NVIDIA GeForce RTX 3090 · GPU-247aa582-0d1b-e161-8156-983ed1fefc57',
      );
      expect(screen.getByTestId('resource-vram')).toHaveTextContent('VRAM: 0.0/24.0 GB');
      expect(screen.getByTestId('resource-temp')).toHaveTextContent('Temp: 40°C');
      expect(screen.getByTestId('shell-resource-strip').textContent).not.toMatch(/not reported/);
      expect(reads).toBe(1);
    } finally {
      globalThis.fetch = original;
    }
  });

  it('should show no attachment chip anywhere in the shell', () => {
    // FPRD section 3.9: the attachment chip is section 5.10's, and in the default path there
    // is nothing attached and no such status.
    render(inTheme(<App />));
    expect(screen.getByTestId('shell-resource-strip').textContent ?? '').not.toMatch(
      /attach|detach|quiesce/i,
    );
  });

  describe('Help (ENH-004): reached through the real app', () => {
    // A backend at the HTTP boundary, so the onboarding flow has a draft to show.
    let original: typeof globalThis.fetch;
    beforeEach(() => {
      original = globalThis.fetch;
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
      globalThis.fetch = original;
    });

    it('should open Help from the last entry of the main nav', async () => {
      // H1: deleting the registry row removes this button.
      render(inTheme(<App initialScreen="runs" />));
      const nav = screen.getByRole('navigation', { name: 'Screens' });
      const buttons = within(nav).getAllByRole('button');
      expect(buttons.at(-1)).toHaveTextContent('Help');
      await userEvent.click(buttons.at(-1)!);
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Help');
      expect(window.location.hash).toBe('#/help');
      expect(buttons.at(-1)).toHaveAttribute('aria-current', 'page');
    });

    it("should open a screen's own Help topic from its page head, with the topic in the URL", async () => {
      // H2: deleting the HelpLink render in PageHead removes this link from every screen.
      render(inTheme(<App initialScreen="checkpoints" />));
      const head = screen.getByTestId('shell-page-head');
      const link = within(head).getByTestId('manual-link-checkpoints');
      expect(link).toHaveAttribute('target', '_blank');
      await userEvent.click(within(head).getByTestId('help-open-checkpoints'));
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Help');
      expect(screen.getByTestId('help-topic-checkpoints')).toHaveAttribute('data-current', 'true');
      expect(screen.getByTestId('help-topic-runs')).not.toHaveAttribute('data-current');
      expect(window.location.hash).toBe('#/help/checkpoints');
      // The Help screen's own head links no topic.
      expect(
        within(screen.getByTestId('shell-page-head')).queryByTestId('page-head-help'),
      ).toBeNull();
    });

    it('should give every screen in the registry a Help link that names a real topic', async () => {
      for (const route of ROUTES) {
        if (route.id === 'help') continue;
        const { unmount } = render(
          inTheme(<App initialScreen={route.id} initialParams={{ runId: 'run-a' }} />),
        );
        await waitFor(() =>
          expect(screen.getAllByTestId('page-head-help').length, route.id).toBeGreaterThan(0),
        );
        unmount();
      }
    });

    it('should land on the topic a bookmarked URL names', () => {
      // H4: without the optional segment the URL reads as no screen and falls back to Runs.
      window.history.replaceState(null, '', '#/help/approvals');
      render(inTheme(<App />));
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Help');
      expect(screen.getByTestId('help-topic-approvals')).toHaveAttribute('data-current', 'true');
    });

    it("should open the onboarding step's own topic from Help with this step", async () => {
      // H3: deleting the per-step link in OnboardPage removes this button.
      render(
        inTheme(<App initialScreen="onboard" initialParams={{ draftId: DRAFT_ID, step: '6' }} />),
      );
      const panel = await screen.findByTestId('onboard-panel');
      await userEvent.click(within(panel).getByRole('button', { name: /Help with this step/ }));
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Help');
      expect(screen.getByTestId('help-topic-regression')).toHaveAttribute('data-current', 'true');
      expect(window.location.hash).toBe('#/help/regression');
    });
  });

  describe('Approvals (ENH-008 A2, D5, D7): the banner and the badge in the real shell', () => {
    const WAITING = {
      approval_id: 'apr_banner',
      action: 'config_write',
      status: 'pending',
      summary: { key: 'log_level', old: 'INFO', new: 'DEBUG' },
      payload: { key: 'log_level' },
      requested_by: 'agent:forge-mcp',
      decided_by: null,
      reason: null,
      result: null,
      error: null,
      created_at: '2026-10-28T12:00:00+00:00',
      decided_at: null,
      expires_at: '2099-10-29T12:00:00+00:00',
    };
    let original: typeof globalThis.fetch;
    beforeEach(() => {
      original = globalThis.fetch;
      useApprovalsStore.setState({ ...INITIAL_APPROVALS_STATE, inFlight: new Set<string>() });
      globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
        const url = new URL(typeof input === 'string' ? input : input.toString(), 'http://local');
        const method = (init?.method ?? 'GET').toUpperCase();
        const reply =
          url.pathname === '/api/v1/approvals' && method === 'GET'
            ? {
                status: 200,
                body: { approvals: [WAITING], count: 1, pending: 1, limit: 200 },
              }
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
    });

    it('should show the banner on every screen but Approvals, between the strip and the page head', async () => {
      for (const route of ROUTES) {
        const { unmount } = render(
          inTheme(<App initialScreen={route.id} initialParams={{ runId: 'run-a' }} />),
        );
        if (route.id === 'approvals') {
          await screen.findByTestId('approval-apr_banner');
          expect(screen.queryByTestId('approvals-banner'), route.id).toBeNull();
        } else {
          const banner = await screen.findByTestId('approvals-banner');
          expect(within(banner).getByTestId('approval-apr_banner')).toBeInTheDocument();
          expect(screen.getByTestId('shell-content').contains(banner), route.id).toBe(true);
          const head = screen.getAllByTestId('shell-page-head')[0]!;
          expect(
            banner.compareDocumentPosition(head) & Node.DOCUMENT_POSITION_FOLLOWING,
            route.id,
          ).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
        }
        unmount();
      }
    });

    it('should float the banner inside the scrolling main, not above it', async () => {
      render(inTheme(<App initialScreen="runs" />));
      const dock = (await screen.findByTestId('approvals-banner')).parentElement!;
      expect(dock).toHaveAttribute('data-testid', 'approvals-banner-dock');
      expect(dock).toHaveClass('sticky', 'top-0');
      // Sticky binds to the nearest scrolling ancestor: that must be shell-main.
      expect(screen.getByTestId('shell-main')).toContainElement(dock);
      expect(screen.getByTestId('shell-main')).toHaveClass('overflow-auto');
    });

    it('should badge the Approvals entry with "{n} waiting", and open Approvals from the banner', async () => {
      render(inTheme(<App initialScreen="runs" />));
      const badge = await screen.findByTestId('nav-badge-approvals');
      expect(badge).toHaveTextContent('1');
      expect(badge).toHaveAttribute('aria-label', '1 waiting');
      expect(badge).toHaveAttribute('title', '1 waiting');
      // Highlighted, not a grey outline (operator 2026-10-28): filled in the warning tone.
      // A circle for one digit, the same pill as the sibling apps: fixed h-5, at least as wide.
      expect(badge).toHaveClass('bg-amber', 'text-white', 'h-5', 'min-w-5', 'rounded-full');
      expect(badge.className).not.toMatch(/\btext-dim\b/);
      const banner = screen.getByTestId('approvals-banner');
      await userEvent.click(within(banner).getByRole('link', { name: 'Open Approvals' }));
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Approvals');
      expect(screen.queryByTestId('approvals-banner')).toBeNull();
      expect(window.location.hash).toBe('#/approvals');
    });
  });
});
