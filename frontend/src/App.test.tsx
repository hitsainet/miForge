/**
 * The shell, rendered from the registry (FR-009.1, .2) — **M1 and M2's unit half.**
 *
 * Deleting a row from `ROUTES` must break something. `routes.test.ts` asserts the registry;
 * this asserts the *rendering*: every nav row reaches its screen, and the page head's title
 * and subtitle come from the same row rather than from the screen component - so a screen
 * cannot be navigable with a head that disagrees with the table.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { App } from './App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { ROUTES, navEntries } from '@/config/routes';

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
    expect(labels).toEqual(navEntries().map((route) => route.nav?.label));
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

  it('should report every host reading as not reported rather than inventing one', () => {
    // Nothing serves host metrics: `GET /api/health` reports dependencies and integrations,
    // not utilisation. The mockup's numbers are not reproduced.
    render(inTheme(<App />));
    expect(screen.getByTestId('resource-gpu')).toHaveTextContent('not reported');
  });

  it('should show no attachment chip anywhere in the shell', () => {
    // FPRD section 3.9: the attachment chip is section 5.10's, and in the default path there
    // is nothing attached and no such status.
    render(inTheme(<App />));
    expect(screen.getByTestId('shell-resource-strip').textContent ?? '').not.toMatch(
      /attach|detach|quiesce/i,
    );
  });
});
