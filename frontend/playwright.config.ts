/**
 * Playwright (foundation task 7.6; ADR-016, ADR-023).
 *
 * **One project per theme**, rather than one spec that toggles. A theme is applied to
 * `<html>` before the first paint (`main.tsx`), so seeding the stored preference in
 * `storageState` exercises the same path a returning operator takes - and a failure then
 * names the theme it failed in, which a toggling spec does not.
 *
 * Chromium only. PPRD 7 and the mockup's 6 both say desktop, single operator; a
 * browser matrix here would be work nobody asked for.
 */
import { defineConfig, devices } from '@playwright/test';

/** Overridable so parallel worktrees do not reuse one another's preview server. */
const PORT = Number(process.env['PLAYWRIGHT_PORT'] ?? 4173);
const THEME_KEY = 'miforge.theme';

/** Seed the stored theme so the first paint is the one under test. */
const storageFor = (theme: 'light' | 'dark') => ({
  cookies: [],
  origins: [
    {
      origin: `http://localhost:${PORT}`,
      localStorage: [{ name: THEME_KEY, value: theme }],
    },
  ],
});

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? 'list' : 'html',
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'dark',
      use: { ...devices['Desktop Chrome'], storageState: storageFor('dark') },
    },
    {
      name: 'light',
      use: { ...devices['Desktop Chrome'], storageState: storageFor('light') },
    },
  ],
  webServer: {
    // `preview` over `dev`: the smoke spec should run against the built bundle, because
    // a Tailwind class that is purged in production and present in dev is exactly the
    // failure a dev-server smoke test cannot see.
    command: `npm run build && npx vite preview --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
