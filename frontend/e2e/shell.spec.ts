/**
 * The shell and the real nav (FR-009.1, .2, .3, .7; foundation 7.6).
 *
 * **This spec is M1 and M2's end-to-end half.** It navigates through the sidebar rather than
 * by URL, so deleting a row from `config/routes.ts` takes it down as well as `routes.test.ts`
 * - which is the whole reason the route table exists as a registry (ADR-016 #1).
 *
 * It also covers the **run index's reachability**, which has no spec of its own: FR-009.19
 * makes it the navigational minimum, and this spec reaches it through the nav, so its route
 * entry is still protected.
 *
 * **No backend is running under `vite preview`**, which is not a limitation here but the
 * point of two of these tests: every screen must render in full with nothing configured, each
 * unavailable region named rather than blank or broken (AC-009.19, AC20).
 */
import { expect, test, type Page } from '@playwright/test';

import { isAllowedFontHost } from './hosts';

/** The nav labels, in the order `ROUTES` declares them. */
const NAV = ['Runs', 'New run', 'Checkpoints', 'Domain packs', 'Onboard a domain'] as const;

async function token(page: Page, name: string): Promise<string> {
  return page.evaluate(
    (variable) => getComputedStyle(document.documentElement).getPropertyValue(variable).trim(),
    name,
  );
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('shell-root')).toBeVisible();
});

test.describe('theme', () => {
  test('applies the seeded theme to <html> before paint', async ({ page }) => {
    const theme = await page.evaluate(() => document.documentElement.dataset['theme'] ?? 'dark');
    expect(['light', 'dark']).toContain(theme);
    await expect(page.locator('html')).toHaveClass(new RegExp(theme));
  });

  test('resolves every shell token in this theme', async ({ page }) => {
    for (const variable of [
      '--mf-bg',
      '--mf-panel',
      '--mf-card',
      '--mf-line',
      '--mf-text',
      '--mf-dim',
      '--mf-accent',
      '--mf-green',
      '--mf-cyan',
      '--mf-amber',
      '--mf-red',
    ]) {
      expect(await token(page, variable), `${variable} unresolved`).not.toBe('');
    }
  });

  test('keeps the accent identical in both themes', async ({ page }) => {
    expect((await token(page, '--mf-accent')).toLowerCase()).toBe('#f97316');
  });
});

test.describe('sidebar', () => {
  test('is 224 px wide, as ADR-012 and FR-009.2 both quote', async ({ page }) => {
    expect((await page.getByTestId('shell-sidebar').boundingBox())?.width).toBe(224);
  });

  test('renders one nav item per registry row, in the registrys order', async ({ page }) => {
    const nav = page.getByRole('navigation', { name: 'Screens' });
    await expect(nav.getByRole('button')).toHaveCount(NAV.length);
    expect(await nav.getByRole('button').allInnerTexts()).toEqual([...NAV]);
  });

  test('reaches every screen through the real nav', async ({ page }) => {
    // M1/M2: deleting a route row removes a nav item and this fails on the missing button.
    for (const label of NAV) {
      await page
        .getByRole('navigation', { name: 'Screens' })
        .getByRole('button', { name: label })
        .click();
      await expect(
        page.getByTestId('shell-page-head').getByRole('heading', { level: 1 }),
      ).toHaveText(label === 'New run' ? 'New run' : label);
      await expect(page.getByTestId('shell-page-head').locator('p')).not.toBeEmpty();
    }
  });

  test('paints the accent tint on the active nav item only', async ({ page }) => {
    const nav = page.getByRole('navigation', { name: 'Screens' });
    const accent = 'rgb(249, 115, 22)';
    const colours = await nav
      .getByRole('button')
      .evaluateAll((nodes) => nodes.map((node) => getComputedStyle(node).color));
    expect(colours.filter((colour) => colour === accent)).toHaveLength(1);
  });
});

test.describe('page head', () => {
  test('opens every screen with a title and exactly one one-line subtitle', async ({ page }) => {
    for (const label of NAV) {
      await page
        .getByRole('navigation', { name: 'Screens' })
        .getByRole('button', { name: label })
        .click();
      const head = page.getByTestId('shell-page-head');
      await expect(head.getByRole('heading', { level: 1 })).toBeVisible();
      await expect(head.locator('p')).toHaveCount(1);
      await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1);
    }
  });

  test('puts no accent on a title or a subtitle', async ({ page }) => {
    const head = page.getByTestId('shell-page-head');
    const accent = 'rgb(249, 115, 22)';
    expect(
      await head.getByRole('heading', { level: 1 }).evaluate((n) => getComputedStyle(n).color),
    ).not.toBe(accent);
    expect(
      await head
        .locator('p')
        .first()
        .evaluate((n) => getComputedStyle(n).color),
    ).not.toBe(accent);
  });
});

test.describe('resource strip', () => {
  test('shows an unread value as not reported, never as zero', async ({ page }) => {
    await expect(page.getByTestId('resource-gpu')).toContainText('not reported');
  });

  test('shows no attachment chip on the core path', async ({ page }) => {
    // FPRD section 3.9: the chip is section 5.10's, and nothing is attached on this path.
    await expect(page.getByTestId('shell-resource-strip')).not.toContainText(/attach|detach/i);
  });
});

test.describe('standalone with nothing configured', () => {
  test('renders every screen in full, naming each unavailable region', async ({ page }) => {
    // AC-009.19 / AC20's screen-shaped cases. No backend is running: every screen must still
    // render its shell, its head and a NAMED state - never a blank panel and never a crash.
    for (const label of NAV) {
      await page
        .getByRole('navigation', { name: 'Screens' })
        .getByRole('button', { name: label })
        .click();
      await expect(page.getByTestId('shell-content')).toBeVisible();
      await expect(page.getByTestId('shell-page-head')).toBeVisible();
      const text = (await page.getByTestId('shell-content').innerText()).trim();
      expect(text.length, `${label} rendered nothing`).toBeGreaterThan(40);
    }
  });

  test('makes no request to any host that is not this origin', async ({ page }) => {
    // ADR-022's fence, at the transport. There is exactly one host this UI may talk to.
    const foreign: string[] = [];
    page.on('request', (request) => {
      const url = new URL(request.url());
      if (url.origin === new URL(page.url()).origin) return;
      if (isAllowedFontHost(request.url())) return;
      foreign.push(request.url());
    });
    for (const label of NAV) {
      await page
        .getByRole('navigation', { name: 'Screens' })
        .getByRole('button', { name: label })
        .click();
      await page.waitForTimeout(200);
    }
    expect(foreign).toEqual([]);
  });

  test('offers none of the closed list anywhere in the product', async ({ page }) => {
    // FPRD section 3.9, asserted across every screen. An item leaves this list only when a
    // requirement arrives for it.
    const forbidden = [
      /quarantined/i,
      /resident in millm/i,
      /export checkpoint/i,
      /load in millm/i,
      /\bdetach\b/i,
      /try the resident model/i,
      /attach and start/i,
      /install plugin/i,
    ];
    for (const label of NAV) {
      await page
        .getByRole('navigation', { name: 'Screens' })
        .getByRole('button', { name: label })
        .click();
      const text = await page.getByTestId('shell-root').innerText();
      for (const pattern of forbidden) {
        expect(text, `${label} offers ${pattern}`).not.toMatch(pattern);
      }
    }
  });
});

test.describe('keyboard', () => {
  test('paints the focus ring on a keyboard-focused nav item', async ({ page }) => {
    await page.keyboard.press('Tab');
    const shadow = await page.evaluate(() =>
      document.activeElement === null ? '' : getComputedStyle(document.activeElement).boxShadow,
    );
    expect(shadow).not.toBe('none');
    expect(shadow).toContain('rgb(249, 115, 22)');
  });

  test('reaches every nav item and the theme toggle by Tab alone', async ({ page }) => {
    const reached: string[] = [];
    for (let i = 0; i < NAV.length + 1; i += 1) {
      await page.keyboard.press('Tab');
      reached.push(
        await page.evaluate(
          () =>
            document.activeElement?.getAttribute('aria-label') ??
            document.activeElement?.textContent?.trim() ??
            '',
        ),
      );
    }
    for (const label of NAV) expect(reached).toContain(label);
    expect(reached.some((name) => /Switch to (light|dark)/.test(name))).toBe(true);
  });
});

test.describe('tab mark', () => {
  // The only check here that proves the icon is actually SERVED. The Python test
  // (`test_favicon_is_the_logo_tile.py`) proves the file is committed, referenced and
  // inside the build context; it cannot prove Vite copies `public/` to the build root,
  // and this spec runs against `vite build && vite preview` rather than the dev server.
  //
  // ⚠ The assertion is on the CONTENT TYPE, not merely on a 200. Before 2026-10-14 a
  // request for a root-level icon was answered by the SPA fallback with index.html at
  // **200 and `text/html`**, so a status-only assertion would have passed against the
  // exact defect this exists to catch. `vite preview` has no SPA-fallback rule for a
  // path with an extension, so what this pins is the build output; `nginx.conf`'s
  // `try_files $uri =404` is the production half, pinned by the Python test.
  test('serves the favicon the document links, as an image', async ({ page }) => {
    const href = await page.locator('link[rel="icon"]').getAttribute('href');
    expect(href).toBe('/favicon.svg');

    const response = await page.request.get(href as string);
    expect(response.status()).toBe(200);
    expect(response.headers()['content-type']).toContain('image/svg+xml');

    const body = await response.text();
    expect(body).toContain('<svg');
    // The accent tile, so a mark replaced by a placeholder fails here too.
    expect(body).toContain('#F97316');
  });
});
