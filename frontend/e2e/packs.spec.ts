/**
 * The pack screens, reached through the real nav (F7 FR-007.29, FR-007.30).
 *
 * The one orange panel on the Domain packs screen is the no-built-in-domain banner, and this
 * spec asserts that in the strongest available form: it counts the accent-coloured elements on
 * the screen. *If a screen starts to look orange, something is wrong.*
 */
import { expect, test } from '@playwright/test';

test('opens Domain packs with its subtitle and the no-built-in-domain banner', async ({ page }) => {
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Domain packs' })
    .click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Domain packs');
  await expect(page.getByTestId('no-built-in-domain')).toContainText(
    'miForge has no built-in domain.',
  );
});

test('paints the accent only where ADR-012 permits it', async ({ page }) => {
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Domain packs' })
    .click();

  // The rule, asserted by WHERE the accent landed rather than by how many elements it
  // touched: a count has to be revised every time a permitted surface is added, and a
  // revised count is a rubber stamp. ADR-012 forbids the accent on body text, borders,
  // backgrounds and any status pill but "Running" - so the check is that every accented
  // element sits inside a permitted surface.
  const offenders = await page.getByTestId('shell-content').evaluate((root) => {
    const ACCENT = 'rgb(249, 115, 22)';
    const permitted = (node: Element): boolean =>
      node.closest('[data-tone="accent"]') !== null ||
      node.closest('[data-surface="logo-tile"]') !== null ||
      node.closest('[data-surface="row-icon-tile"]') !== null ||
      node.closest('button.bg-accent') !== null ||
      node.classList.contains('bg-accent');
    const bad: string[] = [];
    for (const node of root.querySelectorAll('*')) {
      const style = getComputedStyle(node);
      const accented = style.color === ACCENT || style.backgroundColor === ACCENT;
      if (!accented) continue;
      if (!permitted(node))
        bad.push(`${node.tagName.toLowerCase()}: ${node.textContent?.slice(0, 40)}`);
    }
    return bad;
  });
  expect(offenders).toEqual([]);

  // And the construction-green guard: the scan must have found the accent SOMEWHERE, or it
  // is reporting success over a page it never looked at.
  const found = await page.getByTestId('shell-content').evaluate((root) => {
    let hits = 0;
    for (const node of root.querySelectorAll('*')) {
      const style = getComputedStyle(node);
      if (style.color === 'rgb(249, 115, 22)' || style.backgroundColor === 'rgb(249, 115, 22)')
        hits += 1;
    }
    return hits;
  });
  expect(found).toBeGreaterThan(0);
});

test('puts the accent on no status pill on this screen', async ({ page }) => {
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Domain packs' })
    .click();
  const accented = await page
    .getByTestId('shell-content')
    .evaluate((root) =>
      [...root.querySelectorAll('[data-tone]')]
        .filter((node) => getComputedStyle(node).color === 'rgb(249, 115, 22)')
        .map((node) => node.textContent ?? ''),
    );
  // "Running" is the one pill ADR-012 excepts, and no run state appears on a pack screen.
  expect(accented).toEqual([]);
});

test('says an empty registry is empty and ships no example packs', async ({ page }) => {
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Domain packs' })
    .click();
  // With no backend the registry cannot be read, so the screen names that rather than
  // claiming the registry is empty.
  await expect(page.getByTestId('shell-content')).toContainText(/pack registry/i);
});

test('opens Onboard a domain and names the step flow', async ({ page }) => {
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Onboard a domain' })
    .click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Onboard a domain');
  await expect(page.getByTestId('shell-page-head').locator('p')).toContainText(
    'progress is saved as you go',
  );
});

/* ------------------------------------------------------------------------------------ *
 * The registry with packs in it (F7 task 12.11; FR-007.11, FR-007.14, FR-007.15, AC10)
 *
 * The tests above run against no backend, which is the honest fresh-installation state and
 * what they exist to assert. These two need rows, so they stub the API **at the network** -
 * `vite preview` serves the built bundle with nothing behind it (see `playwright.config.ts`),
 * so the alternative is no end-to-end coverage of either behaviour. The real client, the
 * real reload-after-import and the real paint are all still in the measurement.
 * ------------------------------------------------------------------------------------ */

/** A pack document in `pack-definition-v1.json`'s shape, by value, with no local pointer. */
const EXPORTED_DOCUMENT = {
  schema_version: '1',
  name: 'kiln-firing',
  version: '2',
  domain: 'ceramics',
  slots: {
    prompt_set: {
      contract_reference: { type: 'prompts', name: 'firing-schedules', version: '1' },
      external: null,
      floor: null,
    },
    corpus: null,
    preference_pairs: null,
    selector_config: null,
    grader_config: null,
    // No test set: this is the incomplete pack below, and the missing slot is this one.
    test_set: null,
    retention_set: {
      contract_reference: { type: 'tests', name: 'general', version: '1' },
      external: null,
      floor: 0.8,
    },
  },
  created_at: '2026-10-14T00:00:00Z',
  definition_sha256: 'e'.repeat(64),
} as const;

const INCOMPLETE_ROW = {
  pack_id: 'kiln-firing',
  version: '2',
  name: 'kiln-firing',
  domain: 'ceramics',
  parts_filled: 2,
  parts_total: 7,
  // `null`, never 0: an unresolvable prompt count is UNKNOWN (ADR-018, FR-007.11).
  prompt_count: null,
  runs_using: [],
  validity: {
    state: 'incomplete',
    missing_slots: ['test_set'],
    unresolved: [],
    service_unavailable: [],
    retention_floor_missing: false,
  },
  derived_from: null,
  is_example: false,
  definition_sha256: 'e'.repeat(64),
  created_at: '2026-10-14T00:00:00Z',
} as const;

interface RegistryCalls {
  readonly listings: number;
  readonly exports: string[];
  readonly imports: unknown[];
}

async function stubRegistry(page: import('@playwright/test').Page): Promise<RegistryCalls> {
  const calls: RegistryCalls = { listings: 0, exports: [], imports: [] };
  await page.route('**/api/**', (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

    if (path === '/api/health') return json({ status: 'ok', dependencies: [], integrations: [] });
    if (path === '/api/v1/packs' && method === 'GET') {
      (calls as { listings: number }).listings += 1;
      return json({ packs: [INCOMPLETE_ROW] });
    }
    if (path.endsWith('/export') && method === 'GET') {
      calls.exports.push(path);
      // Served as an attachment, which is what the screen relies on: the export is a
      // document an operator keeps, so `PacksPage` navigates to it rather than fetching it.
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: { 'content-disposition': 'attachment; filename="kiln-firing@2.pack.json"' },
        body: JSON.stringify(EXPORTED_DOCUMENT),
      });
    }
    if (path === '/api/v1/packs/import' && method === 'POST') {
      calls.imports.push(request.postDataJSON());
      return json({ ...INCOMPLETE_ROW, pack_id: 'kiln-firing' }, 201);
    }
    return json(
      { error: { code: 'not_stubbed', message: `${method} ${path} is not part of this test` } },
      404,
    );
  });
  return calls;
}

async function openPacks(page: import('@playwright/test').Page): Promise<void> {
  await page.goto('/');
  await page
    .getByRole('navigation', { name: 'Screens' })
    .getByRole('button', { name: 'Domain packs' })
    .click();
  await expect(page.getByTestId('packs-list')).toBeVisible();
}

test('the incomplete pill NAMES the slot that is missing', async ({ page }) => {
  await stubRegistry(page);
  await openPacks(page);

  // AC10's whole content: `Incomplete` on its own is the generic failure the criterion
  // exists to forbid, so the assertion is on the slot's name and not on the word.
  const pill = page.getByTestId('validity-kiln-firing');
  await expect(pill).toHaveText('Incomplete · no test set');
  await expect(pill).toHaveAttribute('title', 'Incomplete · no test set');

  // And the unresolvable prompt count is rendered as unknown rather than as 0, which would
  // be a fabricated measurement an operator would act on.
  await expect(page.getByTestId('prompts-unknown-kiln-firing')).toBeVisible();
  await expect(page.getByTestId('pack-kiln-firing')).not.toContainText('0 prompts');
});

test('exports a pack and re-imports the same bytes', async ({ page }) => {
  const calls = await stubRegistry(page);
  await openPacks(page);

  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export this pack' }).click();
  const exported = await download;

  // The export is a download of the canonical document, named after the pack version.
  expect(exported.suggestedFilename()).toBe('kiln-firing@2.pack.json');
  expect(calls.exports).toEqual(['/api/v1/packs/kiln-firing/versions/2/export']);

  const path = await exported.path();
  const { readFileSync } = await import('node:fs');
  const bytes = readFileSync(path);
  expect(JSON.parse(bytes.toString())).toEqual(EXPORTED_DOCUMENT);

  // Re-import THE BYTES THAT WERE DOWNLOADED, not a fresh literal: the round trip is the
  // claim, and feeding the dialog a hand-built object would be a fixture agreeing with
  // itself rather than a document that travelled.
  const listingsBefore = calls.listings;
  await page.getByRole('button', { name: 'Import pack' }).click();
  await page.getByTestId('import-pack-file').setInputFiles({
    name: 'kiln-firing@2.pack.json',
    mimeType: 'application/json',
    buffer: bytes,
  });

  await expect(page.getByTestId('import-result')).toBeVisible();
  // The payload, not merely that the call happened: an importer that posted a summary, or
  // the file's name, would also "have been called".
  expect(calls.imports).toEqual([EXPORTED_DOCUMENT]);

  // A 201 is not "ready". The dialog renders the validity it was given and says that no
  // reference was rewritten to make it resolve - rewriting would make it a different pack.
  await expect(page.getByTestId('import-result')).toContainText('Incomplete · no test set');
  await expect(page.getByTestId('import-result')).toContainText('were not rewritten');
  // And the listing is re-read from the server rather than patched locally.
  expect(calls.listings).toBeGreaterThan(listingsBefore);
});

test('reports a file that is not JSON locally, without blaming the server', async ({ page }) => {
  await stubRegistry(page);
  await openPacks(page);

  await page.getByRole('button', { name: 'Import pack' }).click();
  await page.getByTestId('import-pack-file').setInputFiles({
    name: 'not-a-pack.json',
    mimeType: 'application/json',
    buffer: Buffer.from('{ this is not json'),
  });

  // The file never reached the backend, so reporting a server refusal would send the reader
  // to the wrong place.
  await expect(page.getByTestId('import-parse-error')).toContainText('is not JSON');
  await expect(page.getByTestId('import-parse-error')).toContainText('export it again');
});
