/**
 * Wait until no web font is still loading.
 *
 * The UI loads Inter and JetBrains Mono from Google Fonts with `display=swap`, so text first
 * paints in a fallback face and re-lays out when the real face lands. `page.goto` returns while
 * the faces are still `loading`. A test that reads geometry before then and again after can see
 * the swap instead of the behaviour it tests. On the public mirror's runner (run 45878920839,
 * 2026-11-26) the faces landed between the two readings:
 * - the sidebar footer grew from 85 to 85.390625 px;
 * - scroll anchoring moved the WSP transcript from 3696 to 3717.
 * Locally the faces land within about 100 ms, before the first reading, so the race never showed.
 *
 * Call this once the content under test is on screen. A face is requested when text that uses
 * it is first laid out, so waiting earlier could return before the request is made. A face that
 * fails to load (no network) also settles, and the fallback then stays.
 */
import type { Page } from '@playwright/test';

export async function fontsSettled(page: Page): Promise<void> {
  // Polled, so a face requested after an earlier one landed is waited for too.
  await page.waitForFunction(() => document.fonts.status === 'loaded');
}
