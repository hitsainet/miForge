import { expect, test } from '@playwright/test';

import { isAllowedFontHost } from './hosts';

// The allow-list behind shell.spec.ts's no-foreign-request test (CodeQL alerts 3-4).
test.describe('isAllowedFontHost', () => {
  test('admits exactly the two Google Fonts hosts', () => {
    expect(isAllowedFontHost('https://fonts.googleapis.com/css2?family=Inter')).toBe(true);
    expect(isAllowedFontHost('https://fonts.gstatic.com/s/inter/v1/x.woff2')).toBe(true);
  });

  test('refuses look-alikes a suffix or substring check would admit', () => {
    for (const url of [
      'https://evilgoogleapis.com/x',
      'https://fonts.googleapis.com.evil.example/x',
      'https://evil.example/?h=fonts.googleapis.com',
      'https://fonts.gstatic.com@evil.example/x',
      'https://www.googleapis.com/x',
      'http://fonts.googleapis.com/x',
      'not a url',
    ]) {
      expect(isAllowedFontHost(url), url).toBe(false);
    }
  });
});
