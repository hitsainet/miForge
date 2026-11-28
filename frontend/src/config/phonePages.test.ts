/**
 * The phone-width overflow audit (`e2e/phone-overflow.spec.ts`) opens every screen in
 * `phonePages.ts`. This pins that list to the live registry: a screen added to `ROUTES` and
 * not to the audit goes red here, instead of shipping unaudited at 390 px.
 */
import { describe, expect, it } from 'vitest';
import { ROUTES } from './routes';
import { PHONE_PAGES } from './phonePages';

describe('[phone overflow audit]', () => {
  it('should open every screen in ROUTES, and no screen that is not in it', () => {
    expect(Object.keys(PHONE_PAGES).sort()).toEqual(ROUTES.map((route) => route.id).sort());
  });

  it("should open each screen at a hash its route's path matches", () => {
    for (const route of ROUTES) {
      const hash = PHONE_PAGES[route.id];
      expect(hash, route.id).toBeDefined();
      const pattern = new RegExp(
        `^#${route.path.replace(/\/:[A-Za-z]+\?/g, '(?:/[^/]+)?').replace(/:[A-Za-z]+/g, '[^/]+')}$`,
      );
      expect(hash, route.id).toMatch(pattern);
    }
  });
});
