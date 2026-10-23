/**
 * The route registry, asserted as a registry (FR-009.1; ADR-016 #1).
 *
 * **This file is M1 and M2's target.** Deleting the `run` or `checkpoints` row from `ROUTES`
 * must make it red - and so must the matching Playwright spec, because the specs navigate
 * through the real nav. A test that only asserted "the module imports" would pass against a
 * screen nobody can reach, which is the failure ADR-016 #1 exists to catch and which a sibling
 * shipped sixteen times over.
 */
import { describe, expect, it } from 'vitest';
import { ROUTES, navEntries, routeFor } from './routes';

describe('[route registry]', () => {
  it('should hold F9s four screens, F7s two, F12s two and F14s one, and nothing else', () => {
    // Nine, attributed. Another is a new screen and needs a requirement (FPRD F-3). F12's are
    // Approvals (FR-012.17) and Settings (FR-012.7, .8; operator 2026-10-20); F14's is Help
    // (the suite's ENH-004, operator 2026-10-20). The mockup's Selectors screen has none and
    // is deliberately absent.
    expect(ROUTES.map((route) => route.id)).toEqual([
      'runs',
      'run',
      'new-run',
      'checkpoints',
      'packs',
      'onboard',
      'approvals',
      'help',
      'settings',
    ]);
    expect(ROUTES.filter((route) => route.owner === 'F14').map((route) => route.id)).toEqual([
      'help',
    ]);
    expect(ROUTES.filter((route) => route.owner === 'F9').map((route) => route.id)).toEqual([
      'runs',
      'run',
      'new-run',
      'checkpoints',
    ]);
    expect(ROUTES.filter((route) => route.owner === 'F7').map((route) => route.id)).toEqual([
      'packs',
      'onboard',
    ]);
    expect(ROUTES.filter((route) => route.owner === 'F12').map((route) => route.id)).toEqual([
      'approvals',
      'settings',
    ]);
  });

  it('should make every screen resolvable from the registry', () => {
    // M1/M2: deleting a row makes one of these resolve to undefined.
    for (const id of [
      'runs',
      'run',
      'new-run',
      'checkpoints',
      'packs',
      'onboard',
      'approvals',
      'help',
      'settings',
    ] as const) {
      const route = routeFor(id);
      expect(route, `no route is registered for ${id}`).toBeDefined();
      expect(typeof route?.element, `${id} has no element`).toBe('function');
    }
  });

  it('should give every screen a non-empty one-line subtitle', () => {
    // ADR-012 requires every page to open with one. The field is in the table precisely so
    // this assertion is possible - a subtitle in a page component cannot be checked here.
    for (const route of ROUTES) {
      expect(route.subtitle.length, `${route.id} has no subtitle`).toBeGreaterThan(0);
      expect(route.subtitle, `${route.id}'s subtitle spans lines`).not.toContain('\n');
    }
  });

  it('should give every screen a distinct path and a distinct id', () => {
    expect(new Set(ROUTES.map((route) => route.id)).size).toBe(ROUTES.length);
    expect(new Set(ROUTES.map((route) => route.path)).size).toBe(ROUTES.length);
  });

  it('should keep the run view out of the nav because it needs a runId', () => {
    expect(routeFor('run')?.nav).toBeNull();
    expect(navEntries().map((route) => route.id)).not.toContain('run');
  });

  it('should put Approvals in the nav, where the pending count is shown (FR-012.17)', () => {
    expect(routeFor('approvals')?.nav?.label).toBe('Approvals');
    expect(navEntries().map((route) => route.id)).toContain('approvals');
  });

  it('should put Help LAST in the list of screens, with a linkable topic (ENH-004 D1, D2)', () => {
    const screens = navEntries().filter((route) => route.nav?.placement !== 'footer');
    expect(screens.at(-1)?.id).toBe('help');
    expect(routeFor('help')?.nav?.label).toBe('Help');
    expect(routeFor('help')?.path).toBe('/help/:topic?');
  });

  it('should put the New run form IN the nav, since it is the primary action', () => {
    expect(routeFor('new-run')?.nav).not.toBeNull();
    expect(navEntries().map((route) => route.id)).toContain('new-run');
  });

  it('should give every nav entry a label and an icon', () => {
    for (const route of navEntries()) {
      expect(route.nav?.label.length, `${route.id} has no nav label`).toBeGreaterThan(0);
      expect(typeof route.nav?.icon, `${route.id} has no nav icon`).not.toBe('undefined');
    }
  });

  it('should put Settings in the footer and every other nav entry in the list of screens', () => {
    // The suite's sidebar footer (operator, 2026-10-20): Settings, the Manual, the version.
    expect(routeFor('settings')?.nav).toMatchObject({ label: 'Settings', placement: 'footer' });
    expect(
      navEntries()
        .filter((route) => route.nav?.placement === 'footer')
        .map((route) => route.id),
    ).toEqual(['settings']);
  });

  it('should register no screen the mockup has and no requirement defines', () => {
    // FPRD section 3.9's closed list, at the registry level. Selectors has no requirement
    // (F3 assigns it in a feature document). Settings left this list when F12's settings
    // surface (FR-012.7, .8) and the operator's footer request (2026-10-20) arrived.
    const ids = ROUTES.map((route) => route.id as string);
    for (const absent of ['selectors', 'install-plugin']) {
      expect(ids, `${absent} is on FPRD 3.9's closed list`).not.toContain(absent);
    }
  });
});
