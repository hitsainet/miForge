/**
 * Every screen in `config/routes.ts`, as the hash the phone-width overflow audit opens
 * (`e2e/phone-overflow.spec.ts`). Pure data, so `phonePages.test.ts` (vitest) can pin it against the
 * live `ROUTES`: a screen added to the registry and not here fails that test, rather than
 * silently going unaudited.
 */

/** Appended to every identifier the audit serves, like the operator's 2026-11-26 tree. */
export const LONG =
  '@0f604ada3f766f9f257460c4c9f0e1d2c3b4a5968778695a4b3c2d1e0f:smoke-20261022-0357';

export const SEEDED_RUN = 'run-seeded-1';
export const WSP_RUN = 'wsp_0123456789ab';

/** Screen id -> the hash it is opened at. Parameters name runs the audit serves. */
export const PHONE_PAGES: Readonly<Record<string, string>> = {
  runs: '#/runs',
  run: `#/runs/${SEEDED_RUN}${LONG}`,
  'new-run': '#/runs/new',
  checkpoints: '#/checkpoints',
  packs: '#/packs',
  onboard: '#/packs/onboard',
  approvals: '#/approvals',
  'wsp-runs': '#/wsp/runs',
  'wsp-run': `#/wsp/runs/${WSP_RUN}${LONG}`,
  'wsp-endpoints': '#/wsp/endpoints',
  help: '#/help',
  settings: '#/settings',
};
