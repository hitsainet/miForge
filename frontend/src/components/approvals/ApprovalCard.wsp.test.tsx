/**
 * The two WSP approval actions have their own words (feature 016, FR-016.37; P62, R2A-12).
 *
 * Without a row the card would show the raw action name, and the approve button would read
 * "Approve" alone, which ADR-012 forbids.
 */
import { describe, expect, it } from 'vitest';
import { ACTION_LABEL, APPROVE_LABEL, actionLabel } from './ApprovalCard';

const WSP_ACTIONS = ['wsp_session', 'wsp_session_external'] as const;

describe('WSP approval labels', () => {
  it.each(WSP_ACTIONS)('%s has an action label and a named approve button', (action) => {
    expect(ACTION_LABEL[action]).toMatch(/weight-space/i);
    expect(actionLabel(action)).toBe(ACTION_LABEL[action]);
    expect(APPROVE_LABEL[action]).toMatch(/^Approve .+/);
  });

  it('the external action says the session leaves through an external endpoint', () => {
    expect(ACTION_LABEL.wsp_session_external).toMatch(/external/i);
    expect(ACTION_LABEL.wsp_session).not.toMatch(/external/i);
  });
});
