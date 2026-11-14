/** The hash codec (task 13.1; AC-009.31, .33). */
import { describe, expect, it } from 'vitest';
import { fromHash, toHash } from './urlState';
import { ROUTES } from './routes';

describe('[urlState]', () => {
  it('should round-trip every registered screen', () => {
    for (const route of ROUTES) {
      const params = route.path.includes(':runId') ? { runId: 'run a/1' } : {};
      const hash = toHash(route.id, params, ROUTES);
      expect(hash, route.id).not.toBeNull();
      expect(fromHash(hash ?? '', ROUTES), route.id).toEqual({ screen: route.id, params });
    }
  });

  it('should prefer a literal segment, so /runs/new is the form and not a run named new', () => {
    expect(fromHash('#/runs/new', ROUTES)).toEqual({ screen: 'new-run', params: {} });
    expect(fromHash('#/runs/news', ROUTES)).toEqual({ screen: 'run', params: { runId: 'news' } });
  });

  it('should carry extra parameters in the query, and the path capture wins a clash', () => {
    const hash = toHash('onboard', { draftId: 'd 1', step: '3' }, ROUTES);
    expect(hash).toBe('#/packs/onboard?draftId=d+1&step=3');
    expect(fromHash(hash ?? '', ROUTES)).toEqual({
      screen: 'onboard',
      params: { draftId: 'd 1', step: '3' },
    });
    expect(fromHash('#/runs/a?runId=b', ROUTES)).toEqual({ screen: 'run', params: { runId: 'a' } });
  });

  it('should refuse a hash it cannot read rather than guess', () => {
    for (const hash of [
      '',
      '#',
      '#/',
      '#/nowhere',
      '#/runs/a/b',
      '#/runs/%E0%A4%A',
      '#/checkpoints/x',
    ]) {
      expect(fromHash(hash, ROUTES), hash).toBeNull();
    }
  });

  it('should build no URL for a screen missing the parameter its path needs', () => {
    expect(toHash('run', {}, ROUTES)).toBeNull();
    expect(toHash('run', { runId: '' }, ROUTES)).toBeNull();
  });

  it('should treat an optional last segment as present or absent (Help topics, ENH-004 D2)', () => {
    expect(toHash('help', {}, ROUTES)).toBe('#/help');
    expect(toHash('help', { topic: 'runs' }, ROUTES)).toBe('#/help/runs');
    expect(toHash('help', { topic: '' }, ROUTES)).toBe('#/help');
    expect(fromHash('#/help', ROUTES)).toEqual({ screen: 'help', params: {} });
    expect(fromHash('#/help/runs', ROUTES)).toEqual({ screen: 'help', params: { topic: 'runs' } });
    expect(fromHash('#/help/new-run', ROUTES)).toEqual({
      screen: 'help',
      params: { topic: 'new-run' },
    });
    // Only one optional segment: anything longer is not Help.
    expect(fromHash('#/help/runs/extra', ROUTES)).toBeNull();
  });
});
