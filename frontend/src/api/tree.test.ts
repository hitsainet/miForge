/**
 * The forest's fetch half, and the run probe FR-009.28 needs.
 *
 * `run_indexed` asks whether an artifact's `run_id` has a run row, and this build serves no
 * run listing - so the only honest answer is one probe per distinct run id, where a
 * `run_not_found` refusal means *not indexed*. The direction of the **default** is the thing
 * these tests pin: a probe that fails for any other reason reports the run as **indexed**,
 * because the opposite default would turn one failed request into a screen full of false
 * "run not indexed" markers.
 */
import { describe, expect, it } from 'vitest';
import { resolveKnownRunIds } from './tree';
import type { ApiResult } from './client';

const notFound = (): Promise<ApiResult<unknown>> =>
  Promise.resolve({
    ok: false,
    failure: {
      kind: 'refusal',
      status: 404,
      code: 'run_not_found',
      message: 'No run with that id.',
      details: {},
    },
  });

const found = (): Promise<ApiResult<unknown>> => Promise.resolve({ ok: true, value: {} });

const brokenTransport = (): Promise<ApiResult<unknown>> =>
  Promise.resolve({ ok: false, failure: { kind: 'transport', status: 0, cause: 'fetch failed' } });

describe('[resolveKnownRunIds]', () => {
  it('should treat run_not_found as not indexed', async () => {
    const known = await resolveKnownRunIds(new Set(['gone']), notFound);
    expect(known.has('gone')).toBe(false);
  });

  it('should treat a found run as indexed', async () => {
    const known = await resolveKnownRunIds(new Set(['here']), found);
    expect(known.has('here')).toBe(true);
  });

  it('should treat any OTHER failure as unknown and report the run indexed', async () => {
    // A transport blip must not mark every node's run missing. The honest reading of "I could
    // not ask" is not "the answer is no".
    const known = await resolveKnownRunIds(new Set(['maybe']), brokenTransport);
    expect(known.has('maybe')).toBe(true);
  });

  it('should probe once per distinct run id, not once per node', async () => {
    const asked: string[] = [];
    await resolveKnownRunIds(new Set(['a', 'b']), (runId) => {
      asked.push(runId);
      return found();
    });
    expect(asked.sort()).toEqual(['a', 'b']);
  });

  it('should ask nothing when there are no runs', async () => {
    let calls = 0;
    const known = await resolveKnownRunIds(new Set(), () => {
      calls += 1;
      return found();
    });
    expect(calls).toBe(0);
    expect(known.size).toBe(0);
  });
});
