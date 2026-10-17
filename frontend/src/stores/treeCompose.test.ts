/**
 * The forest join (FR-009.21-.31), and five of this feature's negative controls.
 *
 * **M6, M7, M8, M9, M10 and M17 all land here**, and three of them pin *other features'*
 * fixes: M8 pins F8's *"not of the run's current head"*, M9 pins F5's `truncated`/`null`
 * distinction, M6 pins the BRD's nullable-but-required `domain` decision. **If any of those
 * three stays green, that earlier fix is unpinned, and that is a finding in the owning
 * feature - not something to patch here.**
 *
 * The fixtures are built from the shapes `services/reindex.py::_checkpoint_payload` and
 * `_adapter_payload` actually produce, read out of that module. A fixture whose fields agree
 * with the composer by construction is how a suite stays green over a real bug, so the field
 * names here are the API's and not the composer's.
 */
import { describe, expect, it } from 'vitest';
import type { CheckpointRow } from '@/types/run';
import type { AdapterRow } from '@/types/tree';
import {
  compose,
  computeDepths,
  resolveBase,
  resolveMerged,
  resolveParent,
  treeOrder,
} from './treeCompose';

function checkpoint(overrides: Partial<CheckpointRow> & { name: string }): CheckpointRow {
  return {
    run_id: 'run-a',
    step: 100,
    domain: 'notary',
    domain_state: 'recorded',
    domain_display: 'notary',
    signal_source: 'corpus',
    parent: null,
    base_model: 'LFM2-1.2B',
    merged_adapter: null,
    weights_sha256: 'w'.repeat(64),
    manifest_sha256: 'm'.repeat(64),
    selector_hash: null,
    grader_hash: null,
    test: { set: 'held-out@3', score: 0.84, n: 320 },
    retention: null,
    path: `checkpoints/${overrides.name}`,
    created_at: '2026-10-14T00:00:00Z',
    ...overrides,
  };
}

function adapterRow(
  overrides: Partial<AdapterRow> & { name: string; base_name: string },
): AdapterRow {
  return {
    run_id: 'run-b',
    step: 200,
    domain: null,
    domain_state: 'not_in_kind',
    domain_display: 'not recorded in this artifact kind',
    base_manifest_sha256: null,
    weights_sha256: 'w'.repeat(64),
    manifest_sha256: 'm'.repeat(64),
    adapter_config: { rank: 8, alpha: 16, target_modules: ['q_proj'] },
    parts: null,
    test: null,
    retention: null,
    path: `adapters/${overrides.name}`,
    created_at: '2026-10-14T00:00:00Z',
    ...overrides,
  };
}

const NO_RUNS = new Set<string>();
const BOTH_RUNS = new Set(['run-a', 'run-b']);

/* ------------------------------------------------------------------------------------ *
 * resolveParent - M9's target
 * ------------------------------------------------------------------------------------ */

describe('[resolveParent]', () => {
  const present = new Map<string, true>([['base', true]]);
  const known = new Set(['base', 'offpage']);

  it('should report a null parent as ROOT, which is an assertion', () => {
    // M9: returning `truncated` here makes a partial tree look complete (F5 FR-005.39).
    expect(resolveParent('child', null, present, known)).toEqual({ target: null, state: 'root' });
  });

  it('should report an unresolvable parent as TRUNCATED, naming the name', () => {
    expect(resolveParent('child', 'gone', present, known)).toEqual({
      target: 'gone',
      state: 'truncated',
    });
  });

  it('should keep root and truncated distinguishable', () => {
    // The assertion M9 breaks: two different states, never one.
    const root = resolveParent('child', null, present, known);
    const truncated = resolveParent('child', 'gone', present, known);
    expect(root.state).not.toBe(truncated.state);
  });

  it('should report a parent present in the store but off this page as NOT_LOADED', () => {
    // A paging artefact must not look like a broken lineage.
    expect(resolveParent('child', 'offpage', present, known)).toEqual({
      target: 'offpage',
      state: 'not_loaded',
    });
  });

  it('should report a self-parent as BROKEN rather than following it', () => {
    expect(resolveParent('loop', 'loop', present, known)).toEqual({
      target: 'loop',
      state: 'broken',
    });
  });

  it('should report a resolvable parent as ok', () => {
    expect(resolveParent('child', 'base', present, known)).toEqual({ target: 'base', state: 'ok' });
  });
});

/* ------------------------------------------------------------------------------------ *
 * resolveBase - M8's target
 * ------------------------------------------------------------------------------------ */

describe('[resolveBase]', () => {
  it('should take no run_id argument at all', () => {
    // M8's mutation requires editing the signature, which makes it visible in review as well
    // as red in the suite. F8 FR-008.34: an adapter is a child of the base ITS OWN MANIFEST
    // names, not of the run's current head.
    expect(resolveBase.length).toBe(4);
    expect(resolveBase.toString()).not.toContain('run_id');
    expect(resolveBase.toString()).not.toContain('runId');
  });

  it('should place an adapter under the base its manifest names', () => {
    const present = new Map<string, true>([['chk-400', true]]);
    expect(resolveBase('ad-1', 'chk-400', present, new Set(['chk-400']))).toEqual({
      target: 'chk-400',
      state: 'ok',
    });
  });

  it('should report an adapter with no base as UNPARENTED, not as a root', () => {
    // Every adapter composes with something, so an absent base is an artifact that cannot be
    // placed rather than one placed at the top.
    expect(resolveBase('ad-1', null, new Map(), new Set())).toEqual({
      target: null,
      state: 'unparented',
    });
  });

  it('should report a self-based adapter as BROKEN', () => {
    expect(resolveBase('ad-1', 'ad-1', new Map(), new Set())).toEqual({
      target: 'ad-1',
      state: 'broken',
    });
  });
});

/* ------------------------------------------------------------------------------------ *
 * resolveMerged - the three verification states
 * ------------------------------------------------------------------------------------ */

describe('[resolveMerged]', () => {
  const present = new Map<string, true>([['ad-1', true]]);
  const known = new Set(['ad-1']);

  it('should report a recorded hash as verified', () => {
    const resolved = resolveMerged(
      'chk',
      { name: 'ad-1', step: 200, manifest_sha256: 'x'.repeat(64) },
      present,
      known,
    );
    expect(resolved.verification).toBe('verified');
    expect(resolved.edge).toEqual({ target: 'ad-1', state: 'ok' });
  });

  it('should report a present-and-null hash as CONFORMING and unverified, not as a defect', () => {
    // E-12. `manifest_sha256` is required AND nullable, so null is a conforming statement
    // that the edge is deliberately unverified - the distinction a boolean collapses.
    const resolved = resolveMerged(
      'chk',
      { name: 'ad-1', step: 200, manifest_sha256: null },
      present,
      known,
    );
    expect(resolved.verification).toBe('unverified_null');
  });

  it('should report an ABSENT hash key as nonconforming, distinctly from null', () => {
    // E-12a. Different artifact, different fact.
    const resolved = resolveMerged('chk', { name: 'ad-1', step: 200 }, present, known);
    expect(resolved.verification).toBe('key_absent');
    expect(resolved.verification).not.toBe('unverified_null');
  });

  it('should report an unfindable adapter as UNRESOLVED, naming it', () => {
    const resolved = resolveMerged(
      'chk',
      { name: 'ad-gone', step: 1, manifest_sha256: null },
      present,
      known,
    );
    expect(resolved.edge).toEqual({ target: 'ad-gone', state: 'unresolved' });
  });
});

/* ------------------------------------------------------------------------------------ *
 * compose - M6, M7, M10's target
 * ------------------------------------------------------------------------------------ */

describe('[compose]', () => {
  it('should give a merged checkpoint BOTH edges', () => {
    // M7, and the single most important assertion in this feature. `merged_adapter` was added
    // to checkpoint-manifest-v1.json because a merged checkpoint has TWO ancestors and
    // `parent` carries only the base; an implementation that drops one undoes that.
    const forest = compose({
      checkpoints: [
        checkpoint({ name: 'base-400' }),
        checkpoint({
          name: 'merged-500',
          parent: 'base-400',
          merged_adapter: { name: 'ad-1', step: 200, manifest_sha256: 'x'.repeat(64) },
        }),
      ],
      adapters: [adapterRow({ name: 'ad-1', base_name: 'base-400' })],
      knownRunIds: BOTH_RUNS,
    });
    const merged = forest.nodes.find((node) => node.name === 'merged-500')!;
    expect(merged.parent).toEqual({ target: 'base-400', state: 'ok' });
    expect(merged.merged).toEqual({ target: 'ad-1', state: 'ok' });
    expect(merged.merged_verification).toBe('verified');
  });

  it('should give a plain checkpoint no merged edge', () => {
    const forest = compose({
      checkpoints: [checkpoint({ name: 'base-400' })],
      adapters: [],
      knownRunIds: BOTH_RUNS,
    });
    expect(forest.nodes[0]?.merged).toBeNull();
    expect(forest.nodes[0]?.merged_verification).toBeNull();
  });

  it('should keep an absent domain key distinct from an asserted null', () => {
    // M6, and it pins the BRD's 2026-10-13 nullable-but-required decision. If the two ever
    // read the same, that decision is unpinned and it is a finding against the contract gate.
    const forest = compose({
      checkpoints: [
        checkpoint({ name: 'a-null', domain: null, domain_state: 'asserted_null' }),
        checkpoint({ name: 'b-absent', domain: null, domain_state: 'absent' }),
      ],
      adapters: [],
      knownRunIds: BOTH_RUNS,
    });
    const asserted = forest.nodes.find((node) => node.name === 'a-null')!;
    const absent = forest.nodes.find((node) => node.name === 'b-absent')!;
    expect(asserted.domain_state).toBe('asserted_null');
    expect(absent.domain_state).toBe('absent');
    expect(asserted.domain_state).not.toBe(absent.domain_state);
  });

  it('should report an unknown domain_state as absent rather than as an assertion', () => {
    // "The record does not say" is honest; "no domain" would be this view asserting something
    // about the artifact that the artifact did not.
    const forest = compose({
      checkpoints: [checkpoint({ name: 'odd', domain: null, domain_state: 'whatever' })],
      adapters: [],
      knownRunIds: BOTH_RUNS,
    });
    expect(forest.nodes[0]?.domain_state).toBe('absent');
  });

  it('should give an adapters domain the not_in_kind state and never read the bases', () => {
    // adapter-manifest-v1.json has no `domain` property. Reading the base's and presenting it
    // as the adapter's is a fabricated value (ADR-018); FPRD F-11 is the finding.
    const forest = compose({
      checkpoints: [checkpoint({ name: 'base-400', domain: 'notary' })],
      adapters: [adapterRow({ name: 'ad-1', base_name: 'base-400' })],
      knownRunIds: BOTH_RUNS,
    });
    const adapter = forest.nodes.find((node) => node.name === 'ad-1')!;
    expect(adapter.domain_state).toBe('not_in_kind');
    expect(adapter.domain).toBeNull();
  });

  it('should keep a checkpoint whose run is not indexed, marked rather than dropped', () => {
    // M10. `reindex` indexes a valid manifest whose run_id has no run row; a checkpoint
    // silently missing from the tree is strictly worse than one whose run is unknown, because
    // R-02.29 promises the tree across runs and an operator cannot see an omission.
    const forest = compose({
      checkpoints: [checkpoint({ name: 'orphan-400', run_id: 'run-gone' })],
      adapters: [],
      knownRunIds: NO_RUNS,
    });
    expect(forest.nodes.map((node) => node.name)).toEqual(['orphan-400']);
    expect(forest.nodes[0]?.run_indexed).toBe(false);
  });

  it('should keep a cycle visible, marking the edge broken rather than recursing', () => {
    const forest = compose({
      checkpoints: [checkpoint({ name: 'a', parent: 'b' }), checkpoint({ name: 'b', parent: 'a' })],
      adapters: [],
      knownRunIds: BOTH_RUNS,
    });
    expect(forest.nodes.map((node) => node.name)).toEqual(['a', 'b']);
    expect(forest.nodes.every((node) => node.parent.state === 'broken')).toBe(true);
  });

  it('should report a test score only when the record has all three required members', () => {
    // `ScoreRecord.required` is [set, score, n]. A record missing one is not a score with a
    // hole in it; it is not a score, and part-rendering it would print a number with no sample.
    const forest = compose({
      checkpoints: [
        checkpoint({ name: 'full' }),
        checkpoint({ name: 'partial', test: { set: 'held-out@3', score: 0.8 } }),
        checkpoint({ name: 'none', test: null }),
      ],
      adapters: [],
      knownRunIds: BOTH_RUNS,
    });
    expect(forest.nodes.find((node) => node.name === 'full')?.test).toEqual({
      set: 'held-out@3',
      score: 0.84,
      n: 320,
    });
    expect(forest.nodes.find((node) => node.name === 'partial')?.test).toBeNull();
    expect(forest.nodes.find((node) => node.name === 'none')?.test).toBeNull();
  });
});

/* ------------------------------------------------------------------------------------ *
 * depth - M17's target
 * ------------------------------------------------------------------------------------ */

describe('[computeDepths]', () => {
  const forest = compose({
    checkpoints: [
      checkpoint({ name: 'c-100', parent: null }),
      checkpoint({ name: 'c-200', parent: 'c-100' }),
      checkpoint({ name: 'c-300', parent: 'c-200' }),
      checkpoint({ name: 'truncated-400', parent: 'never-existed' }),
      checkpoint({
        name: 'merged-500',
        parent: 'c-300',
        merged_adapter: { name: 'ad-1', step: 200, manifest_sha256: null },
      }),
    ],
    adapters: [adapterRow({ name: 'ad-1', base_name: 'c-200' })],
    knownRunIds: BOTH_RUNS,
  });

  it('should walk depth from the edges and emit no depth field to trust', () => {
    // M17's mutation requires ADDING a `depth` field to the response, which is the point: the
    // response has none, so there is nothing a client could read instead of walking.
    expect('depth' in forest).toBe(false);
    for (const node of forest.nodes) expect('depth' in node).toBe(false);

    const depths = computeDepths(forest.nodes);
    expect(depths.get('c-100')).toBe(0);
    expect(depths.get('c-200')).toBe(1);
    expect(depths.get('c-300')).toBe(2);
  });

  it('should take an adapters depth from its base, plus one', () => {
    const depths = computeDepths(forest.nodes);
    expect(depths.get('ad-1')).toBe((depths.get('c-200') ?? -1) + 1);
  });

  it('should take a merged nodes depth from its parent edge, not its adapter edge', () => {
    // The ordering of the tree stays one-dimensional while the lineage stays complete.
    const depths = computeDepths(forest.nodes);
    expect(depths.get('merged-500')).toBe((depths.get('c-300') ?? -1) + 1);
  });

  it('should draw a truncated chain at the top rather than guessing an ancestor', () => {
    const depths = computeDepths(forest.nodes);
    expect(depths.get('truncated-400')).toBe(0);
  });

  it('should terminate on a cycle', () => {
    const cyclic = compose({
      checkpoints: [checkpoint({ name: 'x', parent: 'y' }), checkpoint({ name: 'y', parent: 'x' })],
      adapters: [],
      knownRunIds: BOTH_RUNS,
    });
    expect(() => computeDepths(cyclic.nodes)).not.toThrow();
  });
});

describe('[treeOrder]', () => {
  it('should emit every node exactly once, even through a cycle', () => {
    const forest = compose({
      checkpoints: [
        checkpoint({ name: 'root' }),
        checkpoint({ name: 'child', parent: 'root' }),
        checkpoint({ name: 'x', parent: 'y' }),
        checkpoint({ name: 'y', parent: 'x' }),
      ],
      adapters: [],
      knownRunIds: BOTH_RUNS,
    });
    const ordered = treeOrder(forest.nodes);
    expect(ordered.map((node) => node.name).sort()).toEqual(['child', 'root', 'x', 'y']);
    expect(new Set(ordered.map((node) => node.name)).size).toBe(4);
  });

  it('should put a child immediately after its parent', () => {
    const forest = compose({
      checkpoints: [checkpoint({ name: 'root' }), checkpoint({ name: 'child', parent: 'root' })],
      adapters: [],
      knownRunIds: BOTH_RUNS,
    });
    const names = treeOrder(forest.nodes).map((node) => node.name);
    expect(names).toEqual(['root', 'child']);
  });
});
