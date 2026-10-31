/**
 * Depth from the walked edges (M17) and the render order, over nodes in `GET /api/v1/tree`'s
 * own wire shape passed through the one translation (`toTreeResponse`) - so an adapter's
 * `base` edge is what places it, exactly as the screen receives it.
 */
import { describe, expect, it } from 'vitest';
import { computeDepths, treeOrder } from './treeLayout';
import { toTreeResponse } from '@/api/tree';
import type { TreeNode } from '@/types/tree';

type Edge = { readonly target: string | null; readonly state: string } | null;

function wire(
  name: string,
  edges: { parent?: Edge; base?: Edge; merged?: Edge },
  kind = 'checkpoint',
) {
  return {
    name,
    kind,
    run_id: 'run-a',
    step: 100,
    domain: kind === 'adapter' ? null : 'notary',
    domain_state: kind === 'adapter' ? 'not_in_kind' : 'recorded',
    domain_display: 'notary',
    parent: edges.parent ?? null,
    base: edges.base ?? null,
    base_manifest_sha256: null,
    merged: edges.merged ?? null,
    merged_step: null,
    merged_verification: null,
    test: { set: 'held-out@3', score: 0.84, n: 320 },
    manifest_sha256: 'm'.repeat(64),
    path: `checkpoints/${name}`,
    run_indexed: true,
    index_stale: false,
  };
}

const ok = (target: string) => ({ target, state: 'ok' });
const root = { target: null, state: 'root' };

function forest(nodes: readonly unknown[]): readonly TreeNode[] {
  const result = toTreeResponse({
    nodes,
    edges: [],
    page: { limit: 2000, offset: 0, total: nodes.length },
    index_stale: [],
  });
  if (!result.ok) throw new Error('fixture is malformed');
  return result.value.nodes;
}

const NODES = forest([
  wire('c-100', { parent: root }),
  wire('c-200', { parent: ok('c-100') }),
  wire('c-300', { parent: ok('c-200') }),
  wire('truncated-400', { parent: { target: 'never-existed', state: 'truncated' } }),
  wire('merged-500', { parent: ok('c-300'), merged: ok('ad-1') }),
  wire('ad-1', { base: ok('c-200') }, 'adapter'),
]);

describe('[toTreeResponse]', () => {
  it('should place an adapter by its base edge and a checkpoint by its parent edge', () => {
    const byName = new Map(NODES.map((node) => [node.name, node]));
    expect(byName.get('ad-1')?.parent).toEqual({ target: 'c-200', state: 'ok' });
    expect(byName.get('merged-500')?.merged).toEqual({ target: 'ad-1', state: 'ok' });
  });

  it('should name a node missing the edge its kind needs as malformed, never invent a root', () => {
    const result = toTreeResponse({
      nodes: [wire('ad-x', {}, 'adapter')],
      page: {},
      index_stale: [],
    });
    expect(result.ok).toBe(false);
  });

  it('should name a body that is not a forest as malformed', () => {
    expect(toTreeResponse({}).ok).toBe(false);
  });

  it('should carry the total the query matched, beyond the page', () => {
    const result = toTreeResponse({
      nodes: [wire('c', { parent: root })],
      page: { total: 2400 },
      index_stale: [],
    });
    expect(result.ok && result.value.total).toBe(2400);
  });
});

describe('[computeDepths]', () => {
  it('should walk depth from the edges and read no depth field (M17)', () => {
    for (const node of NODES) expect('depth' in node).toBe(false);
    const depths = computeDepths(NODES);
    expect([depths.get('c-100'), depths.get('c-200'), depths.get('c-300')]).toEqual([0, 1, 2]);
  });

  it('should take an adapters depth from its base, plus one', () => {
    const depths = computeDepths(NODES);
    expect(depths.get('ad-1')).toBe((depths.get('c-200') ?? -1) + 1);
  });

  it('should take a merged nodes depth from its parent edge, not its adapter edge', () => {
    const depths = computeDepths(NODES);
    expect(depths.get('merged-500')).toBe((depths.get('c-300') ?? -1) + 1);
  });

  it('should draw a truncated chain at the top rather than guessing an ancestor', () => {
    expect(computeDepths(NODES).get('truncated-400')).toBe(0);
  });

  it('should terminate on a cycle', () => {
    const cyclic = forest([wire('x', { parent: ok('y') }), wire('y', { parent: ok('x') })]);
    expect(() => computeDepths(cyclic)).not.toThrow();
  });
});

describe('[treeOrder]', () => {
  it('should emit every node exactly once, even through a cycle', () => {
    const nodes = forest([
      wire('root', { parent: root }),
      wire('child', { parent: ok('root') }),
      wire('x', { parent: ok('y') }),
      wire('y', { parent: ok('x') }),
    ]);
    const ordered = treeOrder(nodes);
    expect(ordered.map((node) => node.name).sort()).toEqual(['child', 'root', 'x', 'y']);
    expect(new Set(ordered.map((node) => node.name)).size).toBe(4);
  });

  it('should put a child immediately after its parent', () => {
    const nodes = forest([wire('root', { parent: root }), wire('child', { parent: ok('root') })]);
    expect(treeOrder(nodes).map((node) => node.name)).toEqual(['root', 'child']);
  });
});
