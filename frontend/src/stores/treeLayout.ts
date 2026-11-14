/**
 * Depth walked from the edges, and the render order (FR-009.26, FR-009.30).
 *
 * The forest itself is `GET /api/v1/tree`'s (`services/tree.py`): the edge rules M6-M10 live
 * there, once. *Until 2026-10-18 they were also transcribed here, in a browser-side composer
 * built while the screen believed the route was not mounted; it was, and the composer is
 * deleted.* What stays is layout, which the response deliberately leaves to the client: there
 * is no `depth` field to trust (M17).
 */
import type { TreeNode } from '@/types/tree';

/* ------------------------------------------------------------------------------------ *
 * Depth - walked, never transmitted
 * ------------------------------------------------------------------------------------ */

/**
 * Depth from the root, **computed from the walked edges** (FR-009.26).
 *
 * `TreeResponse` carries no `depth` field, so there is nothing for this to trust. Taking it
 * from a transmitted field is mutation **M17**, which requires adding that field - which is
 * the point. The mockup's indentation came from a stored `depth` and its own handoff warns
 * that "the real tree needs the same, walked from `parent`".
 *
 * A merged node's depth comes from its **`parent`** edge, the base chain, so the ordering of
 * the tree stays one-dimensional while the lineage stays complete; the adapter edge is drawn
 * as a marked second edge. An adapter reached through its base takes base depth + 1.
 */
export function computeDepths(nodes: readonly TreeNode[]): ReadonlyMap<string, number> {
  const byName = new Map(nodes.map((node) => [node.name, node]));
  const depths = new Map<string, number>();

  const depthOf = (name: string, guard: ReadonlySet<string>): number => {
    const cached = depths.get(name);
    if (cached !== undefined) return cached;
    const node = byName.get(name);
    if (node === undefined) return 0;
    const target = node.parent.target;
    // Anything that is not a resolved edge is drawn at the top: a truncated chain, a broken
    // edge and an unparented adapter have no ancestor *in this forest* to indent under, and
    // guessing one would place the node under something that is not its parent.
    if (target === null || node.parent.state !== 'ok' || guard.has(name)) {
      depths.set(name, 0);
      return 0;
    }
    const depth = depthOf(target, new Set([...guard, name])) + 1;
    depths.set(name, depth);
    return depth;
  };

  for (const node of nodes) depthOf(node.name, new Set());
  return depths;
}

/** Nodes in render order: each root followed by its descendants, depth-first by name. */
export function treeOrder(nodes: readonly TreeNode[]): readonly TreeNode[] {
  const children = new Map<string, TreeNode[]>();
  const roots: TreeNode[] = [];
  for (const node of nodes) {
    const target = node.parent.target;
    if (target === null || node.parent.state !== 'ok') {
      roots.push(node);
      continue;
    }
    const bucket = children.get(target) ?? [];
    bucket.push(node);
    children.set(target, bucket);
  }
  const byName = (a: TreeNode, b: TreeNode): number => a.name.localeCompare(b.name);
  const out: TreeNode[] = [];
  const emit = (node: TreeNode, guard: ReadonlySet<string>): void => {
    out.push(node);
    if (guard.has(node.name)) return;
    const next = new Set([...guard, node.name]);
    for (const child of (children.get(node.name) ?? []).sort(byName)) emit(child, next);
  };
  for (const root of roots.sort(byName)) emit(root, new Set());
  // A node reachable only through a cycle would otherwise vanish; FR-009.30 keeps every
  // node visible, so anything the walk did not emit is appended rather than dropped.
  const emitted = new Set(out.map((node) => node.name));
  for (const node of nodes) if (!emitted.has(node.name)) out.push(node);
  return out;
}
