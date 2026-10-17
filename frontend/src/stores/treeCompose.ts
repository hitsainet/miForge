/**
 * The forest join, and depth walked from the edges (FR-009.21-.31, FR-009.26).
 *
 * ## Where this belongs, and why it is here
 *
 * F9's FTDD puts `compose()` in `backend/src/services/tree.py` behind `GET /api/v1/tree`.
 * **Neither exists in this build** - verified by enumerating `create_app().routes`. What
 * exists is `services/reindex.py::checkpoint_tree`, which does the same two-edge join and has
 * **no production caller**; and its payload carries neither the `test` score R-02.29 asks for
 * nor any edge state, so it could not serve FR-009.24 or FR-009.27 even once mounted.
 *
 * So the join is done here, from two live endpoints whose payloads carry every fact the
 * requirement names, and the edge rules below are transcribed from FTDD §3.2 and §3.3
 * **unchanged**, so that when the route lands this module is deleted rather than reconciled.
 * It is a reported seam, not a design decision. See `types/tree.ts` for the full note.
 *
 * ## The five rules that are mutation targets
 *
 * - **`root` is not `truncated`** (M9). One means *this is a root*; the other means *the
 *   record stops here*. Conflating them makes a partial tree look complete, which is F5's
 *   own FR-005.39 fix - so if M9 survives, that fix is unpinned and it is a finding in F5.
 * - **A merged checkpoint gets BOTH edges** (M7), never `elif` and never an early return.
 * - **`resolveBase` cannot see a `run_id`** (M8): it is not a parameter, so joining on the
 *   run's head requires editing the signature - visible in review as well as red in the
 *   suite. F8 FR-008.34: *"as a child of the base its own manifest names - not of the run's
 *   current head."*
 * - **`runIndexed === false` never removes a node** (M10). `reindex` indexes a valid manifest
 *   whose `run_id` has no run row, and a checkpoint silently missing from the tree is
 *   strictly worse than one whose run is unknown.
 * - **An absent `domain` key is not `null`** (M6). Three states on a checkpoint plus a
 *   fourth for an adapter, carried not collapsed.
 */
import type { CheckpointRow } from '@/types/run';
import type {
  AdapterRow,
  DomainState,
  MergedVerification,
  ResolvedEdge,
  TreeNode,
  TreeResponse,
} from '@/types/tree';
import { asDomainState } from '@/types/tree';

/* ------------------------------------------------------------------------------------ *
 * The three resolutions - one function each, pure
 * ------------------------------------------------------------------------------------ */

/**
 * Place a checkpoint by its manifest's `parent`.
 *
 * @param nodeName the child's own fully qualified name.
 * @param parent the manifest's `parent`: a name, or `null` asserting this is a root.
 * @param byName every node in this forest, by name.
 * @param allNames every name known to the query - a superset of `byName` when the listing
 *   was bounded. A name here but not in `byName` is a **paging artefact**, reported as
 *   `not_loaded` so it cannot be mistaken for a broken lineage.
 */
export function resolveParent(
  nodeName: string,
  parent: string | null,
  byName: ReadonlyMap<string, unknown>,
  allNames: ReadonlySet<string>,
): ResolvedEdge {
  // An assertion, not an absence: this IS a root. Returning `truncated` here is M9.
  if (parent === null) return { target: null, state: 'root' };
  if (parent === nodeName) return { target: parent, state: 'broken' };
  if (byName.has(parent)) return { target: parent, state: 'ok' };
  if (allNames.has(parent)) return { target: parent, state: 'not_loaded' };
  // The record stops here. The name that did not resolve is carried so the row can say it.
  return { target: parent, state: 'truncated' };
}

/**
 * Place an adapter by **the base its own manifest names**.
 *
 * There is deliberately **no `runId` parameter**. F8 FR-008.34 places an adapter under the
 * base its manifest names, *not* under the run's current head, and the absence of the
 * argument is what makes M8's mutation require a signature change.
 */
export function resolveBase(
  adapterName: string,
  baseName: string | null,
  byName: ReadonlyMap<string, unknown>,
  allNames: ReadonlySet<string>,
): ResolvedEdge {
  // An adapter with no base is not a root - every adapter composes with something, so an
  // absent base is an artifact that cannot be placed rather than one placed at the top.
  if (baseName === null || baseName.length === 0) return { target: null, state: 'unparented' };
  if (baseName === adapterName) return { target: baseName, state: 'broken' };
  if (byName.has(baseName)) return { target: baseName, state: 'ok' };
  if (allNames.has(baseName)) return { target: baseName, state: 'not_loaded' };
  return { target: baseName, state: 'truncated' };
}

/**
 * The merged checkpoint's **second** ancestor, and whether the edge could be verified.
 *
 * Verification is **three-valued, not a boolean**. `MergedAdapter.required` is
 * `[name, step, manifest_sha256]` *and* `manifest_sha256` is `anyOf [string, null]` with
 * `default: null` - so the key is required and nullable, which gives a hash (verified), a
 * present `null` (**conforming**, deliberately unverified - E-12), and an absent key (a
 * missing required key, nonconforming - E-12a). A boolean collapses the middle into the
 * third and reports a conforming manifest as defective, which is the exact conflation
 * FR-009.17 rule 3 forbids for `domain`.
 */
export function resolveMerged(
  nodeName: string,
  merged: Readonly<Record<string, unknown>>,
  byName: ReadonlyMap<string, unknown>,
  allNames: ReadonlySet<string>,
): { readonly edge: ResolvedEdge; readonly verification: MergedVerification } {
  const verification: MergedVerification = !('manifest_sha256' in merged)
    ? 'key_absent'
    : merged['manifest_sha256'] === null
      ? 'unverified_null'
      : 'verified';

  const name = merged['name'];
  if (typeof name !== 'string' || name.length === 0) {
    return { edge: { target: null, state: 'unresolved' }, verification };
  }
  if (name === nodeName) return { edge: { target: name, state: 'broken' }, verification };
  if (byName.has(name)) return { edge: { target: name, state: 'ok' }, verification };
  if (allNames.has(name)) return { edge: { target: name, state: 'not_loaded' }, verification };
  // Named, so the row can say *which* adapter it could not find.
  return { edge: { target: name, state: 'unresolved' }, verification };
}

/* ------------------------------------------------------------------------------------ *
 * compose
 * ------------------------------------------------------------------------------------ */

function readTest(value: unknown): TreeNode['test'] {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const score = record['score'];
  const n = record['n'];
  const set = record['set'];
  // `ScoreRecord.required` is `[set, score, n]`. A record missing any of the three is not a
  // score with a hole in it; it is not a score. Reported absent rather than part-rendered.
  if (typeof score !== 'number' || typeof n !== 'number' || typeof set !== 'string') return null;
  return { score, n, set };
}

function checkpointDomainState(row: CheckpointRow): DomainState {
  const state = asDomainState(row.domain_state);
  // A state this build does not know is reported `absent` - "the record does not say" -
  // rather than `asserted_null`, which would be this view asserting something about the
  // artifact that the artifact did not say.
  if (state === null) return 'absent';
  // `not_in_kind` is an adapter's state and a checkpoint manifest always has the field.
  return state === 'not_in_kind' ? 'absent' : state;
}

export interface ComposeInput {
  readonly checkpoints: readonly CheckpointRow[];
  readonly adapters: readonly AdapterRow[];
  /** Run ids that have a run row. A node whose run is absent is marked, never dropped. */
  readonly knownRunIds: ReadonlySet<string>;
  /**
   * Every artifact name the query knows, including ones outside this page. Defaults to the
   * composed set, which makes `not_loaded` unreachable - correct for an unbounded listing
   * and wrong the moment paging arrives, which is why it is a parameter.
   */
  readonly allNames?: ReadonlySet<string>;
  /** Names whose index row disagrees with the manifest on disk (FR-009.29). */
  readonly indexStale?: readonly string[];
}

/** The forest. Pure: no fetch, no clock, no store. */
export function compose(input: ComposeInput): TreeResponse {
  const names = new Set<string>();
  for (const row of input.checkpoints) names.add(row.name);
  for (const row of input.adapters) names.add(row.name);
  const byName: ReadonlyMap<string, true> = new Map([...names].map((name) => [name, true]));
  const allNames = input.allNames ?? names;

  const nodes: TreeNode[] = [];

  for (const row of input.checkpoints) {
    const parent = resolveParent(row.name, row.parent, byName, allNames);
    // BOTH edges, and this is the single most important shape in the feature. `merged` is
    // assigned IN ADDITION to `parent`; an `else` or an early return here is M7 and undoes
    // the reason `merged_adapter` was added to the schema on 2026-10-13.
    let merged: ResolvedEdge | null = null;
    let verification: MergedVerification | null = null;
    if (row.merged_adapter !== null && typeof row.merged_adapter === 'object') {
      const resolved = resolveMerged(row.name, row.merged_adapter, byName, allNames);
      merged = resolved.edge;
      verification = resolved.verification;
    }
    nodes.push({
      name: row.name,
      kind: 'checkpoint',
      run_id: row.run_id,
      step: row.step,
      domain: row.domain,
      domain_state: checkpointDomainState(row),
      parent,
      merged,
      merged_verification: verification,
      run_indexed: input.knownRunIds.has(row.run_id),
      test: readTest(row.test),
      manifest_sha256: row.manifest_sha256,
      path: row.path,
    });
  }

  for (const row of input.adapters) {
    nodes.push({
      name: row.name,
      kind: 'adapter',
      run_id: row.run_id,
      step: row.step,
      // `adapter-manifest-v1.json` has NO `domain` property (parsed). The base's domain is
      // not the adapter's and is never read across to fill this - that is a fabricated value
      // (ADR-018), and FPRD F-11 reports R-02.29's field list as unsatisfiable for this kind.
      domain: null,
      domain_state: 'not_in_kind',
      parent: resolveBase(row.name, row.base_name, byName, allNames),
      merged: null,
      merged_verification: null,
      run_indexed: input.knownRunIds.has(row.run_id),
      test: readTest(row.test),
      manifest_sha256: row.manifest_sha256,
      path: row.path,
    });
  }

  const withCycles = markCycles(nodes);
  return {
    nodes: withCycles.sort((a, b) => a.name.localeCompare(b.name)),
    index_stale: input.indexStale ?? [],
  };
}

/**
 * Mark the node at which a parent walk re-enters itself.
 *
 * `resolveParent` catches only a self-parent; a longer cycle (`a → b → a`) is caught here by
 * a visited set, and **every node stays visible** (FR-009.30) - the edge is named broken, the
 * artifact is not hidden. Without this the depth walk below would not terminate.
 */
function markCycles(nodes: readonly TreeNode[]): TreeNode[] {
  const byName = new Map(nodes.map((node) => [node.name, node]));
  return nodes.map((node) => {
    const seen = new Set<string>([node.name]);
    let cursor: TreeNode | undefined = node;
    while (cursor !== undefined) {
      const target = cursor.parent.target;
      if (target === null || cursor.parent.state !== 'ok') break;
      if (seen.has(target)) {
        return { ...node, parent: { target: node.parent.target, state: 'broken' as const } };
      }
      seen.add(target);
      cursor = byName.get(target);
    }
    return node;
  });
}

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
