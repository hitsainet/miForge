/**
 * The artifact forest - R-02.29's two node kinds and three edges (FR-009.21-.31).
 *
 * ## Why this shape lives on the client, and the seam that is owed
 *
 * F9's FTDD puts the forest join in the backend: `services/tree.py::compose`, served by
 * `GET /api/v1/tree`. **Neither exists.** What *does* exist is
 * `services/reindex.py::checkpoint_tree(session)`, which performs the same two-edge join -
 * and it has **no production caller**: the only references to it are the reindex-equivalence
 * snapshot and two tests. It is the fifth capability in this repository implemented with no
 * route (after the four index rebuilders), and ADR-016 #1 is explicit that such a capability
 * is not shipped.
 *
 * `checkpoint_tree`'s payload is also a strict subset of what R-02.29 needs: its nodes carry
 * `{name, kind, run_id, step, domain, domain_state}` and its edges `{child, parent, edge}` -
 * **no `test` score** (R-02.29's fourth fact, FR-009.24) and **no edge states**, so it could
 * not serve FR-009.27's `truncated` / `root` distinction or FR-009.28's unindexed run even
 * once it were mounted.
 *
 * So this build composes the forest in the browser from **two live endpoints** -
 * `GET /api/v1/checkpoints` and `GET /api/v1/adapters` - whose payloads
 * (`services/reindex.py::_checkpoint_payload` / `_adapter_payload`) carry every fact the
 * requirement names: `parent`, `merged_adapter`, `base_name`, `domain` + `domain_state`, and
 * `test`. The composition is one pure function, `stores/treeCompose.ts`, with the edge rules
 * of the FTDD §3.2 and §3.3 transcribed exactly, so that **when the backend route lands the
 * frontend composer is deleted rather than reconciled.**
 *
 * This is recorded as a seam, not presented as the design. It is the one place where this
 * feature does arithmetic the FTDD assigns to the server, and it is done here only because
 * the alternative was a screen that cannot render.
 */

/**
 * How an edge resolved. The FTDD's seven, verbatim, and the distinctions are load-bearing:
 *
 * - `root` is an **assertion** - `parent: null`, this *is* a root;
 * - `truncated` means the record stops here - the named parent does not resolve.
 *
 * Conflating the two "would make a partial tree look complete" (F5 FR-005.39 / its E-11),
 * and that conflation is mutation **M9**. If M9 ever survives, F5's own fix is unpinned.
 */
export type EdgeState =
  'ok' | 'root' | 'truncated' | 'broken' | 'unparented' | 'unresolved' | 'not_loaded';

export const EDGE_STATES: readonly EdgeState[] = [
  'ok',
  'root',
  'truncated',
  'broken',
  'unparented',
  'unresolved',
  'not_loaded',
];

export type ArtifactKind = 'checkpoint' | 'adapter';

/**
 * The three states a `domain` can be in, plus a fourth for an artifact kind that has none.
 *
 * `backend/src/services/reindex.py` already carries `domain_state` beside `domain` for
 * exactly this reason: `null` is an **assertion** ("this artifact is not domain-scoped") and
 * an absent manifest key is an **omission**, and collapsing them is mutation **M6**. An
 * adapter's is a fourth state because `adapter-manifest-v1.json` has **no** `domain`
 * property - reading the base's domain and presenting it as the adapter's would be a
 * fabricated value (ADR-018), and FPRD F-11 reports that R-02.29's field list is
 * unsatisfiable for that kind.
 */
export type DomainState = 'recorded' | 'asserted_null' | 'absent' | 'not_in_kind';

/**
 * The four states, **spelled exactly as `backend/src/models/domain_state.py` spells them**.
 *
 * Not `present`/`null`: this project has already paid for one vocabulary acquiring four
 * spellings (foundation 3.1a's verdict value), and a client enum that renames a server enum
 * is the first of the four. A value off the wire that is none of these is classified
 * `absent` by `asDomainState` **and named**, never silently mapped to `asserted_null`.
 */
export const DOMAIN_STATES: readonly DomainState[] = [
  'recorded',
  'asserted_null',
  'absent',
  'not_in_kind',
];

/** Narrow a wire value, returning `null` for one this build does not know. */
export function asDomainState(value: unknown): DomainState | null {
  return DOMAIN_STATES.includes(value as DomainState) ? (value as DomainState) : null;
}

/** One resolved edge: who it points at, and how the resolution went. */
export interface ResolvedEdge {
  readonly target: string | null;
  readonly state: EdgeState;
}

/** Whether a merged edge could be verified. **Three states, not a boolean.** */
export type MergedVerification =
  /** `manifest_sha256` carried a hash. */
  | 'verified'
  /** The key is present and `null` - **conforming**, deliberately unverified (E-12). */
  | 'unverified_null'
  /** The key is absent - a **missing required key**, nonconforming (E-12a). */
  | 'key_absent';

export interface TreeNode {
  readonly name: string;
  readonly kind: ArtifactKind;
  readonly run_id: string;
  readonly step: number;
  readonly domain: string | null;
  readonly domain_state: DomainState;
  /** The base chain's edge. For an adapter this is its `base.name` edge. */
  readonly parent: ResolvedEdge;
  /**
   * The **second** edge a merged checkpoint has, **in addition** to `parent` - never
   * instead of it. `null` for every other node. Dropping it is mutation **M7**, and it
   * undoes the reason `merged_adapter` was added to `checkpoint-manifest-v1.json`.
   */
  readonly merged: ResolvedEdge | null;
  readonly merged_verification: MergedVerification | null;
  /** `false` when the artifact's `run_id` has no run row. **Never removes the node** (M10). */
  readonly run_indexed: boolean;
  /** The final test score and its sample - R-02.29's fourth fact. `null` when unrecorded. */
  readonly test: { readonly score: number; readonly n: number; readonly set: string } | null;
  readonly manifest_sha256: string | null;
  readonly path: string;
}

/**
 * The forest. **No `depth` field, and that is deliberate** (FR-009.26): there is nothing
 * for a client to trust, so depth is walked from the edges. Mutation **M17** requires adding
 * a field here, which is the point. **No pack field either** (FR-009.24, F-10).
 */
export interface TreeResponse {
  readonly nodes: readonly TreeNode[];
  /** Nodes whose index row disagrees with the manifest on disk (FR-009.29, ADR-004). */
  readonly index_stale: readonly string[];
}

/** One row of `GET /api/v1/adapters` - `services/reindex.py::_adapter_payload`. */
export interface AdapterRow {
  readonly name: string;
  readonly run_id: string;
  readonly step: number;
  readonly domain: string | null;
  readonly domain_state: string;
  readonly domain_display: string;
  readonly base_name: string;
  readonly base_manifest_sha256: string | null;
  readonly weights_sha256: string | null;
  readonly manifest_sha256: string | null;
  readonly adapter_config: Readonly<Record<string, unknown>> | null;
  readonly parts: readonly unknown[] | null;
  readonly test: Readonly<Record<string, unknown>> | null;
  readonly retention: Readonly<Record<string, unknown>> | null;
  readonly path: string;
  readonly created_at: string;
}
