/**
 * The forest, from **`GET /api/v1/tree`** (FR-009.21-.31; task 14.0's inventory, row T-2).
 *
 * *Until 2026-10-18 this module composed the forest in the browser from `GET /checkpoints`,
 * `GET /adapters` and one `GET /runs/{id}` probe per run, and the Checkpoints screen said
 * "GET /api/v1/tree is not mounted in this build". It had been mounted since F9's task 3.0 -
 * `services/tree.py::compose`, the same two-edge join with every edge state, the test score
 * and `run_indexed` from the run table. The screen's own note said that when the route landed
 * "the browser-side composer is deleted rather than reconciled", and it now is: the rules
 * (M6-M10) live once, in the backend, under `test_tree_compose.py`.*
 *
 * ## One translation, stated
 *
 * The wire node carries two base-chain edges - `parent` for a checkpoint, `base` for an
 * adapter (F8: an adapter is a child of the base its own manifest names) - and the screen
 * walks one, `TreeNode.parent`. {@link toTreeResponse} picks the edge the node's kind has. A
 * node missing the edge its kind requires is **malformed**, named, never given an invented
 * `root`.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';
import type { ArtifactKind, ResolvedEdge, TreeNode, TreeResponse } from '@/types/tree';
import { asDomainState } from '@/types/tree';

export const TREE_PATH = `${V1}/tree`;
/** The route's own maximum page (`services/tree.py::MAX_LIMIT`). The screen states the total. */
export const TREE_LIMIT = 2000;

export interface ForestOptions {
  readonly domain?: string;
  readonly limit?: number;
}

export async function fetchForest(options: ForestOptions = {}): Promise<ApiResult<TreeResponse>> {
  const result = await apiGet<unknown>(TREE_PATH, {
    query: { domain: options.domain, limit: options.limit ?? TREE_LIMIT },
  });
  return result.ok ? toTreeResponse(result.value) : result;
}

type Wire = Readonly<Record<string, unknown>>;

function edge(value: unknown): ResolvedEdge | null {
  if (value === null || typeof value !== 'object') return null;
  const record = value as Wire;
  const target = record['target'];
  const state = record['state'];
  if (typeof state !== 'string' || (target !== null && typeof target !== 'string')) return null;
  return { target: target as string | null, state: state as ResolvedEdge['state'] };
}

function malformed(reason: string, body: unknown): ApiResult<TreeResponse> {
  return {
    ok: false,
    failure: {
      kind: 'malformed',
      status: 200,
      reason,
      body: JSON.stringify(body).slice(0, 2000),
    },
  };
}

/** `GET /api/v1/tree`'s body (`schemas/tree.py::TreeResponse`) as the screen walks it. */
export function toTreeResponse(body: unknown): ApiResult<TreeResponse> {
  const wire = body as Wire | null;
  if (wire === null || typeof wire !== 'object' || !Array.isArray(wire['nodes'])) {
    return malformed('the response is not a forest (no nodes)', body);
  }
  const nodes: TreeNode[] = [];
  for (const raw of wire['nodes'] as readonly Wire[]) {
    const kind = raw['kind'] as ArtifactKind;
    const chain = edge(kind === 'adapter' ? raw['base'] : raw['parent']);
    const domainState = asDomainState(raw['domain_state']);
    if (chain === null || domainState === null || typeof raw['name'] !== 'string') {
      return malformed(`node ${String(raw['name'])} carries no ${kind} edge or domain state`, raw);
    }
    const test = raw['test'] as Wire | null;
    nodes.push({
      name: raw['name'],
      kind,
      run_id: String(raw['run_id']),
      step: Number(raw['step']),
      domain: (raw['domain'] as string | null) ?? null,
      domain_state: domainState,
      parent: chain,
      merged: edge(raw['merged']),
      merged_verification: (raw['merged_verification'] ?? null) as TreeNode['merged_verification'],
      run_indexed: raw['run_indexed'] === true,
      test:
        test === null || test === undefined
          ? null
          : { score: Number(test['score']), n: Number(test['n']), set: String(test['set']) },
      manifest_sha256: (raw['manifest_sha256'] as string | null) ?? null,
      path: String(raw['path']),
    });
  }
  const page = wire['page'] as Wire | undefined;
  return {
    ok: true,
    value: {
      nodes,
      index_stale: Array.isArray(wire['index_stale'])
        ? (wire['index_stale'] as unknown[]).map(String)
        : [],
      total: typeof page?.['total'] === 'number' ? (page['total'] as number) : nodes.length,
    },
  };
}
