/**
 * The forest, assembled from the endpoints that exist (FR-009.21, .28).
 *
 * **`GET /api/v1/tree` is not mounted** and `services/reindex.py::checkpoint_tree` - which
 * performs the same join - has no production caller. See `types/tree.ts` for the full note
 * and `stores/treeCompose.ts` for the join itself. This module is the fetch half of that
 * seam: two listings, plus one probe per **distinct run id** to answer FR-009.28.
 *
 * ## Why the run probe, and what it costs
 *
 * `run_indexed` asks whether an artifact's `run_id` has a run row. The honest source is the
 * run table, and this build serves no run listing (`GET /api/v1/runs` is not mounted either),
 * so the only way to answer is `GET /api/v1/runs/{id}` per distinct id: a `run_not_found`
 * refusal means *not indexed*. That is one request per run, not per node - twenty or so for
 * a five-hundred-node forest - and it is stated here rather than hidden because it is a cost
 * the designed endpoint would not have. **The alternative was to assume every run is indexed,
 * which would make FR-009.28's case unreachable and S3's checkpoint look ordinary.**
 *
 * A probe that fails for any reason other than `run_not_found` is treated as **unknown, not
 * absent**: the run is reported indexed so a transport blip cannot mark every node's run
 * missing. The opposite default would turn one failed request into a screen full of false
 * "run not indexed" markers.
 */
import type { ApiFailure } from './errors';
import { fetchAdapters } from './adapters';
import { fetchCheckpoints } from './checkpoints';
import { fetchRun } from './runs';
import type { ApiResult } from './client';
import { compose } from '@/stores/treeCompose';
import type { TreeResponse } from '@/types/tree';

export interface ForestOptions {
  readonly domain?: string;
  readonly limit?: number;
}

/**
 * Fetch and compose the forest.
 *
 * Both listings are required: a forest missing its adapters would satisfy no part of
 * R-02.29's amendment, so a failure in either is the call's failure rather than a partial
 * tree presented as whole.
 */
export async function fetchForest(options: ForestOptions = {}): Promise<ApiResult<TreeResponse>> {
  const [checkpoints, adapters] = await Promise.all([
    fetchCheckpoints({
      ...(options.domain === undefined ? {} : { domain: options.domain }),
      ...(options.limit === undefined ? {} : { limit: options.limit }),
    }),
    fetchAdapters(),
  ]);
  if (!checkpoints.ok) return checkpoints;
  if (!adapters.ok) return adapters;

  const runIds = new Set<string>();
  for (const row of checkpoints.value.checkpoints) runIds.add(row.run_id);
  for (const row of adapters.value.adapters) runIds.add(row.run_id);

  const knownRunIds = await resolveKnownRunIds(runIds);

  return {
    ok: true,
    value: compose({
      checkpoints: checkpoints.value.checkpoints,
      adapters: adapters.value.adapters,
      knownRunIds,
    }),
  };
}

/** One probe per distinct run id. `run_not_found` means not indexed; anything else is unknown. */
export async function resolveKnownRunIds(
  runIds: ReadonlySet<string>,
  probe: (runId: string) => Promise<ApiResult<unknown>> = fetchRun,
): Promise<ReadonlySet<string>> {
  const known = new Set<string>();
  await Promise.all(
    [...runIds].map(async (runId) => {
      const result = await probe(runId);
      if (result.ok || !isRunNotFound(result.failure)) known.add(runId);
    }),
  );
  return known;
}

function isRunNotFound(failure: ApiFailure): boolean {
  return failure.kind === 'refusal' && failure.code === 'run_not_found';
}
