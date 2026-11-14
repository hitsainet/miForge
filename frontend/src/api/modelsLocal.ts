/**
 * `GET /api/v1/models/local` - the model directories a load can read (task 14.0, AC-009.41).
 *
 * The roots are the two `local_dir.resolve_within_roots` confines a submitted path to, so
 * every row is a path preflight accepts. The walk is bounded (`max_depth`), which is why the
 * Path field keeps a typed escape hatch and says so.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

/** One row - `api/v1/endpoints/models_local.py::_describe`, exactly. */
export interface LocalModelRow {
  readonly path: string;
  readonly root: string;
  readonly relative_path: string;
  /** `checkpoint` when the directory carries miForge's manifest, else `directory`. */
  readonly kind: 'checkpoint' | 'directory' | string;
  readonly architecture: string | null;
  readonly has_weights: boolean;
  readonly has_tokenizer: boolean;
  readonly config_path: string;
  readonly modified_at: string;
}

export interface LocalModelListing {
  readonly models: readonly LocalModelRow[];
  readonly total: number;
  readonly limit: number;
  readonly offset: number;
  readonly roots: readonly string[];
  readonly unreadable_roots: readonly {
    readonly root: string;
    readonly code: string;
    readonly reason: string;
  }[];
  readonly max_depth: number;
}

export const LOCAL_MODELS_PATH = `${V1}/models/local`;

export const fetchLocalModels = (
  query: { readonly hasWeights?: boolean; readonly limit?: number } = {},
): Promise<ApiResult<LocalModelListing>> =>
  apiGet<LocalModelListing>(LOCAL_MODELS_PATH, {
    query: { has_weights: query.hasWeights, limit: query.limit },
  });
