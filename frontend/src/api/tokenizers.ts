/**
 * The tokenizer a create run ships (operator request 2026-10-20).
 *
 * A run that creates a model is given a tokenizer; miForge never trains one. Two read-only
 * routes back the New run form's tokenizer fields:
 *
 * - `GET /api/v1/tokenizers/local` - directories under the model roots with tokenizer files at
 *   their root: a tokenizer-only directory, a model directory or a checkpoint.
 * - `GET /api/v1/hub/revision` - a Hub repository's revision resolved to the commit it points
 *   at now, with what that commit holds at its root. Nothing is downloaded.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

/** One row - `api/v1/endpoints/tokenizers.py::_describe`, exactly. */
export interface LocalTokenizerRow {
  readonly path: string;
  readonly root: string;
  readonly relative_path: string;
  /** `tokenizer` (no config.json), `model`, or `checkpoint` (miForge's manifest). */
  readonly kind: 'tokenizer' | 'model' | 'checkpoint' | string;
  readonly tokenizer_files: readonly string[];
  readonly modified_at: string;
}

export interface LocalTokenizerListing {
  readonly tokenizers: readonly LocalTokenizerRow[];
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

/** `api/v1/endpoints/tokenizers.py::check_hub_revision`'s answer. */
export interface HubRevisionCheck {
  readonly repo: string;
  readonly revision: string;
  readonly commit: string;
  /** The revision asked for was already the commit, so it cannot move. */
  readonly is_commit: boolean;
  readonly tokenizer_files: readonly string[];
  readonly has_tokenizer: boolean;
  readonly has_config: boolean;
  readonly has_weights: boolean;
}

export const LOCAL_TOKENIZERS_PATH = `${V1}/tokenizers/local`;
export const HUB_REVISION_PATH = `${V1}/hub/revision`;

export const fetchLocalTokenizers = (
  query: { readonly limit?: number } = {},
): Promise<ApiResult<LocalTokenizerListing>> =>
  apiGet<LocalTokenizerListing>(LOCAL_TOKENIZERS_PATH, { query: { limit: query.limit } });

/** An empty revision is not sent: the route resolves `main`, and says so in its answer. */
export const checkHubRevision = (
  repo: string,
  revision: string,
): Promise<ApiResult<HubRevisionCheck>> =>
  apiGet<HubRevisionCheck>(HUB_REVISION_PATH, {
    query: { repo, revision: revision.length === 0 ? undefined : revision },
  });
