/**
 * F7's pack registry and part surface (F7 task 12.1; FR-007.11, .24, .27).
 *
 * Every response type here is transcribed from the producer - `packs/registry.py`,
 * `packs/validation.py`, `packs/service.py` - and the one that matters is `PartRef`, which
 * is a **discriminated union on this side too** (`types/packs.ts`). A flattened client type
 * would let a component read `type` off an external part and render miDataworks provenance
 * for something miDataworks never published, which is what R-02.40 forbids and what the
 * backend spent two Pydantic models on preventing.
 */
import { apiDelete, apiGet, apiPost, type ApiResult } from './client';
import { V1 } from './v1';
import type { PackListing, PackListingRow, PackValidity, PartRef } from '@/types/packs';

export const PACKS_PATH = `${V1}/packs`;

export const fetchPacks = (): Promise<ApiResult<PackListing>> => apiGet<PackListing>(PACKS_PATH);

const packVersionPath = (packId: string, version: string): string =>
  `${PACKS_PATH}/${encodeURIComponent(packId)}/versions/${encodeURIComponent(version)}`;

export const fetchPack = (packId: string, version: string): Promise<ApiResult<PackListingRow>> =>
  apiGet<PackListingRow>(packVersionPath(packId, version));

export const fetchPackValidity = (
  packId: string,
  version: string,
): Promise<ApiResult<PackValidity>> =>
  apiGet<PackValidity>(`${packVersionPath(packId, version)}/validity`);

export interface SlotLineage {
  readonly slot: string;
  readonly source: string | null;
  readonly reference: PartRef | null;
  readonly resolution: string | null;
  readonly reason: string | null;
  readonly floor: number | null;
  readonly [key: string]: unknown;
}

export const fetchPackLineage = (
  packId: string,
  version: string,
): Promise<ApiResult<{ readonly slots: readonly SlotLineage[] }>> =>
  apiGet<{ readonly slots: readonly SlotLineage[] }>(`${packVersionPath(packId, version)}/lineage`);

export const clonePack = (
  packId: string,
  version: string,
  body: { readonly into: string; readonly version?: string },
): Promise<ApiResult<PackListingRow>> =>
  apiPost<PackListingRow>(`${packVersionPath(packId, version)}/clone`, body);

export const deletePack = (packId: string, version: string): Promise<ApiResult<unknown>> =>
  apiDelete<unknown>(packVersionPath(packId, version));

/** The pack document **by value**, as the canonical bytes `definition_sha256` covers. */
export const packExportUrl = (packId: string, version: string): string =>
  `/api${packVersionPath(packId, version)}/export`;

/**
 * Import an exported pack document.
 *
 * The backend **never rewrites a reference to make it resolve** (its E8): an imported pack
 * whose parts this installation cannot resolve is stored and *reported* unresolvable, because
 * rewriting would make it a different pack. The dialog therefore renders the unresolved list
 * rather than treating a 201 as "ready".
 */
export const importPack = (document: unknown): Promise<ApiResult<PackListingRow>> =>
  apiPost<PackListingRow>(`${PACKS_PATH}/import`, document);

export interface CreatePackBody {
  readonly name: string;
  readonly version?: string;
  readonly domain?: string | null;
  readonly slots?: Readonly<Record<string, unknown>>;
}

export const createPack = (body: CreatePackBody): Promise<ApiResult<PackListingRow>> =>
  apiPost<PackListingRow>(PACKS_PATH, body);

/* ------------------------------------------------------------------------------------ *
 * Parts
 * ------------------------------------------------------------------------------------ */

/**
 * One row of `GET /api/v1/parts` - `services/parts/index.py::part_payload`.
 *
 * **`id`, not `part_id`**: this type said `part_id` while the route has always answered `id`.
 * Nothing read it until task 14.0's slot picker, which is why the disagreement survived.
 */
export interface PartRow {
  readonly id: string;
  /**
   * The **contract** spelling (`batch-record-v1.json`'s `PartReference.kind`), which is
   * exactly a pack slot's name - `packs/slots.py` asserts the two agree at import - so a slot
   * filters this listing by its own name with no translation table on this side.
   */
  readonly part_kind: string;
  readonly source: string;
  /** `null` when the version carries none: such a part cannot pin a slot (`part_unpinnable`). */
  readonly contract_reference: Readonly<Record<string, string>> | null;
  readonly rendered: string | null;
  readonly dataset: string | null;
  readonly version_number: number | null;
  readonly row_count: number | null;
  readonly caveats: readonly unknown[];
  readonly licence_classes: readonly string[];
  readonly imported_at: string | null;
  /** When the part was archived, or `null` while listed (operator decision 2026-10-20). */
  readonly archived_at?: string | null;
  readonly [key: string]: unknown;
}

export const PARTS_PATH = `${V1}/parts`;

/**
 * `PartRow.source` - the INDEX's spelling (`services/parts/tables.py`: `SOURCE_CONTRACT`,
 * `SOURCE_EXTERNAL`), which is **not** a pack reference's `midataworks` / `external`
 * (`PART_REF_SOURCES`). The slot picker compared a row against the reference spelling until
 * 2026-10-20, so on a real backend no reference-less part was ever disabled and every
 * external part was labelled "from miDataworks"; the stub answered the reference spelling,
 * which is why no test saw it.
 */
export const PART_SOURCE_CONTRACT = 'midataworks_contract';
export const PART_SOURCE_EXTERNAL = 'external_files';

/**
 * The part index. **Archived parts are left out** unless `includeArchived` - the route's
 * default, so a picker lists what is available without a filter of its own.
 */
export const fetchParts = (
  options: { readonly partKind?: string; readonly includeArchived?: boolean } = {},
): Promise<ApiResult<{ readonly items: readonly PartRow[]; readonly total: number }>> =>
  apiGet<{ readonly items: readonly PartRow[]; readonly total: number }>(`${V1}/parts`, {
    query: {
      part_kind: options.partKind,
      include_archived: options.includeArchived === true ? true : undefined,
    },
  });

/** What `POST /parts/{id}/archive` and `/unarchive` answer (`services/archive.py`). */
export interface PartArchiveOutcome {
  readonly part_id: string;
  readonly archived: boolean;
  readonly archived_at: string | null;
  readonly changed: boolean;
}

const PART_PATH = (partId: string): string => `${PARTS_PATH}/${encodeURIComponent(partId)}`;

/** Hide a part from the default lists and pickers. Deletes nothing; a pinned pack is untouched. */
export const archivePart = (partId: string): Promise<ApiResult<PartArchiveOutcome>> =>
  apiPost<PartArchiveOutcome>(`${PART_PATH(partId)}/archive`);

/** Return an archived part to the default lists. */
export const unarchivePart = (partId: string): Promise<ApiResult<PartArchiveOutcome>> =>
  apiPost<PartArchiveOutcome>(`${PART_PATH(partId)}/unarchive`);

/**
 * The pinnable reference for an indexed part, **discriminated on `source`**.
 *
 * `display_form` is deliberately absent from this response: a part may be addressed through
 * a *different* pack from the one holding the slot (FR-007.25), so the prefixing pack is not
 * known at that route and inventing one would be the layer confusion R-02.35 forbids. The
 * renderer takes the pack as an argument - see `PartReference.tsx`.
 */
export interface PinnableReference {
  readonly part_id: string;
  readonly ref: PartRef;
  readonly source: string;
  readonly inner_form: string;
  /**
   * The slot document this part fills, from the backend's one writer
   * (`packs/definition.py::slot_document`), with `floor: null`. The authoring screen sends it
   * unchanged (task 14.0) rather than converting `ref` itself - a second converter here would
   * agree with that one until one of them changed.
   */
  readonly slot_document: SlotDocument;
}

/** `pack-definition-v1.json`'s `PackSlot`: exactly one of the two sources, and a floor. */
export interface SlotDocument {
  readonly contract_reference: Readonly<Record<string, string>> | null;
  readonly external: Readonly<Record<string, unknown>> | null;
  readonly floor: number | null;
}

export const fetchPinnableReference = (partId: string): Promise<ApiResult<PinnableReference>> =>
  apiGet<PinnableReference>(`${V1}/pack-part-refs/${encodeURIComponent(partId)}`);
