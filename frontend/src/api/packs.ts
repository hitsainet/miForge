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

export interface PartRow {
  readonly part_id: string;
  readonly part_kind: string;
  readonly contract_reference: Readonly<Record<string, string>> | null;
  readonly rendered?: string | null;
  readonly caveats?: readonly Readonly<Record<string, unknown>>[];
  readonly licence_classes?: readonly string[];
  readonly [key: string]: unknown;
}

export const fetchParts = (
  options: { readonly partKind?: string } = {},
): Promise<ApiResult<{ readonly items: readonly PartRow[]; readonly total: number }>> =>
  apiGet<{ readonly items: readonly PartRow[]; readonly total: number }>(`${V1}/parts`, {
    query: { part_kind: options.partKind },
  });

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
}

export const fetchPinnableReference = (partId: string): Promise<ApiResult<PinnableReference>> =>
  apiGet<PinnableReference>(`${V1}/pack-part-refs/${encodeURIComponent(partId)}`);
