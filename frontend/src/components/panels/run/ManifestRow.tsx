/**
 * One manifest key (FR-009.17 rules 1-3).
 *
 * Three statuses and they are three different facts:
 *
 * - `schema` - a key the schema describes. Its value may be **`null`**, which is an
 *   **assertion**: `domain` is required *and* nullable, so `null` says *this artifact is not
 *   domain-scoped*.
 * - `unrecognised` - a key the schema does not describe. **Rendered, and named.** Both root
 *   objects set `additionalProperties: false`, so such a key makes the document nonconforming;
 *   dropping it from the view is mutation **M13** and is how a nonconforming manifest comes to
 *   look clean.
 * - `missing_required` - a required key the document does **not** carry. An **omission**, and
 *   presented with no value at all, because there is nothing to show. Collapsing this into
 *   present-and-null is the conflation rule 3 forbids.
 *
 * Identifiers and hashes render in JetBrains Mono (ADR-012).
 */
import type { ReactNode } from 'react';
import { MonoId } from '@/components/common/MonoId';
import { AbsentValue } from '@/components/common/AbsentValue';
import type { ManifestRowModel } from '@/types/manifest';

export interface ManifestRowProps {
  readonly row: ManifestRowModel;
  readonly depth?: number;
}

/** A scalar as the view prints it. `null` prints the word, which is the assertion. */
export function renderValue(value: unknown): ReactNode {
  if (value === null) return <span className="text-dim">null</span>;
  if (typeof value === 'string') return <MonoId value={value} />;
  if (typeof value === 'number' || typeof value === 'boolean') {
    return <MonoId value={String(value)} />;
  }
  if (Array.isArray(value)) {
    return value.length === 0 ? (
      <span className="text-dim">[] (an empty list, which is a value and not an absence)</span>
    ) : (
      <MonoId value={JSON.stringify(value)} />
    );
  }
  return <MonoId value={JSON.stringify(value)} />;
}

export function ManifestRow({ row, depth = 0 }: ManifestRowProps): ReactNode {
  return (
    <>
      <div
        className="flex flex-wrap items-baseline gap-2 border-b border-line py-1.5 last:border-b-0"
        style={{ paddingLeft: depth * 16 }}
        data-manifest-key={row.key}
        data-manifest-status={row.status}
      >
        <span className="min-w-56 font-mono text-ident text-dim">{row.key}</span>
        {row.status === 'missing_required' ? (
          <AbsentValue reason="missing_required_key" />
        ) : (
          <span className="min-w-0 break-all">{renderValue(row.value)}</span>
        )}
        {row.status === 'unrecognised' ? (
          <span
            className="text-hint text-amber"
            title={
              'This schema sets additionalProperties: false, so a key it does not describe ' +
              'makes the document nonconforming. The manifest is authoritative, so ' +
              'the key is shown rather than dropped.'
            }
          >
            unrecognised key — this artifact may not conform
          </span>
        ) : null}
      </div>
      {(row.nested ?? []).map((nested) => (
        <ManifestRow key={`${row.key}.${nested.key}`} row={nested} depth={depth + 1} />
      ))}
    </>
  );
}
