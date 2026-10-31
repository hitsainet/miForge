/**
 * One pack version on the registry screen (F7 task 12.4; FR-007.11, FR-007.16, FR-007.27).
 *
 * `7 parts · 640 prompts · used by 3 runs` - and **`prompt_count` may be `null`, which renders
 * as unknown and never as 0** (FR-007.11, ADR-018). Every count travels with its scale.
 *
 * Parts render through `PartReference`, which **takes a typed ref and never a pre-formatted
 * string**, so a component cannot render a part without handling both sources. The source
 * marker is indigo for miDataworks - the suite's reserved colour for it - and a neutral chip
 * otherwise; painting an external part in a reserved colour would say it was a sibling's.
 *
 * **The Example pill is rendered from `is_example`.** F7 shipped no example packs rather than
 * seed unmarked ones that look real, so this pill is expected to be absent in a fresh
 * installation; that is the honest state, not a missing feature.
 */
import type { ReactNode } from 'react';
import { Database } from 'lucide-react';
import { Card } from '@/components/common/Card';
import { IconTile } from '@/components/common/IconTile';
import { MonoId } from '@/components/common/MonoId';
import { Button } from '@/components/common/Button';
import { AbsentValue } from '@/components/common/AbsentValue';
import { NumberWithSample } from '@/components/common/NumberWithSample';
import { ValidityPill } from '@/components/common/ValidityPill';
import { PartReference } from '@/components/common/PartReference';
import type { SlotLineage } from '@/api/packs';
import { SLOT_LABELS, type PackListingRow, type PartRef, type SlotKind } from '@/types/packs';

export interface PackCardProps {
  readonly pack: PackListingRow;
  /** Per-slot provenance from `…/lineage`, when it has been read. */
  readonly lineage?: readonly SlotLineage[];
  readonly onShowLineage: (pack: PackListingRow) => void;
  readonly onExport: (pack: PackListingRow) => void;
}

export function PackCard({ pack, lineage, onShowLineage, onExport }: PackCardProps): ReactNode {
  return (
    <Card as="li" className="min-w-0" data-testid={`pack-${pack.pack_id}`}>
      <div className="flex min-w-0 items-start gap-3">
        {/* accent-ok:row-icon-tile - ADR-012 names "the icon tile on list rows". */}
        <IconTile icon={Database} surface="row-icon-tile" />
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <span className="font-semibold">{pack.name}</span>
            <span className="text-hint text-dim">version {pack.version}</span>
            <ValidityPill validity={pack.validity} data-testid={`validity-${pack.pack_id}`} />
            {pack.is_example ? (
              <span
                className="rounded-full bg-panel px-2 py-0.5 text-hint text-dim"
                data-testid={`example-${pack.pack_id}`}
              >
                Example
              </span>
            ) : null}
          </div>

          <div className="mt-1">
            <MonoId value={pack.pack_id} className="text-dim" />
          </div>

          <div className="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-hint">
            <NumberWithSample
              value={`${pack.parts_filled} of ${pack.parts_total}`}
              sample={pack.parts_total}
              sampleLabel="slots filled"
            />
            {pack.prompt_count === null ? (
              <AbsentValue
                reason="missing_required_key"
                detail="prompt_count could not be resolved"
                data-testid={`prompts-unknown-${pack.pack_id}`}
              />
            ) : (
              <NumberWithSample
                value={pack.prompt_count}
                sample={pack.prompt_count}
                sampleLabel="prompts"
              />
            )}
            <NumberWithSample
              value={pack.runs_using.length}
              sample={pack.runs_using.length}
              sampleLabel="runs using it"
            />
            {pack.domain === null ? (
              <AbsentValue reason="no_domain" />
            ) : (
              <span className="text-dim">domain {pack.domain}</span>
            )}
          </div>

          {pack.derived_from === null ? null : (
            <div className="mt-1 text-hint text-dim">
              Derived from <MonoId value={pack.derived_from} />
            </div>
          )}

          {lineage === undefined ? null : (
            <ul className="mt-3 stack-y-1" data-testid={`lineage-${pack.pack_id}`}>
              {lineage.map((slot) => (
                <li key={slot.slot} className="flex min-w-0 flex-wrap items-center gap-2 text-hint">
                  <span className="min-w-44 text-dim">
                    {SLOT_LABELS[slot.slot as SlotKind] ?? slot.slot}
                  </span>
                  {slot.reference === null ? (
                    <AbsentValue reason="missing_required_key" detail="this slot is empty" />
                  ) : (
                    <PartReference
                      reference={slot.reference as PartRef}
                      /* The pack a part is addressed THROUGH. A part may be addressed
                         through a different pack from the one holding the slot (FR-007.25),
                         so this is the holding pack's name only because the lineage endpoint
                         reports the slots of this pack. An unowned part gets `null` and no
                         segment is invented for it (F7 OQ-9). */
                      pack={pack.name}
                    />
                  )}
                  {slot.resolution === null ? null : (
                    <span className="text-dim">{slot.resolution}</span>
                  )}
                  {slot.floor === null ? null : (
                    <NumberWithSample
                      value={slot.floor.toFixed(3)}
                      sample={1}
                      sampleLabel="floor"
                    />
                  )}
                </li>
              ))}
            </ul>
          )}

          <div className="mt-3 flex flex-wrap gap-2">
            <Button
              onClick={() => onShowLineage(pack)}
              aria-expanded={lineage !== undefined}
              data-testid={`lineage-toggle-${pack.pack_id}`}
            >
              {lineage === undefined
                ? 'Show where every part came from'
                : 'Hide where every part came from'}
            </Button>
            <Button onClick={() => onExport(pack)}>Export this pack</Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
