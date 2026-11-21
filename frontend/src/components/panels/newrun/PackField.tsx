/**
 * The pack, with every part through F7's `PartReference` (FR-009.33; F7 OQ-9).
 *
 * The listing comes from `GET /api/v1/packs`, which carries validity, the filled/total count,
 * the prompt count and the derivation in **one** call - *"if a UI has to make a second call to
 * show a part's caveats or its contract reference, the shape is wrong."*
 *
 * ## The unowned part - **settled 2026-10-14: there is no unowned-parts namespace**
 *
 * F7's **OQ-9** - the mockup's `general/retention/mixed@2`, where `general` is not one of its
 * packs - was *"blocked behind"* this form, and this form is where an operator meets it. The
 * answer is that the mockup is inconsistent: `general` there is the retention set's *name*,
 * read as a pack segment. So a part addressed through no pack renders its **inner form alone**
 * with the prefix named as unknown (`PartReference`'s `pack={null}` path, F7's
 * `rendering.unowned_form`). **F9 invents no namespace** - *"miForge has no built-in domain"*
 * is a banner this product renders.
 *
 * F7's `display_form()` still **refuses** a null pack, and that is not what this screen shows:
 * the refusal is the backend's guard against a pack that was *lost* in nullable storage, which
 * must stay distinguishable from a part that genuinely has none.
 *
 * ## `prompt_count` may be `null` and renders as unknown, never 0
 *
 * FR-007.11 and ADR-018: a count that could not be resolved is not a count of nothing.
 */
import type { ReactNode } from 'react';
import { Field } from '@/components/common/Field';
import { Picker, type PickerOption } from '@/components/common/Picker';
import { useListing } from '@/hooks/useListing';
import { Card } from '@/components/common/Card';
import { AbsentValue } from '@/components/common/AbsentValue';
import { MonoId } from '@/components/common/MonoId';
import { ValidityPill } from '@/components/common/ValidityPill';
import { NumberWithSample } from '@/components/common/NumberWithSample';
import { fetchPacks } from '@/api/packs';
import type { PackListing, PackListingRow } from '@/types/packs';

export interface PackFieldProps {
  readonly packId: string;
  readonly version: string;
  readonly onChange: (packId: string, version: string) => void;
  readonly load?: typeof fetchPacks;
}

/** `pack_id` and `version` travel together: a pack without its version is not a pin. */
export function packKey(row: PackListingRow): string {
  return `${row.pack_id}@${row.version}`;
}

export function splitPackKey(key: string): { readonly packId: string; readonly version: string } {
  const at = key.lastIndexOf('@');
  if (at < 0) return { packId: key, version: '' };
  return { packId: key.slice(0, at), version: key.slice(at + 1) };
}

/**
 * Every pack version as an option: **"name @ version"** to read, `pack_id@version` to submit.
 *
 * An invalid pack is **listed and choosable**, with what is wrong in its detail: this form
 * validates nothing (FR-009.39), and the backend's `require_valid_pack` refuses it by name at
 * check and at submit. Hiding it would make a pack that exists look like one that does not.
 */
export function packOptions(listing: PackListing): readonly PickerOption[] {
  return listing.packs.map((row) => {
    const missing = row.validity.missing_slots;
    const state =
      row.validity.state === 'valid'
        ? 'valid'
        : missing.length > 0
          ? `${row.validity.state}: missing ${missing.join(', ')}`
          : row.validity.state;
    return {
      value: packKey(row),
      label: `${row.name} @ ${row.version}`,
      detail: [
        state,
        `${row.parts_filled} of ${row.parts_total} slots filled`,
        row.domain ?? 'no domain',
        row.is_example ? 'example' : null,
      ]
        .filter((part): part is string => part !== null)
        .join(' · '),
      keywords: row.pack_id,
    };
  });
}

export function PackField({
  packId,
  version,
  onChange,
  load = fetchPacks,
}: PackFieldProps): ReactNode {
  const { value: listing, listing: pickerListing } = useListing(load, packOptions);
  const packs = listing?.packs ?? null;

  const selectedKey = packId.length === 0 ? '' : `${packId}@${version}`;
  const selected = (packs ?? []).find((row) => packKey(row) === selectedKey) ?? null;

  return (
    <>
      <Field
        label="Pack"
        hint="A pack is everything miForge needs to know about a domain: the prompt set, optionally a corpus and preference pairs, a selector and grader configuration, a test set, and a retention set with its floor."
      >
        {({ id, describedBy }) => (
          <Picker
            id={id}
            describedBy={describedBy}
            value={selectedKey}
            onChange={(key) => {
              const parts = splitPackKey(key);
              onChange(parts.packId, parts.version);
            }}
            listing={pickerListing}
            placeholder="Choose a pack version"
            empty="No pack is registered. Author one on Onboard a domain, or import one on Domain packs."
            surface="the pack list"
            data-testid="pack-select"
          />
        )}
      </Field>

      {selected === null ? null : (
        <Card className="mb-4" data-testid="pack-summary">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <MonoId value={selected.pack_id} />
            <ValidityPill validity={selected.validity} />
            {selected.is_example ? (
              <span className="rounded-full bg-panel px-2 py-0.5 text-hint text-dim">Example</span>
            ) : null}
          </div>
          <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-hint sm:grid-cols-2">
            <div>
              <dt className="inline text-dim">parts: </dt>
              <dd className="inline">
                <NumberWithSample
                  value={`${selected.parts_filled} of ${selected.parts_total}`}
                  sample={selected.parts_total}
                  sampleLabel="slots"
                />
              </dd>
            </div>
            <div>
              <dt className="inline text-dim">prompts: </dt>
              <dd className="inline">
                {selected.prompt_count === null ? (
                  /* `null` is unknown, never 0 (FR-007.11, ADR-018). */
                  <AbsentValue
                    reason="missing_required_key"
                    detail="the prompt count could not be read"
                  />
                ) : (
                  <NumberWithSample
                    value={selected.prompt_count}
                    sample={selected.prompt_count}
                    sampleLabel="prompts"
                  />
                )}
              </dd>
            </div>
            <div>
              <dt className="inline text-dim">domain: </dt>
              <dd className="inline">
                {selected.domain === null ? <AbsentValue reason="no_domain" /> : selected.domain}
              </dd>
            </div>
            <div>
              <dt className="inline text-dim">used by: </dt>
              <dd className="inline">
                <NumberWithSample
                  value={selected.runs_using.length}
                  sample={selected.runs_using.length}
                  sampleLabel="runs"
                />
              </dd>
            </div>
          </dl>
          {selected.validity.retention_floor_missing ? (
            <div className="mt-2">
              <AbsentValue reason="no_floor_recorded" variant="block" />
            </div>
          ) : null}
        </Card>
      )}
    </>
  );
}
