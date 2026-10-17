/**
 * One part reference, rendered from a **typed ref and never a pre-formatted string**
 * (F7 task 12.4; FR-007.24, FR-007.25, FR-007.27).
 *
 * ## Two rendering layers, which looked like a conflict and was not
 *
 * `<type>/<name>@<version>` is **miDataworks' contract form**. `<pack>/<type>/<name>@<version>`
 * is **miForge prefixing its own pack identity for display**. Two layers, settled 2026-10-12.
 * So this component takes the prefixing pack as a separate prop: a part may be addressed
 * **through a different pack** from the one holding the slot (FR-007.25), which is why
 * `GET /api/v1/pack-part-refs/{id}` deliberately returns `inner_form` and no `display_form` -
 * it does not know the pack and inventing one would be the layer confusion R-02.35 forbids.
 *
 * ## Why the prop is a union and the switch is exhaustive
 *
 * `PartRef` is discriminated on `source`, matching `backend/src/packs/references.py`, which
 * has **no** `promote()` and no `as_contract_reference()` because R-02.40 forbids miForge
 * claiming miDataworks provenance for anything miDataworks did not publish. A component that
 * read `type` off an unnarrowed ref would render a contract citation for an external part -
 * the exact borrowing those two separate Pydantic models exist to prevent.
 *
 * ## The unowned part - **settled 2026-10-14**
 *
 * **`pack` may be `null`, and then no pack segment is invented.** F7's OQ-9 - the mockup's
 * `general/retention/mixed@2`, where `general` is not one of its packs - is **answered: there
 * is no unowned-parts namespace**, and the mockup is inconsistent (`general` there is the
 * retention set's *name*). The inner form is shown with the prefix named as unknown, which is
 * what the backend's `rendering.unowned_form()` renders.
 *
 * The backend's `display_form()` still **refuses** a null pack, and this component is not that
 * path: the refusal guards against a pack *lost* in nullable storage, and only a caller that
 * names the unowned case - this component, via `pack={null}` - gets a rendering.
 */
import type { ReactNode } from 'react';
import type { PartRef } from '@/types/packs';
import { partRefSource } from '@/types/packs';
import { MonoId } from './MonoId';

/** miDataworks' own citation form: `<type>/<name>@<version>`. */
export function contractForm(ref: {
  readonly type: string;
  readonly name: string;
  readonly version: string;
}): string {
  return `${ref.type}/${ref.name}@${ref.version}`;
}

/**
 * An external part's inner form: `external/<origin>@<first 12 of the content hash>`.
 *
 * **The `external/` marker is load-bearing and was missing here until 2026-10-14.** It sits in
 * the position a contract reference carries its `type`, so the two can never be read as the
 * same shape (FR-007.24) - and `backend/src/packs/rendering.py::external_form` has always
 * carried it, so the two layers disagreed on the string an operator sees for one reference.
 *
 * Without it, an external part whose origin happens to contain a slash renders
 * `retention/mixed@aaaaaaaaaaaa` - which is exactly `<type>/<name>@<version>`, and with the
 * pack prefix applied becomes `mypack/retention/mixed@...`, indistinguishable from a contract
 * citation addressed through a pack. The source chip beside it says otherwise, but the chip is
 * not what an operator copies into a bug report. *The test that was meant to catch this
 * asserted `not.toContain('@3')` against a fixture whose origin was an `https://` URL, so it
 * could not fail against the defect it described - which is why the fixture here is now
 * contract-shaped.*
 */
export function externalForm(ref: {
  readonly origin: string;
  readonly content_sha256: string;
}): string {
  return `external/${ref.origin}@${ref.content_sha256.slice(0, 12)}`;
}

export function innerForm(ref: PartRef): string | null {
  switch (ref.source) {
    case 'midataworks':
      return contractForm(ref);
    case 'external':
      return externalForm(ref);
  }
}

export interface PartReferenceProps {
  /**
   * The typed reference, **never a pre-formatted string**.
   *
   * Named `reference` and **not `ref`**, which React reserves: a prop called `ref` is
   * intercepted by the renderer and never reaches the component, so the first render of a
   * `ref`-propped version read `undefined.source` and threw. Found by the test rather than by
   * reading - which is the point of writing the test before trusting the component.
   */
  readonly reference: PartRef;
  /**
   * The pack this part is addressed **through**, for the display prefix. `null` when it is
   * not known - see the module docstring; no segment is invented for it.
   */
  readonly pack: string | null;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

export function PartReference({
  reference,
  pack,
  className = '',
  ...rest
}: PartReferenceProps): ReactNode {
  const source = partRefSource(reference);
  if (source === null) {
    // A ref whose `source` this build does not admit. Named, not guessed at: picking either
    // branch would read one source's fields off the other's document.
    return (
      <span className={`text-amber ${className}`.trim()} data-part-source="unrecognised" {...rest}>
        A part reference whose source this build does not recognise. It is shown unrendered rather
        than read as either a miDataworks citation or an external provenance record.
      </span>
    );
  }

  const inner = innerForm(reference);
  const display = pack === null ? inner : `${pack}/${inner}`;

  return (
    <span
      className={`inline-flex min-w-0 items-center gap-2 ${className}`.trim()}
      data-part-source={source}
      {...rest}
    >
      <SourceMarker source={source} />
      <MonoId value={display ?? ''} />
      {pack === null ? (
        <span className="text-hint text-dim" title={UNOWNED_PART_SENTENCE}>
          addressed through no pack
        </span>
      ) : null}
    </span>
  );
}

export const UNOWNED_PART_SENTENCE =
  'This part is not addressed through any pack, so there is no pack segment to prefix. ' +
  'miForge has no built-in domain and invents no namespace for an unowned part; the ' +
  'contract or provenance form above is the whole of what is known about it.';

/**
 * Where a part came from.
 *
 * **Indigo for miDataworks**, which is the suite's reserved colour for it (ADR-012), and a
 * neutral chip for anything external - because external parts are not a sibling's state and
 * painting them in a reserved colour would say they were. Indigo has no token in
 * `config/tokens.ts` because nothing else in the shell renders miDataworks state, so it is
 * written as a Tailwind indigo utility here and named in the comment rather than added to the
 * palette for one chip.
 */
export function SourceMarker({
  source,
}: {
  readonly source: 'midataworks' | 'external';
}): ReactNode {
  if (source === 'midataworks') {
    return (
      <span
        className="rounded-full bg-indigo-500/15 px-2 py-0.5 text-hint text-indigo-400"
        data-source-marker="midataworks"
        title="A miDataworks dataset version, cited by its contract reference (BRD-03 R-03.47)."
      >
        miDataworks
      </span>
    );
  }
  return (
    <span
      className="rounded-full bg-panel px-2 py-0.5 text-hint text-dim"
      data-source-marker="external"
      title="A part from outside the suite, carrying its own provenance: an origin, a content hash and when it was taken in (R-02.40)."
    >
      External
    </span>
  );
}
