/**
 * **The one way this UI asks the operator to name something that already exists** (F9 task
 * 14.0, D-14a; operator request 2026-10-18: *"Selection boxes should be dropdowns that list
 * what is available."*).
 *
 * Until then a run, a model directory, a GPU and a stop condition were typed - and a slot of
 * a pack draft could not be filled from the screen at all. Every one of those is now this
 * component over the route that lists them.
 *
 * ## The contract
 *
 * - **It submits the option's `value` and shows its `label`.** A pack is chosen by
 *   "notary @ 3" and submitted as `{pack_id, version}`; a card by its name and submitted as
 *   its UUID. `onChange` receives `value` and nothing else - the label never reaches a body.
 * - **Nothing is preselected** (FR-009.39). An empty `value` is "not chosen", the
 *   placeholder says so, and no first entry is silently owned by the operator.
 * - **An unavailable option is listed, disabled, with its reason** - never hidden and never
 *   swapped (ADR-010). An option missing from a list is indistinguishable from a broken one.
 * - **Loading, empty and failed are states of the component**, so a call site cannot forget
 *   one. Empty says what to do next; failed names the surface that could not be read.
 * - **A value the list no longer carries is shown as such**, never silently cleared: a
 *   restored draft whose pack was deleted says "not in the current list" and the backend's
 *   refusal decides.
 * - **A typed escape hatch only where the list cannot be complete** (`freeText`), and its
 *   reason is on screen and on the element (`data-free-text-reason`), which is what the
 *   ROUTES-derived declaration test reads.
 *
 * ## Accessible by construction
 *
 * The ARIA 1.2 combobox pattern: a text `input` with `role="combobox"`, `aria-expanded`,
 * `aria-controls` and `aria-activedescendant`, over a `role="listbox"` whose rows are
 * `role="option"` with `aria-selected` and `aria-disabled`. Keyboard: Down/Up move through the
 * enabled options, Home/End jump, Enter chooses, Escape closes and restores, typing filters.
 * The input is a real focusable element, so `index.css`'s focus ring applies (FR-009.7).
 *
 * **No dependency** (ADR-011): a combobox library is an ADR amendment, and this is one file.
 * No accent colour (ADR-012): the active row is `panel`, the chosen row is weighted, and
 * neither is orange.
 */
import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { Check } from 'lucide-react';
import {
  DROPDOWN_CLASSES,
  DROPDOWN_LIST_CLASSES,
  DropdownChevron,
  dropdownRowClasses,
} from './Field';
import { FailureNotice } from './FailureNotice';
import type { ApiFailure } from '@/api/errors';

export interface PickerOption {
  /** What is submitted. An identifier: a run id, a path, a UUID, a part id. */
  readonly value: string;
  /** What the operator reads. Never submitted. */
  readonly label: string;
  /** A second line: the version, the status, the size - what tells two rows apart. */
  readonly detail?: string;
  /** Present when the option is listed but cannot be chosen, saying why. */
  readonly disabledReason?: string;
  /** Extra words the search matches, beyond the label, value and detail. */
  readonly keywords?: string;
}

export type PickerListing =
  | { readonly kind: 'loading' }
  | { readonly kind: 'failed'; readonly failure: ApiFailure }
  | { readonly kind: 'ready'; readonly options: readonly PickerOption[] };

export interface PickerProps {
  /** From `Field`, so the label is associated with the control. */
  readonly id?: string;
  readonly describedBy?: string | undefined;
  readonly value: string;
  readonly onChange: (value: string) => void;
  readonly listing: PickerListing;
  /** What "nothing chosen" reads as. Required: see FR-009.39. */
  readonly placeholder: string;
  /** What an empty list means and what to do about it. One sentence. */
  readonly empty: string;
  /** What could not be read, for the failure notice: "the pack registry". */
  readonly surface: string;
  /**
   * The escape hatch: the operator may type a value the list does not carry. **Only where
   * the list cannot be complete**, and `reason` says why on screen.
   */
  readonly freeText?: { readonly reason: string };
  readonly disabled?: boolean;
  /**
   * Called when a row is **chosen** (click or Enter) - not when text is typed. For a control
   * whose choice is an action, like the Runs screen's "open a run": choosing opens it, while
   * a pasted id in the escape hatch waits for its own button.
   */
  readonly onChoose?: (value: string) => void;
  readonly 'data-testid'?: string;
}

const NO_OPTIONS: readonly PickerOption[] = [];

/** Every word of the query must appear somewhere in the option's text. */
export function matches(option: PickerOption, query: string): boolean {
  const haystack = [option.label, option.value, option.detail ?? '', option.keywords ?? '']
    .join(' ')
    .toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter((word) => word.length > 0)
    .every((word) => haystack.includes(word));
}

/**
 * The options matching `query`, **the ones whose label matches first**: typing a name puts
 * that name at the top, even when other rows mention it in their detail ("changes every
 * parameter" under each run type). Stable within each rank, so the list's own order holds.
 */
export function ranked(options: readonly PickerOption[], query: string): readonly PickerOption[] {
  const needle = query.trim().toLowerCase();
  const rank = (option: PickerOption): number => {
    const label = option.label.toLowerCase();
    if (needle.length === 0) return 0;
    if (label.startsWith(needle)) return 0;
    if (label.includes(needle)) return 1;
    return 2;
  };
  return options
    .filter((option) => matches(option, query))
    .map((option, index) => ({ option, index, rank: rank(option) }))
    .sort((a, b) => a.rank - b.rank || a.index - b.index)
    .map((entry) => entry.option);
}

export function Picker({
  id,
  describedBy,
  value,
  onChange,
  listing,
  placeholder,
  empty,
  surface,
  freeText,
  disabled = false,
  onChoose,
  'data-testid': testId,
}: PickerProps): ReactNode {
  const base = useId();
  const inputId = id ?? `${base}-input`;
  const listboxId = `${base}-listbox`;
  const optionId = (index: number): string => `${base}-option-${index}`;

  const [open, setOpen] = useState(false);
  /** What the operator has typed since opening. `null` = not searching. */
  const [query, setQuery] = useState<string | null>(null);
  const [active, setActive] = useState<number>(-1);
  const listRef = useRef<HTMLUListElement | null>(null);

  const options = useMemo(
    () => (listing.kind === 'ready' ? listing.options : NO_OPTIONS),
    [listing],
  );
  const selected = options.find((option) => option.value === value) ?? null;
  const shown = useMemo(
    () => (query === null ? options : ranked(options, query)),
    [options, query],
  );
  const enabledIndices = shown
    .map((option, index) => (option.disabledReason === undefined ? index : -1))
    .filter((index) => index >= 0);

  // Keep the active row in view when the keyboard moves it. `scrollIntoView` is absent in
  // jsdom, which is why it is guarded rather than assumed.
  useEffect(() => {
    if (!open || active < 0) return;
    const row = listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`);
    if (row && typeof row.scrollIntoView === 'function') row.scrollIntoView({ block: 'nearest' });
  }, [open, active]);

  const ready = listing.kind === 'ready';
  const unusable =
    disabled || (!ready && freeText === undefined) || (ready && options.length === 0 && !freeText);

  const close = (): void => {
    setOpen(false);
    setQuery(null);
    setActive(-1);
  };

  const choose = (option: PickerOption): void => {
    if (option.disabledReason !== undefined) return;
    // The VALUE, never the label. This line is what "the picker submits the id" means.
    onChange(option.value);
    onChoose?.(option.value);
    close();
  };

  const openList = (): void => {
    if (unusable) return;
    setOpen(true);
    const at = shown.findIndex((option) => option.value === value);
    setActive(at >= 0 ? at : (enabledIndices[0] ?? -1));
  };

  const move = (direction: 1 | -1): void => {
    if (!open) {
      openList();
      return;
    }
    if (enabledIndices.length === 0) return;
    const position = enabledIndices.indexOf(active);
    const next =
      position < 0
        ? direction === 1
          ? 0
          : enabledIndices.length - 1
        : (position + direction + enabledIndices.length) % enabledIndices.length;
    setActive(enabledIndices[next] ?? -1);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        move(1);
        break;
      case 'ArrowUp':
        event.preventDefault();
        move(-1);
        break;
      case 'Home':
        if (open) {
          event.preventDefault();
          setActive(enabledIndices[0] ?? -1);
        }
        break;
      case 'End':
        if (open) {
          event.preventDefault();
          setActive(enabledIndices[enabledIndices.length - 1] ?? -1);
        }
        break;
      case 'Enter': {
        if (!open) return;
        event.preventDefault();
        const option = shown[active];
        if (option !== undefined) choose(option);
        else close();
        break;
      }
      case 'Escape':
        if (open) {
          event.preventDefault();
          close();
        }
        break;
      case 'Tab':
        close();
        break;
      default:
        break;
    }
  };

  /**
   * What the input shows. Searching: the query. Free text: the value itself, which is what
   * is submitted. Otherwise the chosen option's label - or, for a value no longer listed,
   * the value, so it is visible rather than silently dropped.
   */
  const text =
    freeText !== undefined
      ? value
      : query !== null
        ? query
        : selected !== null
          ? selected.label
          : value;

  const statusLine =
    listing.kind === 'loading'
      ? `Reading ${surface}…`
      : listing.kind === 'ready' && options.length === 0
        ? empty
        : value.length > 0 && ready && selected === null
          ? freeText === undefined
            ? `${value} is not in the current list. The backend will say whether it still exists.`
            : 'Typed, not chosen from the list.'
          : selected?.detail;

  return (
    <div className="relative" data-testid={testId === undefined ? undefined : `${testId}-picker`}>
      {/* The input and its chevron share a box of their own, so the chevron centres on the
          field and not on the status lines below it. */}
      <div className="relative">
        <input
          id={inputId}
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-controls={listboxId}
          aria-autocomplete="list"
          aria-activedescendant={open && active >= 0 ? optionId(active) : undefined}
          aria-describedby={describedBy}
          autoComplete="off"
          spellCheck={false}
          className={`${DROPDOWN_CLASSES} ${unusable ? 'cursor-not-allowed opacity-60' : ''}`.trim()}
          placeholder={listing.kind === 'loading' ? `Reading ${surface}…` : placeholder}
          value={text}
          disabled={unusable}
          data-testid={testId}
          data-picker={freeText === undefined ? 'strict' : 'free'}
          data-value={value}
          {...(freeText === undefined ? {} : { 'data-free-text-reason': freeText.reason })}
          onClick={() => (open ? undefined : openList())}
          onKeyDown={onKeyDown}
          onChange={(event) => {
            const typed = event.target.value;
            if (freeText !== undefined) onChange(typed);
            setQuery(typed);
            setOpen(true);
            setActive(-1);
          }}
          onBlur={close}
        />
        <DropdownChevron open={open} />
      </div>

      {open ? (
        <ul
          id={listboxId}
          ref={listRef}
          role="listbox"
          aria-label={placeholder}
          className={`absolute z-20 mt-1 max-h-72 w-full overflow-auto py-1 ${DROPDOWN_LIST_CLASSES}`}
          data-testid={testId === undefined ? undefined : `${testId}-listbox`}
        >
          {shown.length === 0 ? (
            <li className="px-3 py-2 text-hint text-dim" role="presentation">
              {freeText === undefined
                ? 'Nothing in the list matches what was typed.'
                : 'Nothing in the list matches. The typed value is what will be submitted.'}
            </li>
          ) : (
            shown.map((option, index) => {
              const isSelected = option.value === value;
              const isDisabled = option.disabledReason !== undefined;
              return (
                <li
                  key={option.value}
                  id={optionId(index)}
                  role="option"
                  aria-selected={isSelected}
                  aria-disabled={isDisabled || undefined}
                  data-index={index}
                  data-value={option.value}
                  className={[
                    'cursor-pointer px-3 py-2',
                    dropdownRowClasses({ selected: isSelected, active: index === active }),
                    isDisabled ? 'cursor-not-allowed text-dim' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  // mousedown, not click: a click lands after the input's blur has closed
                  // the list, so the choice would never arrive.
                  onMouseDown={(event) => {
                    event.preventDefault();
                    choose(option);
                  }}
                  onMouseEnter={() => (isDisabled ? undefined : setActive(index))}
                >
                  {/* The chosen row is filled and ticked: a weight change alone is too quiet to
                      find in a long list (operator, 2026-10-31). */}
                  <div className="flex items-center gap-2">
                    <span className="min-w-0 flex-1">{option.label}</span>
                    {isSelected ? (
                      <>
                        <Check size={14} aria-hidden="true" className="shrink-0" />
                        <span className="sr-only"> (chosen)</span>
                      </>
                    ) : null}
                  </div>
                  {option.detail === undefined && option.disabledReason === undefined ? null : (
                    <div className={`text-hint ${isDisabled ? 'text-amber' : 'text-dim'}`}>
                      {option.disabledReason === undefined
                        ? option.detail
                        : `Unavailable: ${option.disabledReason}`}
                    </div>
                  )}
                </li>
              );
            })
          )}
        </ul>
      ) : null}

      {statusLine === undefined ? null : (
        <div
          className="mt-1 text-hint text-dim"
          data-testid={testId === undefined ? undefined : `${testId}-status`}
        >
          {statusLine}
        </div>
      )}
      {freeText === undefined ? null : (
        <div
          className="mt-1 text-hint text-dim"
          data-testid={testId === undefined ? undefined : `${testId}-free-text`}
        >
          Not listed? Type it. {freeText.reason}
        </div>
      )}
      {listing.kind === 'failed' ? (
        <FailureNotice failure={listing.failure} surface={surface} className="mt-2" />
      ) : null}
    </div>
  );
}
