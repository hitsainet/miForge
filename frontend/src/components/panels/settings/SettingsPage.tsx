/**
 * The Settings screen (F12; FR-012.7, .8; operator request 2026-10-20).
 *
 * The suite's sidebars all end the same way - **Settings, the Manual, the version** - and this
 * is the screen that footer's Settings entry opens. It is the operator's view of
 * `GET /api/v1/settings`: every setting with its effective value and where it came from.
 *
 * - **Changeable while miForge runs** - the backend's `RUNTIME_KEYS`. Each row can save an
 *   override (`PUT`) and, when one is set, return to the environment's value (`DELETE`). The
 *   backend validates the value by the field's own type; its refusal is shown on the row.
 * - **From the environment** - everything else, read-only. ADR-020's configuration stays
 *   environment-sourced; secrets arrive redacted and are shown as the server sent them.
 *
 * **An unsaved value survives a reload** like every other entry (AC-009.31): the drafts are
 * one map under `DRAFT_KEYS.settings`, and a row's draft is dropped once it is saved or reset.
 *
 * The control for a row follows the value's type in the payload, so the screen holds no copy
 * of the backend's field list: a boolean is on/off, a number is typed, an address is typed.
 * The one enum is the log level, whose four values the backend's `Literal` fixes.
 *
 * **What each setting does** (ENH-003 section 3.4, 2026-10-20): every row shows the backend's
 * plain-language label and description (`core/settings_catalog.py`), **when a change takes
 * effect**, the environment's value beside the one in force, and the last change - who made it,
 * and the approval when an agent's approved request did. A change may carry a reason, kept in
 * the append-only history the "Recent changes" list reads (`SettingsHistory`).
 *
 * **Grouped, then alphabetical** (2026-10-28): both lists are split into the backend's
 * operator-facing groups (General, Storage, ...) in the order it serves them, and sorted by
 * label within each group (`settingsGroups.ts`).
 */
import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { FailureNotice } from '@/components/common/FailureNotice';
import { Input, Select } from '@/components/common/Field';
import { MonoId } from '@/components/common/MonoId';
import { Pill, type PillTone } from '@/components/common/Pill';
import { formatEt } from '@/components/common/time';
import type { ApiFailure } from '@/api/errors';
import { usePersistentState } from '@/hooks/usePersistentState';
import { DRAFT_KEYS } from '@/stores/draftStorage';
import {
  readSettings,
  resetSetting,
  writeSetting,
  type SettingEffectWhen,
  type SettingRow,
  type SettingsListing,
} from '@/api/settings';
import { SettingsHistory } from './SettingsHistory';
import { showActor, showSetting } from './settingsText';
import { groupSettings } from './settingsGroups';

export { showActor, showSetting };

export const SETTINGS_TITLE = 'Settings';
export const SETTINGS_SUBTITLE =
  "Where miForge's configuration comes from, and the few settings you can change while it runs";

/** `core.config.Settings.log_level`'s `Literal`. The backend refuses anything else. */
export const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR'] as const;

type Editor = 'boolean' | 'log-level' | 'number' | 'address';

function editorFor(key: string, value: unknown): Editor {
  if (key === 'log_level') return 'log-level';
  if (typeof value === 'boolean') return 'boolean';
  if (typeof value === 'number') return 'number';
  return 'address';
}

/** The effect badge's words and tone. Amber only for a change that needs a restart. */
export const EFFECT_BADGE: Readonly<Record<SettingEffectWhen, { label: string; tone: PillTone }>> =
  {
    now: { label: 'Takes effect now', tone: 'green' },
    next_sweep: { label: 'At the next pruning sweep', tone: 'neutral' },
    next_attached_run: { label: 'At the next hot-update run', tone: 'neutral' },
    restart: { label: 'Needs a restart', tone: 'amber' },
  };

/** The row's header: plain label, the key in small mono, the effect badge and the source. */
function RowHead({ name, row }: { readonly name: string; readonly row: SettingRow }): ReactNode {
  const badge = row.effect === undefined ? null : EFFECT_BADGE[row.effect.when];
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        <div className="font-semibold" data-testid={`setting-label-${name}`}>
          {row.label ?? name}
        </div>
        <MonoId value={name} className="text-hint text-dim" />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {badge === null ? null : (
          <span title={row.effect?.text} data-testid={`setting-effect-${name}`}>
            <Pill tone={badge.tone}>{badge.label}</Pill>
          </span>
        )}
        {row.runtime_writable ? (
          <Pill tone={row.source === 'override' ? 'cyan' : 'neutral'}>
            {row.source === 'override' ? 'override' : 'environment'}
          </Pill>
        ) : null}
      </div>
    </div>
  );
}

/** Description, when it takes effect, the environment value and the last change. */
function RowFacts({ name, row }: { readonly name: string; readonly row: SettingRow }): ReactNode {
  const last = row.last_change ?? null;
  return (
    <div className="flex flex-col gap-1 text-body">
      {row.description === undefined ? null : (
        <p className="text-dim" data-testid={`setting-description-${name}`}>
          {row.description}
        </p>
      )}
      {row.effect === undefined ? null : (
        <p className="text-hint text-dim" data-testid={`setting-effect-text-${name}`}>
          {row.effect.text}
        </p>
      )}
      {row.runtime_writable ? (
        <dl
          className="grid grid-cols-[auto_1fr] gap-x-3 text-hint"
          data-testid={`setting-values-${name}`}
        >
          <dt className="text-dim">In force</dt>
          <dd className="min-w-0 wrap-break-word">{showSetting(row.value)}</dd>
          <dt className="text-dim">Environment value</dt>
          <dd className="min-w-0 wrap-break-word">
            {row.env_value === undefined ? 'not reported' : showSetting(row.env_value)}
          </dd>
          <dt className="text-dim">Last changed</dt>
          <dd className="min-w-0 wrap-break-word" data-testid={`setting-last-change-${name}`}>
            {last === null ? (
              'never changed here'
            ) : (
              <span title={last.changed_at}>
                {formatEt(last.changed_at)} by {showActor(last.actor, last.approval_id)}
                {last.reason ? ` - "${last.reason}"` : ''}
              </span>
            )}
          </dd>
        </dl>
      ) : null}
    </div>
  );
}

function draftOf(value: unknown): string {
  return value === null || value === undefined ? '' : String(value);
}

/** The typed draft as the value `PUT` sends, or why it cannot be one. */
function parseDraft(editor: Editor, draft: string): { value: unknown } | { error: string } {
  const text = draft.trim();
  switch (editor) {
    case 'boolean':
      return text === 'true' || text === 'false'
        ? { value: text === 'true' }
        : { error: 'Choose on or off' };
    case 'number': {
      const number = Number(text);
      return text === '' || !Number.isFinite(number)
        ? { error: 'Enter a number' }
        : { value: number };
    }
    case 'log-level':
      return { value: text };
    case 'address':
      return { value: text === '' ? null : text };
  }
}

type Drafts = Readonly<Record<string, string>>;

function decodeDrafts(data: unknown): Drafts | null {
  if (data === null || typeof data !== 'object' || Array.isArray(data)) return null;
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(data as Record<string, unknown>)) {
    if (typeof value === 'string') out[key] = value;
  }
  return out;
}

const NO_DRAFTS: Drafts = {};

/** Where a row's unsaved reason lives in the drafts map, beside its unsaved value. */
export const reasonDraftKey = (name: string): string => `${name}::reason`;

function RuntimeRow({
  name,
  row,
  draft: unsaved,
  reason,
  onDraft,
  onReason,
  onChanged,
}: {
  readonly name: string;
  readonly row: SettingRow;
  /** The operator's unsaved value, or `undefined` to show the effective one. */
  readonly draft: string | undefined;
  /** The operator's unsaved reason for the change, if any. */
  readonly reason: string | undefined;
  /** `undefined` drops the draft. */
  readonly onDraft: (value: string | undefined) => void;
  readonly onReason: (value: string | undefined) => void;
  readonly onChanged: () => Promise<void>;
}): ReactNode {
  const editor = editorFor(name, row.value);
  const draft = unsaved ?? draftOf(row.value);
  const setDraft = (value: string) => onDraft(value);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);

  const send = async (action: () => ReturnType<typeof writeSetting>) => {
    setBusy(true);
    setFailure(null);
    const result = await action();
    setBusy(false);
    if (!result.ok) {
      setFailure(result.failure);
      return;
    }
    onDraft(undefined);
    onReason(undefined);
    await onChanged();
  };

  const save = () => {
    const parsed = parseDraft(editor, draft);
    if ('error' in parsed) {
      setProblem(parsed.error);
      return;
    }
    setProblem(null);
    void send(() => writeSetting(name, parsed.value, reason));
  };

  const testId = `setting-input-${name}`;
  const control =
    editor === 'boolean' ? (
      <Select
        closedEnum="A boolean setting: on or off"
        placeholder="Choose on or off"
        options={[
          { value: 'true', label: 'On' },
          { value: 'false', label: 'Off' },
        ]}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        aria-label={name}
        data-testid={testId}
      />
    ) : editor === 'log-level' ? (
      <Select
        closedEnum="The backend's log_level Literal: DEBUG, INFO, WARNING, ERROR"
        placeholder="Choose a log level"
        options={LOG_LEVELS.map((level) => ({ value: level, label: level }))}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        aria-label={name}
        data-testid={testId}
      />
    ) : (
      <Input
        entry={editor === 'number' ? 'number' : 'address'}
        inputMode={editor === 'number' ? 'decimal' : 'url'}
        placeholder={editor === 'number' ? '' : 'not set'}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        aria-label={name}
        data-testid={testId}
      />
    );

  return (
    <Card as="li" className="flex flex-col gap-3" data-testid={`setting-${name}`}>
      <RowHead name={name} row={row} />
      <RowFacts name={name} row={row} />
      <div className="flex flex-wrap items-center gap-2">
        <div className="min-w-0 flex-1">{control}</div>
        <div className="min-w-0 flex-1">
          <Input
            entry="prose"
            placeholder="Reason (optional)"
            value={reason ?? ''}
            onChange={(event) =>
              onReason(event.target.value === '' ? undefined : event.target.value)
            }
            aria-label={`Reason for changing ${row.label ?? name}`}
            maxLength={2000}
            data-testid={`setting-reason-${name}`}
          />
        </div>
        <Button disabled={busy} onClick={save} data-testid={`setting-save-${name}`}>
          Save override
        </Button>
        {row.source === 'override' ? (
          <Button
            disabled={busy}
            onClick={() => void send(() => resetSetting(name, reason))}
            data-testid={`setting-reset-${name}`}
          >
            Use the environment value
          </Button>
        ) : null}
      </div>
      {problem === null ? null : <p className="text-hint text-red">{problem}</p>}
      {failure === null ? null : <FailureNotice failure={failure} surface={`Saving ${name}`} />}
    </Card>
  );
}

export function SettingsPage(): ReactNode {
  const [listing, setListing] = useState<SettingsListing | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const drafts = usePersistentState<Drafts>(DRAFT_KEYS.settings, NO_DRAFTS, {
    version: 1,
    decode: decodeDrafts,
  });
  const setDrafts = drafts.set;
  const draftFor = (name: string) => (value: string | undefined) =>
    setDrafts((current) => {
      const next: Record<string, string> = { ...current };
      if (value === undefined) delete next[name];
      else next[name] = value;
      return next;
    });

  const load = useCallback(async () => {
    const result = await readSettings();
    if (result.ok) {
      setListing(result.value);
      setFailure(null);
    } else {
      setFailure(result.failure);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const [historyVersion, setHistoryVersion] = useState(0);
  const entries = Object.entries(listing?.settings ?? {});
  const labels: Readonly<Record<string, string>> = Object.fromEntries(
    entries.map(([name, row]) => [name, row.label ?? name]),
  );
  const runtime = entries.filter(([, row]) => row.runtime_writable);
  const fixed = entries.filter(([, row]) => !row.runtime_writable);
  const runtimeGroups = groupSettings(runtime, listing?.groups);
  const fixedGroups = groupSettings(fixed, listing?.groups);

  return (
    <>
      <PageHead title={SETTINGS_TITLE} subtitle={SETTINGS_SUBTITLE} />
      {failure === null ? null : (
        <FailureNotice failure={failure} surface="The settings" className="mb-4" />
      )}
      {listing === null ? null : (
        <>
          <h2 className="mb-1 font-semibold">Changeable while miForge runs</h2>
          <p className="mb-3 text-body text-dim" data-testid="settings-runtime-count">
            {runtime.length} of {listing.count} settings. An override is stored in miForge&apos;s
            database and replaces the environment&apos;s value until you clear it.
          </p>
          <div className="flex flex-col gap-5" data-testid="settings-runtime">
            {runtimeGroups.map((group) => (
              <section key={group.id} data-testid={`settings-runtime-group-${group.id}`}>
                <h3 className="mb-2 text-hint font-semibold uppercase tracking-wide text-dim">
                  {group.label}
                </h3>
                <ul className="flex flex-col gap-3" aria-label={`Runtime settings: ${group.label}`}>
                  {group.rows.map(([name, row]) => (
                    <RuntimeRow
                      key={name}
                      name={name}
                      row={row}
                      draft={drafts.value[name]}
                      reason={drafts.value[reasonDraftKey(name)]}
                      onDraft={draftFor(name)}
                      onReason={draftFor(reasonDraftKey(name))}
                      onChanged={async () => {
                        await load();
                        setHistoryVersion((version) => version + 1);
                      }}
                    />
                  ))}
                </ul>
              </section>
            ))}
          </div>
          <h2 className="mb-1 mt-8 font-semibold">Needs a restart</h2>
          <p className="mb-3 text-body text-dim">
            Read from the environment when miForge starts. Change these in its environment and
            restart the pod named on each one; a restart stops running jobs.
          </p>
          <div className="flex flex-col gap-5" data-testid="settings-fixed">
            {fixedGroups.map((group) => (
              <section key={group.id} data-testid={`settings-fixed-group-${group.id}`}>
                <h3 className="mb-2 text-hint font-semibold uppercase tracking-wide text-dim">
                  {group.label}
                </h3>
                <Card>
                  <dl
                    className="grid grid-cols-[minmax(0,16rem)_1fr] gap-x-4 gap-y-3 text-body"
                    aria-label={`Restart settings: ${group.label}`}
                  >
                    {group.rows.map(([name, row]) => (
                      <div key={name} className="contents" data-testid={`setting-fixed-${name}`}>
                        <dt className="min-w-0 wrap-break-word">
                          <div className="font-semibold">{row.label ?? name}</div>
                          <MonoId value={name} className="text-hint text-dim" />
                        </dt>
                        <dd className="min-w-0 wrap-break-word">
                          <div>{showSetting(row.value)}</div>
                          {row.description === undefined ? null : (
                            <div className="text-hint text-dim">{row.description}</div>
                          )}
                          {row.effect?.pod === undefined ? null : (
                            <div className="text-hint text-dim">Restart {row.effect.pod}</div>
                          )}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </Card>
              </section>
            ))}
          </div>
          <SettingsHistory labels={labels} version={historyVersion} />
        </>
      )}
    </>
  );
}
