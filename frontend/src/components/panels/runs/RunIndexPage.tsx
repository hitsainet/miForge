/**
 * The run index (FR-009.19; FPRD **F-2**; task 14.0, AC-009.43).
 *
 * ## The list
 *
 * **Every run, newest first, from `GET /api/v1/runs`** - a page at a time, filtered by status
 * and run type, each row opening its run. *Until 2026-10-18 there was no such route: the path
 * answered 405, this screen said "this build serves no run listing", and the only way to reach
 * a run was to type its id - so six real runs on the cluster were invisible to the operator who
 * had started them.* The filter vocabularies come from the response itself (`statuses`,
 * `run_types`), so the screen offers exactly what the route accepts and refuses nothing it
 * would accept.
 *
 * ## Opening by identifier stays, as the fallback
 *
 * An id copied from a log line, a manifest or a socket room is faster to paste than to page
 * for, so the box stays - below the list, declared `id-fallback` with that reason. A run opened
 * this way renders through `RunRow`, with its liveness and scores, as before.
 *
 * **A "New run" action belongs here**: R-02.53 defines the form and it is a link, not a dialog.
 * **No resident-model card, no Detach, no attachment chip** - all section 5.10's.
 *
 * ## What survives a reload (task 13.0)
 *
 * The identifier box and **the list of identifiers opened from it**, through
 * `usePersistentState`. The *records* are not stored: each opened id is **re-read from the
 * backend** on load, because a stored record would show a run's state as it was, not as it is.
 * An id the backend answers 404 for is dropped; any other failure keeps it and names the
 * failure, so a backend that is briefly down does not erase the operator's list. A successful
 * open empties the box - it has been submitted. The list's two filters are kept the same way;
 * the list itself is re-read, never stored.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Field } from '@/components/common/Field';
import { Picker, type PickerListing, type PickerOption } from '@/components/common/Picker';
import { useListing } from '@/hooks/useListing';
import { FailureNotice } from '@/components/common/FailureNotice';
import { useNavigation } from '@/contexts/NavigationContext';
import { fetchRun, listRuns, type RunListPage, type RunListRow } from '@/api/runs';
import { RunListItem, formatWhen } from './RunListItem';
import { fetchScores } from '@/api/scores';
import type { ApiFailure } from '@/api/errors';
import type { RunRecord, ScoreSeriesRow } from '@/types/run';
import { latestOfKind } from '@/components/panels/run/StatTiles';
import { RunRow } from './RunRow';
import { usePersistentState } from '@/hooks/usePersistentState';
import { DRAFT_KEYS, asRecord } from '@/stores/draftStorage';
import { ShowArchivedToggle } from '@/components/common/ShowArchivedToggle';
import { PurgeDialog, PurgeHeader, usePurgeSelection } from '@/components/common/PurgeSelection';
import { previewPurge, purgeSelection } from '@/api/purge';

/** What the index keeps across a reload: the box, the ids opened from it, and the filters. */
export interface RunIndexDraft {
  readonly query: string;
  readonly opened: readonly string[];
  /** The list's filters. Empty is "any". */
  readonly status: string;
  readonly runType: string;
  /**
   * Show archived runs in the list **and** the dropdown (operator decision 2026-10-20).
   * Off by default: archived runs are hidden, never deleted.
   */
  readonly showArchived: boolean;
}

export const EMPTY_RUN_INDEX: RunIndexDraft = {
  query: '',
  opened: [],
  status: '',
  runType: '',
  showArchived: false,
};

/** One page of the list. A navigational surface: enough to scan, "show older" for the rest. */
export const RUN_LIST_PAGE = 25;
/** How many runs the "Open a run" dropdown lists - the route's maximum page. */
export const RUN_PICKER_LIMIT = 200;

/** A run as the dropdown shows it: its id, with status, run type and when it was created. */
export function runOptions(page: RunListPage): readonly PickerOption[] {
  return page.runs.map((row) => ({
    value: row.run_id,
    label: `${row.run_id} — ${row.status} · ${row.run_type} · created ${formatWhen(row.created_at)}${(row.archived_at ?? null) === null ? '' : ' · archived'}`,
    keywords: `${row.status} ${row.run_type} ${row.created_at ?? ''}${(row.archived_at ?? null) === null ? '' : ' archived'}`,
    ...(row.stop_reason === null && row.failure_code === null
      ? {}
      : {
          detail:
            row.stop_reason !== null
              ? `stopped: ${row.stop_reason}`
              : `failed: ${row.failure_code}`,
        }),
  }));
}
export const RUN_INDEX_DRAFT_VERSION = 1;

export function decodeRunIndexDraft(data: unknown): RunIndexDraft | null {
  const record = asRecord(data);
  if (record === null) return null;
  const query = typeof record['query'] === 'string' ? record['query'] : '';
  const opened = Array.isArray(record['opened'])
    ? record['opened'].filter((id): id is string => typeof id === 'string' && id.length > 0)
    : [];
  const status = typeof record['status'] === 'string' ? record['status'] : '';
  const runType = typeof record['runType'] === 'string' ? record['runType'] : '';
  const showArchived = record['showArchived'] === true;
  return { query, opened, status, runType, showArchived };
}

interface OpenedRun {
  readonly record: RunRecord;
  readonly test: ScoreSeriesRow | null;
  readonly retention: ScoreSeriesRow | null;
}

export interface RunIndexPageProps {
  /** Injected in tests. The real one is `GET /api/v1/runs`. */
  readonly list?: typeof listRuns;
  /** Injected in tests. The real ones are `POST /api/v1/runs/purge/plan` and `/purge`. */
  readonly preview?: typeof previewPurge;
  readonly purge?: typeof purgeSelection;
}

export function RunIndexPage({
  list = listRuns,
  preview = previewPurge,
  purge = purgeSelection,
}: RunIndexPageProps = {}): ReactNode {
  const navigation = useNavigation();
  const persisted = usePersistentState(DRAFT_KEYS.runsIndex, EMPTY_RUN_INDEX, {
    version: RUN_INDEX_DRAFT_VERSION,
    decode: decodeRunIndexDraft,
  });
  const { query, opened, status, runType, showArchived } = persisted.value;
  const setIndex = persisted.set;
  const setQuery = useCallback(
    (next: string) => setIndex((current) => ({ ...current, query: next })),
    [setIndex],
  );
  /** Records by run id. A cache of answers, re-read on every load and never stored. */
  const [found, setFound] = useState<Readonly<Record<string, OpenedRun>>>({});
  const [failure, setFailure] = useState<ApiFailure | null>(null);

  /** The list: the rows read so far, the last page's answer, and its failure. Never stored. */
  const [listed, setListed] = useState<readonly RunListRow[]>([]);
  const [page, setPage] = useState<RunListPage | null>(null);
  const [listFailure, setListFailure] = useState<ApiFailure | null>(null);
  const [listing, setListing] = useState(false);

  /** The latest request. A slower answer to an older filter must not replace a newer one. */
  const latest = useRef(0);
  const readPage = useCallback(
    async (offset: number) => {
      const ticket = latest.current + 1;
      latest.current = ticket;
      setListing(true);
      const result = await list({
        limit: RUN_LIST_PAGE,
        offset,
        status,
        runType,
        includeArchived: showArchived,
      });
      if (ticket !== latest.current) return;
      setListing(false);
      if (!result.ok) {
        setListFailure(result.failure);
        return;
      }
      setListFailure(null);
      setPage(result.value);
      setListed((current) =>
        offset === 0 ? result.value.runs : [...current, ...result.value.runs],
      );
    },
    [list, status, runType, showArchived],
  );

  /** Bumped after an archive or unarchive, so the dropdown re-reads what is available. */
  const [revision, setRevision] = useState(0);
  /**
   * The dropdown's own read: every run, newest first, unfiltered by the list's status and
   * run-type filters - but **not** by archive: an archived run is left out of the dropdown
   * too unless "Show archived runs" is ticked (operator decision 2026-10-20).
   */
  const readForPicker = useCallback(
    () => list({ limit: RUN_PICKER_LIMIT, offset: 0, includeArchived: showArchived }),
    // `revision` is a dependency only, so an archive makes a new reader and the listing re-runs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [list, showArchived, revision],
  );
  const runPicker = useListing(readForPicker, runOptions);

  /** An archive or unarchive answered: re-read the first page and the dropdown. */
  const afterArchive = useCallback(() => {
    setRevision((current) => current + 1);
    void readPage(0);
  }, [readPage]);

  // The first page, and again whenever a filter changes.
  useEffect(() => {
    void readPage(0);
  }, [readPage]);

  const look = useCallback(
    async (runId: string, submitted: boolean) => {
      const record = await fetchRun(runId);
      if (!record.ok) {
        setFailure(record.failure);
        // A run the backend does not know is dropped from the kept list; a backend that could
        // not answer is not evidence the run is gone, so the id stays.
        if (record.failure.status === 404) {
          setIndex((current) => ({
            ...current,
            opened: current.opened.filter((id) => id !== runId),
          }));
        }
        return;
      }
      const scores = await fetchScores(runId);
      const rows = scores.ok ? scores.value.scores : [];
      setFailure(null);
      setFound((current) => ({
        ...current,
        [record.value.run_id]: {
          record: record.value,
          test: latestOfKind(rows, 'test'),
          retention: latestOfKind(rows, 'retention'),
        },
      }));
      setIndex((current) => ({
        ...current,
        query: submitted ? '' : current.query,
        opened: submitted
          ? [record.value.run_id, ...current.opened.filter((id) => id !== record.value.run_id)]
          : current.opened,
      }));
    },
    [setIndex],
  );

  // Re-read every kept id once, on mount. The records are the backend's current answers.
  const restored = useRef(false);
  useEffect(() => {
    if (restored.current) return;
    restored.current = true;
    for (const runId of opened) void look(runId, false);
  }, [opened, look]);

  const rows = opened
    .map((runId) => found[runId])
    .filter((row): row is OpenedRun => row !== undefined);

  return (
    <>
      <PageHead
        title="Runs"
        subtitle="Push a model toward a domain you have described as a pack, one step at a time"
        actions={
          <Button variant="primary" onClick={() => navigation.go('new-run')}>
            New run
          </Button>
        }
      />

      <Card className="mb-4">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (query.trim().length > 0) void look(query.trim(), true);
          }}
        >
          <Field
            label="Open a run"
            hint="Every run, newest first - type to filter by id, status, run type or date, and choosing one opens it."
          >
            {({ id, describedBy }) => (
              <div className="flex items-start gap-2">
                <div className="min-w-0 flex-1">
                  <Picker
                    id={id}
                    describedBy={describedBy}
                    value={query}
                    onChange={setQuery}
                    // Choosing a run IS opening it (operator, 2026-10-18: "I want the
                    // available runs to be listed in a DROPDOWN").
                    onChoose={(runId) => {
                      setQuery('');
                      navigation.go('run', { runId });
                    }}
                    listing={runPicker.listing}
                    placeholder="Choose a run"
                    empty={
                      showArchived
                        ? 'No run has been started yet. Start one from New run.'
                        : 'No run is listed. Start one from New run, or tick "Show archived runs" below.'
                    }
                    surface="the run list"
                    freeText={{
                      reason: `Paste an id from a log line, a manifest or a socket room and press Open. The list holds the newest ${RUN_PICKER_LIMIT}; older runs are in the paged list below.`,
                    }}
                    data-testid="run-id-input"
                  />
                </div>
                <Button type="submit" variant="primary">
                  Open
                </Button>
              </div>
            )}
          </Field>
        </form>
      </Card>

      <RunList
        rows={listed}
        page={page}
        failure={listFailure}
        reading={listing}
        status={status}
        runType={runType}
        showArchived={showArchived}
        onFilter={(next) => setIndex((current) => ({ ...current, ...next }))}
        onMore={() => void readPage(listed.length)}
        onOpen={(runId) => navigation.go('run', { runId })}
        onArchived={afterArchive}
        preview={preview}
        purge={purge}
      />

      {failure === null ? null : (
        <FailureNotice
          failure={failure}
          surface="that run"
          className="mb-4"
          data-testid="run-open-failure"
        />
      )}

      {rows.length === 0 ? (
        opened.length === 0 ? null : (
          <Card data-testid="run-index-empty">
            {`Re-reading the ${opened.length === 1 ? 'run' : `${opened.length} runs`} you opened here before: ${opened.join(', ')}.`}
          </Card>
        )
      ) : (
        <ul className="stack-y-3" data-testid="run-index-list" aria-label="Opened by identifier">
          {rows.map((row) => (
            <RunRow
              key={row.record.run_id}
              record={row.record}
              latestTest={row.test}
              latestRetention={row.retention}
              onOpen={(runId) => navigation.go('run', { runId })}
            />
          ))}
        </ul>
      )}
    </>
  );
}

interface RunListProps {
  readonly rows: readonly RunListRow[];
  readonly page: RunListPage | null;
  readonly failure: ApiFailure | null;
  readonly reading: boolean;
  readonly status: string;
  readonly runType: string;
  readonly showArchived: boolean;
  readonly onFilter: (next: {
    readonly status?: string;
    readonly runType?: string;
    readonly showArchived?: boolean;
  }) => void;
  readonly onMore: () => void;
  readonly onOpen: (runId: string) => void;
  readonly onArchived: () => void;
  readonly preview: typeof previewPurge;
  readonly purge: typeof purgeSelection;
}

/** The two filter pickers' lists, from the response's own vocabularies. */
function vocabulary(page: RunListPage | null, key: 'statuses' | 'run_types'): PickerListing {
  return page === null
    ? { kind: 'loading' }
    : { kind: 'ready', options: page[key].map((name) => ({ value: name, label: name })) };
}

function RunList({
  rows,
  page,
  failure,
  reading,
  status,
  runType,
  showArchived,
  onFilter,
  onMore,
  onOpen,
  onArchived,
  preview,
  purge,
}: RunListProps): ReactNode {
  const total = page?.total ?? null;
  // Permanent deletion works on the archived rows on screen, which appear only with "Show
  // archived runs" (operator decision and request, 2026-10-20).
  const archivedIds = useMemo(
    () =>
      (showArchived ? rows.filter((row) => (row.archived_at ?? null) !== null) : []).map(
        (row) => row.run_id,
      ),
    [rows, showArchived],
  );
  const selection = usePurgeSelection('run', archivedIds, preview);
  const [purging, setPurging] = useState<readonly string[] | null>(null);
  return (
    <section className="mb-4" aria-label="Every run" data-testid="run-list">
      <div className="mb-3 grid grid-cols-1 gap-x-4 sm:grid-cols-[1fr_1fr_auto] sm:items-end">
        <Field label="Status" hint="Any status when none is chosen.">
          {({ id, describedBy }) => (
            <Picker
              id={id}
              describedBy={describedBy}
              value={status}
              onChange={(next) => onFilter({ status: next })}
              listing={vocabulary(page, 'statuses')}
              placeholder="Any status"
              empty="The list named no statuses."
              surface="the run list"
              data-testid="run-filter-status"
            />
          )}
        </Field>
        <Field label="Run type" hint="Any run type when none is chosen.">
          {({ id, describedBy }) => (
            <Picker
              id={id}
              describedBy={describedBy}
              value={runType}
              onChange={(next) => onFilter({ runType: next })}
              listing={vocabulary(page, 'run_types')}
              placeholder="Any run type"
              empty="The list named no run types."
              surface="the run list"
              data-testid="run-filter-type"
            />
          )}
        </Field>
        <div className="mb-4">
          <Button
            onClick={() => onFilter({ status: '', runType: '' })}
            disabled={status.length === 0 && runType.length === 0}
            data-testid="run-filter-clear"
          >
            Clear filters
          </Button>
        </div>
      </div>
      <div className="mb-3">
        <ShowArchivedToggle
          checked={showArchived}
          onChange={(next) => onFilter({ showArchived: next })}
          noun="runs"
          data-testid="run-show-archived"
        />
      </div>

      {failure !== null ? (
        <FailureNotice failure={failure} surface="the run list" data-testid="run-list-failure" />
      ) : page === null ? (
        <Card data-testid="run-list-loading">Reading the run list…</Card>
      ) : rows.length === 0 ? (
        <Card data-testid="run-list-empty">
          {status.length !== 0 || runType.length !== 0
            ? 'No run matches these filters. Clear them to see every run.'
            : showArchived
              ? 'No run has been started yet. Start one from New run.'
              : 'No run is listed. Start one from New run, or tick "Show archived runs" to see archived ones.'}
        </Card>
      ) : (
        <>
          <div className="mb-2 text-hint text-dim" data-testid="run-list-count">
            Showing {rows.length} of {total} {total === 1 ? 'run' : 'runs'}, newest first
            {showArchived ? ', archived included' : ''}.
          </div>
          <PurgeHeader
            selection={selection}
            noun="runs"
            onDelete={() => setPurging(archivedIds.filter((id) => selection.selected.has(id)))}
          />
          <ul className="stack-y-2" data-testid="run-list-rows">
            {rows.map((row) => (
              <RunListItem
                key={row.run_id}
                row={row}
                onOpen={onOpen}
                onArchived={() => onArchived()}
                purge={selection}
              />
            ))}
          </ul>
          {total !== null && rows.length < total ? (
            <div className="mt-3">
              <Button onClick={onMore} disabled={reading} data-testid="run-list-more">
                {reading ? 'Reading…' : 'Show older runs'}
              </Button>
            </div>
          ) : null}
        </>
      )}
      {purging === null ? null : (
        <PurgeDialog
          kind="run"
          ids={purging}
          noun="runs"
          preview={preview}
          purge={purge}
          onClose={() => setPurging(null)}
          onDone={() => {
            selection.deselectAll();
            selection.refresh();
            onArchived();
          }}
        />
      )}
    </section>
  );
}
