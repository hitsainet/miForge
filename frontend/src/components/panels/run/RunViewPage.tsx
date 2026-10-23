/**
 * R-02.28's screen. **Thin: store, states, panels** (FR-009.4, .10-.18, .20).
 *
 * This component formats nothing. Every number goes through `NumberWithSample`, every absence
 * through `AbsentValue`, and the liveness verdict through `deriveLiveness` - which is what
 * makes the accent rule, the sample rule and the liveness rule each testable in one place
 * rather than in whichever panel last touched them.
 *
 * **The socket is additive.** `useRecordPolling` runs regardless; `useRunSocket` only shortens
 * the latency between polls and reports the transport. A screen that stopped polling because a
 * socket connected would report a stalled run as healthy for as long as the socket stayed open
 * (ADR-007, FR-009.4).
 */
import { useCallback, useMemo, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { Callout } from '@/components/common/Callout';
import { MonoId } from '@/components/common/MonoId';
import { Pill, StatePill } from '@/components/common/Pill';
import { Button } from '@/components/common/Button';
import { FailureNotice } from '@/components/common/FailureNotice';
import { AbsentValue } from '@/components/common/AbsentValue';
import { endedWithoutReason } from '@/stores/runStore';
import { useRecordPolling } from '@/hooks/useRecordPolling';
import { useRunSocket } from '@/hooks/useRunSocket';
import { useNavigation } from '@/contexts/NavigationContext';
import { describeTransport } from '@/stores/transport';
import {
  deriveLiveness,
  describeLiveness,
  savePoints as savePointsOf,
  useRunStore,
} from '@/stores/runStore';
import { StatTiles, latestOfKind } from './StatTiles';
import { RunCurves } from './RunCurves';
import { StepLog } from './StepLog';
import { DegenerateWarning } from './DegenerateWarning';
import { CheckpointManifestPanel } from './ManifestView';
import { ContinueRun } from './ContinueRun';
import {
  HotUpdateTimings,
  hasHotUpdateTimings,
} from '@/components/panels/hotupdate/HotUpdateTimings';
import { RunAttachmentRecord } from '@/components/panels/hotupdate/RunAttachmentRecord';
import { formatEt } from '@/components/common/time';
import { RunArchiveAction } from '@/components/panels/runs/RunArchiveAction';

export function RunViewPage(): ReactNode {
  const navigation = useNavigation();
  const runId = navigation.params['runId'] ?? null;

  const store = useRunStore();
  const open = store.open;
  const load = store.load;
  const applyEvent = store.applyEvent;
  const setTransport = store.setTransport;

  // Open the store on the id the navigation carries. Doing this in render rather than an
  // effect would loop; doing it on every params change is what makes the screen follow a
  // second "open run" without a remount.
  const openIfNeeded = useCallback(() => {
    if (runId !== null && store.runId !== runId) open(runId);
  }, [runId, store.runId, open]);
  openIfNeeded();

  useRecordPolling({ load, transport: store.transport, paused: runId === null });
  useRunSocket({ runId, onEvent: applyEvent, onTransport: setTransport });

  const record = store.record;

  // §5.10 (F11 16.4): the run's own record of miLLM's state, shown on this screen as history.
  // The shell's chip reads miLLM live instead; this block outlives the attachment.
  const attachment = record?.attachment ?? null;

  const liveness = useMemo(
    () => (record === null ? null : deriveLiveness(record, Date.now())),
    [record],
  );
  const scores = store.scores?.scores ?? [];
  const latestTest = latestOfKind(scores, 'test');
  const latestRetention = latestOfKind(scores, 'retention');
  const marks = savePointsOf(store.checkpoints);
  const latestBatch = store.batches[0] ?? null;

  if (runId === null) {
    return (
      <>
        <PageHead
          title="Run"
          subtitle="One run: its loss, its test and retention scores, its save points and its log"
          back={{ screen: 'runs', label: 'All runs' }}
        />
        <Callout tone="info" title="No run is open">
          Open a run from Runs, or start one from New run. This screen needs a run identifier; it
          has no list of its own.
        </Callout>
      </>
    );
  }

  return (
    <>
      <PageHead
        title={runId}
        subtitle="One run: its loss, its test and retention scores, its save points and its log"
        back={{ screen: 'runs', label: 'All runs' }}
        actions={
          <>
            <span className="text-hint text-dim" data-testid="run-transport">
              {describeTransport(store.transport)}
            </span>
            <Button onClick={() => void load()}>Re-read the record</Button>
            {record === null ? null : (
              <RunArchiveAction
                runId={record.run_id}
                status={record.status}
                archivedAt={record.archived_at}
                onDone={() => void load()}
              />
            )}
          </>
        }
      />

      {store.failure === null ? null : (
        <FailureNotice failure={store.failure} surface="this run's record" className="mb-4" />
      )}

      {record === null ? (
        store.phase === 'loading' ? (
          <Card data-testid="run-loading">Reading the run record…</Card>
        ) : null
      ) : (
        <>
          <Card className="mb-4" data-testid="run-liveness">
            <div className="flex flex-wrap items-center gap-3">
              <StatePill state={record.status} data-testid="run-state" />
              {record.archived_at === null || record.archived_at === undefined ? null : (
                <Pill tone="neutral" title={record.archived_at} data-testid="run-archived">
                  {`Archived ${formatEt(record.archived_at)} — hidden from the Runs list; nothing was deleted`}
                </Pill>
              )}
              <span
                className="text-hint text-dim"
                data-liveness={liveness?.kind ?? 'unknown'}
                data-testid="run-liveness-sentence"
              >
                {liveness === null ? '' : describeLiveness(liveness)}
              </span>
            </div>
            {record.failure_code === null ? null : (
              <div className="mt-2 text-hint text-red">
                Failure code <MonoId value={record.failure_code} />
              </div>
            )}
            {record.cancel_requested_at === null ? null : (
              <div className="mt-2 text-hint text-dim">
                A stop was requested at{' '}
                <span title={record.cancel_requested_at}>
                  {formatEt(record.cancel_requested_at)}
                </span>
                . Cancellation is cooperative and takes effect at the next step boundary.
              </div>
            )}
            {/* F6's `stop_reason` is not on this response yet. Named absent rather than
                rendered as a bare "done" - which of the five conditions fired is the one
                thing an operator wants from a finished run (FR-009.19). */}
            {liveness?.kind === 'terminal' && record.stop_reason == null ? (
              <AbsentValue
                reason="missing_required_key"
                variant="block"
                className="mt-2"
                detail={endedWithoutReason(record)}
              />
            ) : null}
            {liveness?.kind !== 'terminal' &&
            liveness !== null &&
            liveness.clockSkewSeconds !== null ? (
              <div className="mt-2 text-hint text-amber" data-testid="run-clock-skew">
                This browser&apos;s clock differs from the server&apos;s by about{' '}
                {liveness.clockSkewSeconds} s. The ages above are the server&apos;s measurements;
                times shown elsewhere may read oddly until the clock is fixed.
              </div>
            ) : null}
            {/* D52: an attached run that ended continues from its last save. */}
            <ContinueRun record={record} onSubmitted={() => void load()} />
            {store.discardedEvents > 0 ? (
              <div className="mt-2 text-hint text-dim" data-testid="run-discarded-events">
                {store.discardedEvents} live update
                {store.discardedEvents === 1 ? '' : 's'} discarded: the run record says this run is{' '}
                {record.status}, and the record is the truth.
              </div>
            ) : null}
          </Card>

          <StatTiles
            currentStep={record.last_step}
            totalSteps={record.total_steps ?? null}
            latestStep={store.steps[0] ?? null}
            latestTest={latestTest}
            latestRetention={latestRetention}
          />

          {latestBatch === null ? null : (
            <DegenerateWarning
              signals={[]}
              health={{
                n: latestBatch.n_prompts,
                accepted: latestBatch.verdicts.accept,
                rejected: latestBatch.verdicts.reject,
                undecided: latestBatch.verdicts.undecided,
              }}
              atStep={latestBatch.step}
            />
          )}

          <div className="mb-4">
            {store.regionFailures['scores'] === undefined &&
            store.regionFailures['steps'] === undefined ? (
              <RunCurves
                steps={store.steps}
                scores={scores}
                savePoints={marks}
                providerDiscontinuities={store.scores?.provider_discontinuities ?? []}
                floor={latestRetention?.floor ?? null}
              />
            ) : (
              <FailureNotice
                failure={store.regionFailures['scores'] ?? store.regionFailures['steps']!}
                surface="the curves"
              />
            )}
          </div>

          {attachment === null ? null : (
            <RunAttachmentRecord attachment={attachment} runStatus={record.status} />
          )}

          {attachment !== null || hasHotUpdateTimings(store.steps) ? (
            <HotUpdateTimings steps={store.steps} />
          ) : null}

          <div className="mb-4">
            {store.regionFailures['batches'] === undefined ? (
              <StepLog
                steps={store.steps}
                scores={scores}
                batches={store.batches}
                savePoints={marks}
              />
            ) : (
              <FailureNotice failure={store.regionFailures['batches']} surface="the step log" />
            )}
          </div>

          {store.regionFailures['checkpoints'] === undefined ? (
            <CheckpointManifestPanel
              runId={runId}
              step={store.checkpoints?.latest_step ?? marks[marks.length - 1] ?? null}
            />
          ) : (
            <FailureNotice
              failure={store.regionFailures['checkpoints']}
              surface="this run's checkpoints"
            />
          )}

          {store.scores === null ? null : (
            <div className="mt-3 text-hint text-dim" data-testid="exposure-window">
              Retention runs every {store.scores.retention_every} steps, so up to{' '}
              {store.scores.exposure_window_steps} steps of training can occur between a breach
              happening and being seen. The floor is a measurement with a lag, not a guarantee.
            </div>
          )}
        </>
      )}
    </>
  );
}
