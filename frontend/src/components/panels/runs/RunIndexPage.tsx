/**
 * The navigational minimum (FR-009.19; FPRD **F-2**).
 *
 * ## The enumeration this screen needs does not exist
 *
 * **There is no `GET /api/v1/runs`.** The live table carries `POST /api/v1/runs`,
 * `POST /api/v1/runs/define` and `GET /api/v1/runs/{run_id}`, and nothing that lists - so a
 * GET at that path answers **405, not 404**, which is worth knowing because a client that
 * read a 405 as "empty list" would present an empty index as a fact. F9's own **F-2** records
 * that *no requirement describes a run list*, which is why nobody built one - and the
 * navigational minimum still needs an enumeration from somewhere.
 *
 * So this screen does the honest thing rather than either of the two dishonest ones. It does
 * **not** fabricate rows, and it does **not** render as broken. It states the gap by name,
 * and it opens a run by identifier - which is all "sufficient to reach a run" actually
 * requires, and which keeps the run view reachable through the real nav for the Playwright
 * specs. A run opened this way renders through the same `RunRow` the list would use, so when
 * the endpoint lands the row component is already the one it feeds.
 *
 * **A "New run" action belongs here and is no longer unowned**: R-02.53 defines the form and
 * it is a link, not a dialog - this list performs no part of a submission.
 *
 * **No resident-model card, no Detach, no attachment chip** - all section 5.10's, and in the
 * default path there is no resident model in another process.
 */
import { useCallback, useState, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { Callout } from '@/components/common/Callout';
import { Button } from '@/components/common/Button';
import { Field, Input } from '@/components/common/Field';
import { MonoId } from '@/components/common/MonoId';
import { FailureNotice } from '@/components/common/FailureNotice';
import { useNavigation } from '@/contexts/NavigationContext';
import { fetchRun, runsSurfaceGaps } from '@/api/runs';
import { fetchScores } from '@/api/scores';
import type { ApiFailure } from '@/api/errors';
import type { RunRecord, ScoreSeriesRow } from '@/types/run';
import { latestOfKind } from '@/components/panels/run/StatTiles';
import { RunRow } from './RunRow';

interface OpenedRun {
  readonly record: RunRecord;
  readonly test: ScoreSeriesRow | null;
  readonly retention: ScoreSeriesRow | null;
}

export function RunIndexPage(): ReactNode {
  const navigation = useNavigation();
  const [query, setQuery] = useState('');
  const [found, setFound] = useState<readonly OpenedRun[]>([]);
  const [failure, setFailure] = useState<ApiFailure | null>(null);

  const look = useCallback(async (runId: string) => {
    const record = await fetchRun(runId);
    if (!record.ok) {
      setFailure(record.failure);
      return;
    }
    const scores = await fetchScores(runId);
    const rows = scores.ok ? scores.value.scores : [];
    setFailure(null);
    setFound((current) => [
      {
        record: record.value,
        test: latestOfKind(rows, 'test'),
        retention: latestOfKind(rows, 'retention'),
      },
      ...current.filter((row) => row.record.run_id !== record.value.run_id),
    ]);
  }, []);

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

      <Callout tone="info" title="This build serves no run listing" className="mb-4">
        <ul className="space-y-1">
          {runsSurfaceGaps().map((gap) => (
            <li key={gap.endpoint}>
              <MonoId value={gap.endpoint} /> is {gap.owner}&apos;s and is not mounted, so{' '}
              {gap.what} is unavailable. Nothing here is fabricated in its place.
            </li>
          ))}
        </ul>
      </Callout>

      <Card className="mb-4">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (query.trim().length > 0) void look(query.trim());
          }}
        >
          <Field
            label="Open a run by its identifier"
            hint="The run_id every manifest, log line and socket room carries. Reading one run needs no listing."
          >
            {({ id, describedBy }) => (
              <div className="flex gap-2">
                <Input
                  id={id}
                  aria-describedby={describedBy}
                  value={query}
                  placeholder="md-notary-2026-09-11"
                  onChange={(event) => setQuery(event.target.value)}
                  data-testid="run-id-input"
                />
                <Button type="submit" variant="primary">
                  Open
                </Button>
              </div>
            )}
          </Field>
        </form>
      </Card>

      {failure === null ? null : (
        <FailureNotice failure={failure} surface="that run" className="mb-4" />
      )}

      {found.length === 0 ? (
        <Card data-testid="run-index-empty">
          No run is open. Enter a run identifier above, or start one from New run.
        </Card>
      ) : (
        <ul className="space-y-3" data-testid="run-index-list">
          {found.map((row) => (
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
