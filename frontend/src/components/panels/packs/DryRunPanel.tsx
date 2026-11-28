/**
 * The dry-run panel (F7 task 12.7; FR-007.20, AC11).
 *
 * Per row: the prompt, the reason and a **three-valued verdict pill** - accept green, reject
 * red, **undecided neutral `dim`**. Not amber, which is the warning colour; never accent.
 *
 * The footer renders **F3's `SelectionHealth` record whole**, with all three counts and their
 * denominator: `Accepted 3, rejected 1, undecided 1 of 5`. **The panel computes no rate of its
 * own** - `acceptance_rate` arrives beside its denominator precisely so a bare percentage is
 * not expressible, and a second place holding the counts is a second answer.
 *
 * **No approve button, no "send for review", no pending count.** Reviewing is miDataworks'
 * queue (R-02.36, BRD-03 R-03.41); this renders what came back. A human-review configuration
 * comes back **all `undecided` with `asynchronous_note` naming why**, and the step still saves
 * - which is the case the test covers, because an all-undecided panel with no explanation
 * reads as a broken dry-run.
 *
 * It also names **the provider that answered and the judge**, separately: a judge-model
 * configuration must visibly not be the model being sampled (R-02.11 - a model grading itself
 * is not a measurement).
 */
import type { ReactNode } from 'react';
import { MonoId } from '@/components/common/MonoId';
import { VerdictPill } from '@/components/common/VerdictPill';
import { SelectionCounts } from '@/components/common/SelectionCounts';
import { NumberWithSample } from '@/components/common/NumberWithSample';
import { DegenerateWarning } from '@/components/panels/run/DegenerateWarning';
import type { DryRunResult } from '@/types/packs';

export interface DryRunPanelProps {
  readonly result: DryRunResult | null;
}

function identity(value: Readonly<Record<string, unknown>> | null): string {
  if (value === null) return 'none';
  const name = value['name'] ?? value['model'] ?? value['provider'];
  return typeof name === 'string' ? name : JSON.stringify(value);
}

export function DryRunPanel({ result }: DryRunPanelProps): ReactNode {
  if (result === null) {
    return (
      <div className="rounded-control bg-panel p-card text-hint text-dim">
        No dry run yet. A dry run answers a sample of this pack&apos;s own prompts and shows a
        verdict with a reason — advisory, and never an input to the pack&apos;s validity.
      </div>
    );
  }

  return (
    <div className="rounded-control bg-panel p-card" data-testid="dry-run-panel">
      <ol className="stack-y-3">
        {result.verdicts.map((verdict, index) => (
          <li key={index} className="text-hint" data-testid={`dry-run-row-${index}`}>
            <div className="font-mono text-ident">{verdict.prompt}</div>
            <div className="mt-1 text-dim">{verdict.completion}</div>
            <div className="mt-1 flex flex-wrap items-center gap-2">
              <VerdictPill verdict={verdict.verdict} />
              <MonoId value={verdict.reason_code} className="text-dim" />
              {verdict.reason === null ? null : <span className="text-dim">{verdict.reason}</span>}
              {verdict.score === null ? null : (
                <NumberWithSample
                  value={verdict.score.toFixed(3)}
                  sample={1}
                  sampleLabel="completion"
                />
              )}
            </div>
          </li>
        ))}
      </ol>

      <div className="mt-3 border-t border-line pt-3">
        {/* F3's record, whole. No rate computed here. */}
        <SelectionCounts health={result.health} data-testid="dry-run-counts" />
        <div className="mt-1 text-hint text-dim">
          Acceptance rate {result.health.acceptance_rate.toFixed(3)}, as the record reports it — the
          denominator is beside it so this is never shown as a bare percentage.
        </div>
        <div className="mt-1 text-hint text-dim" data-testid="dry-run-provider">
          Answered by <MonoId value={identity(result.provider)} />; judged by{' '}
          <MonoId value={identity(result.judge)} />. A judge is deliberately not the model being
          sampled — a model grading itself is not a measurement.
        </div>
        <div className="mt-1 text-hint text-dim">
          Configuration <MonoId value={result.config_sha256} truncateTo={8} /> ({result.config_kind}
          , {result.hash_provenance}). Advisory: a dry run is advice to the author and never an
          input to the pack&apos;s validity.
        </div>
        {result.asynchronous_note === null ? null : (
          <div className="mt-2 text-hint text-dim" data-testid="dry-run-async-note">
            {result.asynchronous_note}
          </div>
        )}
      </div>

      <div className="mt-3">
        <DegenerateWarning signals={result.health.signals} health={result.health} />
      </div>
    </div>
  );
}
