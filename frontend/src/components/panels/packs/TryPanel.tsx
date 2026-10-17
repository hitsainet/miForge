/**
 * The try panel: a seeded sample of one slot, and its score where there is one
 * (F7 task 12.7; FR-007.19).
 *
 * Inset `panel` background. Every number names its scale and sample: **`n` and `population`
 * travel together**, so "5 of 640 prompts" is expressible and a bare "5" is not.
 *
 * The sample is **seeded** so re-opening the step shows the same rows - a sample that changed
 * on every view would read as the data changing - and **stratified through F4's own sampler**,
 * so the panel shows the sample a run would score rather than a head slice.
 *
 * At step 6 the panel answers the one thing an author cannot guess: **whether the proposed
 * floor is already breached.** `floor_breached` is `null` when no floor was proposed, because
 * *"is it breached"* has no answer without one and `false` would read as "it is fine".
 */
import type { ReactNode } from 'react';
import { AbsentValue } from '@/components/common/AbsentValue';
import { NumberWithSample } from '@/components/common/NumberWithSample';
import { MonoId } from '@/components/common/MonoId';
import type { ScoredSample, SlotSample } from '@/types/packs';

export interface TryPanelProps {
  readonly sample: SlotSample | ScoredSample | null;
  readonly 'data-testid'?: string;
}

function isScored(sample: SlotSample | ScoredSample): sample is ScoredSample {
  return 'score' in sample;
}

export function TryPanel({ sample, ...rest }: TryPanelProps): ReactNode {
  if (sample === null) {
    return (
      <div className="rounded-control bg-panel p-card text-hint text-dim" {...rest}>
        Nothing sampled yet. Fill this step&apos;s slot and the panel shows the sample a run would
        actually score.
      </div>
    );
  }

  return (
    <div className="rounded-control bg-panel p-card" data-testid="try-panel" {...rest}>
      <div className="mb-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-hint">
        <MonoId value={sample.set} />
        <NumberWithSample
          value={`${sample.n} of ${sample.population}`}
          sample={sample.population}
          sampleLabel="rows in the set"
        />
        <span className="text-dim">seed {sample.seed} — the same rows every time</span>
      </div>

      {isScored(sample) ? (
        <div className="mb-2 flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <NumberWithSample
            value={sample.score.toFixed(3)}
            sample={sample.n}
            sampleLabel="prompts"
            provider={sample.provider}
            qualifier={`${sample.aggregation} aggregation`}
            data-testid="try-panel-score"
          />
          {sample.floor === null ? (
            <AbsentValue reason="no_floor_recorded" data-testid="try-panel-no-floor" />
          ) : (
            <span
              className={sample.floor_breached === true ? 'text-amber' : 'text-dim'}
              data-testid="try-panel-floor"
              data-breached={String(sample.floor_breached)}
            >
              floor {sample.floor.toFixed(3)} —{' '}
              {sample.floor_breached === null
                ? 'no floor was proposed, so whether it is breached has no answer'
                : sample.floor_breached
                  ? 'already breached before the run starts'
                  : 'not breached by this sample'}
            </span>
          )}
        </div>
      ) : null}

      {sample.prompts.length === 0 ? (
        <AbsentValue
          reason="missing_required_key"
          variant="block"
          detail="the sample came back with no rows"
        />
      ) : (
        <ol className="space-y-2">
          {sample.prompts.map((row, index) => (
            <li key={index} className="text-hint">
              <div className="font-mono text-ident">{row.prompt}</div>
              {row.reference === null ? null : (
                <div className="mt-0.5 text-dim">reference: {row.reference}</div>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
