/**
 * The four tiles (FR-009.14), and **M12's first half**.
 *
 * M12 is `floor ?? 0` in the retention tile. `0` is the worst available rendering of an absent
 * floor: it says *"nothing is protected and that is the recorded policy"*, which is the
 * opposite of *"no floor was recorded"* - and `floor` is **optional** in `ScoreRecord`
 * (`required` is `[set, score, n]`, parsed from the vendored schema), so an absent one is
 * ordinary rather than exceptional.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatTiles, latestOfKind } from './StatTiles';
import type { ScoreSeriesRow, StepRecord } from '@/types/run';

function score(
  overrides: Partial<ScoreSeriesRow> & { kind: string; step: number },
): ScoreSeriesRow {
  return {
    set: 'held-out@3',
    score: 0.84,
    n: 320,
    floor: null,
    floor_layer: null,
    breached: false,
    provider: 'in-process',
    provider_kind: 'in_process',
    provider_model: 'LFM2-1.2B',
    aggregation: 'mean',
    out_of_cadence: false,
    provider_discontinuity: false,
    message: 'Test 0.840 on 320 prompts via in-process',
    ...overrides,
  };
}

const step: StepRecord = {
  step: 590,
  loss: 1.2345,
  grad_norm: null,
  learning_rate: null,
  wall_time_ms: 412,
  batch_source_hashes: null,
};

describe('[StatTiles]', () => {
  it('should name every number with its scale and its sample', () => {
    render(
      <StatTiles
        currentStep={590}
        totalSteps={800}
        latestStep={step}
        latestTest={score({ kind: 'test', step: 500 })}
        latestRetention={score({ kind: 'retention', step: 500, score: 0.77, n: 180, floor: 0.72 })}
      />,
    );
    expect(screen.getByTestId('tile-step')).toHaveTextContent('590 of 800');
    expect(screen.getByTestId('tile-test')).toHaveTextContent('320 prompts');
    expect(screen.getByTestId('tile-test')).toHaveTextContent('in-process');
    expect(screen.getByTestId('tile-retention')).toHaveTextContent('180 prompts');
  });

  it('should name the step a cadenced value came from when it is not the current one', () => {
    // Test and retention run on a cadence and loss does not, so `0.84` beside `step 590` when
    // the score was taken at 500 is a number quoted against the wrong sample.
    render(
      <StatTiles
        currentStep={590}
        totalSteps={800}
        latestStep={step}
        latestTest={score({ kind: 'test', step: 500 })}
        latestRetention={null}
      />,
    );
    expect(screen.getByTestId('tile-test')).toHaveTextContent('at step 500');
    // Loss IS from the current step, so it does not repeat it.
    expect(screen.getByTestId('tile-loss')).not.toHaveTextContent('at step');
  });

  it('should render an absent floor as named-absent and NEVER as zero', () => {
    // M12. A floor of 0 is a policy the run does not have.
    render(
      <StatTiles
        currentStep={590}
        totalSteps={800}
        latestStep={step}
        latestTest={null}
        latestRetention={score({ kind: 'retention', step: 500, score: 0.77, n: 180, floor: null })}
      />,
    );
    const tile = screen.getByTestId('tile-retention');
    expect(tile).toHaveTextContent('No floor recorded');
    expect(tile.textContent ?? '').not.toContain('floor 0.000');
    expect(tile.querySelector('[data-absent="no_floor_recorded"]')).not.toBeNull();
  });

  it('should render the floor when it IS recorded, with its value', () => {
    render(
      <StatTiles
        currentStep={590}
        totalSteps={800}
        latestStep={step}
        latestTest={null}
        latestRetention={score({ kind: 'retention', step: 500, score: 0.77, n: 180, floor: 0.72 })}
      />,
    );
    expect(screen.getByTestId('tile-retention')).toHaveTextContent('floor 0.720');
  });

  it('should say a run with no test yet has none, rather than drawing it at zero', () => {
    // S1. A series with no samples is not a series of zeroes.
    render(
      <StatTiles
        currentStep={10}
        totalSteps={null}
        latestStep={null}
        latestTest={null}
        latestRetention={null}
      />,
    );
    // The inline variant renders the short label with the sentence as its title, so the
    // cell stays narrow while the explanation is one hover away.
    expect(screen.getByTestId('tile-test')).toHaveTextContent('no test at this step');
    expect(screen.getByTestId('tile-loss')).toHaveTextContent('no loss recorded');
  });

  it('should state the step honestly when no response carries a total', () => {
    // `total_steps` is not on GET /api/v1/runs/{id} in this build, so `590 of 800` is not
    // available - and `590 of 590` would be an invented denominator.
    render(
      <StatTiles
        currentStep={590}
        totalSteps={null}
        latestStep={step}
        latestTest={null}
        latestRetention={null}
      />,
    );
    const tile = screen.getByTestId('tile-step');
    expect(tile).toHaveTextContent('590');
    expect(tile).toHaveTextContent('steps recorded');
    expect(tile.textContent ?? '').not.toContain('of 0');
  });

  it('should render no step at all as named-absent rather than step zero', () => {
    render(
      <StatTiles
        currentStep={null}
        totalSteps={null}
        latestStep={null}
        latestTest={null}
        latestRetention={null}
      />,
    );
    expect(screen.getByTestId('tile-step')).toHaveTextContent('no step recorded yet');
  });
});

describe('[latestOfKind]', () => {
  it('should take the newest row of that kind and ignore the other kinds', () => {
    const rows = [
      score({ kind: 'test', step: 100, score: 0.1 }),
      score({ kind: 'retention', step: 500, score: 0.9 }),
      score({ kind: 'test', step: 400, score: 0.4 }),
    ];
    expect(latestOfKind(rows, 'test')?.step).toBe(400);
    expect(latestOfKind(rows, 'retention')?.step).toBe(500);
    expect(latestOfKind(rows, 'nothing')).toBeNull();
  });

  it('should include an out-of-cadence evaluation, which is a real one', () => {
    // Recorded with `out_of_cadence: true` so it never silently joins the step series - but
    // the latest value is still the latest value.
    const rows = [
      score({ kind: 'test', step: 400 }),
      score({ kind: 'test', step: 450, out_of_cadence: true, score: 0.5 }),
    ];
    expect(latestOfKind(rows, 'test')?.score).toBe(0.5);
  });
});
