/**
 * Three series, two kinds of gap (FR-009.10-.13, .20), and **M11, M12 and M18**.
 *
 * The series construction is tested as a pure function (`buildCurve`) and the surrounding
 * copy through the DOM, because Recharts renders nothing measurable inside a
 * `ResponsiveContainer` in jsdom - a chart assertion there would pass whatever the data was,
 * which is the construction-green failure this project keeps finding.
 *
 * - **M11** — save marks from `step % save_every`. Blocked structurally: `SavePointMarks` takes
 *   `steps: number[]` and has no access to run configuration.
 * - **M12** — `floor ?? 0` as a reference line. An absent floor draws **no line** and says so.
 * - **M18** — joining a score series across a provider discontinuity. The break is an explicit
 *   `null` point, never a filtered array, and `connectNulls` is off.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RunCurves, buildCurve, describeBreak } from './RunCurves';
import { SavePointMarks } from './SavePointMarks';
import type { ScoreSeriesRow, StepRecord } from '@/types/run';

const steps: readonly StepRecord[] = [100, 200, 300, 400].map((step) => ({
  step,
  loss: 2 - step / 1000,
  grad_norm: null,
  learning_rate: null,
  wall_time_ms: null,
  batch_source_hashes: null,
}));

function score(step: number, overrides: Partial<ScoreSeriesRow> = {}): ScoreSeriesRow {
  return {
    step,
    kind: 'test',
    set: 'held-out@3',
    score: 0.5 + step / 1000,
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
    message: 'Test',
    ...overrides,
  };
}

describe('[buildCurve]', () => {
  it('should put an explicit null on a step with no test, rather than omitting the step', () => {
    // S1 and the mechanism of every gap on this chart: a filtered array makes Recharts join
    // straight across, which is M18's shape.
    const points = buildCurve(steps, [score(200), score(400)], []);
    const byStep = new Map(points.map((point) => [point.step, point]));
    expect(byStep.get(100)?.test).toBeNull();
    expect(byStep.get(200)?.test).toBeCloseTo(0.7);
    expect(byStep.get(300)?.test).toBeNull();
    // The step is PRESENT with a null value - not absent from the array.
    expect(byStep.has(300)).toBe(true);
  });

  it('should break the series at a provider discontinuity with a null point before it', () => {
    // M18. A score is comparable only against another taken the same way, so the two
    // providers' numbers are never joined into one segment.
    const rows = [score(200), score(300, { provider: 'millm', provider_discontinuity: true })];
    const points = buildCurve(steps, rows, [300]);
    const index = points.findIndex((point) => point.step === 300);
    expect(index).toBeGreaterThan(0);
    const before = points[index - 1]!;
    expect(before.step).toBe(299.5);
    expect(before.test).toBeNull();
    // The new provider's first reading is still plotted - the break is before it, not on it.
    expect(points[index]?.test).toBeCloseTo(0.8);
  });

  it('should keep loss on every recorded step and never synthesise one', () => {
    const points = buildCurve(steps, [], []);
    expect(points.map((point) => point.step)).toEqual([100, 200, 300, 400]);
    expect(points.every((point) => point.loss !== null)).toBe(true);
  });

  it('should carry a null loss through rather than dropping the step', () => {
    const withGap: readonly StepRecord[] = [{ ...steps[0]!, loss: null }, steps[1]!];
    const points = buildCurve(withGap, [], []);
    expect(points[0]?.loss).toBeNull();
    expect(points).toHaveLength(2);
  });
});

describe('[describeBreak]', () => {
  it('should name both providers at a discontinuity', () => {
    // FR-009.13: the break is LABELLED WITH BOTH, so an operator can see which instrument
    // each half was taken with.
    const rows = [
      score(200, { provider: 'in-process' }),
      score(300, { provider: 'millm', provider_discontinuity: true }),
    ];
    expect(describeBreak(rows, 300)).toEqual({ before: 'in-process', after: 'millm' });
  });

  it('should report an unrecorded provider as null rather than guessing the other side', () => {
    const rows = [
      score(200, { provider: null }),
      score(300, { provider: 'millm', provider_discontinuity: true }),
    ];
    expect(describeBreak(rows, 300).before).toBeNull();
  });
});

describe('[SavePointMarks]', () => {
  it('should take only the steps where a checkpoint exists, and see no run configuration', () => {
    // M11's structural guard: the component has no `saveEvery` prop and no access to one, so
    // computing a cadence requires changing its signature.
    const source = SavePointMarks.toString();
    expect(source).not.toContain('save_every');
    expect(source).not.toContain('saveEvery');
    expect(source).not.toContain('%');
  });
});

describe('[RunCurves]', () => {
  it('should name the decimation and the underlying sample', () => {
    // FR-009.20: a decimated curve presented as the raw series is a number quoted without
    // its sample, which ADR-012 forbids.
    render(
      <RunCurves
        steps={steps}
        scores={[score(200)]}
        savePoints={[200]}
        providerDiscontinuities={[]}
        floor={null}
      />,
    );
    expect(screen.getByTestId('curve-decimation')).toHaveTextContent('4 steps');
  });

  it('should draw NO floor line and say so when no floor is recorded', () => {
    // M12's chart half.
    render(
      <RunCurves
        steps={steps}
        scores={[score(200)]}
        savePoints={[]}
        providerDiscontinuities={[]}
        floor={null}
      />,
    );
    expect(screen.getByTestId('curve-no-floor')).toHaveTextContent('No floor recorded');
  });

  it('should mark save points from the checkpoints that exist, naming how many', () => {
    render(
      <RunCurves
        steps={steps}
        scores={[]}
        /* An OUT-OF-CADENCE save: 250 is not a multiple of any cadence these steps imply,
           and a `step % save_every` computation would not mark it. */
        savePoints={[100, 250]}
        providerDiscontinuities={[]}
        floor={0.72}
      />,
    );
    expect(screen.getByTestId('curve-saves')).toHaveTextContent('2 save points');
  });

  it('should say a run with no checkpoint has no save points, not mark a cadence', () => {
    // The other half of M11: a cadence step whose save FAILED must not be marked as though it
    // had succeeded, which is the dangerous direction.
    render(
      <RunCurves
        steps={steps}
        scores={[]}
        savePoints={[]}
        providerDiscontinuities={[]}
        floor={null}
      />,
    );
    expect(screen.getByTestId('curve-no-saves')).toHaveTextContent('never from the save cadence');
  });

  it('should label every provider break with both providers', () => {
    render(
      <RunCurves
        steps={steps}
        scores={[score(200), score(300, { provider: 'millm', provider_discontinuity: true })]}
        savePoints={[]}
        providerDiscontinuities={[300]}
        floor={null}
      />,
    );
    const label = screen.getByTestId('curve-break-300');
    expect(label).toHaveTextContent('in-process');
    expect(label).toHaveTextContent('millm');
    expect(label).toHaveTextContent('not joined');
  });

  it('should say a run with no steps and no scores has none, rather than drawing an empty axis', () => {
    render(
      <RunCurves
        steps={[]}
        scores={[]}
        savePoints={[]}
        providerDiscontinuities={[]}
        floor={null}
      />,
    );
    expect(screen.getByTestId('run-curves-empty')).toHaveTextContent('no steps and no scores yet');
  });
});
