/**
 * The chart's two legend names count their own samples in the singular and the plural (train F4).
 *
 * Recharts draws nothing inside a `ResponsiveContainer` in jsdom (it measures a 0 x 0 parent),
 * so the count-line fix's two legend names survived their controls. This file swaps the
 * container for a fixed 800 x 400 box, so `LineChart` renders its `Legend` and the names are
 * read from the DOM, as the operator sees them. The rest of recharts is the real library.
 */
import { Children, cloneElement, isValidElement, type ReactElement, type ReactNode } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { StepRecord } from '@/types/run';

vi.mock('recharts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('recharts')>();
  function FixedContainer({ children }: { children?: ReactNode }): ReactNode {
    return (
      <div style={{ width: 800, height: 400 }}>
        {Children.map(children, (child) =>
          isValidElement(child)
            ? cloneElement(child as ReactElement<{ width?: number; height?: number }>, {
                width: 800,
                height: 400,
              })
            : child,
        )}
      </div>
    );
  }
  return { ...actual, ResponsiveContainer: FixedContainer };
});

// Imported after the mock is declared (vi.mock is hoisted either way).
const { HotUpdateTimings } = await import('./HotUpdateTimings');

function step(n: number, hold: number | null, commit: number | null): StepRecord {
  return {
    step: n,
    loss: 1,
    grad_norm: 1,
    learning_rate: 1e-5,
    wall_time_ms: 100,
    batch_source_hashes: null,
    quiesce_hold_ms: hold,
    commit_duration_ms: commit,
  };
}

function legendNames(): string[] {
  const chart = screen.getByTestId('hotupdate-timings-chart');
  return Array.from(chart.querySelectorAll('.recharts-legend-item-text')).map(
    (el) => el.textContent ?? '',
  );
}

describe('[HotUpdateTimings] legend names', () => {
  it('should name one commit and one hold in the singular', () => {
    render(<HotUpdateTimings steps={[step(1, 230, 12.6)]} />);
    expect(legendNames()).toEqual(['Commit duration (1 commit)', 'Quiesce hold (1 hold)']);
  });

  it('should name several in the plural, each series counting its own samples', () => {
    render(
      <HotUpdateTimings steps={[step(3, 230, 12.6), step(2, 231, null), step(1, 229, 13.1)]} />,
    );
    expect(legendNames()).toEqual(['Commit duration (2 commits)', 'Quiesce hold (3 holds)']);
  });
});
