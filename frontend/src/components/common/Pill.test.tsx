/**
 * Status pills (foundation tasks 7.2, 7.3; ADR-012, FR-009.8).
 *
 * The two assertions that carry a rule: an unrecognised state **degrades** instead of
 * throwing, and the accent reaches no pill but `"Running"`.
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  ACCENT_PILL_LABEL,
  enforceAccentOnlyForRunning,
  Pill,
  RUN_STATE_TONES,
  StatePill,
} from './Pill';

describe('[Pill]', () => {
  it('should tint its background with its own tone colour', () => {
    render(
      <Pill tone="green" data-testid="p">
        Completed
      </Pill>,
    );
    const pill = screen.getByTestId('p');
    expect(pill).toHaveAttribute('data-tone', 'green');
    expect(pill.style.color).toContain('--mf-green');
    expect(pill.style.background).toContain('color-mix');
  });
});

describe('[StatePill]', () => {
  it('should render the live columns statuses and F6s spelling, and nothing else', () => {
    // Two sources, and they disagree - see the note on RUN_STATE_TONES. `models/run.py`'s
    // RUN_STATUSES is {pending, running, completed, failed, cancelled} and a submitted run
    // defaults to `pending`; F6 FR-006.13 says `queued -> running -> {completed | failed}`.
    // Both spellings are carried so neither renders as unrecognised. The mockup's `held`
    // and `paused` are in neither and are deliberately absent.
    expect(Object.keys(RUN_STATE_TONES).sort()).toEqual([
      'cancelled',
      'completed',
      'failed',
      'pending',
      'queued',
      'running',
    ]);
  });

  it('should render a cancelled run neutral rather than as a failure', () => {
    // An operator stop (R-02.25) and cooperative cancellation (ADR-007) are not failures,
    // and painting them red tells an operator their own action went wrong.
    expect(RUN_STATE_TONES['cancelled']?.tone).toBe('neutral');
    expect(RUN_STATE_TONES['failed']?.tone).toBe('red');
  });

  it('should degrade an unrecognised state instead of throwing', () => {
    // The mockup's version destructures `map[state]` with no fallback and throws; its own
    // 2026-09-19 review found this, and a real enum grows - F6's did. A pill that
    // white-screens on a value the backend legitimately added is unrecoverable.
    expect(() => render(<StatePill state="reaped" data-testid="p" />)).not.toThrow();
    const pill = screen.getByTestId('p');
    expect(pill).toHaveTextContent('reaped (unrecognised)');
    expect(pill).toHaveAttribute('data-tone', 'neutral');
    expect(pill).toHaveAttribute('title', 'Unrecognised state: reaped');
  });

  it('should give only the Running pill the accent', () => {
    const { container } = render(
      <>
        <StatePill state="running" />
        <StatePill state="completed" />
        <StatePill state="queued" />
        <StatePill state="failed" />
      </>,
    );
    const accented = [...container.querySelectorAll('[data-tone="accent"]')].map(
      (node) => node.textContent,
    );
    expect(accented).toEqual([ACCENT_PILL_LABEL]);
  });

  it('should downgrade an accent tone given to any other label, and say so', () => {
    // The runtime net under `accent.test.tsx`. A feature supplying its own tone map
    // cannot smuggle the accent onto a second state.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    render(
      <StatePill
        state="resident"
        tones={{ resident: { tone: 'accent', label: 'Resident' } }}
        data-testid="p"
      />,
    );
    expect(screen.getByTestId('p')).toHaveAttribute('data-tone', 'neutral');
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  it('should pass a non-accent tone through untouched', () => {
    expect(enforceAccentOnlyForRunning({ tone: 'cyan', label: 'Resident in miLLM' })).toBe('cyan');
    expect(enforceAccentOnlyForRunning({ tone: 'accent', label: ACCENT_PILL_LABEL })).toBe(
      'accent',
    );
  });

  it('should render a feature supplied state map without extending the shared one', () => {
    render(
      <StatePill
        state="quarantined"
        tones={{ quarantined: { tone: 'red', label: 'Quarantined' } }}
        data-testid="p"
      />,
    );
    expect(screen.getByTestId('p')).toHaveTextContent('Quarantined');
    // The shared map is untouched: a map every feature appends to is a registry nobody
    // owns.
    expect(RUN_STATE_TONES['quarantined']).toBeUndefined();
  });
});
