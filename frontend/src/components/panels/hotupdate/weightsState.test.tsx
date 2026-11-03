/**
 * ADR-012's colour rule for `weights_state`, as a guard that fails if the pill goes orange.
 *
 * `weights_state` is miLLM's state, so every value renders **cyan**. Two layers are asserted
 * because either alone can be fooled: the tone map (a value re-toned to the accent), and the
 * rendered DOM (`StatePill` downgrades an accent on any label but "Running" to neutral, so a
 * map edit to the accent shows up here as `neutral`, not as orange - and still fails).
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { WEIGHTS_STATE_TONES } from './weightsState';
import { WeightsStatePill } from './WeightsStatePill';

const BRD01_STATES = ['clean', 'writing', 'torn'] as const;

describe('[WeightsStatePill]', () => {
  it('should map every BRD-01 state, and only those', () => {
    expect(Object.keys(WEIGHTS_STATE_TONES).sort()).toEqual([...BRD01_STATES].sort());
  });

  it('should give every weights_state value the cyan tone and never the accent', () => {
    for (const [state, tone] of Object.entries(WEIGHTS_STATE_TONES)) {
      expect(tone.tone, `${state} is miLLM state and must be cyan`).toBe('cyan');
      expect(tone.tone).not.toBe('accent');
    }
  });

  it.each(BRD01_STATES)('should render %s cyan in the DOM, never orange', (state) => {
    render(<WeightsStatePill state={state} />);
    const pill = screen.getByTestId('weights-state-pill');
    expect(pill).toHaveAttribute('data-tone', 'cyan');
    const style = pill.getAttribute('style') ?? '';
    expect(style).toContain('--mf-cyan');
    expect(style).not.toContain('--mf-accent');
  });

  it('should say a torn set refuses serving, in words rather than colour', () => {
    render(<WeightsStatePill state="torn" />);
    expect(screen.getByTestId('weights-state-pill')).toHaveTextContent(/torn/i);
    expect(screen.getByTestId('weights-state-pill')).toHaveTextContent(/serving refused/i);
  });

  it('should render an unreported state as words, never as a guessed clean pill', () => {
    render(<WeightsStatePill state={null} />);
    expect(screen.queryByTestId('weights-state-pill')).toBeNull();
    expect(screen.getByTestId('weights-state-unreported')).toHaveTextContent(
      'weights state unknown',
    );
  });

  it('should render an unknown state neutral and named, never the accent', () => {
    render(<WeightsStatePill state="quarantined" />);
    const pill = screen.getByTestId('weights-state-pill');
    expect(pill).toHaveAttribute('data-tone', 'neutral');
    expect(pill).toHaveTextContent('quarantined (unrecognised)');
  });
});
