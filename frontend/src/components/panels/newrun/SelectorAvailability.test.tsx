/**
 * The selector panel says in words why a type cannot run. The registry sends a code only
 * (`midataworks_not_configured`), and the panel once printed the bare code (copy review,
 * 2026-10-20). Negative control: rendering `row.unavailable_reason` directly turns this red.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SelectorAvailability } from './SelectorAvailability';
import type { SelectorCapability } from '@/api/selectors';

const ROWS: readonly SelectorCapability[] = [
  {
    name: 'rule',
    forms: ['selector'],
    requires: 'none',
    summary: 'regex, length bounds and required terms',
    available: true,
    unavailable_reason: null,
  },
  {
    name: 'human_review',
    forms: ['selector'],
    requires: 'midataworks',
    summary: 'a person decides, through miDataworks',
    available: false,
    unavailable_reason: 'midataworks_not_configured',
  },
];

describe('[SelectorAvailability]', () => {
  it('should give an unavailable type a plain reason, keeping its code', async () => {
    render(<SelectorAvailability load={() => Promise.resolve({ ok: true, value: ROWS })} />);
    const row = await screen.findByText(/a person decides/);
    const item = row.closest('li');
    expect(item).toHaveTextContent(
      'unavailable: miDataworks is not configured (midataworks_not_configured)',
    );
    expect(screen.getByTestId('selector-availability')).toHaveTextContent(
      'Selector types available now',
    );
  });
});
