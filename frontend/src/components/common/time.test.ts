import { describe, expect, it } from 'vitest';
import { formatAge, formatEt } from './time';

const NOW = Date.parse('2026-10-19T13:00:00Z');

describe('[formatEt]', () => {
  it('should render a UTC instant in US Eastern daylight time', () => {
    // EDT is UTC-4: 11:35 UTC is 7:35 AM.
    expect(formatEt('2026-10-18T11:35:22.843344+00:00', NOW)).toBe('Oct 18, 7:35 AM ET');
  });

  it('should follow the switch to standard time', () => {
    // EST is UTC-5 after the first Sunday of November.
    expect(formatEt('2026-11-10T17:05:00Z', NOW)).toBe('Nov 10, 12:05 PM ET');
  });

  it('should add the year when it is not the current one', () => {
    expect(formatEt('2025-12-31T23:30:00Z', NOW)).toBe('Dec 31, 2025, 6:30 PM ET');
  });

  it('should return an unparseable value unchanged and null as empty', () => {
    expect(formatEt('not a time', NOW)).toBe('not a time');
    expect(formatEt(null, NOW)).toBe('');
  });
});

describe('[formatAge]', () => {
  it('should describe the age in the largest sensible unit', () => {
    expect(formatAge('2026-10-19T12:59:50Z', NOW)).toBe('just now');
    expect(formatAge('2026-10-19T12:55:00Z', NOW)).toBe('5 min ago');
    expect(formatAge('2026-10-19T10:00:00Z', NOW)).toBe('3 h ago');
    expect(formatAge('2026-10-17T11:35:00Z', NOW)).toBe('2 days ago');
  });
});
