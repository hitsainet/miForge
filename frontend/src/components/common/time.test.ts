import { describe, expect, it } from 'vitest';
import { formatAge, formatEt, formatExpiresIn, formatLocal } from './time';

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

describe('[formatLocal] (ENH-008 D8b, A5)', () => {
  it('should show the instant in the given zone with its abbreviation, the exact UTC ISO as title', () => {
    const eastern = formatLocal('2026-10-18T11:35:22.843344+00:00', 'America/New_York');
    expect(eastern.text).toMatch(/Oct 18, 2026/);
    expect(eastern.text).toMatch(/7:35/);
    expect(eastern.text).toMatch(/EDT/);
    expect(eastern.title).toBe('2026-10-18T11:35:22.843Z');
    const utc = formatLocal('2026-10-18T11:35:22Z', 'UTC');
    expect(utc.text).toMatch(/11:35/);
    expect(utc.text).toMatch(/UTC/);
  });

  it('should use the browser zone when none is given, and never throw on its options', () => {
    // ECMA-402 refuses timeZoneName beside dateStyle/timeStyle; this call is the guard.
    expect(() => formatLocal('2026-10-18T11:35:22Z')).not.toThrow();
    expect(formatLocal('2026-10-18T11:35:22Z').title).toBe('2026-10-18T11:35:22.000Z');
  });

  it('should show "unknown" for a missing or unparseable value, never "Invalid Date"', () => {
    for (const value of [null, undefined, '', 'not a time']) {
      const shown = formatLocal(value);
      expect(shown.text).toBe('unknown');
      expect(shown.title).toBe('');
    }
  });
});

describe('[formatExpiresIn]', () => {
  it('should count down in minutes, then hours, and say when it has passed', () => {
    expect(formatExpiresIn('2026-10-20T13:00:00Z', NOW)).toBe('expires in 24 h');
    expect(formatExpiresIn('2026-10-19T13:40:00Z', NOW)).toBe('expires in 40 min');
    expect(formatExpiresIn('2026-10-19T13:00:30Z', NOW)).toBe('expires in less than 1 min');
    expect(formatExpiresIn('2026-10-19T12:00:00Z', NOW)).toBe('expired');
    expect(formatExpiresIn(null, NOW)).toBe('never expires');
    expect(formatExpiresIn('garbage', NOW)).toBe('expires unknown');
  });
});
