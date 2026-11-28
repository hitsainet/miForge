import { describe, expect, it } from 'vitest';
import { countOf, nounFor } from './plural';

describe('[plural] countOf', () => {
  it('should use the singular for exactly one and the plural otherwise', () => {
    expect(countOf(1, 'adapter')).toBe('1 adapter');
    expect(countOf(2, 'adapter')).toBe('2 adapters');
    expect(countOf(0, 'adapter')).toBe('0 adapters');
  });

  it('should take an irregular plural', () => {
    expect(countOf(1, 'directory', 'directories')).toBe('1 directory');
    expect(countOf(3, 'directory', 'directories')).toBe('3 directories');
    expect(nounFor(1, 'replay or rerun', 'replays or reruns')).toBe('replay or rerun');
  });

  it('should choose the noun from the count, not from the written number', () => {
    expect(countOf(1000, 'commit', undefined, '1,000')).toBe('1,000 commits');
    expect(countOf(1, 'commit', undefined, '1')).toBe('1 commit');
  });
});
