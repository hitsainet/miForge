/** The hook every persisted form uses (task 13.3; AC-009.34, .35, .37). **M24's target.** */
import { describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { usePersistentState } from './usePersistentState';
import { DRAFT_PREFIX } from '@/stores/draftStorage';

const options = { version: 1, decode: (data: unknown) => (typeof data === 'string' ? data : null) };

describe('[usePersistentState]', () => {
  it('should restore the value after a remount, and report when it was saved', () => {
    const first = renderHook(() => usePersistentState('k', '', options));
    act(() => first.result.current.set('typed'));
    expect(first.result.current.restoredAt).toBeNull();
    first.unmount();
    const second = renderHook(() => usePersistentState('k', '', options));
    expect(second.result.current.value).toBe('typed');
    expect(second.result.current.restoredAt).toEqual(expect.any(Number));
  });

  it('should remove the entry on clear(), and when the value returns to the initial one', () => {
    const view = renderHook(() => usePersistentState('k', '', options));
    act(() => view.result.current.set('typed'));
    expect(window.localStorage.getItem(`${DRAFT_PREFIX}k`)).not.toBeNull();
    act(() => view.result.current.clear());
    expect(view.result.current.value).toBe('');
    expect(window.localStorage.getItem(`${DRAFT_PREFIX}k`)).toBeNull();
    act(() => view.result.current.set('again'));
    act(() => view.result.current.set(''));
    expect(window.localStorage.getItem(`${DRAFT_PREFIX}k`)).toBeNull();
  });

  it('should remove the entry synchronously on clear(), before any re-render', () => {
    const view = renderHook(() => usePersistentState('k', '', options));
    act(() => view.result.current.set('typed'));
    const clear = view.result.current.clear;
    // A submit handler that clears and then navigates away unmounts the form in the same batch.
    act(() => {
      clear();
      view.unmount();
    });
    expect(window.localStorage.getItem(`${DRAFT_PREFIX}k`)).toBeNull();
  });

  it('should read the new key when the key changes, and never write the old value under it', () => {
    const view = renderHook(({ k }) => usePersistentState(k, '', options), {
      initialProps: { k: 'a' as string | null },
    });
    act(() => view.result.current.set('for a'));
    view.rerender({ k: 'b' });
    expect(view.result.current.value).toBe('');
    expect(window.localStorage.getItem(`${DRAFT_PREFIX}b`)).toBeNull();
    view.rerender({ k: 'a' });
    expect(view.result.current.value).toBe('for a');
  });

  it('should work, unpersisted, when storage throws on every call', () => {
    const boom = () => {
      throw new DOMException('denied', 'SecurityError');
    };
    const spies = [
      vi.spyOn(Storage.prototype, 'getItem').mockImplementation(boom),
      vi.spyOn(Storage.prototype, 'setItem').mockImplementation(boom),
      vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(boom),
    ];
    try {
      const view = renderHook(() => usePersistentState('k', '', options));
      act(() => view.result.current.set('typed'));
      expect(view.result.current.value).toBe('typed');
      act(() => view.result.current.clear());
      expect(view.result.current.value).toBe('');
    } finally {
      for (const spy of spies) spy.mockRestore();
    }
  });
});
