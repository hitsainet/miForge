import '@testing-library/jest-dom/vitest';
import { beforeEach } from 'vitest';

/**
 * `ResizeObserver`, which jsdom does not implement.
 *
 * Recharts' `ResponsiveContainer` constructs one on mount, so without this **every component
 * that renders a chart throws** - and the throw is a `ReferenceError` from a vendored module,
 * which reads like a broken test rather than a missing browser API. It is a no-op observer:
 * jsdom reports every element as 0x0, so there is nothing to observe and nothing a size
 * callback could usefully say.
 *
 * **This is why `RunCurves.test.tsx` asserts the chart's COPY and tests the series
 * construction as a pure function** rather than asserting on rendered paths. A chart assertion
 * in jsdom would pass whatever the data was, which is precisely the construction-green failure
 * this project keeps finding - a check that never ran, reporting success.
 */
class NoopResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

if (!('ResizeObserver' in globalThis)) {
  (globalThis as { ResizeObserver?: unknown }).ResizeObserver = NoopResizeObserver;
}

/**
 * Every test starts with empty browser storage and a bare URL (task 13.0).
 *
 * Forms now keep their entries in `localStorage` and the navigation keeps the screen in the
 * URL's hash, and jsdom keeps both for the whole file. Without this, one test's typed batch
 * size is the next test's restored draft - a leak between tests that would read as a form
 * defaulting a value (M19), which is exactly the failure that first showed it.
 */
beforeEach(() => {
  try {
    window.localStorage.clear();
    window.sessionStorage.clear();
  } catch {
    // A test that replaced storage with a throwing double restores it itself.
  }
  window.history.replaceState(null, '', '/');
});
