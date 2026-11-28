/**
 * What is pinned at the top of `shell-main`, and how far down it reaches (Tab-focus fix,
 * 2026-11-26).
 *
 * Several elements stick to the top of the shell's scroll container: the approvals dock on every
 * screen but Approvals, and, below it, a screen's own pinned block (the WSP run page's floating
 * header, the WSP runs list's header). A focused element scrolled into view by Tab must land
 * below all of them. Chrome scrolls a focused element into the scroller's padded viewport and
 * ignores the element's own `scroll-margin` (measured 2026-11-25), so `shell-main` takes
 * `--shell-pinned-height` as its `scroll-padding-top` (`AppShell`).
 *
 * Each pinned element registers itself with `trackPinned` (`pinnedRef` as its `ref` callback). The
 * registry keeps each element's measured height (a `ResizeObserver` follows it) and publishes
 * the **total** on `<html>`; the property is removed when nothing is registered. Pinned elements
 * stack (each page block sits at `top: var(--approvals-dock-height)`), so the total is how far
 * down the pinned stack reaches. There is no list of pinned elements anywhere to keep in step:
 * an element is counted because it registered, and stops counting when it unmounts.
 *
 * **Pinned at the bottom** (2026-11-27): an approval card's decision row sticks to the bottom of
 * its scroller (`shell-main` on Approvals, the dock's region elsewhere), and a control reached by
 * forward Tab was scrolled into view behind it ("All arguments the agent sent", measured on
 * Approvals). Those rows register with `trackPinnedBottom` (`pinnedBottomRef`). They do not stack:
 * each sticks inside its own card, so at most one covers a scroller's bottom at a time, and the
 * registry publishes the **tallest** as `--shell-pinned-bottom-height`, which `shell-main` and the
 * dock's region take as their `scroll-padding-bottom`.
 */

/** The total height of every pinned element: `shell-main`'s scroll padding. */
export const SHELL_PINNED_HEIGHT_VAR = '--shell-pinned-height';

/** The tallest element pinned at a scroller's bottom: the scroll padding below. */
export const SHELL_PINNED_BOTTOM_HEIGHT_VAR = '--shell-pinned-bottom-height';

const heights = new Map<Element, number>();
const bottomHeights = new Map<Element, number>();

function publishTallestBottom(): void {
  const root = document.documentElement.style;
  if (bottomHeights.size === 0) {
    root.removeProperty(SHELL_PINNED_BOTTOM_HEIGHT_VAR);
    return;
  }
  root.setProperty(SHELL_PINNED_BOTTOM_HEIGHT_VAR, `${Math.max(...bottomHeights.values())}px`);
}

function publishTotal(): void {
  const root = document.documentElement.style;
  if (heights.size === 0) {
    root.removeProperty(SHELL_PINNED_HEIGHT_VAR);
    return;
  }
  let total = 0;
  for (const height of heights.values()) total += height;
  root.setProperty(SHELL_PINNED_HEIGHT_VAR, `${total}px`);
}

/**
 * Registers a pinned element (a `ref` callback; React 19 runs the returned cleanup on unmount).
 * `onHeight`, when given, hears each measured height too (the dock publishes its own for the
 * blocks that sit below it).
 */
export function trackPinned(element: HTMLElement, onHeight?: (height: number) => void): () => void {
  const read = (): void => {
    const height = element.getBoundingClientRect().height;
    heights.set(element, height);
    onHeight?.(height);
    publishTotal();
  };
  read();
  const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(read);
  observer?.observe(element);
  return () => {
    observer?.disconnect();
    heights.delete(element);
    publishTotal();
  };
}

/** `trackPinned` as a `ref` callback: `<div ref={pinnedRef}>` registers the element as pinned. */
export function pinnedRef(element: HTMLElement | null): (() => void) | undefined {
  return element === null ? undefined : trackPinned(element);
}

/** Registers an element pinned at its scroller's bottom (a `ref` callback's body). */
export function trackPinnedBottom(element: HTMLElement): () => void {
  const read = (): void => {
    bottomHeights.set(element, element.getBoundingClientRect().height);
    publishTallestBottom();
  };
  read();
  const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(read);
  observer?.observe(element);
  return () => {
    observer?.disconnect();
    bottomHeights.delete(element);
    publishTallestBottom();
  };
}

/** `trackPinnedBottom` as a `ref` callback. */
export function pinnedBottomRef(element: HTMLElement | null): (() => void) | undefined {
  return element === null ? undefined : trackPinnedBottom(element);
}
