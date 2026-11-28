/**
 * One sidebar nav entry (foundation task 7.3; ADR-012).
 *
 * The active item is ADR-012's first permitted accent surface: *"the active nav item
 * (tint background, orange text and icon)"*. An inactive item is `text` on `panel` with
 * no accent anywhere - that asymmetry is the one thing `NavItem.test.tsx` and
 * `accent.test.ts` both check, because "the nav is orange" and "the active nav item is
 * orange" look identical in a screenshot of a one-item nav.
 *
 * `aria-current="page"` rather than a class alone: a screen reader user needs to know
 * which screen they are on, and a tint is not announced.
 */

import type { ReactNode } from 'react';
import type { ShellNavItem } from './types';

export interface NavItemProps {
  readonly item: ShellNavItem;
  readonly active: boolean;
  readonly onSelect: (id: string) => void;
  /** Icon only, the label as a tooltip, as miStudio's and miLLM's collapsed items. */
  readonly collapsed?: boolean;
}

export function NavItem({ item, active, onSelect, collapsed = false }: NavItemProps): ReactNode {
  const Icon = item.icon;
  // accent-ok:active-nav - ADR-012's first surface: tint background, accent text and
  // icon, on the ACTIVE item only. The inactive branch carries no accent token at all.
  const activeClasses = 'bg-accent-tint text-accent';
  const inactiveClasses = 'text-text hover:bg-card';

  const content = (
    <>
      <span className="relative shrink-0">
        <Icon size={17} aria-hidden="true" />
        {/* Collapsed, the waiting count shrinks to a dot on the icon (miStudio's rail). */}
        {collapsed && item.badge !== undefined && item.badge.count > 0 ? (
          <span
            className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-amber"
            aria-hidden="true"
          />
        ) : null}
      </span>
      {/* sr-only rather than absent, so the item keeps its accessible name collapsed. */}
      <span className={collapsed ? 'sr-only' : 'truncate'}>{item.label}</span>
      {/* A waiting count is a call to act, so it is filled amber - the warning tone, not
          the accent (ADR-012 keeps orange for the active item and the logo tile). */}
      {item.badge === undefined || item.badge.count === 0 ? null : (
        <span
          className={`${collapsed ? 'sr-only' : 'ml-auto'} h-5 min-w-5 rounded-full bg-amber px-1.5 text-center text-[11px] leading-5 font-semibold text-white`}
          title={item.badge.title}
          aria-label={item.badge.title}
          data-testid={`nav-badge-${item.id}`}
        >
          {item.badge.count}
        </span>
      )}
    </>
  );
  const classes =
    'mb-0.5 flex w-full items-center gap-3 rounded-control px-3 py-2.5 text-left text-body ' +
    (collapsed ? 'justify-center ' : '') +
    (active ? activeClasses : inactiveClasses);
  const title = collapsed ? item.label : undefined;

  if (item.href !== undefined) {
    return (
      <a
        href={item.href}
        className={classes}
        title={title}
        aria-current={active ? 'page' : undefined}
        onClick={(event) => {
          // Left-click navigates in-app; a modifier or middle click is left to the
          // browser so "open in a new tab" keeps working on a real href.
          if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
          event.preventDefault();
          onSelect(item.id);
        }}
      >
        {content}
      </a>
    );
  }

  return (
    <button
      type="button"
      className={classes}
      title={title}
      aria-current={active ? 'page' : undefined}
      onClick={() => onSelect(item.id)}
    >
      {content}
    </button>
  );
}
