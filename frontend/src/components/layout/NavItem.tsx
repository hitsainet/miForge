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
}

export function NavItem({ item, active, onSelect }: NavItemProps): ReactNode {
  const Icon = item.icon;
  // accent-ok:active-nav - ADR-012's first surface: tint background, accent text and
  // icon, on the ACTIVE item only. The inactive branch carries no accent token at all.
  const activeClasses = 'bg-accent-tint text-accent';
  const inactiveClasses = 'text-text hover:bg-card';

  const content = (
    <>
      <Icon size={17} aria-hidden="true" />
      <span className="truncate">{item.label}</span>
      {item.badge === undefined || item.badge.count === 0 ? null : (
        <span
          className="ml-auto rounded-full border border-line px-2 text-hint text-dim"
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
    (active ? activeClasses : inactiveClasses);

  if (item.href !== undefined) {
    return (
      <a
        href={item.href}
        className={classes}
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
      aria-current={active ? 'page' : undefined}
      onClick={() => onSelect(item.id)}
    >
      {content}
    </button>
  );
}
