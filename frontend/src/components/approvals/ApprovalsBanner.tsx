/**
 * The approvals banner (ENH-008 D5, §2.3): on every screen but Approvals, while any agent
 * request waits, above the screen's content and below the resource strip.
 *
 * - a region labelled "Agent requests waiting for your approval" (`approvals-banner`);
 * - its heading counts the waiting requests and is announced politely when it changes;
 * - at most {@link BANNER_MAX_CARDS} full cards, the soonest to expire first, then
 *   "and N more waiting — Open Approvals";
 * - **Collapse** reduces it to the heading. The choice lives in `sessionStorage` (wrapped: it
 *   can throw or be empty) together with the ids waiting when it was collapsed, so a request
 *   that arrives later opens it again;
 * - **Approve all settings changes (N)** when two or more settings changes wait (ENH-006 D79,
 *   `ApproveAllSettings`), counted over every waiting request, not only the cards shown;
 * - nothing at all when nothing waits and the last read succeeded, and the reason when the
 *   read failed - it never goes quietly dark.
 *
 * `App.tsx` mounts it in `Shell` and leaves it out on the Approvals screen, which shows the
 * same cards. The data is the one store the sidebar badge reads.
 */
import { useState, type ReactNode } from 'react';
import { ShieldCheck } from 'lucide-react';
import { describeFailure } from '@/api/errors';
import { useOptionalNavigation } from '@/contexts/NavigationContext';
import { useApprovalsStore } from '@/stores/approvalsStore';
import { ApproveAllSettings } from './ApproveAllSettings';
import { WaitingList } from './WaitingList';

export const BANNER_MAX_CARDS = 3;
export const BANNER_REGION_LABEL = 'Agent requests waiting for your approval';
export const BANNER_COLLAPSE_KEY = 'miforge.approvals-banner.collapsed';
/** Sticky inside the scrolling `<main>`; above page content, under dialogs. */
export const BANNER_DOCK_CLASS = 'sticky top-0 z-30 mb-6 bg-bg';

interface Collapsed {
  /** The ids waiting when the operator collapsed the banner. */
  readonly seen: readonly string[];
}

function readCollapsed(): Collapsed | null {
  try {
    const raw = window.sessionStorage.getItem(BANNER_COLLAPSE_KEY);
    if (raw === null) return null;
    const parsed: unknown = JSON.parse(raw);
    if (
      parsed !== null &&
      typeof parsed === 'object' &&
      Array.isArray((parsed as { seen?: unknown }).seen)
    ) {
      return { seen: (parsed as { seen: unknown[] }).seen.map(String) };
    }
    return null;
  } catch {
    return null;
  }
}

function writeCollapsed(value: Collapsed | null): void {
  try {
    if (value === null) window.sessionStorage.removeItem(BANNER_COLLAPSE_KEY);
    else window.sessionStorage.setItem(BANNER_COLLAPSE_KEY, JSON.stringify(value));
  } catch {
    // Private windows and blocked storage: the banner still works, it just forgets.
  }
}

export function bannerHeading(count: number): string {
  return `${count} agent request${count === 1 ? '' : 's'} waiting for your approval`;
}

function OpenApprovals({ children = 'Open Approvals' }: { readonly children?: string }): ReactNode {
  const navigation = useOptionalNavigation();
  return (
    <a
      href="#/approvals"
      className="font-medium underline"
      onClick={(event) => {
        if (navigation === null) return;
        event.preventDefault();
        navigation.go('approvals');
      }}
    >
      {children}
    </a>
  );
}

export function ApprovalsBanner(): ReactNode {
  const waiting = useApprovalsStore((state) => state.waiting);
  const failure = useApprovalsStore((state) => state.failure);
  const [collapsed, setCollapsedState] = useState<Collapsed | null>(readCollapsed);

  const rows = waiting ?? [];
  if (rows.length === 0 && failure === null) return null;

  // A request the operator has not seen yet opens the banner again.
  const isCollapsed =
    collapsed !== null && rows.every((row) => collapsed.seen.includes(row.approval_id));
  const setCollapsed = (value: Collapsed | null): void => {
    writeCollapsed(value);
    setCollapsedState(value);
  };
  const shown = rows.slice(0, BANNER_MAX_CARDS);
  const more = rows.length - shown.length;

  // The dock floats (operator 2026-10-28): sticky at the top of `shell-main`, the scrolling
  // element, so the waiting requests never scroll out of sight. It is opaque (`bg-bg`) because
  // the region's amber tint is translucent, and sits above page content but under dialogs
  // (z-50). The region caps its height and scrolls inside, so it never hides the whole screen.
  return (
    <div className={BANNER_DOCK_CLASS} data-testid="approvals-banner-dock">
      <section
        role="region"
        aria-label={BANNER_REGION_LABEL}
        data-testid="approvals-banner"
        className="flex max-h-[40vh] flex-col gap-3 overflow-y-auto rounded-card border border-amber/60 bg-amber/10 p-card"
      >
        {failure === null ? null : (
          <p className="text-body text-red" role="alert" data-testid="approvals-banner-failure">
            Could not read the approval queue: {describeFailure(failure)}
          </p>
        )}
        {rows.length === 0 ? null : (
          <>
            <div className="flex flex-wrap items-center gap-3">
              <ShieldCheck size={18} className="text-amber" aria-hidden="true" />
              <h2
                className="font-semibold"
                aria-live="polite"
                data-testid="approvals-banner-heading"
              >
                {bannerHeading(rows.length)}
              </h2>
              <OpenApprovals />
              <button
                type="button"
                className="ml-auto text-hint text-dim hover:underline"
                aria-expanded={!isCollapsed}
                onClick={() =>
                  setCollapsed(isCollapsed ? null : { seen: rows.map((row) => row.approval_id) })
                }
                data-testid="approvals-banner-toggle"
              >
                {isCollapsed ? 'Expand' : 'Collapse'}
              </button>
            </div>
            {isCollapsed ? null : (
              <>
                <ApproveAllSettings waiting={rows} where="banner" />
                <WaitingList
                  approvals={shown}
                  label="Waiting agent requests"
                  data-testid="approvals-banner-cards"
                />
                {more <= 0 ? null : (
                  <p className="text-body" data-testid="approvals-banner-more">
                    and {more} more waiting — <OpenApprovals />
                  </p>
                )}
              </>
            )}
          </>
        )}
      </section>
    </div>
  );
}
