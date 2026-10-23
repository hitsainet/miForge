/**
 * The application root: **the router, rendered from `config/routes.ts`** (FR-009.1, .2).
 *
 * This file is the one registration point on the frontend, and it is deliberately the whole
 * of it: the nav comes from `navEntries()`, the screen comes from `routeFor()`, and the page
 * head's title and subtitle come from the same row. **There is no second shell and no second
 * list of screens** - two lists of screens is the condition ADR-016 #1 exists to make
 * impossible, and the foundation's harness marked its own placeholder nav so that F9 would
 * replace it rather than add to it.
 *
 * `App` composes foundation task 7.3's layout components and defines no layout of its own
 * (FR-009.2): 224 px sidebar, the logo tile, the page head with its one-line subtitle, the
 * resource strip. The sidebar's footer (Settings, the Manual, the version) is a row of the
 * same table with `nav.placement: 'footer'`, plus `APP_VERSION`.
 *
 * **Deleting a row from `ROUTES` makes `routes.test.ts` and the matching Playwright spec go
 * red** - the specs navigate through this nav rather than by URL. That is M1 and M2.
 *
 * `ROUTES` is also handed to the `NavigationProvider`, whose rows' `path`s are what the URL
 * carries (task 13.0): a reload keeps the screen. With no `initialScreen` the first screen is
 * read from the URL, and an unknown URL falls back to Runs.
 */
import { useEffect, type ReactNode } from 'react';
import { Activity } from 'lucide-react';
import { AppShell } from '@/components/layout/AppShell';
import { PageHead } from '@/components/layout/PageHead';
import { Callout } from '@/components/common/Callout';
import type { ShellNavItem } from '@/components/layout/types';
import { hostReadings } from '@/components/layout/hostReadings';
import { NavigationProvider, useNavigation, type ScreenId } from '@/contexts/NavigationContext';
import { ROUTES, navEntries, routeFor } from '@/config/routes';
import { AttachmentChip } from '@/components/panels/hotupdate/AttachmentChip';
import { fetchHostResources } from '@/api/host';
import { fetchAttachmentStatus } from '@/api/attachment';
import { usePolledRead } from '@/hooks/usePolledRead';
import { APPROVALS_POLL_MS, useApprovalsStore } from '@/stores/approvalsStore';
import { version as packageVersion } from '../package.json';

/** How often the strip re-reads the host. Fast enough to watch a run load the card. */
export const HOST_POLL_MS = 5_000;
/** How often the chip re-reads miLLM's attachment state through the backend. */
export const ATTACHMENT_POLL_MS = 15_000;

/**
 * The version the sidebar footer shows, **read from `package.json`** rather than typed a second
 * time: the suite versioning rule (operator decides bumps; 0.1.0 until the feature MVP) moves
 * one field, and `version.test.ts` holds it equal to the backend's `pyproject.toml`.
 */
export const APP_VERSION = `v${packageVersion}`;

/** The nav, derived from the registry. Not a list. `pending` badges the Approvals entry. */
function shellNav(pending: number): readonly ShellNavItem[] {
  return navEntries().map((route) => ({
    id: route.id,
    label: route.nav?.label ?? route.title,
    icon: route.nav?.icon ?? Activity,
    ...(route.nav?.placement === undefined ? {} : { placement: route.nav.placement }),
    ...(route.id === 'approvals'
      ? {
          badge: {
            count: pending,
            title: `${pending} pending approval${pending === 1 ? '' : 's'}`,
          },
        }
      : {}),
  }));
}

/**
 * F12: the approval queue is read on every screen, so the sidebar's count is current wherever
 * the operator is - an agent's request should not wait unseen behind another screen.
 */
function useApprovalsPolling(): number {
  const load = useApprovalsStore((state) => state.load);
  const pending = useApprovalsStore((state) => state.listing?.pending ?? 0);
  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), APPROVALS_POLL_MS);
    return () => clearInterval(timer);
  }, [load]);
  return pending;
}

function Screen(): ReactNode {
  const navigation = useNavigation();
  const route = routeFor(navigation.screen);

  if (route === undefined) {
    // Unreachable through the nav, which only offers ids from the registry - and refused by
    // name rather than rendered blank, because a blank screen is indistinguishable from a
    // screen whose data has not arrived.
    return (
      <>
        <PageHead title="miForge" subtitle="That screen is not in the route registry" />
        <Callout tone="error" title={`No screen is registered as "${navigation.screen}"`}>
          The registry lists {ROUTES.map((entry) => entry.id).join(', ')}. A screen not in it is not
          reachable, which is what makes the reachability test meaningful.
        </Callout>
      </>
    );
  }

  const Element = route.element;
  return <Element />;
}

/**
 * The resource strip's right-hand slot (F11 16.4): miLLM's attachment state **now**, read
 * through `GET /api/v1/attachment` on every screen, or nothing when no serving process is
 * configured (the standalone default). The strip and the shell stay ignorant of miLLM
 * (ADR-022); the chip is §5.10's component.
 */
function AttachmentSlot(): ReactNode {
  const { value, failure } = usePolledRead(fetchAttachmentStatus, ATTACHMENT_POLL_MS);
  if (value === null || !value.configured) return null;
  return <AttachmentChip status={value} outdated={failure !== null} />;
}

function Shell(): ReactNode {
  const navigation = useNavigation();
  const nav = shellNav(useApprovalsPolling());
  const host = usePolledRead(fetchHostResources, HOST_POLL_MS);
  // The nav highlights the row the screen belongs to. The run view has no nav row, so Runs
  // stays current while it is open - which is where the operator came from.
  const activeId = nav.some((item) => item.id === navigation.screen)
    ? navigation.screen
    : navigation.screen === 'run'
      ? 'runs'
      : null;

  return (
    <AppShell
      nav={nav}
      activeId={activeId}
      onNavigate={(id) => navigation.go(id as ScreenId)}
      version={APP_VERSION}
      resources={hostReadings(host.value, host.failure)}
      resourceTrailing={<AttachmentSlot />}
      // A new screen, or another run in the run view, opens at the top. In-screen parameters
      // (written with `replace`) are not part of the key, so they keep the operator's place.
      scrollKey={`${navigation.screen}/${navigation.params['runId'] ?? ''}`}
    >
      <Screen />
    </AppShell>
  );
}

export interface AppProps {
  readonly initialScreen?: ScreenId;
  readonly initialParams?: Readonly<Record<string, string>>;
}

export function App({ initialScreen, initialParams }: AppProps = {}): ReactNode {
  return (
    <NavigationProvider
      routes={ROUTES}
      {...(initialScreen === undefined ? {} : { initialScreen })}
      {...(initialParams === undefined ? {} : { initialParams })}
    >
      <Shell />
    </NavigationProvider>
  );
}
