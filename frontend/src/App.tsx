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
 * resource strip.
 *
 * **Deleting a row from `ROUTES` makes `routes.test.ts` and the matching Playwright spec go
 * red** - the specs navigate through this nav rather than by URL. That is M1 and M2.
 */
import type { ReactNode } from 'react';
import { Activity, Cpu, HardDrive, MemoryStick, Thermometer } from 'lucide-react';
import { AppShell } from '@/components/layout/AppShell';
import { PageHead } from '@/components/layout/PageHead';
import { Callout } from '@/components/common/Callout';
import type { ResourceReading, ShellNavItem } from '@/components/layout/types';
import { NavigationProvider, useNavigation, type ScreenId } from '@/contexts/NavigationContext';
import { ROUTES, navEntries, routeFor } from '@/config/routes';
import { AttachmentChip } from '@/components/panels/hotupdate/AttachmentChip';
import { useAttachmentStore } from '@/stores/attachmentStore';

/**
 * Host readings, all `null`.
 *
 * **Nothing serves them.** The mockup's strip shows CPU, RAM, GPU, VRAM and temperature, and
 * no requirement and no endpoint defines host metrics - `GET /api/health` reports dependencies
 * and integrations, not utilisation. So every reading renders as **not reported** rather than
 * as the mockup's numbers, which is the same rule `AbsentValue` enforces everywhere else: a
 * missing value is named, never printed as 0.
 */
const UNREPORTED_RESOURCES: readonly ResourceReading[] = [
  { id: 'cpu', label: 'CPU', icon: Cpu, value: null, unit: '%' },
  { id: 'ram', label: 'RAM', icon: MemoryStick, value: null, unit: 'GB' },
  { id: 'gpu', label: 'GPU', icon: Activity, value: null, unit: '%' },
  { id: 'vram', label: 'VRAM', icon: HardDrive, value: null, unit: 'GB' },
  { id: 'temp', label: 'Temp', icon: Thermometer, value: null, unit: '°C' },
];

export const APP_VERSION = 'v0.1.0';

/** The nav, derived from the registry. Not a list. */
function shellNav(): readonly ShellNavItem[] {
  return navEntries().map((route) => ({
    id: route.id,
    label: route.nav?.label ?? route.title,
    icon: route.nav?.icon ?? Activity,
  }));
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
 * The resource strip's right-hand slot (F11 16.4): the attachment chip while an attached run
 * is open, nothing otherwise. The strip and the shell stay ignorant of miLLM (ADR-022).
 */
function AttachmentSlot(): ReactNode {
  const attachment = useAttachmentStore((state) => state.attachment);
  return attachment === null ? null : <AttachmentChip attachment={attachment} />;
}

function Shell(): ReactNode {
  const navigation = useNavigation();
  const nav = shellNav();
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
      resources={UNREPORTED_RESOURCES}
      resourceTrailing={<AttachmentSlot />}
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
      {...(initialScreen === undefined ? {} : { initialScreen })}
      {...(initialParams === undefined ? {} : { initialParams })}
    >
      <Shell />
    </NavigationProvider>
  );
}
