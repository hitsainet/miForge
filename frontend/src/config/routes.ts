/**
 * **The live route registry** (FR-009.1; ADR-016 #1; ADR-011).
 *
 * The sidebar's nav, the router and every screen-reachability test read **this array and
 * nothing else**. A screen not in this table is not reachable and therefore does not exist.
 *
 * ## Why a table and not a router's own registry
 *
 * **ADR-011 lists no router.** Its frontend stack is React 18, TypeScript strict, Vite,
 * Tailwind 3, Zustand 5, `socket.io-client`, `lucide-react` and Recharts; the mockup switches
 * screens with `useState`. Adding `react-router-dom` is an **ADR amendment the operator owns**,
 * not a feature decision. A declarative table needs no dependency - and it gives the
 * reachability test something to assert against, which is the whole point: ADR-016 #1 requires
 * a test that **fails when the wiring is removed**, and *"the component file imports"* is not
 * wiring. Deleting a row here must turn `routes.test.ts` **and** the matching Playwright spec
 * red, because the specs navigate through the real nav rather than by URL (**M1**, **M2**).
 *
 * **What the operator loses, and might want back** (FPRD **F-6**): there is no deep link. A run
 * that takes hours is exactly the thing somebody wants to bookmark, and `path` on every row
 * exists so that adopting a router is a change to `NavigationContext` and `App.tsx` alone - the
 * URLs are already here.
 *
 * ## Six rows, and why each one is allowed to exist
 *
 * **Four are F9's** (FPRD section 1.4): `runs`, `run`, `checkpoints`, `new-run`. The fourth
 * exists because **R-02.53 arrived** - *that is the only reason an entry may be added* (FPRD
 * **F-3**). **Two are F7's** (its task 12.9): `packs` and `onboard`, which R-02.39 defines.
 *
 * Adding a seventh is adding a screen, which needs a requirement. The screens the mockup has
 * and this product does not - Selectors, Settings - are **deliberately absent**: F3 assigns a
 * Selectors screen to F9 in a feature document with no requirement behind it, and the Settings
 * screen has no `R-02.x` and contradicts ADR-020's environment-sourced configuration.
 *
 * `nav: null` keeps the run view out of the sidebar while keeping it in the registry: it needs
 * a `runId`, so there is nothing for a nav item to navigate to. The form's `nav` is **set**,
 * because it is reachable directly and is the product's primary action.
 */
import type { ComponentType } from 'react';
import { Database, GitBranch, Hammer, LayoutGrid, Sparkles } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { ScreenId } from '@/contexts/NavigationContext';
import { RunIndexPage } from '@/components/panels/runs/RunIndexPage';
import { RunViewPage } from '@/components/panels/run/RunViewPage';
import { CheckpointTreePage } from '@/components/panels/tree/CheckpointTreePage';
import { NewRunForm } from '@/components/panels/newrun/NewRunForm';
import { PacksPage } from '@/components/panels/packs/PacksPage';
import { OnboardPage } from '@/components/panels/packs/OnboardPage';

export interface RouteNav {
  readonly label: string;
  readonly icon: LucideIcon;
}

export interface RouteEntry {
  readonly id: ScreenId;
  /** The URL this screen would have. Unused today; see FPRD F-6 in the module docstring. */
  readonly path: string;
  /** `null` for a screen reachable only by navigation, because it needs a parameter. */
  readonly nav: RouteNav | null;
  readonly title: string;
  /**
   * The one-line subtitle every page opens with (ADR-012).
   *
   * It lives **here, not in the page component**, because a field in a table is checkable:
   * `routes.test.ts` asserts every entry has a non-empty one, which a convention cannot.
   */
  readonly subtitle: string;
  readonly element: ComponentType;
  /** Which feature owns the screen. Read by the test, so a row cannot be added unattributed. */
  readonly owner: 'F9' | 'F7';
}

export const ROUTES: readonly RouteEntry[] = [
  {
    id: 'runs',
    path: '/runs',
    nav: { label: 'Runs', icon: LayoutGrid },
    title: 'Runs',
    subtitle: 'Push a model toward a domain you have described as a pack, one step at a time',
    element: RunIndexPage,
    owner: 'F9',
  },
  {
    id: 'run',
    path: '/runs/:runId',
    // Needs a runId, so there is nothing for a sidebar item to navigate to.
    nav: null,
    title: 'Run',
    subtitle: 'One run: its loss, its test and retention scores, its save points and its log',
    element: RunViewPage,
    owner: 'F9',
  },
  {
    id: 'new-run',
    path: '/runs/new',
    nav: { label: 'New run', icon: Hammer },
    title: 'New run',
    subtitle:
      'One definition: the model, the pack, the signal, the configuration, the cadences and at least one stop condition',
    element: NewRunForm,
    owner: 'F9',
  },
  {
    id: 'checkpoints',
    path: '/checkpoints',
    nav: { label: 'Checkpoints', icon: GitBranch },
    title: 'Checkpoints',
    subtitle: 'Every saved step is a loadable model. A branch can start from any checkpoint.',
    element: CheckpointTreePage,
    owner: 'F9',
  },
  {
    id: 'packs',
    path: '/packs',
    nav: { label: 'Domain packs', icon: Database },
    title: 'Domain packs',
    subtitle: 'A pack is everything miForge needs to know about a domain',
    element: PacksPage,
    owner: 'F7',
  },
  {
    id: 'onboard',
    path: '/packs/onboard',
    nav: { label: 'Onboard a domain', icon: Sparkles },
    title: 'Onboard a domain',
    subtitle: 'Seven steps · progress is saved as you go',
    element: OnboardPage,
    owner: 'F7',
  },
];

/** The nav items, in table order. The sidebar maps over **this**, not over a second list. */
export function navEntries(): readonly RouteEntry[] {
  return ROUTES.filter((route) => route.nav !== null);
}

export function routeFor(id: ScreenId): RouteEntry | undefined {
  return ROUTES.find((route) => route.id === id);
}
