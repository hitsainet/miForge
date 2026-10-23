/**
 * Screen selection, with no router (FR-009.1; ADR-011) - **and the screen is in the URL**
 * (F9 task 13.0; operator request 2026-10-17; closes FPRD F-6).
 *
 * **ADR-011 lists no router.** Its frontend stack is React 18, TypeScript strict, Vite,
 * Tailwind CSS 4, Zustand 5, `socket.io-client`, `lucide-react` and Recharts, and adding
 * `react-router-dom` is an ADR amendment the operator owns rather than a feature decision.
 * The mockup switches screens with `useState`; this is that, with the screen set and the
 * parameters typed, and with one place that performs the switch so `config/routes.ts` stays
 * the only registry.
 *
 * ## The URL (task 13.2)
 *
 * Until 2026-10-17 nothing was written anywhere, so **a reload always landed on Runs** and
 * lost the open run, the open manifest and the onboarding step. Given `routes` (the app passes
 * `ROUTES`), the provider now:
 *
 * - **reads** the hash for its first state when no `initialScreen` is given - an unknown or
 *   malformed hash falls back to the default screen, and the URL is rewritten to match;
 * - **writes** the hash from the state: `go()` **pushes** a history entry, so back and forward
 *   walk screens; `go(..., {replace: true})` - a parameter changing within a screen, such as
 *   the onboarding step - **replaces** it, so back does not walk keystrokes;
 * - **follows** `popstate` and `hashchange`, so back, forward, a typed URL and a bookmark all
 *   move the screen.
 *
 * The codec is `config/urlState.ts`. Without `routes` the provider is URL-free, which is what
 * a component test that mounts one screen wants.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { fromHash, toHash, type PathRoute } from '@/config/urlState';

/** The screens. Extending this is adding a screen, which needs a requirement (FPRD F-3). */
export type ScreenId =
  | 'runs'
  | 'run'
  | 'checkpoints'
  | 'new-run'
  | 'packs'
  | 'onboard'
  | 'approvals'
  | 'help'
  | 'settings';

export type ScreenParams = Readonly<Record<string, string>>;

export interface GoOptions {
  /** Replace the current history entry rather than adding one. For in-screen parameters. */
  readonly replace?: boolean;
}

export interface Navigation {
  readonly screen: ScreenId;
  readonly params: ScreenParams;
  readonly go: (screen: ScreenId, params?: ScreenParams, options?: GoOptions) => void;
}

const NavigationCtx = createContext<Navigation | null>(null);

export interface NavigationProviderProps {
  readonly initialScreen?: ScreenId;
  readonly initialParams?: ScreenParams;
  /** The route table whose paths the URL carries. Absent: the provider never touches the URL. */
  readonly routes?: readonly PathRoute<ScreenId>[];
  readonly children: ReactNode;
}

export const DEFAULT_SCREEN: ScreenId = 'runs';

interface NavState {
  readonly screen: ScreenId;
  readonly params: ScreenParams;
}

function readLocation(routes: readonly PathRoute<ScreenId>[]): NavState | null {
  try {
    return fromHash(window.location.hash, routes);
  } catch {
    return null;
  }
}

function sameState(a: NavState, b: NavState): boolean {
  if (a.screen !== b.screen) return false;
  const keysA = Object.keys(a.params);
  if (keysA.length !== Object.keys(b.params).length) return false;
  return keysA.every((key) => a.params[key] === b.params[key]);
}

export function NavigationProvider({
  initialScreen,
  initialParams = {},
  routes,
  children,
}: NavigationProviderProps): ReactNode {
  const [state, setState] = useState<NavState>(() => {
    if (initialScreen !== undefined) return { screen: initialScreen, params: initialParams };
    const fromUrl = routes === undefined ? null : readLocation(routes);
    return fromUrl ?? { screen: DEFAULT_SCREEN, params: initialParams };
  });
  /** How the next state change reaches the URL. `replace` for the first write: normalising. */
  const historyMode = useRef<'push' | 'replace'>('replace');

  const go = useCallback((screen: ScreenId, params: ScreenParams = {}, options?: GoOptions) => {
    historyMode.current = options?.replace === true ? 'replace' : 'push';
    setState({ screen, params });
  }, []);

  // State -> URL. Removing this effect is M23: the URL never changes and a reload lands on Runs.
  useEffect(() => {
    if (routes === undefined) return;
    const mode = historyMode.current;
    historyMode.current = 'push';
    const hash = toHash(state.screen, state.params, routes);
    if (hash === null || hash === window.location.hash) return;
    try {
      if (mode === 'replace') window.history.replaceState(window.history.state, '', hash);
      else window.history.pushState(window.history.state, '', hash);
    } catch {
      // A sandboxed frame can refuse history writes. The screen still works; it just will
      // not survive a reload, which is how it behaved before the URL carried it.
    }
  }, [state, routes]);

  // URL -> state: back, forward, a typed URL, a bookmark.
  useEffect(() => {
    if (routes === undefined) return;
    const follow = (): void => {
      const next = readLocation(routes) ?? { screen: DEFAULT_SCREEN, params: {} };
      historyMode.current = 'replace';
      setState((current) => (sameState(current, next) ? current : next));
    };
    window.addEventListener('popstate', follow);
    window.addEventListener('hashchange', follow);
    return () => {
      window.removeEventListener('popstate', follow);
      window.removeEventListener('hashchange', follow);
    };
  }, [routes]);

  const value = useMemo<Navigation>(
    () => ({ screen: state.screen, params: state.params, go }),
    [state.screen, state.params, go],
  );

  return <NavigationCtx.Provider value={value}>{children}</NavigationCtx.Provider>;
}

/**
 * Read the navigation.
 *
 * Throws outside a provider rather than returning a no-op. A silent no-op would make every
 * button on a mis-mounted screen inert - which is the mockup's own recorded bug, where
 * `Primary` took no `onClick` and every primary action did nothing (handoff section 6).
 */
export function useNavigation(): Navigation {
  const value = useContext(NavigationCtx);
  if (value === null) {
    throw new Error(
      'useNavigation was called outside a NavigationProvider. Wrap the tree in <NavigationProvider>.',
    );
  }
  return value;
}

/**
 * The navigation, or `null` outside a provider - **for the page head only**, which a component
 * test may render with no provider: its per-screen Help link (ENH-004) then simply does not
 * render. Anything that navigates uses {@link useNavigation}, which throws.
 */
export function useOptionalNavigation(): Navigation | null {
  return useContext(NavigationCtx);
}
