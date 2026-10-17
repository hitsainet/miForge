/**
 * Screen selection, with no router (FR-009.1; ADR-011; FPRD F-6).
 *
 * **ADR-011 lists no router.** Its frontend stack is React 18, TypeScript strict, Vite,
 * Tailwind 3, Zustand 5, `socket.io-client`, `lucide-react` and Recharts, and adding
 * `react-router-dom` is an ADR amendment the operator owns rather than a feature decision.
 * The mockup switches screens with `useState`; this is that, with the screen set and the
 * parameters typed, and with one place that performs the switch so `config/routes.ts` stays
 * the only registry.
 *
 * **What the operator loses, recorded rather than hidden** (FPRD F-6): there is no deep link.
 * A run that takes hours is exactly the thing somebody wants to bookmark, and they cannot.
 * The `path` on every route entry exists so that adopting a router is a change to this file
 * and `App.tsx` alone - the table already holds the URLs.
 */
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';

/** The screens. Extending this is adding a screen, which needs a requirement (FPRD F-3). */
export type ScreenId = 'runs' | 'run' | 'checkpoints' | 'new-run' | 'packs' | 'onboard';

export type ScreenParams = Readonly<Record<string, string>>;

export interface Navigation {
  readonly screen: ScreenId;
  readonly params: ScreenParams;
  readonly go: (screen: ScreenId, params?: ScreenParams) => void;
}

const NavigationCtx = createContext<Navigation | null>(null);

export interface NavigationProviderProps {
  readonly initialScreen?: ScreenId;
  readonly initialParams?: ScreenParams;
  readonly children: ReactNode;
}

export function NavigationProvider({
  initialScreen = 'runs',
  initialParams = {},
  children,
}: NavigationProviderProps): ReactNode {
  const [state, setState] = useState<{ screen: ScreenId; params: ScreenParams }>({
    screen: initialScreen,
    params: initialParams,
  });

  const go = useCallback((screen: ScreenId, params: ScreenParams = {}) => {
    setState({ screen, params });
  }, []);

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
