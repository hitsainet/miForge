/**
 * Light/dark theme (foundation tasks 7.1, 7.3, 7.6; ADR-011's `darkMode: 'class'`).
 *
 * **Why the shell owns a theme at all.** The mockup is dark only - its handoff 2 gives
 * one palette and 6 lists its gaps without mentioning a light mode - but ADR-011
 * mandates Tailwind's class strategy and foundation 7.6 wants a Playwright smoke spec per
 * shell region in **light and dark**. A class strategy with nothing that sets the class is
 * a configuration that has never been exercised, so the toggle is what makes 7.6's "light
 * and dark" a real path rather than a claim.
 *
 * `contexts/` is for the shell's concerns, not for feature data (ADR-017, F9 FTDD 7.2:
 * "no global store, no context for data"). A theme is the shell's.
 *
 * **Reading the stored preference cannot throw the app down.** `localStorage` is
 * unavailable in a sandboxed iframe and throws on access in some privacy modes, so every
 * read and write is guarded and an unreadable store degrades to the default rather than
 * to a blank page.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

export type Theme = 'light' | 'dark';

/** Dark is the default: every screen in the mockup and the documents is dark. */
export const DEFAULT_THEME: Theme = 'dark';

export const THEME_STORAGE_KEY = 'miforge.theme';

export interface ThemeContextValue {
  readonly theme: Theme;
  readonly setTheme: (theme: Theme) => void;
  readonly toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function isTheme(value: unknown): value is Theme {
  return value === 'light' || value === 'dark';
}

/** The stored preference, or `null` when there is none or the store is unreachable. */
export function readStoredTheme(): Theme | null {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isTheme(stored) ? stored : null;
  } catch {
    return null;
  }
}

function writeStoredTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // A preference that cannot be saved is a lost convenience, not a failure. Refusing
    // the toggle here would be the fail-closed rule applied where no correctness is at
    // stake (ADR-013) - the theme does not change what any number means.
  }
}

/**
 * Put the theme on `<html>` where Tailwind's class strategy looks for it.
 *
 * Exported because `main.tsx` calls it **before** React mounts: applying it in an effect
 * alone paints one frame of the wrong theme, which is the flash every class-strategy app
 * ships once.
 */
export function applyThemeClass(theme: Theme, root: HTMLElement = document.documentElement): void {
  root.classList.toggle('dark', theme === 'dark');
  root.classList.toggle('light', theme === 'light');
  root.dataset['theme'] = theme;
}

export function ThemeProvider({
  children,
  initialTheme,
}: {
  children: ReactNode;
  /** For tests and for `main.tsx`, which has already read storage. */
  initialTheme?: Theme;
}): ReactNode {
  const [theme, setThemeState] = useState<Theme>(
    () => initialTheme ?? readStoredTheme() ?? DEFAULT_THEME,
  );

  useEffect(() => {
    applyThemeClass(theme);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    writeStoredTheme(next);
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState((current) => {
      const next: Theme = current === 'dark' ? 'light' : 'dark';
      writeStoredTheme(next);
      return next;
    });
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({ theme, setTheme, toggleTheme }),
    [theme, setTheme, toggleTheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

/**
 * Read the theme.
 *
 * Throws outside a provider rather than returning the default. A silent default is the
 * `getattr` default that always fires (ADR-018): the component would render in dark, the
 * toggle would do nothing, and nothing would say why.
 */
export function useTheme(): ThemeContextValue {
  const value = useContext(ThemeContext);
  if (value === null) {
    throw new Error(
      'useTheme was called outside ThemeProvider. Wrap the tree in <ThemeProvider> - ' +
        'a default here would render the wrong theme with no signal.',
    );
  }
  return value;
}
