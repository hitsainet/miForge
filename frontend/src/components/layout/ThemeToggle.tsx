/**
 * The light/dark toggle (foundation tasks 7.3, 7.6).
 *
 * **Not in the mockup, and added deliberately.** ADR-011 requires Tailwind's
 * `darkMode: 'class'` and foundation 7.6 requires a Playwright smoke spec per shell
 * region in **light and dark** - and a class strategy with nothing that sets the class is
 * a configuration no test has exercised. This is the control that makes the second theme
 * a real path.
 *
 * **It names its action** (ADR-012: "buttons name their action"), which for a toggle
 * means naming the state it moves *to*, not the state it is in. "Switch to light" is
 * unambiguous; a sun glyph alone is not, and "Dark" could be read either way.
 */

import type { ReactNode } from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

export function ThemeToggle(): ReactNode {
  const { theme, toggleTheme } = useTheme();
  const next = theme === 'dark' ? 'light' : 'dark';
  const Icon = theme === 'dark' ? Sun : Moon;
  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="inline-flex items-center gap-1.5 rounded-control px-2 py-1 text-hint text-dim hover:bg-card hover:text-text"
      aria-label={`Switch to ${next} theme`}
      data-testid="theme-toggle"
      data-theme-now={theme}
    >
      <Icon size={14} aria-hidden="true" />
      Switch to {next}
    </button>
  );
}
