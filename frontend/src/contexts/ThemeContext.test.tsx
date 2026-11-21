/**
 * The theme (foundation tasks 7.1, 7.3, 7.6; ADR-011).
 *
 * The assertion that is not about convenience: a `localStorage` that throws must not take
 * the application down. It throws on access in some privacy modes and in a sandboxed
 * iframe, and a theme preference is the least important thing in the product to fail
 * over.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  applyThemeClass,
  DEFAULT_THEME,
  readStoredTheme,
  THEME_STORAGE_KEY,
  ThemeProvider,
  useTheme,
} from './ThemeContext';

function Probe() {
  const { theme, toggleTheme } = useTheme();
  return (
    <button type="button" onClick={toggleTheme}>
      now {theme}
    </button>
  );
}

beforeEach(() => {
  window.localStorage.clear();
  document.documentElement.className = '';
  delete document.documentElement.dataset['theme'];
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('[ThemeProvider]', () => {
  it('should default to dark, as every document and the mockup show', () => {
    expect(DEFAULT_THEME).toBe('dark');
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    expect(screen.getByRole('button')).toHaveTextContent('now dark');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });

  it('should put the class on <html> where Tailwinds class strategy looks', () => {
    applyThemeClass('light');
    expect(document.documentElement.classList.contains('dark')).toBe(false);
    expect(document.documentElement.classList.contains('light')).toBe(true);
    expect(document.documentElement.dataset['theme']).toBe('light');
  });

  it('should toggle and persist the preference', async () => {
    render(
      <ThemeProvider initialTheme="dark">
        <Probe />
      </ThemeProvider>,
    );
    await userEvent.click(screen.getByRole('button'));
    expect(screen.getByRole('button')).toHaveTextContent('now light');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light');
    expect(document.documentElement.classList.contains('dark')).toBe(false);
  });

  it('should read a stored preference back', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light');
    expect(readStoredTheme()).toBe('light');
  });

  it('should ignore a stored value that is not a theme', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'solarized');
    expect(readStoredTheme()).toBeNull();
  });

  it('should survive a localStorage that throws on read and on write', async () => {
    const getItem = vi.spyOn(window.localStorage, 'getItem').mockImplementation(() => {
      throw new Error('SecurityError');
    });
    const setItem = vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
      throw new Error('SecurityError');
    });

    expect(readStoredTheme()).toBeNull();
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    await userEvent.click(screen.getByRole('button'));
    // The toggle still works; only the persistence is lost.
    expect(screen.getByRole('button')).toHaveTextContent('now light');

    getItem.mockRestore();
    setItem.mockRestore();
  });

  it('should refuse by name when useTheme is called outside the provider', () => {
    // ADR-018: a silent default here would render the wrong theme, make the toggle inert,
    // and say nothing about why.
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(() => render(<Probe />)).toThrow(/outside ThemeProvider/);
    error.mockRestore();
  });
});
