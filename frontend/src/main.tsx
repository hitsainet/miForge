/**
 * The browser entry point (foundation task 7.1).
 *
 * `applyThemeClass` runs **before** React mounts. Applying it in an effect alone paints
 * one frame in the wrong theme - the flash every Tailwind class-strategy app ships once -
 * and `index.html` already carries `class="dark"` so the very first paint is right even
 * before this line runs.
 */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import {
  applyThemeClass,
  DEFAULT_THEME,
  readStoredTheme,
  ThemeProvider,
} from './contexts/ThemeContext';
import './index.css';

const initialTheme = readStoredTheme() ?? DEFAULT_THEME;
applyThemeClass(initialTheme);

const container = document.getElementById('root');
if (container === null) {
  // Refuse with a reason rather than let `createRoot(null!)` fail somewhere else
  // (ADR-018). This is unreachable unless index.html changed, which is the point.
  throw new Error('#root is missing from index.html; miForge cannot mount.');
}

createRoot(container).render(
  <StrictMode>
    <ThemeProvider initialTheme={initialTheme}>
      <App />
    </ThemeProvider>
  </StrictMode>,
);
