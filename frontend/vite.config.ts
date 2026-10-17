/**
 * Vite + Vitest for the miForge admin UI (foundation task 7.1; ADR-011, ADR-016).
 *
 * One config for both, as miStudio does, so there is a single place where the alias
 * table lives. Two aliases that disagree is the frontend's version of ADR-005's "two
 * canonicalisers are two answers".
 */
import path from 'node:path';
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    // The backend is a separate process (ADR-017). Proxying keeps the browser on one
    // origin, which matters because ADR-022 fences the core path off every sibling
    // endpoint and a browser `fetch` is on that path: there is exactly one host the
    // UI may talk to, and it is miForge's own.
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/socket.io': { target: 'http://localhost:8000', ws: true, changeOrigin: true },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    // Tests sit beside the file they test (ADR-011, ADR-017). `e2e/` is Playwright's
    // and must not be picked up by Vitest - the two runners share the word "test".
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['**/node_modules/**', '**/dist/**', 'e2e/**'],
  },
});
