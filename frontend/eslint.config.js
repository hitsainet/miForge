/**
 * ESLint for miForge (ADR-011, ADR-018).
 *
 * Two project-specific rules beyond the recommended sets, both enforcing a decision
 * written down elsewhere:
 *
 * - `no-restricted-imports` bans `millm`, `mistudio` and `midataworks` (ADR-001). The
 *   backend has a test with a self-test for this; a lint rule is the frontend's cheapest
 *   equivalent, and `api/client.test.ts` asserts the stronger property - that no request
 *   leaves for a host that is not miForge.
 * - `no-restricted-syntax` bans a hex colour literal in a component (ADR-012, and F9's
 *   FTID 6.3: "no hex literal in any F9 component"). A hex is invisible to the accent
 *   scan, which looks for the token NAME, so a literal would make the accent rule
 *   unenforceable. `src/config/tokens.ts` is where the hexes live.
 */
import js from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';

export default tseslint.config(
  { ignores: ['dist', 'coverage', 'playwright-report', 'test-results', 'node_modules'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // eslint-plugin-react-hooks 7 (suite baseline, 2026-10-21) adds the React Compiler's
      // advisory rules to `recommended`. They flagged 20 EXISTING, deliberate patterns on the
      // day of the upgrade - the latest-value ref (`ref.current = x` in render), the
      // reset-then-load effect (`setValue(null)` before a fetch) and `Date.now()` as a render
      // default - none of which the upgrade changed. This app does not run the React
      // Compiler, so they are WARNINGS (still printed by `npm run lint`), and the sites are
      // tracked debt in 0xcc/reviews/deps_sync_2026-10-21.md rather than rewritten blind.
      'react-hooks/set-state-in-effect': 'warn',
      'react-hooks/refs': 'warn',
      'react-hooks/purity': 'warn',
      'react-refresh/only-export-components': 'off',
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: ['millm', 'millm/*', 'mistudio', 'mistudio/*', 'midataworks', 'midataworks/*'],
              message:
                'ADR-001: miForge imports no sibling code. The siblings are checked out side ' +
                'by side on this machine, so an absolute import resolves locally and passes ' +
                'every manual review. Reach a sibling over HTTP, through its one client module.',
            },
          ],
        },
      ],
    },
  },
  {
    files: ['src/components/**/*.tsx', 'src/components/**/*.ts'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector: "Literal[value=/^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/]",
          message:
            'ADR-012: components use the tokens from config/tokens.ts, never a hex literal. ' +
            'A hex is invisible to config/accent.test.tsx, which scans for the token name.',
        },
      ],
    },
  },
);
