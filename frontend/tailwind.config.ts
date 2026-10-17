/**
 * Tailwind for miForge (foundation task 7.1, 7.2; ADR-011, ADR-012).
 *
 * `darkMode: 'class'` per ADR-011, matching both siblings.
 *
 * **Every colour is a CSS variable, not a hex.** That is what makes one component tree
 * serve light and dark (foundation 7.6 wants a smoke spec in both) without a single
 * conditional in a component, and it is what lets `accent.test.ts` enforce the accent
 * rule by scanning for the *name* `accent`: a hex literal is invisible to a grep for a
 * token, so if components wrote `#F97316` the rule would be unenforceable. F9's FTID
 * 6.3 states the consequence as a prohibition - "no hex literal in any F9 component" -
 * and this config is what makes obeying it easy.
 *
 * The variables themselves are declared in `src/index.css`, and `tokens.test.ts` asserts
 * that file agrees with `config/tokens.ts` value for value, so there is one source of
 * truth with a test rather than two copies with a convention.
 */
import type { Config } from 'tailwindcss';
import { FONTS, SHELL, TYPE } from './src/config/tokens';

const px = (n: number) => `${n}px`;

export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'var(--mf-bg)',
        panel: 'var(--mf-panel)',
        card: 'var(--mf-card)',
        line: 'var(--mf-line)',
        text: 'var(--mf-text)',
        dim: 'var(--mf-dim)',
        accent: {
          DEFAULT: 'var(--mf-accent)',
          hover: 'var(--mf-accent-hover)',
          tint: 'var(--mf-accent-tint)',
        },
        green: 'var(--mf-green)',
        cyan: 'var(--mf-cyan)',
        red: 'var(--mf-red)',
        amber: 'var(--mf-amber)',
      },
      fontFamily: {
        sans: [FONTS.sans],
        mono: [FONTS.mono],
      },
      fontSize: {
        // The handoff's type scale, named by role rather than by size, so a heading that
        // is "16px semibold" cannot drift to 15 because a utility was closer to hand.
        title: [px(TYPE.pageTitlePx), { lineHeight: '1.2' }],
        heading: [px(TYPE.sectionHeadingPx), { lineHeight: '1.4' }],
        body: [px(TYPE.bodyPx), { lineHeight: '1.5' }],
        hint: [px(TYPE.hintPx), { lineHeight: '1.45' }],
        ident: [px(TYPE.monoPx), { lineHeight: '1.45' }],
      },
      spacing: {
        sidebar: px(SHELL.sidebarWidthPx),
        card: px(SHELL.cardPaddingPx),
        'tree-indent': px(SHELL.treeIndentPx),
        'icon-tile': px(SHELL.iconTilePx),
      },
      maxWidth: {
        content: px(SHELL.contentMaxWidthPx),
      },
      borderRadius: {
        card: px(SHELL.cardRadiusPx),
        control: px(SHELL.controlRadiusPx),
      },
    },
  },
  plugins: [],
} satisfies Config;
