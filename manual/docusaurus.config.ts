import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// THE PUBLISHED ADDRESS. `url` + `baseUrl` is the manual's home page, and the admin UI's
// sidebar links to exactly that string (frontend/src/config/manual.ts). A test in the
// frontend compares the two, so moving the site means changing both in one commit.
const SITE_URL = 'https://hitsainet.github.io';
const BASE_URL = '/miForge/';

const config: Config = {
  title: 'miForge Manual',
  tagline: 'miForge makes weights: create them, change them, or add adapters that compose with them',
  favicon: 'img/favicon.svg',

  // GitHub Pages, published from the PUBLIC hitsainet mirror by
  // .github/workflows/deploy-manual.yml (actions/deploy-pages), never by `npm run deploy`.
  url: SITE_URL,
  baseUrl: BASE_URL,

  organizationName: 'hitsainet',
  projectName: 'miForge',
  trailingSlash: false,

  // A broken link, a broken anchor or a broken Markdown link FAILS THE BUILD. The CI
  // build check (.github/workflows/manual-ci.yml) relies on this: a green build is the
  // proof that every internal link resolves.
  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/hitsainet/miForge/tree/main/manual/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'miForge Manual',
      logo: {alt: 'miForge', src: 'img/logo.svg'},
      items: [
        {type: 'docSidebar', sidebarId: 'manualSidebar', position: 'left', label: 'Manual'},
        {to: '/getting-started/first-run', label: 'First run', position: 'left'},
        {to: '/operators/approvals', label: 'Operators', position: 'left'},
        {to: '/reference/mcp-tools', label: 'Reference', position: 'left'},
        {href: 'https://github.com/hitsainet/miForge', label: 'GitHub', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Manual',
          items: [
            {label: 'What miForge is', to: '/'},
            {label: 'Your first run', to: '/getting-started/first-run'},
            {label: 'Troubleshooting', to: '/troubleshooting'},
            {label: 'Glossary', to: '/glossary'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'MCP tools', to: '/reference/mcp-tools'},
            {label: 'REST API', to: '/reference/rest-api'},
            {label: 'Artifacts and schemas', to: '/reference/artifacts'},
          ],
        },
        {
          title: 'The suite',
          items: [
            {label: 'GitHub', href: 'https://github.com/hitsainet/miForge'},
            {label: 'miStudio manual', href: 'https://hitsainet.github.io/miStudio/'},
            {label: 'hitsai.net', href: 'https://hitsai.net'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Human in the Stream, LLC. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'json', 'yaml'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
