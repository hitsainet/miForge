import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  manualSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting started',
      items: ['getting-started/concepts', 'getting-started/installation', 'getting-started/first-run'],
    },
    {
      type: 'category',
      label: 'Using miForge',
      items: [
        'guides/packs',
        'guides/runs',
        'guides/run-types',
        'guides/signals-and-judging',
        'guides/scoring',
        'guides/checkpoints',
        'guides/chains',
      ],
    },
    {
      type: 'category',
      label: 'Operators',
      items: [
        'operators/approvals',
        'operators/settings',
        'operators/deployment',
        'operators/hot-update',
      ],
    },
    {
      type: 'category',
      label: 'Integrations',
      items: ['integrations/mcp-server', 'integrations/suite'],
    },
    {
      type: 'category',
      label: 'Reference',
      items: ['reference/mcp-tools', 'reference/rest-api', 'reference/artifacts'],
    },
    'troubleshooting',
    'glossary',
  ],
};

export default sidebars;
