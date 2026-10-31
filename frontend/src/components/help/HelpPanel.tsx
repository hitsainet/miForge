/**
 * The Help screen (ENH-004, 2026-10-20): short summaries, each ending in its manual section.
 *
 * Origin: miDataworks (Onegaishimas/miDataworks) `frontend/src/components/help/HelpPanel.tsx`
 * @ 32cf406. Mode: adapt (`docs/REUSE.md`). Kept: the line linking the manual home, the topic
 * chips, one card per topic (`#help-<id>`) with one manual link, the asked-for topic marked
 * `data-current` and scrolled into view. Changed: the topic is the URL's (`#/help/<topic>`,
 * D2), so it survives a reload and can be bookmarked; a chip navigates rather than scrolls.
 */
import { useEffect, useRef, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { HELP_TOPICS } from '@/config/help';
import { MANUAL_BASE_URL } from '@/config/manual';
import { useNavigation } from '@/contexts/NavigationContext';
import { ManualLink } from './ManualLink';

export const HELP_TITLE = 'Help';
export const HELP_SUBTITLE =
  'What each screen does and how to get from a model and a domain to a better model';

export function HelpPanel(): ReactNode {
  const navigation = useNavigation();
  const current = navigation.params['topic'];
  const refs = useRef<Record<string, HTMLElement | null>>({});

  useEffect(() => {
    if (current === undefined) return;
    const node = refs.current[current];
    // jsdom has no layout, and so no scrollIntoView.
    if (node !== null && node !== undefined && typeof node.scrollIntoView === 'function') {
      node.scrollIntoView({ block: 'start' });
    }
  }, [current]);

  return (
    <>
      <PageHead title={HELP_TITLE} subtitle={HELP_SUBTITLE} help={null} />
      <div className="mb-4 text-body text-dim">
        These are short summaries. The{' '}
        <a
          href={MANUAL_BASE_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="text-text underline underline-offset-2"
          data-testid="manual-home"
        >
          miForge manual
          <span className="sr-only"> (opens in a new tab)</span>
        </a>{' '}
        explains every screen in full.
      </div>
      <nav aria-label="Help topics" className="mb-6 flex flex-wrap gap-2">
        {HELP_TOPICS.map((topic) => (
          <button
            key={topic.id}
            type="button"
            onClick={() => navigation.go('help', { topic: topic.id }, { replace: true })}
            aria-current={topic.id === current ? 'true' : undefined}
            className={`rounded-full border px-3 py-1 text-hint ${
              topic.id === current ? 'border-text text-text' : 'border-line text-dim'
            }`}
            data-testid={`help-chip-${topic.id}`}
          >
            {topic.title}
          </button>
        ))}
      </nav>
      <div className="flex flex-col gap-4">
        {HELP_TOPICS.map((topic) => (
          <section
            key={topic.id}
            id={`help-${topic.id}`}
            ref={(node) => {
              refs.current[topic.id] = node;
            }}
            aria-labelledby={`help-${topic.id}-title`}
            data-current={topic.id === current ? 'true' : undefined}
            data-testid={`help-topic-${topic.id}`}
          >
            <Card className={topic.id === current ? 'border-text' : ''}>
              <h2 id={`help-${topic.id}-title`} className="mb-2 font-semibold">
                {topic.title}
              </h2>
              {topic.body.map((paragraph) => (
                <p key={paragraph} className="mb-2 text-body">
                  {paragraph}
                </p>
              ))}
              {topic.steps === undefined ? null : (
                <ol className="mb-2 ml-5 list-decimal text-body">
                  {topic.steps.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              )}
              <ManualLink
                page={topic.manual}
                label="Read more in the manual"
                data-testid={`help-manual-${topic.id}`}
              />
            </Card>
          </section>
        ))}
      </div>
    </>
  );
}
