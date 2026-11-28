/** The Help screen (ENH-004): one card per topic, chips that select one, an unknown topic. */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HELP_FOR_ONBOARD_STEP, HELP_FOR_SCREEN, HELP_TOPICS, HELP_TOPIC_IDS } from '@/config/help';
import { NavigationProvider } from '@/contexts/NavigationContext';
import { ROUTES } from '@/config/routes';
import { HelpPanel } from './HelpPanel';

const mount = (topic?: string) =>
  render(
    <NavigationProvider
      routes={ROUTES}
      initialScreen="help"
      initialParams={topic === undefined ? {} : { topic }}
    >
      <HelpPanel />
    </NavigationProvider>,
  );

describe('[HelpPanel]', () => {
  it('should show every topic as a card, with a glossary among them', () => {
    mount();
    expect(HELP_TOPICS.length).toBeGreaterThanOrEqual(12);
    for (const topic of HELP_TOPICS) {
      expect(screen.getByTestId(`help-topic-${topic.id}`)).toHaveTextContent(topic.title);
    }
    expect(screen.getByTestId('help-topic-glossary')).toHaveTextContent('Words used here');
    expect(new Set(HELP_TOPIC_IDS).size).toBe(HELP_TOPICS.length);
  });

  it('should mark the asked-for topic and move the mark when a chip is chosen', async () => {
    mount('packs');
    expect(screen.getByTestId('help-topic-packs')).toHaveAttribute('data-current', 'true');
    await userEvent.click(screen.getByTestId('help-chip-approvals'));
    expect(screen.getByTestId('help-topic-approvals')).toHaveAttribute('data-current', 'true');
    expect(screen.getByTestId('help-topic-packs')).not.toHaveAttribute('data-current');
    expect(window.location.hash).toBe('#/help/approvals');
  });

  it('should show the screen with nothing marked for an unknown topic, never blank', () => {
    mount('no-such-topic');
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Help');
    expect(document.querySelectorAll('[data-current="true"]')).toHaveLength(0);
  });

  it('should map every screen and every onboarding step to a topic that exists', () => {
    for (const route of ROUTES) {
      const topic = HELP_FOR_SCREEN[route.id];
      if (route.id === 'help') expect(topic).toBeNull();
      else expect(HELP_TOPIC_IDS, route.id).toContain(topic);
    }
    for (let step = 1; step <= 7; step += 1) {
      expect(HELP_TOPIC_IDS, `step ${step}`).toContain(HELP_FOR_ONBOARD_STEP[step]);
    }
  });
});
