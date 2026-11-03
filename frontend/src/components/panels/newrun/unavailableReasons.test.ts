import { describe, expect, it } from 'vitest';
import { unavailableSentence } from './unavailableReasons';
import { objectiveName } from './DefinitionFields';

describe('[unavailableSentence]', () => {
  it('should put a plain sentence first and keep the code to search for', () => {
    expect(unavailableSentence('inference_provider_unavailable')).toBe(
      'no generation provider is available (inference_provider_unavailable)',
    );
    expect(unavailableSentence('midataworks_not_configured')).toBe(
      'miDataworks is not configured (midataworks_not_configured)',
    );
  });

  it('should show an unknown code as itself, and say when there is none', () => {
    expect(unavailableSentence('something_new')).toBe('something_new');
    expect(unavailableSentence(null)).toBe('no reason given');
  });
});

describe('[objectiveName]', () => {
  it('should spell out each objective with its acronym once', () => {
    expect(objectiveName('sft')).toBe('supervised fine-tuning (SFT)');
    expect(objectiveName('dpo')).toBe('direct preference optimization (DPO)');
    expect(objectiveName('grpo')).toBe('group relative policy optimization (GRPO)');
    expect(objectiveName('other')).toBe('other');
    expect(objectiveName(null)).toBe('not recorded');
  });
});
