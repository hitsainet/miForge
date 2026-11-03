/**
 * The pack screens' panels (F7 tasks 12.3-12.8).
 *
 * The two that carry a rule:
 *
 * - **`DryRunPanel` renders F3's `SelectionHealth` whole** - all three counts and their
 *   denominator - and **computes no rate of its own**. `undecided` is neutral `dim`, not amber,
 *   and there is **no approve button, no "send for review" and no pending count.**
 * - **`TryPanel`'s numbers name their scale**: `n` and `population` travel together, and at
 *   step 6 `floor_breached: null` means *"no floor was proposed"* rather than *"it is fine"*.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TryPanel } from './TryPanel';
import { DryRunPanel } from './DryRunPanel';
import { StepList, STEP_TITLES } from './StepList';
import { NoBuiltInDomainBanner } from './NoBuiltInDomainBanner';
import { PackCard } from './PackCard';
import type { DraftRecord, DryRunResult, PackListingRow, ScoredSample } from '@/types/packs';

const health = {
  n: 5,
  accepted: 3,
  rejected: 1,
  undecided: 1,
  acceptance_rate: 0.6,
  reason_histogram: { matched: 3, too_short: 1 },
  grader_stats: null,
  signals: [],
};

function dryRun(overrides: Partial<DryRunResult> = {}): DryRunResult {
  return {
    provider: { name: 'in-process' },
    judge: null,
    config_sha256: 'c'.repeat(64),
    config_kind: 'rule',
    hash_provenance: 'local',
    health,
    verdicts: [
      {
        prompt: 'p1',
        completion: 'c1',
        verdict: 'accept',
        reason_code: 'matched',
        reason: null,
        score: null,
        admitted: true,
      },
      {
        prompt: 'p2',
        completion: 'c2',
        verdict: 'reject',
        reason_code: 'too_short',
        reason: 'under 10 chars',
        score: null,
        admitted: false,
      },
      {
        prompt: 'p3',
        completion: 'c3',
        verdict: 'undecided',
        reason_code: 'queued',
        reason: null,
        score: null,
        admitted: false,
      },
    ],
    asynchronous_note: null,
    advisory: true,
    ...overrides,
  };
}

describe('[DryRunPanel]', () => {
  it('should render all three counts and their denominator, whole', () => {
    render(<DryRunPanel result={dryRun()} />);
    expect(screen.getByTestId('dry-run-counts')).toHaveTextContent(
      'Accepted 3, rejected 1, undecided 1 of 5',
    );
  });

  it('should render undecided NEUTRAL, not amber and never accent', () => {
    render(<DryRunPanel result={dryRun()} />);
    const tones = [...screen.getByTestId('dry-run-panel').querySelectorAll('[data-tone]')].map(
      (node) => ({ label: node.textContent, tone: node.getAttribute('data-tone') }),
    );
    expect(tones.find((row) => row.label === 'Undecided')?.tone).toBe('neutral');
    expect(tones.some((row) => row.tone === 'accent')).toBe(false);
    expect(tones.some((row) => row.tone === 'amber')).toBe(false);
  });

  it('should offer no approve button, no send-for-review and no pending count', () => {
    // Reviewing is miDataworks' queue (R-02.36, BRD-03 R-03.41); this renders what came back.
    render(<DryRunPanel result={dryRun()} />);
    const text = screen.getByTestId('dry-run-panel').textContent ?? '';
    expect(screen.queryByRole('button', { name: /approve/i })).toBeNull();
    expect(text).not.toMatch(/send for review/i);
    expect(text).not.toMatch(/\bpending\b/i);
  });

  it('should render the servers rate beside its denominator and compute none of its own', () => {
    render(<DryRunPanel result={dryRun()} />);
    expect(screen.getByTestId('dry-run-panel')).toHaveTextContent('Acceptance rate 0.600');
    expect(DryRunPanel.toString()).not.toContain('/ result.health.n');
  });

  it('should name the provider that answered and the judge, separately', () => {
    // R-02.11 as a constraint on this panel: a judge must visibly not be the model sampled -
    // a model grading itself is not a measurement.
    render(<DryRunPanel result={dryRun({ judge: { name: 'judge-7b' } })} />);
    const line = screen.getByTestId('dry-run-provider');
    expect(line).toHaveTextContent('in-process');
    expect(line).toHaveTextContent('judge-7b');
  });

  it('should explain an all-undecided panel rather than letting it read as broken', () => {
    // The human-review case. The step still saves; the note names which configuration kind
    // answered asynchronously and why.
    render(
      <DryRunPanel
        result={dryRun({
          health: { ...health, accepted: 0, rejected: 0, undecided: 5 },
          asynchronous_note:
            'A human-review configuration answers from a queue, so every verdict is undecided.',
        })}
      />,
    );
    expect(screen.getByTestId('dry-run-async-note')).toHaveTextContent('answers from a queue');
  });

  it('should render F3s signal with F3s own sentence when one fired', () => {
    render(
      <DryRunPanel
        result={dryRun({
          health: {
            ...health,
            accepted: 5,
            rejected: 0,
            undecided: 0,
            signals: [
              {
                name: 'selection_accepts_everything',
                message:
                  'A selector that accepts everything is not selecting; one that rejects everything will stall the run.',
                details: {},
              },
            ],
          },
        })}
      />,
    );
    expect(screen.getByTestId('signal-selection_accepts_everything')).toHaveTextContent(
      'is not selecting',
    );
  });

  it('should say nothing has been tried yet rather than rendering an empty table', () => {
    render(<DryRunPanel result={null} />);
    expect(screen.getByText(/No dry run yet/)).toBeInTheDocument();
  });
});

describe('[TryPanel]', () => {
  const sample: ScoredSample = {
    slot: 'retention_set',
    set: 'retain@2',
    n: 5,
    population: 640,
    seed: 7,
    prompts: [{ prompt: 'p1', reference: 'r1' }],
    score: 0.68,
    provider: 'in-process',
    aggregation: 'mean',
    floor: 0.72,
    floor_breached: true,
  };

  it('should name n and the population together, so a bare count is not expressible', () => {
    render(<TryPanel sample={sample} />);
    expect(screen.getByTestId('try-panel')).toHaveTextContent('5 of 640');
    expect(screen.getByTestId('try-panel')).toHaveTextContent('rows in the set');
  });

  it('should say the seed, so the sample reads as stable rather than as changing data', () => {
    render(<TryPanel sample={sample} />);
    expect(screen.getByTestId('try-panel')).toHaveTextContent('the same rows every time');
  });

  it('should say a proposed floor is already breached', () => {
    // The one thing an author cannot guess (FR-007.19).
    render(<TryPanel sample={sample} />);
    expect(screen.getByTestId('try-panel-floor')).toHaveTextContent('already breached');
  });

  it('should say a null floor_breached has no answer, never that it is fine', () => {
    render(<TryPanel sample={{ ...sample, floor: 0.72, floor_breached: null }} />);
    expect(screen.getByTestId('try-panel-floor')).toHaveTextContent('has no answer');
  });

  it('should name an absent floor rather than rendering zero', () => {
    render(<TryPanel sample={{ ...sample, floor: null, floor_breached: null }} />);
    expect(screen.getByTestId('try-panel-no-floor')).toHaveTextContent('no floor recorded');
  });

  it('should name the provider and the aggregation beside the score', () => {
    // Two scores aggregated differently are not the same measurement.
    render(<TryPanel sample={sample} />);
    expect(screen.getByTestId('try-panel-score')).toHaveTextContent('mean aggregation');
    expect(screen.getByTestId('try-panel-score')).toHaveTextContent('in-process');
  });
});

describe('[StepList]', () => {
  const draft: DraftRecord = {
    draft_id: 'd1',
    name: 'md-notary',
    version: '1',
    domain: 'notary',
    current_step: 3,
    step_count: 7,
    // Step 5 is DONE while the current step is 3: going back must not make later work look
    // lost, which a "less than the current step" comparison would do.
    step_states: { '1': 'done', '2': 'done', '3': 'current', '5': 'done' },
    slots: {},
    created_at: '',
    updated_at: '',
    last_dry_run: null,
  };

  it('should take each steps state from the server, not from a number comparison', () => {
    render(<StepList draft={draft} onGo={() => {}} />);
    const items = [...screen.getByTestId('step-list').querySelectorAll('[data-step]')];
    const states = new Map(
      items.map((node) => [node.getAttribute('data-step'), node.getAttribute('data-step-state')]),
    );
    expect(states.get('3')).toBe('current');
    expect(states.get('5')).toBe('done');
    expect(states.get('4')).toBe('todo');
  });

  it('should mark the current step with aria-current', () => {
    render(<StepList draft={draft} onGo={() => {}} />);
    expect(
      screen.getByTestId('step-list').querySelector('[aria-current="step"]')?.textContent,
    ).toContain('3.');
  });

  it('should mark the two skippable steps as optional, from the backends own STEPS', () => {
    render(<StepList draft={draft} onGo={() => {}} />);
    const optional = [...screen.getByTestId('step-list').querySelectorAll('[data-step]')]
      .filter((node) => (node.textContent ?? '').includes('optional'))
      .map((node) => node.getAttribute('data-step'));
    // `packs/authoring.py::STEPS` marks 3 (answers) and 4 (judgment) skippable.
    expect(optional).toEqual(['3', '4']);
  });

  it('should title all seven steps', () => {
    expect(Object.keys(STEP_TITLES)).toHaveLength(7);
  });
});

describe('[NoBuiltInDomainBanner]', () => {
  it('should be the accent callout, carrying the mockups sentence', () => {
    render(<NoBuiltInDomainBanner />);
    const banner = screen.getByTestId('no-built-in-domain');
    expect(banner.getAttribute('data-tone')).toBe('accent');
    expect(banner).toHaveTextContent('miForge has no built-in domain.');
    expect(banner).toHaveTextContent('references to parts, never copies');
  });
});

describe('[PackCard]', () => {
  const pack: PackListingRow = {
    pack_id: 'md-notary@1',
    version: '1',
    name: 'md-notary',
    domain: 'notary',
    parts_filled: 7,
    parts_total: 7,
    prompt_count: 640,
    runs_using: ['run-a', 'run-b', 'run-c'],
    validity: {
      state: 'valid',
      missing_slots: [],
      unresolved: [],
      service_unavailable: [],
      retention_floor_missing: false,
    },
    derived_from: null,
    is_example: false,
    definition_sha256: 'd'.repeat(64),
    created_at: '2026-10-14T00:00:00Z',
  };

  it('should name every count with its scale', () => {
    render(<PackCard pack={pack} onShowLineage={() => {}} onExport={() => {}} />);
    const card = screen.getByTestId('pack-md-notary@1');
    expect(card).toHaveTextContent('7 of 7');
    expect(card).toHaveTextContent('640');
    expect(card).toHaveTextContent('3');
    expect(card).toHaveTextContent('runs using it');
  });

  it('should render an unresolvable prompt count as unknown, NEVER as zero', () => {
    // FR-007.11, ADR-018: a count that could not be resolved is not a count of nothing.
    render(
      <PackCard
        pack={{ ...pack, prompt_count: null }}
        onShowLineage={() => {}}
        onExport={() => {}}
      />,
    );
    expect(screen.getByTestId('prompts-unknown-md-notary@1')).toHaveTextContent(
      'prompt_count could not be resolved',
    );
  });

  it('should show the Example pill only when the row says so', () => {
    const { unmount } = render(
      <PackCard pack={pack} onShowLineage={() => {}} onExport={() => {}} />,
    );
    expect(screen.queryByTestId('example-md-notary@1')).toBeNull();
    unmount();
    render(
      <PackCard
        pack={{ ...pack, is_example: true }}
        onShowLineage={() => {}}
        onExport={() => {}}
      />,
    );
    expect(screen.getByTestId('example-md-notary@1')).toHaveTextContent('Example');
  });

  it('should name the missing slot on an incomplete pack', () => {
    render(
      <PackCard
        pack={{
          ...pack,
          validity: { ...pack.validity, state: 'incomplete', missing_slots: ['test_set'] },
        }}
        onShowLineage={() => {}}
        onExport={() => {}}
      />,
    );
    expect(screen.getByTestId('validity-md-notary@1')).toHaveTextContent('no test set');
  });

  it('should render a null domain as no domain', () => {
    render(
      <PackCard pack={{ ...pack, domain: null }} onShowLineage={() => {}} onExport={() => {}} />,
    );
    expect(screen.getByTestId('pack-md-notary@1')).toHaveTextContent('no domain');
  });

  it('should render each slots part through PartReference, with a source marker', () => {
    render(
      <PackCard
        pack={pack}
        lineage={[
          {
            slot: 'prompt_set',
            source: 'midataworks_contract',
            reference: { source: 'midataworks', type: 'prompt-set', name: 'notary', version: '3' },
            resolution: 'resolved',
            reason: null,
            floor: null,
          },
          {
            slot: 'retention_set',
            source: 'external_files',
            reference: {
              source: 'external',
              origin: 'https://example.invalid/r.jsonl',
              content_sha256: 'a'.repeat(64),
              taken_in_at: '2026-10-14T00:00:00Z',
            },
            resolution: 'resolved',
            reason: null,
            floor: 0.72,
          },
        ]}
        onShowLineage={() => {}}
        onExport={() => {}}
      />,
    );
    const lineage = screen.getByTestId('lineage-md-notary@1');
    expect(lineage).toHaveTextContent('md-notary/prompt-set/notary@3');
    expect(lineage.querySelector('[data-source-marker="midataworks"]')).not.toBeNull();
    expect(lineage.querySelector('[data-source-marker="external"]')).not.toBeNull();
    expect(lineage).toHaveTextContent('0.720');
  });

  it('should name an empty slot rather than rendering a blank row', () => {
    render(
      <PackCard
        pack={pack}
        lineage={[
          {
            slot: 'corpus',
            source: null,
            reference: null,
            resolution: null,
            reason: null,
            floor: null,
          },
        ]}
        onShowLineage={() => {}}
        onExport={() => {}}
      />,
    );
    expect(screen.getByTestId('lineage-md-notary@1')).toHaveTextContent('this slot is empty');
  });
});
