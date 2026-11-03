/**
 * The components that carry a rule (FR-009.6, .8, .23; M12, M16 in part).
 *
 * Each of these exists to make a violation **inexpressible rather than forbidden**:
 * `NumberWithSample` has no prop combination that renders a bare number; `AbsentValue` accepts
 * no value and so cannot print a zero; `EdgeMark`'s map is `satisfies Record<EdgeState, …>` so
 * an unhandled member is a typecheck failure. These tests assert the *absence* as much as the
 * presence, because a required prop is only a rule if nothing routes around it.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { NumberWithSample, formatCount, formatGb, shortProvider } from './NumberWithSample';
import { ABSENT_LABELS, ABSENT_REASONS, ABSENT_SENTENCES, AbsentValue } from './AbsentValue';
import { KindMarker } from './KindMarker';
import { EDGE_PRESENTATION, EdgeMark } from './EdgeMark';
import { VERDICT_VALUES, VerdictPill } from './VerdictPill';
import { SelectionCounts, selectionCountsText } from './SelectionCounts';
import { ValidityPill, validityLabel } from './ValidityPill';
import { PartReference, contractForm, externalForm, innerForm } from './PartReference';
import { EDGE_STATES } from '@/types/tree';
import type { PackValidity, PartRef } from '@/types/packs';

describe('[NumberWithSample]', () => {
  it('should require a sample and a sample label on every number', () => {
    // The structural half of FR-009.6: there is no prop combination that renders a bare
    // number, so a reviewer does not have to remember the rule.
    render(<NumberWithSample value="0.84" sample={320} sampleLabel="prompts" data-testid="n" />);
    const node = screen.getByTestId('n');
    expect(node).toHaveTextContent('0.84');
    expect(node).toHaveTextContent('320 prompts');
  });

  it('should name the provider and the step when they are given', () => {
    render(
      <NumberWithSample
        value="0.84"
        sample={320}
        sampleLabel="prompts"
        provider="in-process"
        atStep={500}
        data-testid="n"
      />,
    );
    expect(screen.getByTestId('n')).toHaveTextContent('in-process');
    expect(screen.getByTestId('n')).toHaveTextContent('at step 500');
  });

  it('should put a long provider id on its own breakable line, shortened, with the whole id kept', () => {
    // Live 2026-10-20: the retention tile's provider id (a model plus a 40-character revision)
    // sat inside the parenthetical as one unbreakable token and spilled out of the tile.
    const id =
      'in_process:hf:LiquidAI/LFM2.5-1.2B-Instruct@0f604ada3f766f9f257460c4c9f0b5d6f69d431b';
    render(
      <NumberWithSample
        value="0.875"
        sample={8}
        sampleLabel="prompts"
        provider={id}
        data-testid="n"
      />,
    );
    const line = screen.getByTestId('number-provider');
    expect(line).toHaveTextContent('via in-process · LiquidAI/LFM2.5-1.2B-Instruct @0f604ad');
    expect(line).toHaveAttribute('title', id);
    expect(line.className).toContain('block');
    expect(line.className).toContain('wrap-anywhere');
    // Not inside the parenthetical any more.
    expect(screen.getByTestId('n').textContent).toContain('(8 prompts)');
  });

  it('should show a provider id it cannot read exactly as written', () => {
    expect(shortProvider('in-process')).toBe('in-process');
    expect(shortProvider('https://judge.example/v1')).toBe('https://judge.example/v1');
    expect(shortProvider('endpoint:vllm-a')).toBe('endpoint · vllm-a');
  });

  it('should carry a qualifier such as a retention floor', () => {
    render(
      <NumberWithSample
        value="0.77"
        sample={180}
        sampleLabel="prompts"
        qualifier="floor 0.720"
        data-testid="n"
      />,
    );
    expect(screen.getByTestId('n')).toHaveTextContent('floor 0.720');
  });

  it('should separate thousands so 50000 steps does not read as 5000', () => {
    expect(formatCount(50_000)).toBe('50,000');
  });

  it('should render bytes as decimal GB with one decimal', () => {
    expect(formatGb(11_600_000_000)).toBe('11.6');
  });
});

describe('[AbsentValue]', () => {
  it('should accept no value, so it cannot print a zero', () => {
    // M12's structural guard. `floor ?? 0` at a call site is the mutation; the control only
    // bites if every absence comes through here.
    const source = AbsentValue.toString();
    expect(source).not.toContain('props.value');
  });

  it('should have one sentence and one label per reason, and exactly eight reasons', () => {
    expect(ABSENT_REASONS).toHaveLength(8);
    for (const reason of ABSENT_REASONS) {
      expect(ABSENT_SENTENCES[reason].length, reason).toBeGreaterThan(20);
      expect(ABSENT_LABELS[reason].length, reason).toBeGreaterThan(0);
    }
    expect(new Set(Object.values(ABSENT_SENTENCES)).size).toBe(8);
  });

  it('should render every reason without printing a number', () => {
    for (const reason of ABSENT_REASONS) {
      const { unmount } = render(<AbsentValue reason={reason} data-testid="a" />);
      const node = screen.getByTestId('a');
      expect(node.getAttribute('data-absent')).toBe(reason);
      expect(node.textContent ?? '', reason).not.toMatch(/\b0\b/);
      expect(node.textContent ?? '', reason).not.toBe('—');
      unmount();
    }
  });

  it('should say no domain for an asserted null, in those words', () => {
    // R-02.29's amendment: the tree "shows that as no domain rather than as a blank that
    // reads like missing data", and the backend's render_domain uses the same two words.
    render(<AbsentValue reason="no_domain" data-testid="a" />);
    expect(screen.getByTestId('a')).toHaveTextContent('no domain');
  });

  it('should name a missing required key as an omission, not as no value', () => {
    render(<AbsentValue reason="missing_required_key" variant="block" data-testid="a" />);
    expect(screen.getByTestId('a')).toHaveTextContent(/does not conform/);
  });

  it('should keep lineage_stops_here distinct from a root in its own copy', () => {
    expect(ABSENT_SENTENCES.lineage_stops_here).toContain('not a root');
  });
});

describe('[KindMarker]', () => {
  it('should take the kind from kind, never from the name', () => {
    // FR-009.23: a view that parses a name to learn what something is renders a hand-named
    // artifact wrongly. A checkpoint called `…adapter-400` is a checkpoint with an odd name.
    render(<KindMarker kind="checkpoint" name="LFM2:run-a:adapter-400" data-testid="k" />);
    const node = screen.getByTestId('k');
    expect(node.getAttribute('data-kind')).toBe('checkpoint');
    expect(node).toHaveTextContent('Checkpoint');
    expect(node).toHaveTextContent('name anomaly');
    expect(node.getAttribute('data-name-anomaly')).toBe('true');
  });

  it('should report no anomaly when kind and name agree', () => {
    for (const [kind, name] of [
      ['adapter', 'LFM2:run-b:adapter-200'],
      ['checkpoint', 'LFM2:run-a:000400'],
    ] as const) {
      const { unmount } = render(<KindMarker kind={kind} name={name} data-testid="k" />);
      expect(screen.getByTestId('k').getAttribute('data-name-anomaly'), name).toBe('false');
      unmount();
    }
  });

  it('should report an adapter whose name lacks the segment as an anomaly, not a checkpoint', () => {
    render(<KindMarker kind="adapter" name="hand-named-thing" data-testid="k" />);
    expect(screen.getByTestId('k')).toHaveTextContent('Adapter');
    expect(screen.getByTestId('k').getAttribute('data-name-anomaly')).toBe('true');
  });
});

describe('[EdgeMark]', () => {
  it('should present all seven edge states distinctly', () => {
    expect(Object.keys(EDGE_PRESENTATION).sort()).toEqual([...EDGE_STATES].sort());
    const labels = Object.values(EDGE_PRESENTATION).map((row) => row.label);
    expect(new Set(labels).size).toBe(7);
    const sentences = Object.values(EDGE_PRESENTATION).map((row) => row.sentence);
    expect(new Set(sentences).size).toBe(7);
  });

  it('should keep root and truncated visibly different', () => {
    // M9's rendering half. "One means the root, the other the record stops here."
    expect(EDGE_PRESENTATION.root.label).not.toBe(EDGE_PRESENTATION.truncated.label);
    expect(EDGE_PRESENTATION.truncated.sentence).toContain('NOT a root');
  });

  it('should keep not_loaded from reading as a broken lineage', () => {
    expect(EDGE_PRESENTATION.not_loaded.sentence).toContain('paging boundary');
  });

  it('should render every state with its target named', () => {
    for (const state of EDGE_STATES) {
      const { unmount } = render(<EdgeMark state={state} target="base-400" data-testid="e" />);
      const node = screen.getByTestId('e');
      expect(node.getAttribute('data-edge'), state).toBe(state);
      expect(node, state).toHaveTextContent('base-400');
      unmount();
    }
  });

  it('should paint no edge state with the accent', () => {
    for (const row of Object.values(EDGE_PRESENTATION)) {
      expect(row.tone).not.toContain('accent');
    }
  });
});

describe('[VerdictPill]', () => {
  it('should offer three verdicts and render undecided NEUTRAL, never amber or accent', () => {
    // Amber is the warning colour and an undecided candidate is not a warning: it is a
    // candidate at a review queue nobody has answered yet.
    expect([...VERDICT_VALUES]).toEqual(['accept', 'reject', 'undecided']);
    render(
      <>
        <VerdictPill verdict="accept" data-testid="v-accept" />
        <VerdictPill verdict="reject" data-testid="v-reject" />
        <VerdictPill verdict="undecided" data-testid="v-undecided" />
      </>,
    );
    expect(screen.getByTestId('v-accept').getAttribute('data-tone')).toBe('green');
    expect(screen.getByTestId('v-reject').getAttribute('data-tone')).toBe('red');
    expect(screen.getByTestId('v-undecided').getAttribute('data-tone')).toBe('neutral');
  });

  it('should degrade an unrecognised verdict rather than throwing', () => {
    expect(() => render(<VerdictPill verdict="deferred" data-testid="v" />)).not.toThrow();
    expect(screen.getByTestId('v')).toHaveTextContent('deferred (unrecognised)');
  });
});

describe('[SelectionCounts]', () => {
  it('should render all three counts and their denominator, and compute no rate', () => {
    // "Accepted 41 of 64" on a batch with 12 pending is a lie about an unattended queue.
    const health = { n: 64, accepted: 41, rejected: 11, undecided: 12 };
    expect(selectionCountsText(health)).toBe('Accepted 41, rejected 11, undecided 12 of 64');
    render(<SelectionCounts health={health} data-testid="c" />);
    expect(screen.getByTestId('c')).toHaveTextContent('undecided 12 of 64');
    // No percentage is computed here: the rate lives on the record beside its denominator.
    expect(SelectionCounts.toString()).not.toContain('/ health.n');
    expect(selectionCountsText.toString()).not.toContain('%');
  });
});

describe('[ValidityPill]', () => {
  const base: PackValidity = {
    state: 'valid',
    missing_slots: [],
    unresolved: [],
    service_unavailable: [],
    retention_floor_missing: false,
  };

  it('should name every missing slot rather than the first', () => {
    // AC10's whole content: an incomplete pack NAMES the slot.
    const label = validityLabel({
      ...base,
      state: 'incomplete',
      missing_slots: ['test_set', 'retention_set'],
    });
    expect(label).toContain('no test set');
    expect(label).toContain('no retention set');
  });

  it('should paint an unconfigured service neutral, not red: it is no fault of the pack', () => {
    render(
      <ValidityPill
        validity={{
          ...base,
          state: 'service_unavailable',
          service_unavailable: [
            { slot: 'test_set', reference: null, service: 'midataworks', reason: 'unset' },
          ],
        }}
        data-testid="p"
      />,
    );
    const pill = screen.getByTestId('p');
    expect(pill.getAttribute('data-tone')).toBe('neutral');
    expect(pill).toHaveTextContent('midataworks not configured');
  });

  it('should degrade an unrecognised validity state instead of throwing', () => {
    // The mockup's pill destructured `map[state]` with no fallback and threw.
    expect(() =>
      render(<ValidityPill validity={{ ...base, state: 'quarantined' }} data-testid="p" />),
    ).not.toThrow();
    expect(screen.getByTestId('p')).toHaveTextContent('quarantined (unrecognised)');
  });
});

describe('[PartReference]', () => {
  const contract: PartRef = {
    source: 'midataworks',
    type: 'prompt-set',
    name: 'notary',
    version: '3',
  };
  const external: PartRef = {
    source: 'external',
    origin: 'https://example.invalid/set.jsonl',
    content_sha256: 'a'.repeat(64),
    taken_in_at: '2026-10-14T00:00:00Z',
  };

  it('should render the contract form for a miDataworks reference', () => {
    expect(contractForm(contract)).toBe('prompt-set/notary@3');
    expect(innerForm(contract)).toBe('prompt-set/notary@3');
  });

  it('should render provenance for an external reference and never a contract citation', () => {
    // R-02.40: miForge never claims miDataworks provenance for something they did not
    // publish. The two shapes are different types and this is the rendering half.
    expect(externalForm(external)).toBe('external/https://example.invalid/set.jsonl@aaaaaaaaaaaa');
    expect(innerForm(external)).not.toContain('@3');
  });

  it('should carry the external marker even when the origin is itself contract-shaped', () => {
    // **The check the test above could not make**, because its origin is an `https://` URL and
    // so could never look like `<type>/<name>@<version>` whatever the renderer did. This one
    // picks the mockup's own string: `retention/mixed` is two segments and a slash, so without
    // the `external/` marker the rendering is byte-for-byte a contract citation, and with the
    // pack prefix it is `general/retention/mixed@...` - the mockup line that opened OQ-9.
    const shaped: PartRef = {
      source: 'external',
      origin: 'retention/mixed',
      content_sha256: 'b'.repeat(64),
      taken_in_at: '2026-10-14T00:00:00Z',
    };
    expect(externalForm(shaped)).toBe('external/retention/mixed@bbbbbbbbbbbb');
    expect(innerForm(shaped)).not.toBe('retention/mixed@bbbbbbbbbbbb');
    // And the same string the backend's `external_form` produces, which is the authority for
    // this shape: the two layers must agree on what an operator sees for one reference.
    expect(innerForm(shaped)).toBe(`external/${shaped.origin}@${'b'.repeat(12)}`);
  });

  it('should render an unowned part as its inner form alone, with no segment invented', () => {
    // F7's OQ-9, **settled 2026-10-14**: there is no unowned-parts namespace. The backend's
    // `unowned_form()` is this same rendering, and `display_form()` still refuses a null pack
    // so that a pack which was merely *lost* cannot render as one that never existed.
    expect(innerForm(contract)).toBe('prompt-set/notary@3');
    render(<PartReference reference={contract} pack={null} data-testid="unowned" />);
    expect(screen.getByTestId('unowned')).toHaveTextContent('prompt-set/notary@3');
    expect(screen.getByTestId('unowned')).toHaveTextContent('addressed through no pack');
    expect(screen.getByTestId('unowned').textContent).not.toContain('general/');
  });

  it('should prefix the pack it is addressed THROUGH, as a second layer', () => {
    // `<pack>/<type>/<name>@<version>` is miForge's display form; `<type>/<name>@<version>`
    // is miDataworks' contract form. Two layers, settled 2026-10-12.
    render(<PartReference reference={contract} pack="md-notary" data-testid="p" />);
    expect(screen.getByTestId('p')).toHaveTextContent('md-notary/prompt-set/notary@3');
  });

  it('should invent NO pack segment for a part addressed through no pack', () => {
    // F7's OQ-9, unresolved and the operator's. The honest behaviour exists; the answer does
    // not, and writing `general` into a picker would be answering it by writing a string.
    render(<PartReference reference={contract} pack={null} data-testid="p" />);
    const node = screen.getByTestId('p');
    expect(node).toHaveTextContent('prompt-set/notary@3');
    expect(node).toHaveTextContent('addressed through no pack');
    expect(node.textContent ?? '').not.toContain('general/');
  });

  it('should mark a miDataworks part indigo and an external part neutral', () => {
    render(
      <>
        <PartReference reference={contract} pack="p" data-testid="c" />
        <PartReference reference={external} pack="p" data-testid="e" />
      </>,
    );
    expect(
      screen.getByTestId('c').querySelector('[data-source-marker="midataworks"]')?.className,
    ).toContain('indigo');
    expect(
      screen.getByTestId('e').querySelector('[data-source-marker="external"]')?.className,
    ).not.toContain('indigo');
  });

  it('should refuse to render a ref whose source it does not recognise', () => {
    const odd = {
      source: 'borrowed',
      type: 'prompt-set',
      name: 'x',
      version: '1',
    } as unknown as PartRef;
    render(<PartReference reference={odd} pack="p" data-testid="p" />);
    const node = screen.getByTestId('p');
    expect(node.getAttribute('data-part-source')).toBe('unrecognised');
    expect(node).toHaveTextContent('does not recognise');
  });
});
