/**
 * A node's four facts (FR-009.24, .25), and **M5, M9's rendering half and M17**.
 *
 * - **M5** — rendering a null `domain` as `domain ?? ''`. R-02.29's amendment is explicit that
 *   the tree *"shows that as no domain rather than as a blank that reads like missing data"*.
 * - **M9** — a truncated chain and a `parent: null` root must render **differently**.
 * - **M17** — depth from a transmitted field. The node type has none, so the mutation requires
 *   adding one.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DomainFact, ParentFact, TestFact, TreeNode } from './TreeNode';
import { TreeEdges, VERIFICATION_SENTENCES } from './TreeEdges';
import type { TreeNode as TreeNodeModel } from '@/types/tree';

function node(overrides: Partial<TreeNodeModel> & { name: string }): TreeNodeModel {
  return {
    kind: 'checkpoint',
    run_id: 'run-a',
    step: 400,
    domain: 'notary',
    domain_state: 'recorded',
    parent: { target: null, state: 'root' },
    merged: null,
    merged_verification: null,
    run_indexed: true,
    test: { set: 'held-out@3', score: 0.84, n: 320 },
    manifest_sha256: 'm'.repeat(64),
    path: 'checkpoints/x',
    ...overrides,
  };
}

describe('[TreeNode]', () => {
  it('should render exactly R-02.29s four facts plus kind, and no pack', () => {
    render(
      <TreeNode
        node={node({
          name: 'LFM2:run-a:000400',
          parent: { target: 'LFM2:run-a:000300', state: 'ok' },
        })}
      />,
    );
    const row = screen.getByTestId('tree-node-LFM2:run-a:000400');
    expect(row).toHaveTextContent('LFM2:run-a:000400');
    // The parent as its FULL qualified name. The mockup abbreviates to `house-sql:400`, and
    // an abbreviation that collides reads as the wrong ancestor.
    expect(row).toHaveTextContent('LFM2:run-a:000300');
    expect(row).toHaveTextContent('notary');
    expect(row).toHaveTextContent('320 prompts');
    expect(row).toHaveTextContent('Checkpoint');
    // FPRD F-10: checkpoint-manifest-v1.json has no pack field, deliberately.
    expect(row.textContent ?? '').not.toMatch(/\bpack\b/i);
  });

  it('should render a null domain as NO DOMAIN, never as a blank', () => {
    // M5.
    render(<DomainFact node={node({ name: 'n', domain: null, domain_state: 'asserted_null' })} />);
    const absence = screen.getByTestId('domain-null');
    expect(absence).toHaveTextContent('no domain');
    expect(absence.textContent ?? '').not.toBe('');
    expect(absence.textContent ?? '').not.toBe('—');
  });

  it('should render an ABSENT domain key as a missing required key, not as no domain', () => {
    // E-3, and the rendering half of M6. `domain` is required AND nullable: null is an
    // assertion, an absent key is an omission, and they are different and worse.
    render(<DomainFact node={node({ name: 'n', domain: null, domain_state: 'absent' })} />);
    const absence = screen.getByTestId('domain-absent');
    expect(absence.getAttribute('data-absent')).toBe('missing_required_key');
  });

  it('should keep the asserted-null and absent renderings distinguishable', () => {
    const { container } = render(
      <>
        <DomainFact node={node({ name: 'a', domain: null, domain_state: 'asserted_null' })} />
        <DomainFact node={node({ name: 'b', domain: null, domain_state: 'absent' })} />
      </>,
    );
    const reasons = [...container.querySelectorAll('[data-absent]')].map((n) =>
      n.getAttribute('data-absent'),
    );
    expect(new Set(reasons).size).toBe(2);
  });

  it('should render an adapters domain as not recorded for this artifact kind', () => {
    // FPRD F-11: adapter-manifest-v1.json has no `domain` property, so R-02.29's own field
    // list is unsatisfiable for one of the two kinds it was amended to cover. The base's
    // domain is NOT read across.
    render(
      <DomainFact
        node={node({ name: 'ad', kind: 'adapter', domain: null, domain_state: 'not_in_kind' })}
      />,
    );
    expect(screen.getByTestId('domain-not-in-kind')).toHaveTextContent(
      'not recorded for this artifact kind',
    );
  });

  it('should render a test score with its sample, or name its absence', () => {
    const { unmount } = render(<TestFact node={node({ name: 'n' })} />);
    expect(screen.getByText(/320 prompts/)).toBeInTheDocument();
    unmount();

    render(<TestFact node={node({ name: 'n', test: null })} />);
    expect(screen.getByText(/no final test score recorded/)).toBeInTheDocument();
  });

  it('should render a root and a truncated chain DIFFERENTLY', () => {
    // M9's rendering half: one means the root, the other that the record stops here.
    const { container } = render(
      <>
        <TreeNode node={node({ name: 'rooted', parent: { target: null, state: 'root' } })} />
        <TreeNode node={node({ name: 'cut', parent: { target: 'gone', state: 'truncated' } })} />
      </>,
    );
    expect(container.textContent).toContain('this is a root');
    // Not just in the edge marker - in the facts column too, which is what a reader scans.
    expect(container.textContent).toContain('lineage stops here');
    expect(container.querySelector('[data-testid="parent-truncated"]')).not.toBeNull();
  });

  it('should distinguish all five non-ok parent states in the facts column', () => {
    // Each names itself, so none of them reads like a resolved parent.
    const rendered = (state: TreeNodeModel['parent']['state'], target: string | null) => {
      const { container, unmount } = render(
        <ParentFact node={node({ name: 'n', parent: { target, state } })} />,
      );
      const text = container.textContent ?? '';
      unmount();
      return text;
    };
    const texts = [
      rendered('root', null),
      rendered('ok', 'base-400'),
      rendered('truncated', 'gone'),
      rendered('not_loaded', 'offpage'),
      rendered('broken', 'loop'),
      rendered('unparented', null),
    ];
    expect(new Set(texts).size).toBe(6);
  });

  it('should name an unindexed run and say reindex is the recovery', () => {
    // FR-009.28 / S3. A checkpoint silently missing from the tree is strictly worse.
    render(<TreeNode node={node({ name: 'orphan', run_indexed: false, run_id: 'run-gone' })} />);
    const notice = screen.getByTestId('run-not-indexed');
    expect(notice).toHaveTextContent('Run not indexed');
    expect(notice).toHaveTextContent('reindex');
    expect(notice).toHaveTextContent('run-gone');
  });

  it('should carry no depth field for a renderer to trust', () => {
    // M17: the mutation requires ADDING one.
    expect('depth' in node({ name: 'n' })).toBe(false);
  });
});

describe('[TreeEdges]', () => {
  it('should draw BOTH edges for a merged checkpoint', () => {
    // M7's rendering half. A merged checkpoint showing one ancestor is a defect, not a
    // simplification.
    render(
      <TreeEdges
        node={node({
          name: 'merged',
          parent: { target: 'base-400', state: 'ok' },
          merged: { target: 'ad-1', state: 'ok' },
          merged_verification: 'verified',
        })}
      />,
    );
    expect(screen.getByTestId('edge-parent-merged')).toHaveTextContent('base-400');
    expect(screen.getByTestId('edge-merged-merged')).toHaveTextContent('ad-1');
  });

  it('should draw only the parent edge for an unmerged checkpoint', () => {
    render(<TreeEdges node={node({ name: 'plain', parent: { target: 'base', state: 'ok' } })} />);
    expect(screen.queryByTestId('edge-merged-plain')).toBeNull();
  });

  it('should report a conforming null hash as unverified and NOT as nonconforming', () => {
    // E-12 vs E-12a: three states, and collapsing the middle reports a conforming manifest
    // as defective.
    render(
      <TreeEdges
        node={node({
          name: 'm1',
          merged: { target: 'ad-1', state: 'ok' },
          merged_verification: 'unverified_null',
        })}
      />,
    );
    const mark = screen.getByTestId('edge-merged-m1').querySelector('[data-merged-verification]');
    expect(mark?.getAttribute('data-merged-verification')).toBe('unverified_null');
    expect(mark?.textContent).toContain('conforming');
    expect(mark?.className).not.toContain('text-red');
  });

  it('should report an absent hash key as nonconforming, in red', () => {
    render(
      <TreeEdges
        node={node({
          name: 'm2',
          merged: { target: 'ad-1', state: 'ok' },
          merged_verification: 'key_absent',
        })}
      />,
    );
    const mark = screen.getByTestId('edge-merged-m2').querySelector('[data-merged-verification]');
    expect(mark?.textContent).toContain('nonconforming');
    expect(mark?.className).toContain('text-red');
  });

  it('should have a distinct sentence for each of the three verification states', () => {
    expect(new Set(Object.values(VERIFICATION_SENTENCES)).size).toBe(3);
    expect(VERIFICATION_SENTENCES.unverified_null).toContain('conforming');
    expect(VERIFICATION_SENTENCES.key_absent).toContain('does not conform');
  });
});
