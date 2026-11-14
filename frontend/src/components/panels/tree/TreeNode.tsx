/**
 * Exactly R-02.29's four facts, plus kind (FR-009.24, .25).
 *
 * The four: the **fully qualified name**, the **parent as its full qualified name**, the
 * **domain**, and the **final test score with its sample**. Plus `step`, which every rendered
 * form derives from.
 *
 * ## Two things it deliberately does not show
 *
 * - **The pack.** The mockup's Checkpoints row shows "the pack that produced it" and
 *   `checkpoint-manifest-v1.json` has **no pack field** - deliberately (R-02.35), and
 *   R-02.29 asks for the domain, not the pack. Sourcing it by joining `run_id` to a run row
 *   would fail for an unindexed checkpoint and would make the tree unreindexable, since packs
 *   are the one artifact `reindex` cannot rebuild. FPRD **F-10.**
 * - **An abbreviated parent.** The mockup shortens it to `house-sql:400`; the manifest's
 *   `parent` is a full `name` and that is what is rendered, because an abbreviation that
 *   collides reads as the wrong ancestor.
 *
 * ## The domain has four states and each gets its own presentation
 *
 * `recorded` is the value. `asserted_null` renders **"no domain"** - R-02.29's own amendment
 * says the tree *"shows that as no domain rather than as a blank that reads like missing
 * data"*. `absent` renders as a **missing required key**, which is a different and worse
 * thing. `not_in_kind` renders as **not recorded for this artifact kind**, because
 * `adapter-manifest-v1.json` has no `domain` property and the base's domain is not the
 * adapter's.
 *
 * Rendering `domain ?? ''` is mutation **M5**.
 *
 * *One divergence, reported:* the backend's `models/domain_state.render_domain` renders the
 * same four states as `<value>` / "no domain" / "not recorded" / "not recorded in this
 * artifact kind", and `domain_display` is on the wire. This component renders through
 * `AbsentValue` instead, because FR-009.6 makes that the one absence renderer and FR-009.25
 * requires the `absent` case be named as a **missing required key** - which "not recorded"
 * does not say. So one four-state vocabulary now has two renderings, and that is a finding
 * for the operator rather than something to settle by picking one here.
 */
import type { ReactNode } from 'react';
import { MonoId } from '@/components/common/MonoId';
import { AbsentValue } from '@/components/common/AbsentValue';
import { NumberWithSample } from '@/components/common/NumberWithSample';
import { KindMarker } from '@/components/common/KindMarker';
import type { TreeNode as TreeNodeModel } from '@/types/tree';

export interface TreeNodeProps {
  readonly node: TreeNodeModel;
}

export function DomainFact({ node }: TreeNodeProps): ReactNode {
  switch (node.domain_state) {
    case 'recorded':
      return node.domain === null ? (
        // `recorded` with no value cannot happen through the API's own CHECK constraint, and
        // if it ever arrives it is a defect in the record rather than "no domain".
        <AbsentValue
          reason="missing_required_key"
          detail="domain_state said recorded with no value"
        />
      ) : (
        <span>{node.domain}</span>
      );
    case 'asserted_null':
      return <AbsentValue reason="no_domain" data-testid="domain-null" />;
    case 'absent':
      return (
        <AbsentValue reason="missing_required_key" detail="domain" data-testid="domain-absent" />
      );
    case 'not_in_kind':
      return <AbsentValue reason="not_recorded_for_kind" data-testid="domain-not-in-kind" />;
  }
}

export function TestFact({ node }: TreeNodeProps): ReactNode {
  if (node.test === null) {
    return <AbsentValue reason="no_test_at_this_step" detail="no final test score recorded" />;
  }
  return (
    <NumberWithSample
      value={node.test.score.toFixed(3)}
      sample={node.test.n}
      sampleLabel="prompts"
      qualifier={node.test.set}
    />
  );
}

/**
 * The parent fact, and **a truncated chain reads differently from a root** (FR-009.27).
 *
 * This was wrong on its first writing and the test caught it: a `truncated` edge has a
 * non-null `target`, so printing the target alone rendered *"parent: gone"* - identical in
 * shape to a resolved parent, with the distinction visible only in the edge marker below. A
 * reader scanning the facts column would have read a partial tree as complete, which is
 * exactly what F5's FR-005.39 fix exists to prevent and what M9 is the control for.
 *
 * So each non-`ok` state names itself here too: `root` is an assertion, `truncated` says the
 * record stops, `not_loaded` says the artifact is off this page, and `broken` and `unparented`
 * say the edge was not followed.
 */
export function ParentFact({ node }: TreeNodeProps): ReactNode {
  const { target, state } = node.parent;
  if (state === 'root') return <span className="text-dim">none — this is a root</span>;
  if (state === 'ok') return <MonoId value={target ?? ''} />;
  if (state === 'not_loaded') {
    return (
      <AbsentValue
        reason="target_not_loaded"
        {...(target === null ? {} : { detail: target })}
        data-testid="parent-not-loaded"
      />
    );
  }
  if (state === 'unparented') {
    return (
      <AbsentValue reason="missing_required_key" detail="no base is recorded for this adapter" />
    );
  }
  if (state === 'broken') {
    return (
      <span className="text-red" data-testid="parent-broken">
        broken edge <MonoId value={target ?? 'unnamed'} />
      </span>
    );
  }
  // `truncated` and `unresolved`: the record stops here, and the name that did not resolve is
  // carried so the row can say which one it was.
  return (
    <AbsentValue
      reason="lineage_stops_here"
      {...(target === null ? {} : { detail: target })}
      data-testid="parent-truncated"
    />
  );
}

export function TreeNode({ node }: TreeNodeProps): ReactNode {
  return (
    <div className="min-w-0" data-testid={`tree-node-${node.name}`}>
      <div className="flex min-w-0 flex-wrap items-baseline gap-2">
        <MonoId value={node.name} className="text-text" />
        <KindMarker kind={node.kind} name={node.name} />
        <span className="text-hint text-dim">step {node.step}</span>
      </div>
      <dl className="mt-1 grid grid-cols-1 gap-x-6 gap-y-0.5 text-hint sm:grid-cols-3">
        <div>
          <dt className="inline text-dim">parent: </dt>
          <dd className="inline">
            <ParentFact node={node} />
          </dd>
        </div>
        <div>
          <dt className="inline text-dim">domain: </dt>
          <dd className="inline">
            <DomainFact node={node} />
          </dd>
        </div>
        <div>
          <dt className="inline text-dim">test: </dt>
          <dd className="inline">
            <TestFact node={node} />
          </dd>
        </div>
      </dl>
      {node.run_indexed ? null : (
        <div className="mt-1">
          <AbsentValue
            reason="run_not_indexed"
            variant="block"
            detail={node.run_id}
            data-testid="run-not-indexed"
          />
        </div>
      )}
    </div>
  );
}
