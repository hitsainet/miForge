/**
 * A node's edges — one, or **two for a merged checkpoint** (FR-009.22, .27, .30).
 *
 * **A merged checkpoint shows both ancestors**: `parent` to the base it came from, and
 * `merged_adapter` to the adapter baked into it. Showing one is a defect, not a
 * simplification - it is mutation **M7**, and it undoes the reason `merged_adapter` was added
 * to `checkpoint-manifest-v1.json` on 2026-10-13: *a merged checkpoint has two ancestors, and
 * `parent` carries only the base.*
 *
 * The merged edge is drawn as a **marked second edge** so the tree's ordering stays
 * one-dimensional (depth comes from the `parent` chain, FR-009.26) while the lineage stays
 * complete.
 *
 * ## The verification state is three-valued
 *
 * `MergedAdapter.manifest_sha256` is **required and nullable**, so: a hash (verified), present
 * and `null` (**conforming**, deliberately unverified), and absent (a missing required key,
 * nonconforming). A boolean would report the middle as the third - reporting a conforming
 * manifest as defective, which is the conflation FR-009.17 rule 3 forbids for `domain`.
 */
import type { ReactNode } from 'react';
import { EdgeMark } from '@/components/common/EdgeMark';
import type { MergedVerification, TreeNode } from '@/types/tree';

export const VERIFICATION_SENTENCES = {
  verified:
    'The merged adapter is named with its manifest hash, so the edge can be verified against the artifact.',
  unverified_null:
    'The merged adapter is named with manifest_sha256 explicitly null. That is conforming — the schema makes the key required and nullable — and it means the edge is deliberately unverified, not that the manifest is defective.',
  key_absent:
    'The merged adapter record has no manifest_sha256 key at all. The schema requires it, so this manifest does not conform; the edge is shown and named rather than dropped.',
} as const satisfies Readonly<Record<MergedVerification, string>>;

export interface TreeEdgesProps {
  readonly node: TreeNode;
}

export function TreeEdges({ node }: TreeEdgesProps): ReactNode {
  return (
    <div
      className="flex flex-wrap items-center gap-x-4 gap-y-1"
      data-testid={`tree-edges-${node.name}`}
    >
      <EdgeMark
        state={node.parent.state}
        target={node.parent.target}
        data-testid={`edge-parent-${node.name}`}
      />
      {/* IN ADDITION, never instead. See the module docstring. */}
      {node.merged === null ? null : (
        <span
          className="flex flex-wrap items-center gap-2"
          data-testid={`edge-merged-${node.name}`}
        >
          <span className="text-hint text-dim">merged adapter:</span>
          <EdgeMark state={node.merged.state} target={node.merged.target} />
          {node.merged_verification === null ? null : (
            <span
              className={
                node.merged_verification === 'key_absent'
                  ? 'text-hint text-red'
                  : 'text-hint text-dim'
              }
              data-merged-verification={node.merged_verification}
              title={VERIFICATION_SENTENCES[node.merged_verification]}
            >
              {node.merged_verification === 'verified'
                ? 'hash recorded'
                : node.merged_verification === 'unverified_null'
                  ? 'hash explicitly null — conforming, unverified'
                  : 'manifest_sha256 missing — nonconforming'}
            </span>
          )}
        </span>
      )}
    </div>
  );
}
