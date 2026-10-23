/**
 * One row of the forest: indent, facts, edges, and the owners' actions (FR-009.26, .31).
 *
 * **Indent is `depth * 28 px`, and `depth` is walked** (`stores/treeCompose.ts::computeDepths`).
 * No transmitted depth is trusted - `TreeResponse` carries no such field, so taking it from one
 * is mutation **M17** and requires adding the field, which is the point. The mockup's
 * indentation came from a stored `depth` and its own handoff warns that the real tree needs it
 * walked from `parent`.
 *
 * **Actions belong to their owners.** "Branch from here" is F5's `POST .../branch`; "Merge
 * into weights" is F8's `POST /api/v1/adapters/{name}/merge`. This row offers them and renders
 * their refusals; it defines no capability and re-derives no condition. R-02.45's refusal - a
 * merge is refused when no checkpoint of the unmerged base exists - is **F8's refusal,
 * rendered**.
 *
 * **No Export, no "Load in miLLM", no Delete** (FPRD section 3.9): nothing defines export,
 * no requirement obliges miForge to push a checkpoint into a serving process, and no
 * checkpoint lifecycle exists anywhere (FPRD F-12), so this row offers no delete.
 */
import { useState, type ReactNode } from 'react';
import { Button } from '@/components/common/Button';
import { IconTile } from '@/components/common/IconTile';
import { GitBranch, Layers } from 'lucide-react';
import { FailureNotice } from '@/components/common/FailureNotice';
import { SHELL } from '@/config/tokens';
import type { ApiFailure } from '@/api/errors';
import { mergeAdapter } from '@/api/adapters';
import type { TreeNode as TreeNodeModel } from '@/types/tree';
import { TreeNode } from './TreeNode';
import { TreeEdges } from './TreeEdges';

export interface TreeRowProps {
  readonly node: TreeNodeModel;
  readonly depth: number;
  readonly indexStale: boolean;
  readonly onOpenManifest: (node: TreeNodeModel) => void;
}

export function TreeRow({ node, depth, indexStale, onOpenManifest }: TreeRowProps): ReactNode {
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const merge = async (): Promise<void> => {
    const result = await mergeAdapter(node.name);
    if (result.ok) {
      const body = result.value as { readonly note?: unknown };
      setFailure(null);
      setNote(typeof body.note === 'string' ? body.note : 'The merge was admitted.');
    } else {
      setNote(null);
      setFailure(result.failure);
    }
  };

  return (
    <li
      role="treeitem"
      aria-level={depth + 1}
      aria-selected={false}
      tabIndex={-1}
      className="rounded-card border border-line bg-card p-card"
      style={{ marginLeft: depth * SHELL.treeIndentPx }}
      data-testid={`tree-row-${node.name}`}
      data-depth={depth}
      data-index-stale={indexStale ? 'true' : 'false'}
    >
      <div className="flex min-w-0 items-start gap-3">
        {/* accent-ok:row-icon-tile - ADR-012 names "the icon tile on list rows". The glyph
            inside KindMarker is deliberately NOT accent; the mockup painted that one orange
            and FTID 6.3 names it as a bug. */}
        <IconTile icon={node.kind === 'adapter' ? Layers : GitBranch} surface="row-icon-tile" />
        <div className="min-w-0 flex-1">
          <TreeNode node={node} />
          <div className="mt-2">
            <TreeEdges node={node} />
          </div>
          {indexStale ? (
            <div className="mt-2 text-hint text-amber" data-testid={`index-stale-${node.name}`}>
              The index row for this artifact disagrees with its manifest. The manifest wins; open
              it to see the value as written.
            </div>
          ) : null}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button onClick={() => onOpenManifest(node)}>Read the manifest</Button>
            {node.kind === 'adapter' ? (
              <Button onClick={() => void merge()}>Merge into weights</Button>
            ) : null}
          </div>
          {note === null ? null : (
            <div className="mt-2 text-hint text-dim" data-testid={`merge-note-${node.name}`}>
              {note}
            </div>
          )}
          {failure === null ? null : (
            <FailureNotice failure={failure} surface={`merging ${node.name}`} className="mt-2" />
          )}
        </div>
      </div>
    </li>
  );
}
