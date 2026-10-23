/**
 * R-02.29's screen: the forest across runs (FR-009.21-.31).
 *
 * **Depth is computed here from the walked edges** and the forest is **invalidated on a save
 * event, never polled** - a tree of artifacts changes when an artifact is written, which is a
 * discrete event, and polling it would be a request per interval for an answer that does not
 * change between saves.
 *
 * The forest is **`GET /api/v1/tree`** (`services/tree.py`'s join, every edge state, the test
 * score and `run_indexed`). *Until 2026-10-18 this screen joined it in the browser and said
 * the route was not mounted; it was. Task 14.0 removed both the composer and the note.*
 *
 * `role="tree"` with arrow-key navigation (FR-009.7). The tree is a keyboard-navigable
 * structure, not a div list - which is what the mockup's was.
 *
 * **The open manifest is in the URL** (`#/checkpoints?node=<name>`, task 13.0), so a reload
 * re-opens it once the forest has been read. A name the forest no longer holds is dropped
 * from the URL rather than shown as an error: the forest is the answer to "what exists".
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { FailureNotice } from '@/components/common/FailureNotice';
import { ManifestView } from '@/components/panels/run/ManifestView';
import { fetchCheckpointManifest } from '@/api/checkpoints';
import { fetchAdapter } from '@/api/adapters';
import { fetchForest } from '@/api/tree';
import type { ApiFailure } from '@/api/errors';
import { computeDepths, treeOrder } from '@/stores/treeLayout';
import type { TreeNode as TreeNodeModel, TreeResponse } from '@/types/tree';
import { TreeRow } from './TreeRow';
import { useNavigation } from '@/contexts/NavigationContext';

export function CheckpointTreePage(): ReactNode {
  const navigation = useNavigation();
  const nodeFromNav = navigation.params['node'] ?? null;
  const [forest, setForest] = useState<TreeResponse | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [loading, setLoading] = useState(true);
  const [manifest, setManifest] = useState<{
    readonly node: TreeNodeModel;
    readonly document: Readonly<Record<string, unknown>> | null;
    readonly failure: ApiFailure | null;
  } | null>(null);
  const listRef = useRef<HTMLUListElement | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    const result = await fetchForest();
    if (result.ok) {
      setForest(result.value);
      setFailure(null);
    } else {
      setFailure(result.failure);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  // Memoised on its own, so the two memos below are not invalidated on every render by a
  // fresh `[]` from the `??`. ESLint caught this; the symptom would have been the depth walk
  // and the order walk re-running on every keystroke anywhere in the tree.
  const nodes = useMemo(() => forest?.nodes ?? [], [forest]);
  const depths = useMemo(() => computeDepths(nodes), [nodes]);
  const ordered = useMemo(() => treeOrder(nodes), [nodes]);
  const stale = useMemo(() => new Set(forest?.index_stale ?? []), [forest]);

  const openManifest = useCallback(async (node: TreeNodeModel) => {
    const result =
      node.kind === 'checkpoint'
        ? await fetchCheckpointManifest(node.run_id, node.step)
        : await fetchAdapter(node.name);
    if (result.ok) {
      const body = result.value as { readonly manifest?: unknown };
      const document =
        body.manifest !== null && typeof body.manifest === 'object' && !Array.isArray(body.manifest)
          ? (body.manifest as Readonly<Record<string, unknown>>)
          : (result.value as Readonly<Record<string, unknown>>);
      setManifest({ node, document, failure: null });
    } else {
      setManifest({ node, document: null, failure: result.failure });
    }
  }, []);

  const go = navigation.go;
  const showManifest = useCallback(
    (node: TreeNodeModel) => {
      go('checkpoints', { node: node.name }, { replace: true });
      void openManifest(node);
    },
    [go, openManifest],
  );

  // URL -> open manifest, once, after the forest is read.
  const reopened = useRef(false);
  useEffect(() => {
    if (reopened.current || forest === null) return;
    reopened.current = true;
    if (nodeFromNav === null) return;
    const node = nodes.find((candidate) => candidate.name === nodeFromNav);
    if (node === undefined) go('checkpoints', {}, { replace: true });
    else void openManifest(node);
  }, [forest, nodes, nodeFromNav, go, openManifest]);

  /** Arrow-key movement across the rows, which is what `role="tree"` promises. */
  const onKeyDown = (event: React.KeyboardEvent<HTMLUListElement>): void => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    const items = [...(listRef.current?.querySelectorAll<HTMLElement>('[role="treeitem"]') ?? [])];
    if (items.length === 0) return;
    const current = items.findIndex((item) => item === document.activeElement);
    const next = event.key === 'ArrowDown' ? current + 1 : current - 1;
    const target = items[Math.max(0, Math.min(items.length - 1, next === -1 ? 0 : next))];
    if (target !== undefined) {
      event.preventDefault();
      target.focus();
    }
  };

  return (
    <>
      <PageHead
        title="Checkpoints"
        subtitle="Every saved step is a loadable model. A branch can start from any checkpoint."
        actions={<Button onClick={() => void reload()}>Re-read the forest</Button>}
      />

      {failure === null ? null : (
        <FailureNotice failure={failure} surface="the checkpoint forest" className="mb-4" />
      )}

      {loading && forest === null ? (
        <Card data-testid="tree-loading">Reading the checkpoint and adapter forest…</Card>
      ) : ordered.length === 0 ? (
        <Card data-testid="tree-empty">
          No checkpoint and no adapter is indexed. A run writes its first checkpoint at its first
          save; a copied-in artifact appears after <code className="font-mono">reindex</code>.
        </Card>
      ) : (
        <>
          <div className="mb-2 text-hint text-dim" data-testid="tree-count">
            {ordered.length} artifact{ordered.length === 1 ? '' : 's'}:{' '}
            {ordered.filter((node) => node.kind === 'checkpoint').length} checkpoints,{' '}
            {ordered.filter((node) => node.kind === 'adapter').length} adapters.
            {forest !== null && forest.total > ordered.length
              ? ` Showing ${ordered.length} of ${forest.total}: the forest is larger than one page, and an edge to a node beyond it reads "not loaded".`
              : null}
          </div>
          <ul
            ref={listRef}
            role="tree"
            aria-label="Checkpoint and adapter forest"
            className="stack-y-2"
            onKeyDown={onKeyDown}
            data-testid="tree"
          >
            {ordered.map((node) => (
              <TreeRow
                key={node.name}
                node={node}
                depth={depths.get(node.name) ?? 0}
                indexStale={stale.has(node.name)}
                onOpenManifest={showManifest}
              />
            ))}
          </ul>
        </>
      )}

      {manifest === null ? null : (
        <div className="mt-5" data-testid="tree-manifest">
          {manifest.failure === null ? (
            <ManifestView
              document={manifest.document}
              manifestSha256={manifest.node.manifest_sha256}
              indexStale={stale.has(manifest.node.name)}
            />
          ) : (
            <FailureNotice
              failure={manifest.failure}
              surface={`the manifest for ${manifest.node.name}`}
            />
          )}
        </div>
      )}
    </>
  );
}
