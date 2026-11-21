/**
 * Where an opened manifest renders, and how it closes (operator bug, 2026-10-28).
 *
 * "Read the manifest" used to render the manifest **after the whole forest**, so on a list
 * longer than a screen the click appeared to do nothing. These tests pin that the manifest
 * renders **inside the row it belongs to**, ahead of every later row, that the row's button
 * toggles it, and that only one is open at a time (the URL names one node).
 *
 * The forest has three rows so "inside the right row" and "after the list" cannot agree by
 * construction: a manifest for the middle row appended after the list would sit after row 3.
 *
 * Stubbed at `fetch`, the HTTP boundary.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import type { ReactNode } from 'react';
import { NavigationProvider, useNavigation } from '@/contexts/NavigationContext';
import { CheckpointTreePage } from './CheckpointTreePage';

const STEPS = [10, 20, 30] as const;
const nameOf = (step: number): string => `run-a/step-${step}`;

function wireNode(step: number): Record<string, unknown> {
  return {
    name: nameOf(step),
    kind: 'checkpoint',
    run_id: 'run-a',
    step,
    domain: 'notary',
    domain_state: 'recorded',
    parent: { target: null, state: 'root' },
    base: null,
    merged: null,
    merged_verification: null,
    test: null,
    manifest_sha256: String(step).padStart(64, '0'),
    path: `/data/run-a/step-${step}`,
    run_indexed: true,
  };
}

let calls: string[] = [];
/** A manifest path whose answer the test releases by hand, to order two answers. */
let held: Map<string, () => void> = new Map();
let hold: Set<string> = new Set();

beforeEach(() => {
  calls = [];
  held = new Map();
  hold = new Set();
  vi.stubGlobal('fetch', (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://local').pathname;
    calls.push(url);
    const respond = (body: unknown): Response =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    if (url === '/api/v1/tree') {
      return Promise.resolve(
        respond({
          nodes: STEPS.map(wireNode),
          edges: [],
          page: { limit: 2000, offset: 0, total: STEPS.length },
          index_stale: [],
        }),
      );
    }
    const match = /^\/api\/v1\/runs\/run-a\/checkpoints\/(\d+)$/.exec(url);
    if (match !== null) {
      const body = { run_id: 'run-a', step: Number(match[1]), manifest: { kind: 'checkpoint' } };
      if (hold.has(url)) {
        return new Promise<Response>((resolve) => held.set(url, () => resolve(respond(body))));
      }
      return Promise.resolve(respond(body));
    }
    return Promise.resolve(new Response('{}', { status: 404 }));
  });
});

afterEach(() => vi.unstubAllGlobals());

async function settle(): Promise<void> {
  for (let index = 0; index < 4; index += 1) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
}

/** Shows the `node` URL parameter the page writes, so the tests can read it. */
function NodeParam(): ReactNode {
  const navigation = useNavigation();
  return <output data-testid="node-param">{navigation.params['node'] ?? ''}</output>;
}

function mount(params: Record<string, string> = {}): void {
  render(
    <NavigationProvider initialScreen="checkpoints" initialParams={params}>
      <CheckpointTreePage />
      <NodeParam />
    </NavigationProvider>,
  );
}

const row = (step: number): HTMLElement => screen.getByTestId(`tree-row-${nameOf(step)}`);
const toggle = (step: number): HTMLElement => screen.getByTestId(`manifest-toggle-${nameOf(step)}`);

/** True when `a` comes before `b` in document order. */
const precedes = (a: Node, b: Node): boolean =>
  (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;

describe('[CheckpointTreePage] manifest placement', () => {
  it('should render the manifest inside the row it belongs to, before every later row', async () => {
    mount();
    await settle();
    fireEvent.click(toggle(20));
    await settle();

    const panel = screen.getByTestId('tree-manifest');
    expect(panel.getAttribute('data-manifest-for')).toBe(nameOf(20));
    // Inside row 2's own list item - not after the list, not in a neighbour.
    expect(row(20)).toContainElement(panel);
    expect(within(row(10)).queryByTestId('tree-manifest')).toBeNull();
    expect(within(row(30)).queryByTestId('tree-manifest')).toBeNull();
    expect(screen.getByTestId('tree')).toContainElement(panel);
    expect(precedes(panel, row(30))).toBe(true);
    // The manifest that renders is the one fetched for this row.
    expect(calls).toContain('/api/v1/runs/run-a/checkpoints/20');
    expect(panel).not.toHaveTextContent(/Reading the manifest/);
  });

  it('should close the manifest on a second click and drop it from the URL', async () => {
    mount();
    await settle();
    expect(toggle(20)).toHaveTextContent('Read the manifest');
    expect(toggle(20).getAttribute('aria-expanded')).toBe('false');

    fireEvent.click(toggle(20));
    await settle();
    expect(toggle(20)).toHaveTextContent('Close the manifest');
    expect(toggle(20).getAttribute('aria-expanded')).toBe('true');
    const panel = screen.getByTestId('tree-manifest');
    expect(toggle(20).getAttribute('aria-controls')).toBe(panel.id);
    expect(screen.getByTestId('node-param')).toHaveTextContent(nameOf(20));

    fireEvent.click(toggle(20));
    await settle();
    expect(screen.queryByTestId('tree-manifest')).toBeNull();
    expect(toggle(20)).toHaveTextContent('Read the manifest');
    expect(toggle(20).getAttribute('aria-expanded')).toBe('false');
    expect(screen.getByTestId('node-param')).toHaveTextContent(/^$/);
  });

  it('should move the open manifest to another row, one at a time', async () => {
    mount();
    await settle();
    fireEvent.click(toggle(10));
    await settle();
    fireEvent.click(toggle(30));
    await settle();
    const panels = screen.getAllByTestId('tree-manifest');
    expect(panels).toHaveLength(1);
    expect(row(30)).toContainElement(panels[0] ?? null);
    expect(toggle(10)).toHaveTextContent('Read the manifest');
    expect(screen.getByTestId('node-param')).toHaveTextContent(nameOf(30));
  });

  it('should discard an answer that arrives after its manifest was closed', async () => {
    hold.add('/api/v1/runs/run-a/checkpoints/10');
    mount();
    await settle();
    fireEvent.click(toggle(10));
    await settle();
    expect(within(row(10)).getByTestId('tree-manifest')).toHaveTextContent(/Reading the manifest/);
    fireEvent.click(toggle(10));
    await settle();
    held.get('/api/v1/runs/run-a/checkpoints/10')?.();
    await settle();
    expect(screen.queryByTestId('tree-manifest')).toBeNull();
    expect(toggle(10)).toHaveTextContent('Read the manifest');
  });

  it('should reopen the manifest the URL names inside that row', async () => {
    mount({ node: nameOf(30) });
    await settle();
    const panel = screen.getByTestId('tree-manifest');
    expect(row(30)).toContainElement(panel);
    expect(toggle(30)).toHaveTextContent('Close the manifest');
  });
});
