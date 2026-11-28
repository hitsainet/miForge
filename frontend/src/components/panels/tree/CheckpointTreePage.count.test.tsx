/**
 * The checkpoint tree's count line (train F4, 2026-11-27).
 *
 * The page read "3 artifacts: 2 checkpoints, 1 adapters." These tests render the page through
 * its real loader, stubbed only at `fetch`, with forests whose counts differ by kind so that a
 * line taking one count's noun for another cannot pass: one checkpoint beside two adapters,
 * then two checkpoints beside one adapter, then a single checkpoint.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { NavigationProvider } from '@/contexts/NavigationContext';
import { CheckpointTreePage } from './CheckpointTreePage';

function checkpoint(step: number): Record<string, unknown> {
  return {
    name: `run-a/step-${step}`,
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

function adapter(step: number): Record<string, unknown> {
  return {
    ...checkpoint(step),
    name: `run-b:adapter/step-${step}`,
    kind: 'adapter',
    run_id: 'run-b',
    domain: null,
    domain_state: 'not_in_kind',
    parent: null,
    base: { target: 'run-a/step-10', state: 'ok' },
    path: `/data/run-b/step-${step}`,
  };
}

function serve(nodes: readonly Record<string, unknown>[]): void {
  vi.stubGlobal('fetch', (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://local').pathname;
    if (url === '/api/v1/tree') {
      return Promise.resolve(
        new Response(
          JSON.stringify({
            nodes,
            edges: [],
            page: { limit: 2000, offset: 0, total: nodes.length },
            index_stale: [],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      );
    }
    return Promise.resolve(new Response('{}', { status: 404 }));
  });
}

afterEach(() => vi.unstubAllGlobals());

async function countLine(nodes: readonly Record<string, unknown>[]): Promise<string> {
  serve(nodes);
  render(
    <NavigationProvider initialScreen="checkpoints" initialParams={{}}>
      <CheckpointTreePage />
    </NavigationProvider>,
  );
  for (let index = 0; index < 4; index += 1) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
  return (screen.getByTestId('tree-count').textContent ?? '').replace(/\s+/g, ' ').trim();
}

describe('[CheckpointTreePage] count line', () => {
  it('should read one adapter in the singular', async () => {
    expect(await countLine([checkpoint(10), checkpoint(20), adapter(5)])).toBe(
      '3 artifacts: 2 checkpoints, 1 adapter.',
    );
  });

  it('should read one checkpoint in the singular', async () => {
    expect(await countLine([checkpoint(10), adapter(5), adapter(6)])).toBe(
      '3 artifacts: 1 checkpoint, 2 adapters.',
    );
  });

  it('should read one artifact in the singular and none in the plural', async () => {
    expect(await countLine([checkpoint(10)])).toBe('1 artifact: 1 checkpoint, 0 adapters.');
  });
});
