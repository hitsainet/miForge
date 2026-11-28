/**
 * The WSP run page's copy buttons and floating header (operator request 2026-11-22): each step's
 * text box copies its whole content, one button copies the whole page, and the header stays at
 * the top of the scroll. The real floating is measured in `e2e/wsp-run-copy.spec.ts`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { NavigationProvider } from '@/contexts/NavigationContext';
import type { ApiResult } from '@/api/client';
import type { WspProgress, WspRunDocument, WspTranscript } from '@/api/wsp';
import { WSP_RUN_DOCUMENT, WSP_RUN_ID, WSP_RUN_PROGRESS } from '@/testing/stubBackend';
import { WspRunPage } from './WspRunPage';

const ok = <T,>(value: T): Promise<ApiResult<T>> => Promise.resolve({ ok: true, value });
const flush = () => act(() => vi.advanceTimersByTimeAsync(0));

const LONG = Array.from({ length: 300 }, (_, i) => `line ${i}  <b>&amp;</b>`).join('\n');
const ENTRIES = [
  { role: 'user', content: LONG },
  {
    role: 'assistant',
    content: '{"tool":"submit"}',
    reasoning: { text: 'why\n  indented', source: 'reasoning_content' },
    tool_calls: [
      { id: 'a', name: 'python', arguments: { code: 'print(1)' } },
      { id: 'b', name: 'submit', arguments: { answer: 'done' } },
    ],
  },
  { role: 'assistant', content: '', tool_calls: [] },
];

let written: string[];

beforeEach(() => {
  vi.useFakeTimers();
  written = [];
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText: vi.fn((text: string) => Promise.resolve(void written.push(text))) },
  });
});

afterEach(() => {
  vi.useRealTimers();
  Reflect.deleteProperty(navigator, 'clipboard');
});

async function mount(
  options: { status?: string; transcript?: () => Promise<ApiResult<WspTranscript>> } = {},
) {
  const status = options.status ?? 'completed';
  render(
    <NavigationProvider initialScreen="wsp-run" initialParams={{ runId: WSP_RUN_ID }}>
      <WspRunPage
        fetchRun={() =>
          ok({
            ...(WSP_RUN_DOCUMENT as unknown as WspRunDocument),
            status,
            request: { kind: 'repair', mode: 'session', task: 'Repair\nthe probe.' },
            verdict: { value: 'not_significant', reasons: ['auroc_not_above_parent'] },
          })
        }
        fetchProgress={() =>
          ok({
            ...(WSP_RUN_PROGRESS as unknown as WspProgress),
            status,
            live: status === 'running',
          })
        }
        fetchCalls={() => ok({ run_id: WSP_RUN_ID, status, live: false, calls: [], more: false })}
        fetchTranscript={
          options.transcript ??
          (() =>
            ok({
              run_id: WSP_RUN_ID,
              transcript_sha256: 'c'.repeat(64),
              entries: ENTRIES,
              partial: false,
            }))
        }
      />
    </NavigationProvider>,
  );
  await flush();
  await flush();
}

async function press(name: string): Promise<string | undefined> {
  fireEvent.click(screen.getByRole('button', { name }));
  await flush();
  return written[written.length - 1];
}

describe("each step's text boxes", () => {
  it('should copy the whole content, exactly as written', async () => {
    await mount();
    expect(await press('Copy step 1 content')).toBe(LONG);
  });

  it('should copy formatted JSON, the reasoning and each call, each on its own', async () => {
    await mount();
    expect(await press('Copy step 2 content')).toBe('{\n  "tool": "submit"\n}');
    expect(await press('Copy step 2 reasoning')).toBe('why\n  indented');
    expect(await press('Copy step 2 call 1 arguments')).toBe('{\n  "code": "print(1)"\n}');
    expect(await press('Copy step 2 call 2 arguments')).toBe('{\n  "answer": "done"\n}');
  });

  it('should give every box a button, and an empty step none', async () => {
    await mount();
    const steps = screen.getAllByTestId('wsp-transcript-entry');
    for (const step of steps) {
      const boxes = step.querySelectorAll('pre').length;
      expect(within(step).queryAllByTestId('copy-button', {})).toHaveLength(boxes);
    }
    expect(steps[2]!.querySelector('pre')).toBeNull();
  });

  it('should say Copied, then offer Copy again', async () => {
    await mount();
    await press('Copy step 1 content');
    expect(screen.getAllByTestId('copy-button')[0]).toHaveTextContent('Copied');
    await act(() => vi.advanceTimersByTimeAsync(2_500));
    expect(screen.getByRole('button', { name: 'Copy step 1 content' })).toHaveTextContent('Copy');
  });
});

describe('the whole page', () => {
  it('should copy every box, value and heading on the page', async () => {
    await mount();
    const copied = (await press('Copy page')) ?? '';
    const page = screen.getByTestId('wsp-run-page');
    const squeeze = (value: string) => value.replace(/\s+/g, ' ').trim();
    for (const box of page.querySelectorAll('pre')) expect(copied).toContain(box.textContent);
    for (const part of page.querySelectorAll('h1, h2, dt')) {
      expect(copied).toContain(squeeze(part.textContent));
    }
    expect(copied).toContain(WSP_RUN_ID);
    expect(copied).toContain('Kind and mode: repair · session');
    expect(copied).toContain('Repair\nthe probe.');
    expect(copied).toContain('"value": "not_significant"');
    expect(copied).toContain('why\n  indented'); // collapsed on screen, still copied
  });

  it("should not copy the buttons' own labels", async () => {
    await mount();
    const copied = (await press('Copy page')) ?? '';
    expect(copied).not.toMatch(/^(Copy|Copy page|Re-read the run|All WSP runs)$/m);
    expect(copied).not.toContain('Re-read the run');
    expect(copied).not.toContain('In the manual');
  });

  it('should keep separate words apart', async () => {
    await mount();
    const copied = (await press('Copy page')) ?? '';
    const pill = screen.getByTestId('wsp-run-state').textContent;
    const stopped = screen.getByTestId('wsp-run-stop-reason').textContent;
    expect(copied).toContain(`${pill} ${stopped}`);
  });

  it('should wait for the transcript of an ended run, so a copy is never short of it', async () => {
    await mount({ transcript: () => new Promise(() => undefined) });
    expect(screen.getByTestId('wsp-copy-page')).toBeDisabled();
  });

  it('should copy a live run as it stands', async () => {
    await mount({ status: 'running' });
    const copied = (await press('Copy page')) ?? '';
    expect(copied).toContain('The worker writes the transcript when the run ends');
  });

  it('should still copy a run whose transcript cannot be read', async () => {
    await mount({
      transcript: () =>
        Promise.resolve({
          ok: false,
          failure: { kind: 'transport', status: 0, cause: 'connection refused' },
        }),
    });
    expect(await press('Copy page')).toContain(WSP_RUN_ID);
  });
});

describe('the header', () => {
  it('should float, opaque, below the approvals dock, with the page button in it', async () => {
    await mount();
    const header = screen.getByTestId('wsp-run-header');
    expect(header).toHaveClass('sticky', 'top-(--approvals-dock-height,0px)', 'z-20', 'bg-bg');
    expect(within(header).getByTestId('shell-page-head')).toBeInTheDocument();
    expect(within(header).getByTestId('wsp-copy-page')).toBeInTheDocument();
    expect(within(header).getByTestId('wsp-run-updated')).toBeInTheDocument();
  });
});
