import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { AttachmentChip, servingHost } from './AttachmentChip';
import { STALE_ATTACHMENT_COPY } from './weightsState';
import type { AttachmentStatus } from '@/api/attachment';
import { App } from '@/App';
import { ThemeProvider } from '@/contexts/ThemeContext';

// 2026-10-18T11:35:22Z is 7:35 AM in New York (EDT, UTC-4).
const OBSERVED = '2026-10-18T11:35:22.843344+00:00';
const NOW = Date.parse('2026-10-18T11:40:00Z');

const ATTACHED: AttachmentStatus = {
  configured: true,
  reachable: true,
  serving_base_url: 'http://millm-backend.millm.svc.cluster.local:8000',
  attached: true,
  holder: 'trainer:nightly-forge',
  resident_model: 'LiquidAI/LFM2.5-1.2B-Instruct',
  weights_state: 'clean',
  stale: false,
  observed_at: OBSERVED,
};

const DETACHED: AttachmentStatus = {
  ...ATTACHED,
  attached: false,
  holder: null,
  stale: null,
  resident_model: 'Llama-3.1-8B-Instruct',
};

describe('[AttachmentChip]', () => {
  it('should show the serving process, the holder, the model and the cyan state', () => {
    render(<AttachmentChip status={ATTACHED} nowMs={NOW} />);
    expect(screen.getByTestId('attachment-chip')).toHaveAttribute('data-attachment', 'attached');
    expect(screen.getByTestId('attachment-serving')).toHaveTextContent(
      'millm-backend.millm.svc.cluster.local:8000',
    );
    expect(screen.getByTestId('attachment-holder')).toHaveTextContent(
      'held by trainer:nightly-forge',
    );
    expect(screen.getByTestId('attachment-model')).toHaveTextContent('LFM2.5-1.2B-Instruct');
    expect(screen.getByTestId('weights-state-pill')).toHaveAttribute('data-tone', 'cyan');
    expect(screen.queryByTestId('attachment-stale')).toBeNull();
  });

  it('should say not attached when miLLM holds no attachment', () => {
    render(<AttachmentChip status={DETACHED} nowMs={NOW} />);
    const chip = screen.getByTestId('attachment-chip');
    expect(chip).toHaveAttribute('data-attachment', 'detached');
    expect(screen.getByTestId('attachment-detached')).toHaveTextContent(
      'Not attached to miLLM at millm-backend.millm.svc.cluster.local:8000',
    );
    // A detached chip names no model or state as though something were attached to it.
    expect(screen.queryByTestId('attachment-model')).toBeNull();
    expect(screen.queryByTestId('weights-state-pill')).toBeNull();
    expect(chip.textContent ?? '').not.toMatch(/Attached to/);
  });

  it('should show the check time in US Eastern, with the ISO instant on hover', () => {
    render(<AttachmentChip status={DETACHED} nowMs={NOW} />);
    const observed = screen.getByTestId('attachment-observed');
    expect(observed).toHaveTextContent('checked Oct 18, 7:35 AM ET');
    expect(observed).toHaveAttribute('title', OBSERVED);
    expect(observed.textContent ?? '').not.toMatch(/T11:35/);
  });

  it('should say how old the state is when the latest check failed', () => {
    render(<AttachmentChip status={ATTACHED} outdated nowMs={NOW} />);
    expect(screen.getByTestId('attachment-observed')).toHaveTextContent(
      'last checked Oct 18, 7:35 AM ET (5 min ago)',
    );
  });

  it('should name an unreachable miLLM rather than report an attachment', () => {
    render(
      <AttachmentChip
        status={{
          configured: true,
          reachable: false,
          reason: 'weights_surface_unreachable',
          serving_base_url: 'http://millm-backend.millm.svc.cluster.local:8000',
          observed_at: OBSERVED,
        }}
        nowMs={NOW}
      />,
    );
    const chip = screen.getByTestId('attachment-chip');
    expect(chip).toHaveAttribute('data-attachment', 'unreachable');
    expect(chip).toHaveTextContent('not reachable');
    expect(chip).toHaveAttribute('title', 'weights_surface_unreachable');
  });

  it('should name the operator clear-stale action, and that miForge cannot perform it', () => {
    render(<AttachmentChip status={{ ...ATTACHED, stale: true }} nowMs={NOW} />);
    const copy = screen.getByTestId('attachment-stale');
    expect(copy).toHaveTextContent(STALE_ATTACHMENT_COPY);
    expect(STALE_ATTACHMENT_COPY).toContain('clear-stale');
    expect(STALE_ATTACHMENT_COPY).toMatch(/operator-only/);
    expect(STALE_ATTACHMENT_COPY).toMatch(/miForge cannot clear it/);
  });

  it('should contain no "not reported" copy in any state', () => {
    const { unmount } = render(
      <AttachmentChip
        status={{ ...ATTACHED, resident_model: null, weights_state: null }}
        nowMs={NOW}
      />,
    );
    expect(screen.getByTestId('attachment-chip').textContent ?? '').not.toMatch(/not reported/);
    unmount();
  });

  it('should fall back to the raw URL when it does not parse', () => {
    expect(servingHost('not a url')).toBe('not a url');
  });
});

describe('[AttachmentChip] in the shell', () => {
  let originalFetch: typeof globalThis.fetch;
  let attachmentReads = 0;

  function serveAttachment(body: unknown): void {
    attachmentReads = 0;
    globalThis.fetch = ((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/api/v1/attachment')) attachmentReads += 1;
      const payload = url.endsWith('/api/v1/attachment') ? body : {};
      return Promise.resolve(
        new Response(JSON.stringify(payload), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    }) as typeof globalThis.fetch;
  }

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  const inTheme = (element: React.ReactElement) => (
    <ThemeProvider initialTheme="dark">{element}</ThemeProvider>
  );

  it('should show the live attachment state in the strip on any screen', async () => {
    serveAttachment(ATTACHED);
    render(inTheme(<App initialScreen="checkpoints" />));
    const strip = screen.getByTestId('shell-resource-strip');
    await waitFor(() =>
      expect(strip.querySelector('[data-testid="attachment-chip"]')).not.toBeNull(),
    );
    expect(screen.getByTestId('attachment-chip')).toHaveAttribute('data-attachment', 'attached');
    expect(attachmentReads).toBe(1);
  });

  it('should show not attached when miLLM reports no attachment', async () => {
    serveAttachment(DETACHED);
    render(inTheme(<App initialScreen="runs" />));
    await waitFor(() =>
      expect(screen.getByTestId('attachment-chip')).toHaveAttribute('data-attachment', 'detached'),
    );
  });

  it('should show no chip when no serving process is configured', async () => {
    serveAttachment({ configured: false });
    render(inTheme(<App initialScreen="runs" />));
    await waitFor(() => expect(attachmentReads).toBe(1));
    expect(screen.queryByTestId('attachment-chip')).toBeNull();
  });
});
