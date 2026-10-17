import { afterEach, describe, expect, it } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { AttachmentChip, servingHost } from './AttachmentChip';
import { STALE_ATTACHMENT_COPY, type AttachmentBlock } from './weightsState';
import { App } from '@/App';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { useAttachmentStore } from '@/stores/attachmentStore';

const BLOCK: AttachmentBlock = {
  serving_base_url: 'http://millm-backend.millm.svc:8000',
  resident_model: 'LiquidAI/LFM2.5-1.2B-Instruct',
  weights_state: 'clean',
  holder: null,
  stale: false,
};

afterEach(() => {
  act(() => useAttachmentStore.getState().show(null));
});

describe('[AttachmentChip]', () => {
  it('should show the serving process, the resident model and the cyan state', () => {
    render(<AttachmentChip attachment={BLOCK} />);
    expect(screen.getByTestId('attachment-serving')).toHaveTextContent(
      'millm-backend.millm.svc:8000',
    );
    expect(screen.getByTestId('attachment-model')).toHaveTextContent('LFM2.5-1.2B-Instruct');
    expect(screen.getByTestId('weights-state-pill')).toHaveAttribute('data-tone', 'cyan');
    expect(screen.queryByTestId('attachment-holder')).toBeNull();
    expect(screen.queryByTestId('attachment-stale')).toBeNull();
  });

  it('should say when the state was observed, and when it never was', () => {
    const { unmount } = render(<AttachmentChip attachment={BLOCK} />);
    expect(screen.getByTestId('attachment-observed')).toHaveTextContent('state not yet observed');
    unmount();
    render(
      <AttachmentChip
        attachment={{ ...BLOCK, attached: false, observed_at: '2026-10-16T22:00:00+00:00' }}
      />,
    );
    expect(screen.getByTestId('attachment-observed')).toHaveTextContent('2026-10-16T22:00:00');
    expect(screen.getByTestId('attachment-detached')).toBeInTheDocument();
  });

  it('should name the holder when it is not this run', () => {
    render(<AttachmentChip attachment={{ ...BLOCK, holder: 'trainer:nightly-forge' }} />);
    expect(screen.getByTestId('attachment-holder')).toHaveTextContent(
      'held by trainer:nightly-forge',
    );
  });

  it('should say not reported rather than guess a model or a state', () => {
    render(<AttachmentChip attachment={{ ...BLOCK, resident_model: null, weights_state: null }} />);
    expect(screen.getByTestId('attachment-model')).toHaveTextContent('not reported');
    expect(screen.getByTestId('weights-state-unreported')).toBeInTheDocument();
  });

  it('should name the operator clear-stale action, and that miForge cannot perform it', () => {
    render(<AttachmentChip attachment={{ ...BLOCK, stale: true }} />);
    const copy = screen.getByTestId('attachment-stale');
    expect(copy).toHaveTextContent(STALE_ATTACHMENT_COPY);
    expect(STALE_ATTACHMENT_COPY).toContain('clear-stale');
    expect(STALE_ATTACHMENT_COPY).toMatch(/operator-only/);
    expect(STALE_ATTACHMENT_COPY).toMatch(/miForge cannot clear it/);
  });

  it('should fall back to the raw URL when it does not parse', () => {
    expect(servingHost('not a url')).toBe('not a url');
  });
});

describe('[AttachmentChip] in the shell', () => {
  const inTheme = (element: React.ReactElement) => (
    <ThemeProvider initialTheme="dark">{element}</ThemeProvider>
  );

  it('should put the chip in the resource strip slot only while an attachment is shown', () => {
    render(inTheme(<App initialScreen="checkpoints" />));
    const strip = screen.getByTestId('shell-resource-strip');
    expect(strip.querySelector('[data-testid="attachment-chip"]')).toBeNull();

    act(() => useAttachmentStore.getState().show(BLOCK));
    expect(strip.querySelector('[data-testid="attachment-chip"]')).not.toBeNull();

    act(() => useAttachmentStore.getState().show(null));
    expect(strip.querySelector('[data-testid="attachment-chip"]')).toBeNull();
  });
});
