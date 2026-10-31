/**
 * The error envelope, decoded and rendered (FR-009.5; ADR-013, ADR-021).
 *
 * **No generic branch over the code.** An unrecognised refusal renders the code itself: the
 * code is the only diagnosable thing in the response, and a friendly sentence in its place
 * discards it. And an **unconfigured optional sibling is not an error** - it goes to
 * `Unavailable`, which names the service and the capabilities lost, because absent-and-normal
 * rendered as a fault teaches an operator to distrust the screen (ADR-010).
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FailureNotice, renderDetail } from './FailureNotice';
import { decodeErrorEnvelope, transportFailure } from '@/api/errors';

describe('[FailureNotice]', () => {
  it('should render a refusals own code rather than a paraphrase of it', () => {
    const failure = decodeErrorEnvelope(409, {
      error: {
        code: 'checkpoint_hash_mismatch',
        message: 'Recomputed digest does not match the manifest. Nothing was changed.',
        details: { expected: 'a'.repeat(64), actual: 'b'.repeat(64) },
      },
    });
    render(<FailureNotice failure={failure} surface="verifying the checkpoint" />);
    expect(screen.getByTestId('failure-code')).toHaveTextContent('checkpoint_hash_mismatch');
    expect(screen.getByText(/Nothing was changed/)).toBeInTheDocument();
  });

  it('should render every detail rather than summarising them', () => {
    // `budget_exceeded` without both figures is a refusal an operator cannot act on.
    const failure = decodeErrorEnvelope(400, {
      error: {
        code: 'budget_exceeded',
        message: 'This run does not fit.',
        details: { estimate_bytes: 25_000_000_000, free_bytes: 21_300_000_000 },
      },
    });
    render(<FailureNotice failure={failure} surface="starting the run" />);
    const details = screen.getByTestId('failure-details');
    expect(details).toHaveTextContent('estimate_bytes');
    expect(details).toHaveTextContent('free_bytes');
  });

  it('should route an unconfigured service to Unavailable, naming it, not as an error', () => {
    const failure = decodeErrorEnvelope(503, {
      error: {
        code: 'midataworks_not_configured',
        message: 'Set MIDATAWORKS_BASE_URL to use the review queue.',
        details: { capabilities_unavailable: ['human review'] },
      },
    });
    render(<FailureNotice failure={failure} surface="the review queue" data-testid="f" />);
    const node = screen.getByTestId('f');
    expect(node.getAttribute('data-service')).toBe('midataworks');
    expect(node).toHaveTextContent('midataworks is not configured');
    expect(node).toHaveTextContent('human review');
    expect(node).toHaveTextContent(/continues to work/);
  });

  it('should name the region a transport failure degraded and say what to do next', () => {
    render(
      <FailureNotice failure={transportFailure('TypeError: fetch failed')} surface="the curves" />,
    );
    expect(screen.getByText(/^The curves is unavailable/)).toBeInTheDocument();
    expect(screen.getByText(/Check that the backend is running/)).toBeInTheDocument();
  });

  it('should show the body of a malformed response rather than guessing at it', () => {
    const failure = decodeErrorEnvelope(500, '<html>nginx</html>');
    render(<FailureNotice failure={failure} surface="the forest" />);
    expect(screen.getByText(/nginx/)).toBeInTheDocument();
  });

  it('should render a detail list, an object and a null without collapsing them', () => {
    expect(renderDetail(['a', 'b'])).toBe('a, b');
    expect(renderDetail(null)).toBe('null');
    expect(renderDetail({ a: 1 })).toBe('{"a":1}');
  });
});
