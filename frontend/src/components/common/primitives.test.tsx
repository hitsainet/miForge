/**
 * The design-system primitives (foundation tasks 7.2, 7.3; ADR-012).
 *
 * `Card`, `Button`, `MonoId`, `IconTile`, `Callout` and `Unavailable` in one file because
 * each has two or three assertions and they share the card grammar they are asserting
 * against. The per-rule tests that carry a *design rule* get their own files -
 * `Pill.test.tsx` for the accent-on-pills rule, `../../config/accent.test.tsx` for the
 * accent allowlist, `../../focusRing.test.tsx` for 7.5.
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Database, Flame } from 'lucide-react';
import { Card, CARD_CLASSES } from './Card';
import { Button } from './Button';
import { MonoId, MONO_CLASSES, shortenIdentifier } from './MonoId';
import { IconTile } from './IconTile';
import { Callout } from './Callout';
import { Select } from './Field';
import { Unavailable } from './Unavailable';
import { decodeErrorEnvelope, type NotConfiguredFailure } from '@/api/errors';
import { SHELL } from '@/config/tokens';

describe('[Card]', () => {
  it('should carry the card grammar: 12 px radius, a 1 px line border, 20 px padding', () => {
    render(<Card data-testid="c">Body</Card>);
    const classes = screen.getByTestId('c').className;
    // The utilities are generated from `config/tokens.ts`, so this asserts the grammar
    // rather than a literal: `rounded-card` IS 12 px because SHELL says so.
    expect(classes).toContain('rounded-card');
    expect(classes).toContain('border-line');
    expect(classes).toContain('p-card');
    expect(CARD_CLASSES).toContain('bg-card');
  });

  it('should render as a different element when asked, for a list or a region', () => {
    render(
      <ul>
        <Card as="li" data-testid="row">
          A row
        </Card>
      </ul>,
    );
    expect(screen.getByTestId('row').tagName).toBe('LI');
  });

  it('should accept an inline style for the one thing utilities cannot express', () => {
    // The checkpoint tree indents by `depth * 28 px`, which is per-row and arbitrary.
    render(
      <Card data-testid="c" style={{ marginLeft: SHELL.treeIndentPx * 2 }}>
        A node two levels deep
      </Card>,
    );
    expect(screen.getByTestId('c').style.marginLeft).toBe('56px');
  });
});

describe('[Button]', () => {
  it('should accept an onClick, which the mockups Primary did not', () => {
    // The handoff's own 2026-09-19 review: "`Primary` accepts no `onClick` at all, so
    // *every* primary action is inert".
    const onClick = vi.fn();
    render(
      <Button variant="primary" onClick={onClick}>
        Create and start
      </Button>,
    );
    return userEvent.click(screen.getByRole('button')).then(() => {
      expect(onClick).toHaveBeenCalledOnce();
    });
  });

  it('should default to type=button so it cannot submit a form by accident', () => {
    render(<Button>Branch from here</Button>);
    expect(screen.getByRole('button')).toHaveAttribute('type', 'button');
  });

  it('should default to the ghost variant, so accent is opt-in', () => {
    render(<Button data-testid="b">Export</Button>);
    expect(screen.getByTestId('b').className).toContain('border-line');
  });

  it('should hide a decorative icon from assistive technology', () => {
    const { container } = render(
      <Button variant="primary" icon={Flame}>
        Create and start
      </Button>,
    );
    expect(container.querySelector('svg')).toHaveAttribute('aria-hidden', 'true');
    expect(screen.getByRole('button')).toHaveAccessibleName('Create and start');
  });
});

describe('[MonoId]', () => {
  it('should render an identifier in the mono face', () => {
    render(<MonoId value="LFM2-1.2B:md-notary:step-000400" data-testid="id" />);
    expect(screen.getByTestId('id').className).toContain('font-mono');
    expect(MONO_CLASSES).toContain('text-ident');
  });

  it('should render the full value by default', () => {
    // A truncated identifier is a decision, and the default must not be the one that
    // hides information.
    const hash = 'a'.repeat(64);
    render(<MonoId value={hash} data-testid="id" />);
    expect(screen.getByTestId('id')).toHaveTextContent(hash);
    expect(screen.getByTestId('id')).not.toHaveAttribute('title');
  });

  it('should keep the whole value reachable when it truncates', () => {
    // Verifying a digest against a manifest is a thing this product asks the operator to
    // do, so a shortened hash with no way to read the rest is a value they cannot check.
    const hash = '7c1e4f00000000000000a0b1';
    render(<MonoId value={hash} truncateTo={6} data-testid="id" />);
    expect(screen.getByTestId('id')).toHaveTextContent('7c1e4f…00a0b1');
    expect(screen.getByTestId('id')).toHaveAttribute('title', hash);
    expect(screen.getByTestId('id')).toHaveAttribute('aria-label', hash);
  });

  it('should not truncate when truncating would not shorten', () => {
    expect(shortenIdentifier('step-400', 6)).toBe('step-400');
    expect(shortenIdentifier('anything', undefined)).toBe('anything');
  });
});

describe('[IconTile]', () => {
  it('should be 36 px square, per the handoffs shape rule', () => {
    render(<IconTile icon={Database} surface="row-icon-tile" data-testid="t" />);
    const tile = screen.getByTestId('t');
    expect(tile.style.width).toBe(`${SHELL.iconTilePx}px`);
    expect(tile.style.height).toBe(`${SHELL.iconTilePx}px`);
    expect(SHELL.iconTilePx).toBe(36);
  });

  it('should require its surface to be named, with no accent default', () => {
    render(<IconTile icon={Database} surface="neutral" data-testid="t" />);
    // A tile that defaulted to an accent surface would *become* one by being written,
    // which is the opposite of an allowlist.
    expect(screen.getByTestId('t')).toHaveAttribute('data-surface', 'neutral');
    expect(screen.getByTestId('t').className).not.toContain('accent');
  });
});

describe('[Callout]', () => {
  it('should tint with the accent only for the product statement', () => {
    const { container } = render(
      <Callout tone="accent" title="miForge has no built-in domain.">
        Delete the examples whenever you like; nothing depends on them.
      </Callout>,
    );
    expect(container.firstElementChild?.className).toContain('bg-accent-tint');
  });

  it('should not tint a warning or an error with the accent', () => {
    // An orange warning makes warnings and identity the same colour, which is how an
    // operator stops reading either.
    for (const tone of ['warning', 'error', 'info'] as const) {
      const { container, unmount } = render(<Callout tone={tone}>Something to fix</Callout>);
      expect(container.firstElementChild?.className).not.toContain('accent');
      unmount();
    }
  });

  it('should announce a warning and an error, and not an informational banner', () => {
    const { unmount } = render(<Callout tone="warning">Reduce the batch size.</Callout>);
    expect(screen.getByRole('alert')).toBeInTheDocument();
    unmount();
    render(<Callout tone="accent">A statement.</Callout>);
    expect(screen.queryByRole('alert')).toBeNull();
  });
});

describe('[Unavailable]', () => {
  const failure = decodeErrorEnvelope(409, {
    error: {
      code: 'midataworks_not_configured',
      message: 'midataworks is not configured; set MIDATAWORKS_BASE_URL to enable contract parts.',
      details: {
        service: 'midataworks',
        capabilities_unavailable: ['contract parts', 'human-review selector'],
      },
    },
  }) as NotConfiguredFailure;

  it('should name the service and list what it costs', () => {
    render(<Unavailable failure={failure} surface="contract parts" data-testid="u" />);
    expect(screen.getByTestId('u')).toHaveTextContent('midataworks is not configured');
    expect(screen.getByText('contract parts')).toBeInTheDocument();
    expect(screen.getByText('human-review selector')).toBeInTheDocument();
  });

  it('should carry the backends own message, which says what to do next', () => {
    render(<Unavailable failure={failure} />);
    expect(screen.getByText(/MIDATAWORKS_BASE_URL/)).toBeInTheDocument();
  });

  it('should be a status, not an error: standalone is the supported configuration', () => {
    // AC20. Painting a supported configuration red teaches the operator to ignore red.
    render(<Unavailable failure={failure} data-testid="u" />);
    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.getByTestId('u').className).not.toContain('red');
  });

  it('should say the capabilities were not stated rather than guess at them', () => {
    const bare = decodeErrorEnvelope(409, {
      error: { code: 'millm_not_configured', message: 'no base URL', details: {} },
    }) as NotConfiguredFailure;
    render(<Unavailable failure={bare} />);
    expect(screen.getByText(/did not state which capabilities/)).toBeInTheDocument();
  });
});

describe('[Select] reads as a dropdown (operator, 2026-10-31)', () => {
  it('should replace the browser arrow with the suite chevron and sit on the card surface', () => {
    render(
      <Select
        aria-label="Log level"
        closedEnum="test"
        placeholder="Choose a log level"
        options={[
          { value: 'INFO', label: 'INFO' },
          { value: 'DEBUG', label: 'DEBUG' },
        ]}
      />,
    );
    const select = screen.getByRole('combobox', { name: 'Log level' });
    expect(select).toHaveClass('appearance-none', 'bg-card', 'pr-9');
    expect(select.parentElement?.querySelector('[data-testid="dropdown-chevron"]')).not.toBeNull();
  });
});
