/**
 * The seven `EdgeState` values, each a distinct presentation (FR-009.27, .30; FTID 3.6).
 *
 * The map is `satisfies Record<EdgeState, …>`, so **adding a state without a presentation is
 * a typecheck failure** rather than a fall-through to a default. The distinctions that carry
 * weight:
 *
 * - `root` ≠ `truncated`. One is an assertion, the other is *the record stops here*, and
 *   conflating them makes a partial tree look complete (F5 FR-005.39). **M9.**
 * - `not_loaded` ≠ `truncated`. A paging artefact must not look like a broken lineage.
 * - `broken` keeps the node visible and names the edge (FR-009.30): no infinite recursion,
 *   no silent drop.
 *
 * None of the seven is accent. ADR-012 permits the accent on seven surfaces and an edge
 * marker is not one of them.
 */
import type { ReactNode } from 'react';
import type { EdgeState } from '@/types/tree';

interface EdgePresentation {
  readonly label: string;
  readonly sentence: string;
  readonly tone: string;
  readonly glyph: string;
}

export const EDGE_PRESENTATION = {
  ok: {
    label: 'from',
    sentence: 'Descends from the named artifact, which is present in this forest.',
    tone: 'text-dim',
    glyph: '└',
  },
  root: {
    label: 'root',
    sentence:
      'A root. Its manifest records parent: null, which asserts that it descends from nothing in this installation.',
    tone: 'text-dim',
    glyph: '•',
  },
  truncated: {
    label: 'lineage stops here',
    sentence:
      'The record stops here: the named ancestor does not resolve in this installation. This is NOT a root — the chain above it is unknown, not empty.',
    tone: 'text-amber',
    glyph: '⋯',
  },
  broken: {
    label: 'broken edge',
    sentence:
      'A malformed edge — a cycle, a self-parent, or an adapter naming itself as its base. The edge is not followed; the artifact is still shown.',
    tone: 'text-red',
    glyph: '✕',
  },
  unparented: {
    label: 'no base recorded',
    sentence:
      'This adapter names no base. Every adapter composes with something, so an absent base is an artifact that cannot be placed rather than one placed at the top.',
    tone: 'text-red',
    glyph: '?',
  },
  unresolved: {
    label: 'merged adapter not found',
    sentence:
      'The adapter this checkpoint names as merged into it is not in this installation. The edge is shown, named, and not followed.',
    tone: 'text-amber',
    glyph: '⊕',
  },
  not_loaded: {
    label: 'not on this page',
    sentence:
      'The named artifact exists but was outside this listing. A paging boundary, not a broken lineage.',
    tone: 'text-dim',
    glyph: '↑',
  },
} as const satisfies Readonly<Record<EdgeState, EdgePresentation>>;

export interface EdgeMarkProps {
  readonly state: EdgeState;
  /** The name at the other end, when there is one. Rendered in mono by the caller. */
  readonly target?: string | null;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

export function EdgeMark({ state, target, className = '', ...rest }: EdgeMarkProps): ReactNode {
  const presentation = EDGE_PRESENTATION[state];
  return (
    <span
      className={`inline-flex items-center gap-1 text-hint ${presentation.tone} ${className}`.trim()}
      data-edge={state}
      title={presentation.sentence}
      {...rest}
    >
      <span aria-hidden="true">{presentation.glyph}</span>
      <span>{presentation.label}</span>
      {target === undefined || target === null ? null : (
        <span className="font-mono text-ident">{target}</span>
      )}
    </span>
  );
}
