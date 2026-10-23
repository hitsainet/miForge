/**
 * What an artifact **is**, from `kind` and never from its name (FR-009.23).
 *
 * F8 names an adapter `<base_model>:<run_id>:adapter-<step>`, so the string is
 * distinguishable - and **a view that parses the name to learn what something is renders a
 * hand-named artifact wrongly.** Both are shown: the kind comes from `kind`, and where the
 * name disagrees with it the disagreement is **named as a name anomaly** rather than allowed
 * to decide. A checkpoint called `…adapter-400` is a checkpoint whose name is odd, not an
 * adapter, and that is what this renders.
 */
import type { ReactNode } from 'react';
import { GitBranch, Layers } from 'lucide-react';
import type { ArtifactKind } from '@/types/tree';

const KINDS = {
  checkpoint: { label: 'Checkpoint', icon: GitBranch },
  adapter: { label: 'Adapter', icon: Layers },
} as const satisfies Readonly<
  Record<ArtifactKind, { readonly label: string; readonly icon: unknown }>
>;

/** The segment F8's naming convention puts in an adapter's name. */
export const ADAPTER_NAME_SEGMENT = 'adapter-';

export interface KindMarkerProps {
  readonly kind: ArtifactKind;
  readonly name: string;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

export function KindMarker({ kind, name, className = '', ...rest }: KindMarkerProps): ReactNode {
  const entry = KINDS[kind];
  const Icon = entry.icon;
  const nameSuggestsAdapter = name.includes(ADAPTER_NAME_SEGMENT);
  const anomaly = nameSuggestsAdapter !== (kind === 'adapter');

  return (
    <span
      className={`inline-flex items-center gap-1.5 text-hint ${className}`.trim()}
      data-kind={kind}
      data-name-anomaly={anomaly ? 'true' : 'false'}
      {...rest}
    >
      {/* The glyph is never accent: ADR-012 permits the accent on a row icon tile, not on a
          row glyph, and the mockup painted this one orange - FTID 6.3 names it as a bug. */}
      <Icon size={14} className="text-dim" aria-hidden="true" />
      <span className="text-dim">{entry.label}</span>
      {anomaly ? (
        <span className="text-amber" title={anomalySentence(kind)}>
          name anomaly
        </span>
      ) : null}
    </span>
  );
}

export function anomalySentence(kind: ArtifactKind): string {
  return kind === 'adapter'
    ? `This artifact's kind is "adapter" and its name carries no "${ADAPTER_NAME_SEGMENT}" segment. The kind is what it is; the name is only what it is called.`
    : `This artifact's kind is "checkpoint" and its name carries an "${ADAPTER_NAME_SEGMENT}" segment. The kind is what it is; the name is only what it is called.`;
}
