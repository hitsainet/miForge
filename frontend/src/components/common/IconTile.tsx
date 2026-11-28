/**
 * Icon tiles (foundation tasks 7.2, 7.3; ADR-012).
 *
 * > Icon tiles 36 px square, 8 px radius, `accentTint` background.
 *
 * Two of ADR-012's seven permitted accent surfaces are tiles - *the logo tile* (solid
 * accent, white glyph) and *the icon tile on list rows* (tint background, accent glyph) -
 * so both live here and nowhere else. Any other tinted square on any screen is a
 * `neutral` tile, and the accent test will say so.
 *
 * **Size.** `SHELL.iconTilePx` is 36, from the handoff's shape rule. The mockup draws its
 * *logo* tile at 32 (`w-8 h-8`) while drawing every list-row tile at 36 (`w-9 h-9`); the
 * handoff's rule says 36 for "icon tiles" without excepting the logo, so both are 36
 * here. That 4 px is judged a mockup inconsistency rather than a deliberate difference -
 * the handoff's 2 is the normative list and nothing in it distinguishes the two.
 *
 * **The glyph is always `aria-hidden`.** A tile is decoration beside a label that already
 * carries the meaning; announcing "flame" before a nav label is noise, and a tile whose
 * *only* content is an icon with no adjacent text is a missing label, not a missing
 * `aria-label` - fix it at the call site.
 */

import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import { SHELL } from '@/config/tokens';

export type IconTileSurface = 'logo-tile' | 'row-icon-tile' | 'neutral';

export interface IconTileProps {
  readonly icon: LucideIcon;
  /**
   * Which of ADR-012's surfaces this tile is. Required, with no default: a tile that
   * defaulted to an accent surface would be a tile that *became* one by being written,
   * which is the opposite of an allowlist.
   */
  readonly surface: IconTileSurface;
  readonly sizePx?: number;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

export function IconTile({
  icon: Icon,
  surface,
  sizePx = SHELL.iconTilePx,
  className = '',
  ...rest
}: IconTileProps): ReactNode {
  // accent-ok:logo-tile - solid accent with a white glyph, per ADR-012 and the mockup's
  // sidebar. accent-ok:row-icon-tile - `accentTint` background with an accent glyph.
  const surfaceClasses: Record<IconTileSurface, string> = {
    'logo-tile': 'bg-accent text-white',
    'row-icon-tile': 'bg-accent-tint text-accent',
    neutral: 'bg-panel text-dim',
  };

  return (
    <span
      className={`inline-flex shrink-0 items-center justify-center rounded-control ${surfaceClasses[surface]} ${className}`.trim()}
      style={{ width: sizePx, height: sizePx }}
      data-surface={surface}
      {...rest}
    >
      <Icon size={Math.round(sizePx / 2)} aria-hidden="true" />
    </span>
  );
}
