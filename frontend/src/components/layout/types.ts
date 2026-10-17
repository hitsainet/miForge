/**
 * The shell's props (foundation task 7.3).
 *
 * **Why the nav is a prop and not a module-level table.** F9's FR-009.1 makes **one**
 * route table the live registry: *"The sidebar's nav, the router and every
 * screen-reachability test read that table and nothing else. A screen not in the table is
 * not reachable and therefore does not exist."* That table is F9's
 * `frontend/src/config/routes.ts`.
 *
 * So the shell must not own a second one. If `Sidebar` imported its own list of screens,
 * there would be two places a screen can be declared, and ADR-016 #1's reachability test
 * - the rule this whole project's review discipline is built on - would be asserting over
 * the wrong one. The shell therefore takes the nav it is given and renders it. F9's
 * FTID 1.3 records the same direction from its side: `components/layout/` is
 * **consume**, *"foundation 7.3's shell. Compose; do not re-create."*
 */

import type { LucideIcon } from 'lucide-react';

/** One nav entry, as F9's route table will project it. */
export interface ShellNavItem {
  /** Stable id. Compared against `activeId`; never the label, which is copy. */
  readonly id: string;
  /** Sentence case, no all-caps (ADR-012). Names the screen, not the action. */
  readonly label: string;
  readonly icon: LucideIcon;
  /** For a real anchor when a router exists. Absent means a button. */
  readonly href?: string;
}

/**
 * One reading in the resource strip.
 *
 * **Every number names its scale and sample** (ADR-012). The shape is what enforces it:
 * `unit` is required, so a bare `41` cannot be rendered, and `of` carries the total for a
 * reading that is a fraction of a budget - `17.9 / 125.6 GB`, not `17.9 GB`, because the
 * second says nothing about how much is left.
 *
 * `value: null` is **not reported** - rendered as such, never as `0`. A GPU reading of 0 %
 * and a GPU that could not be read are different facts, and this node's driver reaches the
 * "Driver/library version mismatch" state often enough that the distinction is not
 * hypothetical (AGENT.md, Cautions).
 */
export interface ResourceReading {
  readonly id: string;
  readonly label: string;
  readonly icon: LucideIcon;
  readonly value: number | null;
  readonly unit: string;
  readonly of?: number;
  /** Decimal places. Defaults to 1, or 0 when the unit is a percentage. */
  readonly precision?: number;
}
