/**
 * §5.10's serving-process state, as F9 renders it (F11 task 16.4; F11 FPRD §4; ADR-012).
 *
 * **`weights_state` is ANOTHER APPLICATION'S state, so every value renders CYAN - never the
 * miForge accent.** ADR-012 reserves cyan for miLLM state and allows the accent on exactly
 * one status pill, "Running". A `torn` set is alarming, and the temptation is to colour it
 * like an alarm in miForge's own palette; that would make a miLLM fact read as a miForge
 * one. The label carries the severity; the colour carries the owner.
 *
 * The three values are BRD-01's (R-01.20/R-01.22): `clean`, `writing`, `torn`. An unknown
 * value falls through `StatePill`'s default branch (neutral, named unrecognised) - never
 * the accent, and never a white screen.
 */
import type { StateTone } from '@/components/common/Pill';

export const WEIGHTS_STATE_TONES: Readonly<Record<string, StateTone>> = {
  clean: { tone: 'cyan', label: 'Weights clean' },
  writing: { tone: 'cyan', label: 'Weights writing' },
  torn: { tone: 'cyan', label: 'Weights torn — serving refused' },
};

/**
 * The stale-attachment copy (F11 FR-011.30, FR-011.41; R-02.34; BRD-01 R-01.29).
 *
 * It names the operator action - by name, never by endpoint path, because the frontend
 * names no sibling endpoint (ADR-022; `test_standalone_without_siblings.py`) - and says
 * miForge cannot perform it: `clear-stale` is
 * **operator-only** on miLLM's side, and miForge's `trainer` identity is refused it by the
 * protocol as well as by F11's own code (which contains no such call - a test asserts it).
 */
export const STALE_ATTACHMENT_COPY =
  'The attachment is stale: miLLM has not heard a heartbeat inside its window. An operator ' +
  "must clear it with miLLM's operator-only `clear-stale` action, using an operator token. " +
  'miForge cannot clear it and will not try - its trainer token is refused that call. ' +
  'After the clear, re-attach; check weights_state before resuming.';

/**
 * The run's §5.10 block on `GET /api/v1/runs/{id}` (`runs.py::run_payload`). `null` for every
 * acquisition mode but `attach` - that `null` is the assertion that nothing is attached.
 *
 * The live fields are miLLM's status **as the trainer last observed it** - written into
 * the run record by the trainer (migration 0020; operator decision 2026-10-16) and served
 * read-only. `null` renders "not reported", never a guessed `clean`; `observed_at` says how
 * old the observation is.
 */
export interface AttachmentBlock {
  readonly serving_base_url: string;
  readonly resident_model: string | null;
  readonly weights_state: string | null;
  /** Who holds the attachment when it is not this run. `null` = this run, or not reported. */
  readonly holder: string | null;
  readonly stale: boolean | null;
  /** `false` once the run has detached; `null` before the first observation. */
  readonly attached?: boolean | null;
  /** ISO time of the trainer's observation, or `null`. */
  readonly observed_at?: string | null;
}
