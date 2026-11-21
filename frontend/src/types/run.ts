/**
 * The run record and its series, as the **live** API carries them (F9 FR-009.4, .14, .15).
 *
 * Every field below was read out of `backend/src/api/v1/endpoints/` or
 * `backend/src/workers/heartbeat.py`, not out of a feature document. Four of this
 * project's recorded defects are a document asserting a field its cited source does not
 * have, so these types are a transcription of the producer rather than of the spec:
 *
 * - `heartbeat` is `workers.heartbeat.liveness_of`'s return value, whose docstring states
 *   that **no key is ever absent** - "a missing key is indistinguishable from a healthy
 *   one, which is the failure this block exists to prevent." The optionality below is on
 *   the *values*, exactly as that function reports them.
 * - `ScoreSeriesRow` is `api/v1/endpoints/scoring.py::_score_payload`.
 * - `StepRecord` is `api/v1/endpoints/runs_acquisition.py::_step_payload`.
 * - `BatchRecord` is `api/v1/endpoints/signal_batches.py::_row_payload`.
 *
 * **Nullability is the producer's.** `age_seconds` is `None` only when the column is NULL
 * and is reported rather than rendered as zero - "a run that has never beaten is not a
 * fresh run" - so these are `number | null`, never `number` with a default.
 */
import type { PartRef } from './packs';
import type { AttachmentBlock } from '@/components/panels/hotupdate/weightsState';

/* ------------------------------------------------------------------------------------ *
 * The run record
 * ------------------------------------------------------------------------------------ */

/**
 * The liveness block, **served** rather than computed in the browser.
 *
 * `lapsed` and `lapse_threshold_seconds` both come from the server deliberately: the two
 * alternatives were written and withdrawn here (trust the socket - shows a stalled run as
 * live; derive a threshold from the 60 s throttle - shows a **healthy** run as stalled
 * through every test pass and every save, because 60 s is a *minimum interval between
 * writes*, not a maximum gap). So the client reads the janitor's own answer and never
 * chooses a second threshold. See `stores/runStore.ts`.
 */
export interface HeartbeatBlock {
  readonly heartbeat_at: string | null;
  readonly age_seconds: number | null;
  readonly longest_gap_seconds: number | null;
  /** The janitor's threshold, in seconds. Provisional upstream; never re-derived here. */
  readonly lapse_threshold_seconds: number;
  /** The server's verdict. A terminal run is never `lapsed` - it is finished. */
  readonly lapsed: boolean;
}

/** `GET /api/v1/runs/{run_id}` - `runs.py::run_payload`. */
export interface RunRecord {
  readonly run_id: string;
  readonly status: string;
  readonly run_type: string | null;
  readonly last_step: number | null;
  readonly cancel_requested_at: string | null;
  readonly failure_code: string | null;
  readonly heartbeat: HeartbeatBlock;
  /**
   * F6's extension of this route (its FR-006.10). **Optional on the client because the
   * route does not serve it yet** - `run_payload` carries seven keys and this is not among
   * them. Declared so the screen can render it the moment F6 lands it, and read through
   * `AbsentValue` until then rather than rendered as a bare "done".
   */
  readonly stop_reason?: string | null;
  /** F6's. Not served today; see `stop_reason`. */
  readonly total_steps?: number | null;
  /**
   * §5.10's block (F11 16.4). `null` for every acquisition mode but `attach`; optional so a
   * backend that predates it reads as "nothing attached" rather than as a broken record.
   */
  readonly attachment?: AttachmentBlock | null;
  /**
   * How the run acquired its model, off its recorded definition (`load` | `create` |
   * `attach`), or `null` for a row with none. The run view offers "Continue" for `attach`
   * (D52). Optional so an older backend reads as "unknown", never as a mode.
   */
  readonly acquisition_mode?: string | null;
  /**
   * When the run was archived (hidden from the default lists), or `null`. Optional so an
   * older backend reads as "not archived". The run view offers Unarchive when set.
   */
  readonly archived_at?: string | null;
}

/* ------------------------------------------------------------------------------------ *
 * Series
 * ------------------------------------------------------------------------------------ */

/** One row of `GET /api/v1/runs/{id}/scores` - `scoring.py::_score_payload`. */
export interface ScoreSeriesRow {
  readonly step: number;
  readonly kind: string;
  readonly set: string;
  readonly score: number;
  readonly n: number;
  readonly floor: number | null;
  readonly floor_layer: string | null;
  readonly breached: boolean;
  readonly provider: string | null;
  readonly provider_kind: string | null;
  readonly provider_model: string | null;
  readonly aggregation: string | null;
  readonly out_of_cadence: boolean;
  readonly provider_discontinuity: boolean;
  readonly message: string;
}

/** `GET /api/v1/runs/{id}/scores`. */
export interface ScoreSeries {
  readonly run_id: string;
  readonly scores: readonly ScoreSeriesRow[];
  readonly test_every: number;
  readonly retention_every: number;
  readonly exposure_window_steps: number;
  readonly provider_discontinuities: readonly number[];
}

/** One row of `GET /api/v1/runs/{id}/steps` - `runs_acquisition.py::_step_payload`. */
export interface StepRecord {
  readonly step: number;
  readonly loss: number | null;
  readonly grad_norm: number | null;
  readonly learning_rate: number | null;
  readonly wall_time_ms: number | null;
  readonly batch_source_hashes: readonly string[] | null;
  /**
   * §5.10 only (R-02.14). `null` on every standalone step - the assertion that no bracket
   * was held. Optional so an older backend's payload still type-checks as "not attached".
   */
  readonly quiesce_hold_ms?: number | null;
  readonly commit_duration_ms?: number | null;
}

/** `GET /api/v1/runs/{id}/steps`. Keyset-paginated, newest first. */
export interface StepPage {
  readonly run_id: string;
  readonly steps: readonly StepRecord[];
  readonly count: number;
  readonly limit: number;
  readonly next_before_step: number | null;
}

/**
 * One row of `GET /api/v1/runs/{id}/batches` - `signal_batches.py::_row_payload`.
 *
 * `verdicts` is **three-valued** and all three counts always travel, so a pending
 * candidate at a human-review queue can never be rendered as a rejection.
 */
export interface BatchRecord {
  readonly batch_sha256: string;
  readonly step: number;
  readonly source: string;
  readonly resolved_model: string | null;
  readonly parts: readonly PartRef[] | null;
  readonly n_prompts: number;
  readonly verdicts: {
    readonly accept: number;
    readonly reject: number;
    readonly undecided: number;
  };
  readonly selection_health: {
    readonly candidates_generated: number | null;
    readonly candidates_admitted: number | null;
    readonly acceptance_rate: number | null;
    readonly exhausted: boolean;
  };
  readonly selector_config_sha256: string | null;
  readonly grader_config_sha256: string | null;
  readonly created_at: string;
}

/** `GET /api/v1/runs/{id}/batches`. */
export interface BatchPage {
  readonly run_id: string;
  readonly batches: readonly BatchRecord[];
  readonly count: number;
  readonly limit: number;
  readonly next_before: string | null;
}

/** One row of `GET /api/v1/runs/{id}/checkpoints` - `services.reindex._checkpoint_payload`. */
export interface CheckpointRow {
  readonly name: string;
  readonly run_id: string;
  readonly step: number;
  readonly domain: string | null;
  readonly domain_state: string;
  readonly domain_display: string;
  readonly signal_source: string | null;
  readonly parent: string | null;
  readonly base_model: string | null;
  readonly merged_adapter: Readonly<Record<string, unknown>> | null;
  readonly weights_sha256: string | null;
  readonly manifest_sha256: string | null;
  readonly selector_hash: string | null;
  readonly grader_hash: string | null;
  readonly test: Readonly<Record<string, unknown>> | null;
  readonly retention: Readonly<Record<string, unknown>> | null;
  readonly path: string;
  readonly created_at: string;
}

/** `GET /api/v1/runs/{id}/checkpoints`. `latest_step` is the symlink's, from disk. */
export interface RunCheckpoints {
  readonly run_id: string;
  readonly checkpoints: readonly CheckpointRow[];
  readonly latest_step: number | null;
  readonly steps_on_disk: readonly number[];
}
