/**
 * R-02.4-.7's signal sources, **from the live registry** (FR-009.33's "Signal source" row).
 *
 * Availability is computed per call, so configuring a generation provider makes
 * self-generation available without a restart. An unavailable source is **listed with its
 * reason**, never omitted and never a mute disabled control.
 *
 * `requires` is what makes the form honest about the rest of the definition: a source that
 * `needs_selector` tells the operator a selector configuration is part of this run before
 * they submit, rather than after F6 refuses it.
 */
import { apiGet, type ApiResult } from './client';
import { V1 } from './v1';

export interface SignalSourceRow {
  readonly name: string;
  readonly part_kind: string | null;
  readonly objective: string | null;
  readonly requires: {
    readonly part_kind: string | null;
    readonly selector: boolean;
    readonly grader: boolean;
    readonly generation_provider: boolean;
  };
  readonly available: boolean;
  readonly unavailable_reason: string | null;
  readonly summary: string;
}

export interface SignalSourceListing {
  readonly sources: readonly SignalSourceRow[];
  readonly count: number;
  readonly available: number;
}

export const SIGNAL_SOURCES_PATH = `${V1}/signal-sources`;

export const fetchSignalSources = (): Promise<ApiResult<SignalSourceListing>> =>
  apiGet<SignalSourceListing>(SIGNAL_SOURCES_PATH);
