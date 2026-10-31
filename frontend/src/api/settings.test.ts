/**
 * The settings client calls the routes the backend declares, with the operator's identity.
 *
 * No `X-Miforge-Agent`: a write from the Settings screen is the operator's and must not be held
 * for approval as an agent's would be (FR-012.8).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readSettings, readSettingsHistory, resetSetting, writeSetting } from './settings';

let calls: { url: string; init: RequestInit | undefined }[] = [];

beforeEach(() => {
  calls = [];
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return Promise.resolve(
      new Response(JSON.stringify({ key: 'log_level', value: 'DEBUG', source: 'override' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  });
});

afterEach(() => vi.unstubAllGlobals());

const headerNames = (init: RequestInit | undefined): string[] =>
  Object.keys((init?.headers ?? {}) as Record<string, string>).map((h) => h.toLowerCase());

describe('[settings client]', () => {
  it('should GET the settings route', async () => {
    await readSettings();
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe('/api/v1/settings');
    expect(calls[0]?.init?.method).toBe('GET');
  });

  it('should PUT {value} to the key, once, with no agent header', async () => {
    await writeSetting('checkpoint_lifecycle_keep_last_n', 12);
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe('/api/v1/settings/checkpoint_lifecycle_keep_last_n');
    expect(calls[0]?.init?.method).toBe('PUT');
    expect(JSON.parse(String(calls[0]?.init?.body))).toEqual({ value: 12 });
    expect(headerNames(calls[0]?.init)).not.toContain('x-miforge-agent');
  });

  it('should DELETE the key to clear an override', async () => {
    await resetSetting('log_level');
    expect(calls[0]?.url).toBe('/api/v1/settings/log_level');
    expect(calls[0]?.init?.method).toBe('DELETE');
  });

  it('should send a reason only when one is given, trimmed', async () => {
    await writeSetting('log_level', 'DEBUG', '  why not  ');
    await writeSetting('log_level', 'DEBUG', '   ');
    await resetSetting('log_level', 'done & dusted');
    expect(JSON.parse(String(calls[0]?.init?.body))).toEqual({ value: 'DEBUG', reason: 'why not' });
    expect(JSON.parse(String(calls[1]?.init?.body))).toEqual({ value: 'DEBUG' });
    expect(calls[2]?.url).toBe('/api/v1/settings/log_level?reason=done+%26+dusted');
  });

  it('should GET the history with the screen limit', async () => {
    await readSettingsHistory();
    expect(calls[0]?.url).toBe('/api/v1/settings/history?limit=50');
    expect(calls[0]?.init?.method).toBe('GET');
  });
});
