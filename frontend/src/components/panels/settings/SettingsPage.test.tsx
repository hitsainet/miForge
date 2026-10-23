/**
 * The Settings screen (F12; FR-012.7, .8): the footer's Settings entry opens it.
 *
 * Stubbed at `fetch`, the HTTP boundary, with the backend's own payload shape.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SETTINGS, SETTINGS_HISTORY } from '@/testing/stubBackend';
import { DRAFT_KEYS, DRAFT_PREFIX } from '@/stores/draftStorage';
import { EFFECT_BADGE, SettingsPage, reasonDraftKey, showActor, showSetting } from './SettingsPage';

let calls: { method: string; url: string; body: unknown }[] = [];
let refuse = false;

beforeEach(() => {
  calls = [];
  refuse = false;
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => {
    const method = (init?.method ?? 'GET').toUpperCase();
    const body = init?.body === undefined ? undefined : JSON.parse(String(init.body));
    calls.push({ method, url: String(input), body });
    if (method !== 'GET' && refuse) {
      return Promise.resolve(
        new Response(
          JSON.stringify({
            error: { code: 'setting_invalid', message: 'Not a valid integer.', details: {} },
          }),
          { status: 422, headers: { 'Content-Type': 'application/json' } },
        ),
      );
    }
    const url = String(input);
    const payload =
      method !== 'GET'
        ? { key: 'k', value: 1, source: 'override' }
        : url.startsWith('/api/v1/settings/history')
          ? SETTINGS_HISTORY
          : SETTINGS;
    return Promise.resolve(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  });
});

afterEach(() => vi.unstubAllGlobals());

async function settle(): Promise<void> {
  for (let index = 0; index < 4; index += 1) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
}

const writes = () => calls.filter((call) => call.method !== 'GET');

describe('[SettingsPage]', () => {
  it('should list the runtime settings as editable rows and the rest read-only', async () => {
    render(<SettingsPage />);
    await settle();
    const runtime = screen.getByRole('list', { name: 'Runtime settings' });
    expect(within(runtime).getAllByRole('listitem')).toHaveLength(4);
    expect(screen.getByTestId('settings-runtime-count')).toHaveTextContent(/^4 of 5 settings/);
    const fixed = screen.getByTestId('settings-fixed');
    expect(fixed).toHaveTextContent('database_url');
    // Served redacted, shown as served.
    expect(fixed).toHaveTextContent('postgresql://miforge:<redacted>@db/miforge');
    expect(within(fixed).queryByRole('textbox')).toBeNull();
    // The source is named: the override row says so.
    expect(screen.getByTestId('setting-checkpoint_lifecycle_keep_last_n')).toHaveTextContent(
      'override',
    );
  });

  it('should save a typed number as a number, once, then re-read', async () => {
    render(<SettingsPage />);
    await settle();
    const input = screen.getByTestId('setting-input-checkpoint_lifecycle_keep_last_n');
    fireEvent.change(input, { target: { value: '12' } });
    await userEvent.click(screen.getByTestId('setting-save-checkpoint_lifecycle_keep_last_n'));
    await settle();
    expect(writes()).toEqual([
      {
        method: 'PUT',
        url: '/api/v1/settings/checkpoint_lifecycle_keep_last_n',
        body: { value: 12 },
      },
    ]);
    // The listing and the history, read once on mount and once after the save.
    expect(calls.filter((call) => call.method === 'GET').map((call) => call.url)).toEqual([
      '/api/v1/settings',
      '/api/v1/settings/history?limit=50',
      '/api/v1/settings',
      '/api/v1/settings/history?limit=50',
    ]);
  });

  it('should keep an unsaved value in storage and drop it once the save succeeds', async () => {
    render(<SettingsPage />);
    await settle();
    const input = screen.getByTestId('setting-input-checkpoint_lifecycle_keep_last_n');
    fireEvent.change(input, { target: { value: '12' } });
    await settle();
    const stored = () => window.localStorage.getItem(DRAFT_PREFIX + DRAFT_KEYS.settings);
    expect(stored()).toContain('"checkpoint_lifecycle_keep_last_n":"12"');
    await userEvent.click(screen.getByTestId('setting-save-checkpoint_lifecycle_keep_last_n'));
    await settle();
    // Saved: the row shows what the server now reports (the stub still says 10), not the
    // stale draft, and nothing is left to be restored as if unsaved.
    expect(input).toHaveValue('10');
    expect(stored()).toBeNull();
  });

  it('should save a boolean as true/false and an empty address as null', async () => {
    render(<SettingsPage />);
    await settle();
    fireEvent.change(screen.getByTestId('setting-input-checkpoint_lifecycle_enabled'), {
      target: { value: 'true' },
    });
    await userEvent.click(screen.getByTestId('setting-save-checkpoint_lifecycle_enabled'));
    await userEvent.click(screen.getByTestId('setting-save-millm_base_url'));
    await settle();
    expect(writes().map((call) => call.body)).toEqual([{ value: true }, { value: null }]);
  });

  it('should refuse a non-number before sending anything', async () => {
    render(<SettingsPage />);
    await settle();
    fireEvent.change(screen.getByTestId('setting-input-checkpoint_lifecycle_keep_last_n'), {
      target: { value: 'ten' },
    });
    await userEvent.click(screen.getByTestId('setting-save-checkpoint_lifecycle_keep_last_n'));
    expect(writes()).toEqual([]);
    expect(screen.getByText('Enter a number')).toBeInTheDocument();
  });

  it('should offer a reset only on an override, and send DELETE for it', async () => {
    render(<SettingsPage />);
    await settle();
    expect(screen.queryByTestId('setting-reset-log_level')).toBeNull();
    await userEvent.click(screen.getByTestId('setting-reset-checkpoint_lifecycle_keep_last_n'));
    await settle();
    expect(writes()).toEqual([
      {
        method: 'DELETE',
        url: '/api/v1/settings/checkpoint_lifecycle_keep_last_n',
        body: undefined,
      },
    ]);
  });

  it("should show the backend's refusal on the row", async () => {
    refuse = true;
    render(<SettingsPage />);
    await settle();
    await userEvent.click(screen.getByTestId('setting-save-log_level'));
    await settle();
    expect(screen.getByTestId('setting-log_level')).toHaveTextContent('setting_invalid');
  });

  it('should say what each setting does, when a change takes effect, and its last change', async () => {
    render(<SettingsPage />);
    await settle();
    expect(screen.getByTestId('setting-label-checkpoint_lifecycle_keep_last_n')).toHaveTextContent(
      'Keep the last N saves',
    );
    expect(
      screen.getByTestId('setting-description-checkpoint_lifecycle_keep_last_n'),
    ).toHaveTextContent("How many of each run's most recent saves pruning always keeps.");
    expect(screen.getByTestId('setting-effect-checkpoint_lifecycle_keep_last_n')).toHaveTextContent(
      EFFECT_BADGE.next_sweep.label,
    );
    expect(screen.getByTestId('setting-effect-log_level')).toHaveTextContent('Takes effect now');
    const values = screen.getByTestId('setting-values-checkpoint_lifecycle_keep_last_n');
    expect(values).toHaveTextContent('In force10');
    expect(values).toHaveTextContent('Environment value12');
    expect(
      screen.getByTestId('setting-last-change-checkpoint_lifecycle_keep_last_n'),
    ).toHaveTextContent(
      'Oct 20, 10:05 AM ET by agent:forge-mcp, approved as apr_0123 - "the volume is filling"',
    );
    expect(screen.getByTestId('setting-last-change-log_level')).toHaveTextContent(
      'never changed here',
    );
    // The restart section names the setting plainly and the pod to restart.
    const fixed = screen.getByTestId('setting-fixed-database_url');
    expect(fixed).toHaveTextContent('Database address');
    expect(fixed).toHaveTextContent('Restart miforge-backend');
  });

  it('should send the reason with a save and with a reset, and drop it after', async () => {
    render(<SettingsPage />);
    await settle();
    fireEvent.change(screen.getByTestId('setting-reason-log_level'), {
      target: { value: 'chasing a slow save' },
    });
    await settle();
    const stored = () => window.localStorage.getItem(DRAFT_PREFIX + DRAFT_KEYS.settings);
    expect(stored()).toContain(`"${reasonDraftKey('log_level')}":"chasing a slow save"`);
    await userEvent.click(screen.getByTestId('setting-save-log_level'));
    fireEvent.change(screen.getByTestId('setting-reason-checkpoint_lifecycle_keep_last_n'), {
      target: { value: 'back to normal' },
    });
    await userEvent.click(screen.getByTestId('setting-reset-checkpoint_lifecycle_keep_last_n'));
    await settle();
    expect(writes()).toEqual([
      {
        method: 'PUT',
        url: '/api/v1/settings/log_level',
        body: { value: 'INFO', reason: 'chasing a slow save' },
      },
      {
        method: 'DELETE',
        url: '/api/v1/settings/checkpoint_lifecycle_keep_last_n?reason=back+to+normal',
        body: undefined,
      },
    ]);
    expect(screen.getByTestId('setting-reason-log_level')).toHaveValue('');
  });

  it('should list the recent changes from the history route', async () => {
    render(<SettingsPage />);
    await settle();
    const history = screen.getByTestId('settings-history');
    const change = within(history).getByTestId('settings-change-7');
    // Labelled with the plain name the rows use, never the raw key.
    expect(change).toHaveTextContent('Keep the last N saves');
    expect(change).toHaveTextContent('12 → 10');
    expect(change).toHaveTextContent('By agent:forge-mcp, approved as apr_0123');
    expect(change).toHaveTextContent('"the volume is filling"');
  });

  it('should name the writer plainly', () => {
    expect(showActor('operator')).toBe('you (the operator)');
    expect(showActor('agent:forge-mcp', 'apr_1')).toBe('agent:forge-mcp, approved as apr_1');
    expect(showActor(null)).toBe('unknown');
  });

  it('should show an unset value as "not set", never blank', () => {
    expect(showSetting(null)).toBe('not set');
    expect(showSetting([])).toBe('none');
    expect(showSetting(['a', 'b'])).toBe('a, b');
  });
});
