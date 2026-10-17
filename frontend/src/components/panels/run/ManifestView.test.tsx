/**
 * The manifest view (FR-009.17, FR-009.18), and **M13 and M14**.
 *
 * - **M13** — dropping unrecognised keys from the view. Both schemas set
 *   `additionalProperties: false`, so such a key makes the document nonconforming, and the
 *   manifest is authoritative (ADR-004): dropping it is how a nonconforming manifest comes to
 *   look clean.
 * - **M14** — guessing the schema when `kind` is unrecognised. A manifest read against the
 *   wrong schema marks real keys unrecognised and reports real omissions as fine.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ManifestView } from './ManifestView';

const checkpoint = {
  kind: 'miforge.checkpoint-manifest',
  schema_version: '1',
  name: 'LFM2-1.2B:run-a:000400',
  run_id: 'run-a',
  step: 400,
  domain: null,
  signal_source: 'corpus',
  parent: 'LFM2-1.2B:run-a:000300',
  optimizer: { name: 'adamw_8bit', state: 'present', config_hash: 'c'.repeat(64) },
  weights_sha256: 'w'.repeat(64),
  base_model: 'LFM2-1.2B',
  created_at: '2026-10-14T00:00:00Z',
  test: { set: 'held-out@3', score: 0.84, n: 320, provider: 'in-process' },
};

describe('[ManifestView]', () => {
  it('should render every key on disk, including one the schema does not describe', () => {
    // M13.
    render(<ManifestView document={{ ...checkpoint, hand_edited: 'yes' }} />);
    const row = screen
      .getByTestId('manifest-view')
      .querySelector('[data-manifest-key="hand_edited"]');
    expect(row).not.toBeNull();
    expect(row?.getAttribute('data-manifest-status')).toBe('unrecognised');
    expect(row?.textContent).toContain('may not conform');
  });

  it('should render a present-and-null value as the word null, which is an assertion', () => {
    render(<ManifestView document={checkpoint} />);
    const row = screen.getByTestId('manifest-view').querySelector('[data-manifest-key="domain"]');
    expect(row?.getAttribute('data-manifest-status')).toBe('schema');
    expect(row?.textContent).toContain('null');
  });

  it('should name an absent required key as missing, distinctly from present-and-null', () => {
    const { weights_sha256, ...without } = checkpoint;
    void weights_sha256;
    render(<ManifestView document={without} />);
    const row = screen
      .getByTestId('manifest-view')
      .querySelector('[data-manifest-key="weights_sha256"]');
    expect(row?.getAttribute('data-manifest-status')).toBe('missing_required');
    expect(row?.textContent).toContain('missing required key');

    const nulled = screen
      .getByTestId('manifest-view')
      .querySelector('[data-manifest-key="domain"]');
    expect(nulled?.getAttribute('data-manifest-status')).not.toBe(
      row?.getAttribute('data-manifest-status'),
    );
  });

  it('should refuse to pick a schema for an unrecognised kind and say which it could not', () => {
    // M14.
    render(<ManifestView document={{ ...checkpoint, kind: 'miforge.something-new' }} />);
    const notice = screen.getByTestId('manifest-schema-unselected');
    expect(notice).toHaveTextContent('miforge.something-new');
    expect(notice).toHaveTextContent('unschematised');
    // And nothing is judged against a guessed schema.
    expect(
      screen.getByTestId('manifest-view').querySelectorAll('[data-manifest-status="unrecognised"]'),
    ).toHaveLength(0);
    expect(
      screen
        .getByTestId('manifest-view')
        .querySelectorAll('[data-manifest-status="missing_required"]'),
    ).toHaveLength(0);
  });

  it('should treat an absent kind as its own case and say the omission is conforming', () => {
    // `kind` is NOT in checkpoint-manifest-v1.json's required list (parsed), so a conforming
    // checkpoint manifest may omit it - a third case FR-009.17 rule 4 does not anticipate.
    const { kind, ...without } = checkpoint;
    void kind;
    render(<ManifestView document={without} />);
    expect(screen.getByTestId('manifest-schema-unselected')).toHaveTextContent('conforming');
  });

  it('should name the schema file it read the document against', () => {
    render(<ManifestView document={checkpoint} />);
    expect(screen.getByTestId('manifest-schema')).toHaveTextContent(
      'docs/schemas/checkpoint-manifest-v1.json',
    );
  });

  it('should render an adapter manifest against the adapter schema', () => {
    render(
      <ManifestView
        document={{
          kind: 'miforge.adapter-manifest',
          schema_version: '1',
          name: 'LFM2:run-b:adapter-200',
          run_id: 'run-b',
          step: 200,
          base: { name: 'LFM2:run-a:000400', manifest_sha256: 'm'.repeat(64) },
          adapter: { rank: 8, alpha: 16, target_modules: ['q_proj'], scaling: 'alpha_over_rank' },
          weights_sha256: 'w'.repeat(64),
          created_at: '2026-10-14T00:00:00Z',
        }}
      />,
    );
    expect(screen.getByTestId('manifest-schema')).toHaveTextContent('adapter-manifest-v1.json');
    // `domain` is not a key of the adapter schema and is not reported missing for it.
    expect(
      screen.getByTestId('manifest-view').querySelector('[data-manifest-key="domain"]'),
    ).toBeNull();
  });

  it('should render the score provider as an ordinary manifest key, never as joined', () => {
    // AC-009.6, rewritten after `75ccb05` put `provider` on `ScoreRecord`. Labelling it
    // "joined" would now be false - the mistake the rule exists to prevent.
    render(<ManifestView document={checkpoint} />);
    const nested = screen
      .getByTestId('manifest-view')
      .querySelector('[data-manifest-key="provider"]');
    expect(nested?.getAttribute('data-manifest-status')).toBe('schema');
    expect(screen.getByTestId('manifest-not-in').textContent ?? '').not.toContain('provider');
  });

  it('should list what is genuinely NOT in the manifest, with its real source', () => {
    render(<ManifestView document={checkpoint} />);
    const list = screen.getByTestId('manifest-not-in');
    expect(list).toHaveTextContent('pack');
    expect(list).toHaveTextContent("adapter's domain");
  });

  it('should say the manifest wins when the index row disagrees', () => {
    render(<ManifestView document={checkpoint} indexStale />);
    expect(screen.getByTestId('manifest-view')).toHaveTextContent('which wins');
    expect(screen.getByTestId('manifest-view')).toHaveTextContent('reindex');
  });

  it('should ask for a selection rather than rendering an empty table', () => {
    render(<ManifestView document={null} />);
    expect(screen.getByTestId('manifest-view-empty')).toHaveTextContent('Select a save point');
  });
});
