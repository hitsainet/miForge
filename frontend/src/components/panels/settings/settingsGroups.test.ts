/**
 * Grouping and ordering the settings list (2026-10-28). The page test drives the served
 * payload; these pin the edges a well-formed payload never reaches.
 */
import { describe, expect, it } from 'vitest';
import type { SettingRow } from '@/api/settings';
import { OTHER_GROUP, groupSettings } from './settingsGroups';

const row = (label: string | undefined, group?: string): SettingRow => ({
  value: null,
  source: 'env',
  runtime_writable: false,
  ...(label === undefined ? {} : { label }),
  ...(group === undefined ? {} : { group }),
});

const GROUPS = [
  { id: 'general', label: 'General' },
  { id: 'storage', label: 'Storage' },
  { id: 'jobs', label: 'Jobs' },
] as const;

const shape = (grouped: ReturnType<typeof groupSettings>) =>
  grouped.map((group) => [group.id, group.rows.map(([name]) => name)]);

describe('[settingsGroups]', () => {
  it('should follow the served group order, not the order rows arrive in', () => {
    const grouped = groupSettings(
      [
        ['b', row('Bee', 'storage')],
        ['a', row('Ay', 'general')],
      ],
      GROUPS,
    );
    expect(shape(grouped)).toEqual([
      ['general', ['a']],
      ['storage', ['b']],
    ]);
  });

  it('should sort by label within a group, ignoring case, with the name breaking a tie', () => {
    const grouped = groupSettings(
      [
        ['z_key', row('alpha', 'general')],
        ['m_key', row('Zulu', 'general')],
        ['a_key', row('Mike', 'general')],
        ['tie_b', row('Same', 'general')],
        ['tie_a', row('same', 'general')],
        ['unlabelled', row(undefined, 'general')],
      ],
      GROUPS,
    );
    expect(shape(grouped)).toEqual([
      ['general', ['z_key', 'a_key', 'tie_a', 'tie_b', 'unlabelled', 'm_key']],
    ]);
  });

  it('should put an unknown or missing group last, under Other, and drop empty groups', () => {
    const grouped = groupSettings(
      [
        ['x', row('X', 'retired_group')],
        ['y', row('Y')],
        ['s', row('S', 'storage')],
      ],
      GROUPS,
    );
    expect(shape(grouped)).toEqual([
      ['storage', ['s']],
      ['other', ['x', 'y']],
    ]);
    expect(grouped.at(-1)?.label).toBe(OTHER_GROUP.label);
  });

  it('should show everything under Other when an older backend serves no groups', () => {
    const grouped = groupSettings([
      ['b', row('B', 'storage')],
      ['a', row('A', 'general')],
    ]);
    expect(shape(grouped)).toEqual([['other', ['a', 'b']]]);
  });
});
