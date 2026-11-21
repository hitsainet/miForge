/**
 * A transcript entry as text (fix 2026-11-20): the engine's JSON turns are laid out readably,
 * and the parse is guarded, so malformed or hostile content is shown as written and never throws.
 */
import { describe, expect, it } from 'vitest';
import { entryView, readable } from './transcript';

describe('[transcript] readable', () => {
  it('should pretty-print a JSON object the engine wrote as text', () => {
    const out = readable('{"action":"python","code":"print(1)"}');
    expect(out.structured).toBe(true);
    expect(out.text).toBe('{\n  "action": "python",\n  "code": "print(1)"\n}');
  });

  it('should show malformed JSON exactly as written, without throwing', () => {
    for (const text of ['{"action": "python", ', '{not json}', '[1, 2', '{"a": NaN}']) {
      expect(() => readable(text)).not.toThrow();
      expect(readable(text)).toEqual({ text, structured: false });
    }
  });

  it('should leave prose and JSON scalars as written', () => {
    expect(readable('I will edit the bias.')).toEqual({
      text: 'I will edit the bias.',
      structured: false,
    });
    expect(readable('42')).toEqual({ text: '42', structured: false });
    expect(readable('"quoted"')).toEqual({ text: '"quoted"', structured: false });
  });
});

describe('[transcript] entryView', () => {
  it('should read a message, its JSON content and its tool calls (both shapes)', () => {
    const view = entryView({
      role: 'assistant',
      content: '{"action":"submit"}',
      tool_calls: [
        { id: '1', name: 'python', arguments: '{"code":"W[0] += 1"}' },
        { id: '2', type: 'function', function: { name: 'submit', arguments: '{bad' } },
      ],
    });
    expect(view.role).toBe('assistant');
    expect(view.structured).toBe(true);
    expect(view.text).toContain('"action": "submit"');
    expect(view.calls).toEqual([
      { name: 'python', args: '{\n  "code": "W[0] += 1"\n}' },
      { name: 'submit', args: '{bad' },
    ]);
  });

  it('should read the engine header, event lines, raw lines and anything else', () => {
    expect(entryView({ miwsp_transcript: 1, provider: 'miwsp_engine' }).role).toBe('header');
    const event = entryView({ event: 'refused', kind: 'two_actions', n: 1 });
    expect(event.role).toBe('event: refused');
    expect(event.text).toContain('"kind": "two_actions"');
    expect(entryView({ raw: '{"cut' })).toEqual({
      role: 'raw',
      text: '{"cut',
      structured: false,
      calls: [],
      reasoning: null,
    });
    expect(entryView('a string').text).toBe('"a string"');
    expect(entryView(null).text).toBe('null');
    expect(entryView({ odd: true }).role).toBe('entry');
  });

  it('should join content blocks by their text', () => {
    const view = entryView({
      role: 'assistant',
      content: [
        { type: 'text', text: 'first' },
        { type: 'tool_use', id: 't', name: 'python', input: {} },
      ],
    });
    expect(view.text.startsWith('first\n\n')).toBe(true);
    expect(view.text).toContain('"tool_use"');
  });
});

describe('[transcript] reasoning (2026-11-21)', () => {
  it('should keep the reasoning apart from the content', () => {
    const view = entryView({
      role: 'assistant',
      content: 'answer',
      reasoning: { text: 'because', source: 'inline_think' },
    });
    expect(view.text).toBe('answer');
    expect(view.reasoning).toEqual({ text: 'because', source: 'inline_think' });
  });

  it('should show a reasoning of an unknown shape whole, and none as null', () => {
    expect(entryView({ role: 'assistant', content: 'x', reasoning: 42 }).reasoning).toEqual({
      text: '42',
      source: 'unrecognised shape',
    });
    expect(entryView({ role: 'assistant', content: 'x' }).reasoning).toBeNull();
    expect(entryView({ event: 'thinking', sent: {} }).reasoning).toBeNull();
  });
});

describe('[transcript] paging events (2026-11-21)', () => {
  it('should show a ledger row as one readable line', () => {
    const view = entryView({
      event: 'ledger',
      row: {
        turn: 6,
        kind: 'edit',
        action: 'write_tensor name=bias values=-4.535',
        before: 'aaaaaaaa',
        after: 'bbbbbbbb',
        change: 'bias 1/1 |d|=0.3002',
        numbers: {},
      },
    });
    expect(view.role).toBe('event: ledger');
    expect(view.structured).toBe(false);
    expect(view.text).toBe(
      'turn 6 | edit: write_tensor name=bias values=-4.535 | weights aaaaaaaa > bbbbbbbb | changed vs start: bias 1/1 |d|=0.3002',
    );
    const measured = entryView({
      event: 'ledger',
      row: {
        turn: 5,
        kind: 'measurement',
        action: 'forward_practice_metrics rule=max',
        before: 'aaaaaaaa',
        after: 'aaaaaaaa',
        change: '-',
        numbers: { auroc: '0.9903', far: '0.001403' },
      },
    });
    expect(measured.text).toContain('weights aaaaaaaa |');
    expect(measured.text.endsWith('auroc=0.9903 far=0.001403')).toBe(true);
  });

  it('should list the pages a page_out event paged, and a recall', () => {
    const view = entryView({
      event: 'page_out',
      turn: 5,
      reason: 'every',
      pages: [{ page: 'p1', turn: 1, tool: 'read_tensor', bytes: 12052 }],
    });
    expect(view.text).toBe(
      'before turn 5 (every): paged out p1 (read_tensor, turn 1, 12052 bytes)',
    );
    expect(entryView({ event: 'recall', turn: 7, page: 'p1', found: true }).text).toBe(
      'turn 7: recall p1, returned in full',
    );
    expect(entryView({ event: 'recall', turn: 8, page: 'p9', found: false }).text).toBe(
      'turn 8: recall p9, refused: no such page',
    );
  });

  it('should fall back to the JSON for an unexpected shape, and never produce markup', () => {
    const odd = entryView({ event: 'page_out', pages: 'not a list' });
    expect(odd.structured).toBe(true);
    expect(odd.text).toContain('"pages": "not a list"');
    const hostile = entryView({
      event: 'ledger',
      row: { turn: 1, kind: 'edit', action: '<img src=x onerror=alert(1)>', numbers: {} },
    });
    expect(hostile.text).toContain('<img src=x onerror=alert(1)>'); // text, rendered as a text node
  });
});
