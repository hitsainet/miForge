/**
 * A WSP transcript entry as text (fix 2026-11-20).
 *
 * The transcript is the editor's conversation as the worker wrote it, and **every line is
 * untrusted data**: the editor model chose most of its words. So nothing here produces markup -
 * each function returns strings, and the screen renders them as text nodes (React escapes them;
 * no `dangerouslySetInnerHTML` anywhere on the path).
 *
 * The two providers write the same neutral message shapes - `{role, content, tool_calls?,
 * tool_call_id?}` - and `miwsp_engine` adds a header line (`{miwsp_transcript: 1, ...}`), event
 * lines (`{event: 'refused' | 'handoff' | 'thinking' | 'repeated', ...}`) and, on an assistant
 * line that came with reasoning, `reasoning: {text, source}`. The route serves a line that is not
 * one JSON object as `{raw: <text>}`.
 *
 * **Paging (2026-11-21)** adds three event lines, each shown as a readable line rather than its
 * JSON (`eventText`): `ledger` (one experiment-ledger row the engine derived from a tool result),
 * `page_out` (the results replaced by a page stub in the editor's history) and `recall` (the
 * editor asked for a page back). Their fields are still untrusted text, shown as text; an
 * unexpected shape falls back to the JSON, as every other event does.
 *
 * **The engine's assistant turns are JSON text** (its structured protocol answers one action as a
 * JSON object). `readable` pretty-prints such text, and **guards the parse**: content that is not
 * JSON, or is JSON of a scalar, is shown exactly as written - a malformed turn never throws and
 * never takes the screen down.
 */

export interface ToolCallView {
  readonly name: string;
  readonly args: string;
}

export interface EntryView {
  /** `system`, `user`, `assistant`, `tool`, an event's name, `header`, or `raw`. */
  readonly role: string;
  readonly text: string;
  /** True when `text` is pretty-printed JSON the entry carried as text. */
  readonly structured: boolean;
  readonly calls: readonly ToolCallView[];
  /**
   * The editor's reasoning on an engine assistant line (`reasoning: {text, source}`,
   * 2026-11-21), kept apart from the content: it is shown to a reviewer, and the engine never
   * parsed it as an action. `null` when the line carries none.
   */
  readonly reasoning: { readonly text: string; readonly source: string } | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Pretty-print a value that is already parsed. Never throws (a cycle cannot arrive as JSON). */
function pretty(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

/**
 * Text that holds a JSON object or array, pretty-printed; any other text exactly as written.
 * The parse is guarded: malformed JSON is shown raw.
 */
export function readable(text: string): { readonly text: string; readonly structured: boolean } {
  const trimmed = text.trim();
  if (!(trimmed.startsWith('{') || trimmed.startsWith('['))) return { text, structured: false };
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return { text, structured: false };
  }
  if (typeof parsed !== 'object' || parsed === null) return { text, structured: false };
  return { text: pretty(parsed), structured: true };
}

/** A message's content: a string, or content blocks (each block's text, else the block). */
function contentText(content: unknown): { readonly text: string; readonly structured: boolean } {
  if (typeof content === 'string') return readable(content);
  if (content === null || content === undefined) return { text: '', structured: false };
  if (Array.isArray(content)) {
    const parts = content.map((block) =>
      isRecord(block) && typeof block['text'] === 'string'
        ? readable(block['text']).text
        : pretty(block),
    );
    return { text: parts.join('\n\n'), structured: false };
  }
  return { text: pretty(content), structured: true };
}

function callsOf(value: unknown): readonly ToolCallView[] {
  if (!Array.isArray(value)) return [];
  return value.map((call) => {
    if (!isRecord(call)) return { name: '(unnamed)', args: pretty(call) };
    // The neutral shape is {id, name, arguments}; OpenAI's wire shape nests it in `function`.
    const fn = isRecord(call['function']) ? call['function'] : call;
    const name = typeof fn['name'] === 'string' ? fn['name'] : '(unnamed)';
    const args = fn['arguments'];
    return { name, args: typeof args === 'string' ? readable(args).text : pretty(args ?? {}) };
  });
}

/** The line's `reasoning {text, source}` as text, or `null`; any other shape is shown whole. */
function reasoningOf(value: unknown): EntryView['reasoning'] {
  if (value === undefined || value === null) return null;
  if (isRecord(value) && typeof value['text'] === 'string') {
    const source = typeof value['source'] === 'string' ? value['source'] : 'unnamed source';
    return { text: value['text'], source };
  }
  return { text: pretty(value), source: 'unrecognised shape' };
}

const word = (value: unknown): string =>
  typeof value === 'string' || typeof value === 'number' ? String(value) : pretty(value);

/**
 * A paging event as one readable line (2026-11-21), or `null` for any other event or an
 * unexpected shape (then the event is shown as its JSON).
 */
export function eventText(event: string, rest: Readonly<Record<string, unknown>>): string | null {
  if (event === 'ledger' && isRecord(rest['row'])) {
    const row = rest['row'];
    const numbers = isRecord(row['numbers'])
      ? Object.entries(row['numbers'])
          .map(([k, v]) => `${k}=${word(v)}`)
          .join(' ')
      : '';
    const weights =
      row['before'] === row['after']
        ? word(row['after'])
        : `${word(row['before'])} > ${word(row['after'])}`;
    return [
      `turn ${word(row['turn'])}`,
      `${word(row['kind'])}: ${word(row['action'])}`,
      `weights ${weights}`,
      `changed vs start: ${word(row['change'])}`,
      numbers === '' ? null : numbers,
    ]
      .filter((part) => part !== null)
      .join(' | ');
  }
  if (event === 'page_out' && Array.isArray(rest['pages'])) {
    const pages = rest['pages'].map((p) =>
      isRecord(p)
        ? `${word(p['page'])} (${word(p['tool'])}, turn ${word(p['turn'])}, ${word(p['bytes'])} bytes)`
        : pretty(p),
    );
    return `before turn ${word(rest['turn'])} (${word(rest['reason'])}): paged out ${pages.join(', ')}`;
  }
  if (event === 'recall') {
    const found = rest['found'] === true ? 'returned in full' : 'refused: no such page';
    return `turn ${word(rest['turn'])}: recall ${word(rest['page'])}, ${found}`;
  }
  return null;
}

/** One served entry as text. Any shape is accepted; an unknown one is shown whole. */
export function entryView(entry: unknown): EntryView {
  const none = { calls: [], reasoning: null } as const;
  if (!isRecord(entry)) {
    return { role: 'entry', text: pretty(entry), structured: false, ...none };
  }
  if (typeof entry['raw'] === 'string' && Object.keys(entry).length === 1) {
    return { role: 'raw', text: entry['raw'], structured: false, ...none };
  }
  if ('miwsp_transcript' in entry) {
    return { role: 'header', text: pretty(entry), structured: true, ...none };
  }
  if (typeof entry['event'] === 'string') {
    const { event, ...rest } = entry;
    const line = eventText(String(event), rest);
    if (line !== null) {
      return { role: `event: ${String(event)}`, text: line, structured: false, ...none };
    }
    return { role: `event: ${String(event)}`, text: pretty(rest), structured: true, ...none };
  }
  if (typeof entry['role'] === 'string') {
    const content = contentText(entry['content']);
    return {
      role: entry['role'],
      text: content.text,
      structured: content.structured,
      calls: callsOf(entry['tool_calls']),
      reasoning: reasoningOf(entry['reasoning']),
    };
  }
  return { role: 'entry', text: pretty(entry), structured: true, ...none };
}
