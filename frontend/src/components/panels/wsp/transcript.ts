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
 * **Hosted models (feature 020).** Three more event lines read as one line each: `refusal` ("turn 4:
 * the editor refused (category: cyber)", or "no category given"), `timing` (each request's gap
 * from the previous one in its conversation, and its duration) and `handoff` (its number, the
 * conversation it begins in an append-only session, then the restarted message's text).
 * `sessionFacts` reads the header's `history_mode`, the handoffs and the refusal for the run page.
 *
 * **The answer is the editor's own account (plan A2, 2026-11-22).** An assistant turn that submits
 * is marked `answer`, so the screen can say it is not checked, and the worker's
 * `submitted_measured` event is one readable line the screen also puts beside it.
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
  /** True for an assistant turn that submits: its figures are the editor's, not a measurement. */
  readonly answer: boolean;
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

const signed = (value: number): string => `${value < 0 ? '' : '+'}${value.toFixed(6)}`;

/** The worker's measurement of the submitted weights as one line, or `null` (plan A2). */
function measuredText(event: Readonly<Record<string, unknown>>): string | null {
  if (event['measured'] === false) {
    const why =
      event['reason'] === 'no_time'
        ? 'no time was left before the deadline'
        : `the measurement failed (${word(event['error'] ?? event['reason'] ?? 'no cause recorded')})`;
    return `not measured by the worker: ${why}`;
  }
  const { now, parent } = event;
  if (!isRecord(now) || !isRecord(parent)) return null;
  const figure = (name: string, digits: number): string | null => {
    const a = now[name];
    const b = parent[name];
    return typeof a === 'number' && typeof b === 'number'
      ? `${a.toFixed(digits)} (parent ${b.toFixed(digits)})`
      : null;
  };
  const auroc = figure('auroc', 6);
  const recall = figure('recall', 4);
  const alarms = figure('false_alarms', 0);
  if (auroc === null || recall === null || alarms === null) return null;
  let difference = '';
  const against = event['auroc_against_parent'];
  if (isRecord(against)) {
    const { delta, low, high } = against;
    if (typeof delta === 'number' && typeof low === 'number' && typeof high === 'number') {
      const verdict =
        against['within_noise'] === true
          ? 'within noise'
          : low > 0
            ? "above the parent's"
            : "below the parent's";
      difference = `, difference ${signed(delta)}, 95% interval ${signed(low)} to ${signed(high)}: ${verdict}`;
    }
  }
  const same =
    event['changed'] === false ? "the submitted weights are the parent's, unchanged; " : '';
  return (
    `${same}measured by the worker on practice validation: AUROC ${auroc}${difference}; ` +
    `recall ${recall}; false alarms ${alarms}`
  );
}

/** The words for one claimed figure's name. */
const FIGURE_NAMES: Readonly<Record<string, string>> = {
  auroc: 'AUROC',
  recall: 'recall',
  false_alarms: 'false alarms',
};

/**
 * FU-2 (operator, 2026-11-25): the editor's claimed figures (`submit`'s `figures`) beside the
 * measured ones, from the `submitted_measured` line's `claims`. `differs` is true when a stated
 * figure does not match, false when every one matches, and null when nothing was compared. A line
 * with no `claims` (a run recorded before FU-2) reads "no figures stated". Prose is not parsed.
 */
export function claimsLine(event: Readonly<Record<string, unknown>>): {
  readonly differs: boolean | null;
  readonly text: string;
} {
  const claims = event['claims'];
  if (!isRecord(claims) || claims['stated'] !== true) {
    return { differs: null, text: 'The editor stated no figures in its submit.' };
  }
  const checks = Array.isArray(claims['checks']) ? claims['checks'].filter(isRecord) : [];
  const stated = checks
    .map(
      (check) =>
        `${FIGURE_NAMES[String(check['figure'])] ?? word(check['figure'])} ${word(check['claimed'])}`,
    )
    .join(', ');
  if (claims['compared'] !== true) {
    const why =
      event['measured'] !== true
        ? 'the submitted weights were not measured'
        : `they name the ${word(claims['stated_split'])} split, and the worker measured ${word(event['split'])}`;
    return { differs: null, text: `The editor stated ${stated}; not compared: ${why}.` };
  }
  const differing = checks.filter((check) => check['matches'] === false);
  if (differing.length === 0) {
    return {
      differs: false,
      text: `The editor's stated figures match the measurement: ${stated}.`,
    };
  }
  const detail = differing
    .map(
      (check) =>
        `${FIGURE_NAMES[String(check['figure'])] ?? word(check['figure'])} stated ${word(check['claimed'])}, measured ${word(check['measured'])}`,
    )
    .join('; ');
  return {
    differs: true,
    text: `The editor's figures differ from the measurement: ${detail}.`,
  };
}

/**
 * The label beside the answer the run kept, from the run document's `submitted_check` (debt D4,
 * 2026-11-28). `check` is `unknown` when the document has none (a run from before FU-2, an edit
 * run without version-4 tools, an inspect that spent a forward pass): the generic label. `false`
 * when the worker did not measure the submitted weights; `unobserved` (fix 2, 2026-11-27) is
 * that case for an inspect that answered before any observation tool ran - its transcript has
 * no `submitted_measured` line, so this label is the only place the screen says so. `true` when
 * the worker measured them: the figures beside it are a measurement, the prose still is not.
 */
export function answerLabel(
  check: unknown,
  hasMeasuredLine: boolean,
): { readonly measured: 'true' | 'false' | 'unobserved' | 'unknown'; readonly text: string } {
  if (!isRecord(check) || typeof check['measured'] !== 'boolean') {
    return { measured: 'unknown', text: "The editor's own account, not checked." };
  }
  if (check['measured']) {
    return {
      measured: 'true',
      text: "The editor's own account. The worker measured the submitted weights; its words are not checked.",
    };
  }
  if (!hasMeasuredLine) {
    return {
      measured: 'unobserved',
      text:
        "The editor's own account, not measured: it answered before any observation tool ran " +
        '(no forward pass), so nothing it reports rests on a measurement.',
    };
  }
  return {
    measured: 'false',
    text: "The editor's own account, not checked: the worker did not measure the submitted weights.",
  };
}

/** Whether a turn submits: a native `submit` call, or the engine's `{"tool": "submit"}` JSON. */
function submits(entry: Readonly<Record<string, unknown>>): boolean {
  if (entry['role'] !== 'assistant') return false;
  if (callsOf(entry['tool_calls']).some((call) => call.name === 'submit')) return true;
  const content = entry['content'];
  if (typeof content !== 'string') return false;
  try {
    const action: unknown = JSON.parse(content);
    return isRecord(action) && action['tool'] === 'submit';
  } catch {
    return false;
  }
}

/**
 * A paging or measurement event as one readable line (2026-11-21), or `null` for any other event or an
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
  if (event === 'submitted_measured') return measuredText(rest);
  if (event === 'refusal') {
    // FR-020.17, FR-020.18: the category as given (it may be null), never guessed.
    const category = typeof rest['category'] === 'string' ? rest['category'] : null;
    return `turn ${word(rest['turn'])}: ${refusalText(category)}`;
  }
  if (event === 'timing') {
    // FR-020.14: the start-to-start gap in this conversation (null for its first) and the
    // duration, in seconds as the engine measured them.
    const seconds = (value: unknown): string | null =>
      typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(1)} s` : null;
    const gap = seconds(rest['gap_s']);
    const took = seconds(rest['duration_s']) ?? 'not recorded';
    return `turn ${word(rest['turn'])}: ${gap === null ? "the conversation's first request" : `${gap} after the previous request`}, took ${took}`;
  }
  if (event === 'handoff') {
    // FR-020.10: an append-only session names the conversation the handoff begins.
    const conversation =
      rest['conversation'] === undefined
        ? ''
        : `, conversation ${word(rest['conversation'])} begins`;
    const messages = Array.isArray(rest['messages']) ? rest['messages'] : [];
    const restarted = messages[messages.length - 1];
    const body =
      isRecord(restarted) && restarted['content'] !== undefined
        ? contentText(restarted['content']).text
        : '';
    return `handoff ${word(rest['n'])}${conversation}${body === '' ? '' : `\n\n${body}`}`;
  }
  if (event === 'recall') {
    const found = rest['found'] === true ? 'returned in full' : 'refused: no such page';
    return `turn ${word(rest['turn'])}: recall ${word(rest['page'])}, ${found}`;
  }
  return null;
}

/** A refusal in words (FR-020.18): "The editor refused (category: cyber)". */
export function refusalText(category: string | null): string {
  return `The editor refused (category: ${category ?? 'no category given'})`;
}

/** What the run page reads from a transcript (feature 020, FR-020.21). */
export interface SessionFacts {
  /** The header's `history_mode` (`append_only`), `paging` for an engine header without it, or
   * null when there is no engine header (another provider's transcript). */
  readonly historyMode: string | null;
  /** Each `handoff` event: its number and, in an append-only session, the conversation it begins. */
  readonly handoffs: readonly { readonly n: string; readonly conversation: string | null }[];
  /** The `refusal` event (its turn and category), or null. */
  readonly refusal: { readonly turn: string; readonly category: string | null } | null;
}

/**
 * The history mode, the handoffs and the refusal a transcript records (FR-020.6, FR-020.10,
 * FR-020.17). Every value is read from the entries as data; an unexpected shape is skipped.
 */
export function sessionFacts(entries: readonly unknown[]): SessionFacts {
  let historyMode: string | null = null;
  const handoffs: { n: string; conversation: string | null }[] = [];
  let refusal: SessionFacts['refusal'] = null;
  for (const entry of entries) {
    if (!isRecord(entry)) continue;
    if ('miwsp_transcript' in entry) {
      historyMode = typeof entry['history_mode'] === 'string' ? entry['history_mode'] : 'paging';
    } else if (entry['event'] === 'handoff') {
      handoffs.push({
        n: word(entry['n']),
        conversation: entry['conversation'] === undefined ? null : word(entry['conversation']),
      });
    } else if (entry['event'] === 'refusal' && refusal === null) {
      refusal = {
        turn: word(entry['turn']),
        category: typeof entry['category'] === 'string' ? entry['category'] : null,
      };
    }
  }
  return { historyMode, handoffs, refusal };
}

/** One served entry as text. Any shape is accepted; an unknown one is shown whole. */
export function entryView(entry: unknown): EntryView {
  const none = { calls: [], reasoning: null, answer: false } as const;
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
      answer: submits(entry),
    };
  }
  return { role: 'entry', text: pretty(entry), structured: true, ...none };
}
