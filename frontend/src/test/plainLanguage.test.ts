/**
 * The Help text is plain (ENH-004 D8, 2026-10-20): no internal identifiers, and no acronym the
 * reader has not been told the meaning of.
 *
 * 1. **Internal identifiers** - requirement, decision and planning ids - are found with the
 *    manual's own publish rules, read from `manual/publish-rules.json` (category `internal`), so
 *    the app and the manual share one list rather than two that drift.
 * 2. **Acronyms** (`\b[A-Z]{2,}\b`) must be spelled out in parentheses at their first use in a
 *    topic - "graphics card (GPU)" - and may then be used bare. Product names are mixed case
 *    (miForge, miLLM) and never match.
 *
 * Fail-open protections, as `noUnfinishedWorkCopy` has them: the scan must read a minimum amount
 * of text and at least three rules, and a synthetic topic carrying `FR-009.1` and a bare `AUROC`
 * must be caught - the negative control is part of the test, not a one-off.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { HELP_TOPICS, type HelpTopic } from '@/config/help';

interface Rule {
  readonly id: string;
  readonly category: string;
  readonly pattern: string;
  readonly flags: string;
}

const RULES_FILE = path.resolve(__dirname, '../../../manual/publish-rules.json');

const RULES: readonly { id: string; regex: RegExp }[] = (
  JSON.parse(readFileSync(RULES_FILE, 'utf8')) as { forbidden: Rule[] }
).forbidden
  .filter((rule) => rule.category === 'internal')
  .map((rule) => ({ id: rule.id, regex: new RegExp(rule.pattern, rule.flags.replace('g', '')) }));

const ACRONYM = /\b[A-Z]{2,}\b/g;

function textOf(topic: HelpTopic): string {
  return [topic.title, ...topic.body, ...(topic.steps ?? [])].join('\n');
}

/** Every problem in one topic, as readable sentences. */
export function problemsIn(topic: HelpTopic): string[] {
  const text = textOf(topic);
  const problems: string[] = [];
  for (const rule of RULES) {
    const hit = rule.regex.exec(text);
    if (hit !== null) problems.push(`${topic.id}: internal identifier "${hit[0]}" (${rule.id})`);
  }
  const seen = new Set<string>();
  for (const match of text.matchAll(ACRONYM)) {
    const word = match[0];
    if (seen.has(word)) continue;
    seen.add(word);
    const at = match.index ?? 0;
    const spelledOut =
      text.slice(at - 1, at) === '(' &&
      text.slice(at + word.length, at + word.length + 1) === ')' &&
      /\w\s*$/.test(text.slice(0, at - 1));
    if (!spelledOut) {
      problems.push(`${topic.id}: "${word}" is not spelled out at its first use`);
    }
  }
  return problems;
}

describe('[Help plain language]', () => {
  it('should read the shared rules and a real amount of text', () => {
    expect(RULES.length).toBeGreaterThanOrEqual(3);
    expect(RULES.map((rule) => rule.id)).toEqual(
      expect.arrayContaining(['planning-tree', 'requirement-id', 'decision-id']),
    );
    const scanned = HELP_TOPICS.map(textOf).join('\n').length;
    expect(scanned).toBeGreaterThan(2000);
  });

  it('should catch an internal id and a bare acronym (negative control)', () => {
    const planted: HelpTopic = {
      id: 'planted',
      title: 'Planted',
      body: ['This satisfies FR-009.1 and reports AUROC.'],
      manual: 'glossary',
    };
    const problems = problemsIn(planted);
    expect(problems.some((p) => p.includes('FR-009') && p.includes('requirement-id'))).toBe(true);
    expect(problems.some((p) => p.includes('"AUROC"'))).toBe(true);
    // ...and accepts the same acronym once it is spelled out.
    expect(
      problemsIn({ ...planted, body: ['The area under the curve (AUROC), then AUROC again.'] }),
    ).toEqual([]);
  });

  it('should find no internal id and no unexplained acronym in any topic', () => {
    expect(HELP_TOPICS.flatMap(problemsIn)).toEqual([]);
  });
});
