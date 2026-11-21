/**
 * The draft envelope and the secret rule (task 13.3; AC-009.36, .38). **M25's target.**
 *
 * The secret assertions go through `writeDraft` and read the raw `localStorage` string back,
 * because that string is the thing that would sit in a browser profile - a test of
 * `stripSecrets` alone would pass against a `writeDraft` that stopped calling it.
 */
import { describe, expect, it } from 'vitest';
import {
  DRAFT_MAX_AGE_MS,
  DRAFT_PREFIX,
  isSecretKey,
  isSecretValue,
  readDraft,
  writeDraft,
} from './draftStorage';

const asIs = (data: unknown) => data;

describe('[draftStorage] the secret rule', () => {
  it('should never write a credential-named field, at any depth (M25)', () => {
    writeDraft('form', 1, {
      repo: 'org/model',
      hfToken: 'abc',
      api_key: 'abc',
      apiKey: 'abc',
      password: 'abc',
      clientSecret: 'abc',
      'x-auth': 'abc',
      authorization: 'abc',
      credentials: { user: 'u' },
      nested: { list: [{ condition: 'step_count', accessToken: 'abc' }] },
    });
    const raw = window.localStorage.getItem(`${DRAFT_PREFIX}form`) ?? '';
    for (const name of [
      'hfToken',
      'api_key',
      'apiKey',
      'password',
      'clientSecret',
      'x-auth',
      'authorization',
      'credentials',
      'accessToken',
    ]) {
      expect(raw, name).not.toContain(name);
    }
    expect(raw).not.toContain('abc');
    // And the ordinary fields beside them are kept.
    expect(raw).toContain('org/model');
    expect(raw).toContain('step_count');
  });

  it('should never write a credential-shaped value, whatever the field is called (M25)', () => {
    const values = [
      `hf_${'a'.repeat(34)}`,
      `sk-${'b'.repeat(40)}`,
      `ghp_${'c'.repeat(36)}`,
      `github_pat_${'d'.repeat(40)}`,
      'Bearer eyJhbGciOiJIUzI1NiJ9.e30',
      'https://sean:hunter2@huggingface.co/org/model',
    ];
    writeDraft('form', 1, { path: values[0], notes: values.slice(1), repo: 'org/model' });
    const raw = window.localStorage.getItem(`${DRAFT_PREFIX}form`) ?? '';
    for (const value of values) expect(raw, value).not.toContain(value);
    expect(raw).toContain('org/model');
  });

  it('should match whole words, so ordinary fields are not mistaken for secrets', () => {
    for (const name of [
      'tokenizer',
      'tokenizerPath',
      'monkey',
      'keyboardLayout',
      'runId',
      'packId',
      'gpuRequest',
      'adapterTargetModules',
      'domain_given',
      'retention_floor',
    ]) {
      expect(isSecretKey(name), name).toBe(false);
    }
    for (const name of [
      'token',
      'hfToken',
      'HF_TOKEN',
      'apiKey',
      'api-key',
      'key',
      'Password',
      'auth',
    ]) {
      expect(isSecretKey(name), name).toBe(true);
    }
    for (const value of [
      'org/model',
      'main',
      '/data/models/llama',
      '3e-4',
      'https://huggingface.co/org/model',
    ]) {
      expect(isSecretValue(value), value).toBe(false);
    }
  });
});

describe('[draftStorage] the envelope', () => {
  it('should round-trip a draft with its saved time', () => {
    writeDraft('form', 3, { a: '1' }, 1_000);
    expect(readDraft('form', 3, asIs, 2_000)).toEqual({ data: { a: '1' }, savedAt: 1_000 });
  });

  it('should discard and remove an unparseable, other-version, expired or rejected entry', () => {
    const key = `${DRAFT_PREFIX}form`;
    const cases: [string, (data: unknown) => unknown][] = [
      ['{', asIs],
      [JSON.stringify({ v: 2, savedAt: 0, data: {} }), asIs],
      [JSON.stringify({ v: 1, savedAt: 'yesterday', data: {} }), asIs],
      [JSON.stringify({ v: 1, savedAt: 0, data: {} }), asIs],
      [JSON.stringify({ v: 1, savedAt: DRAFT_MAX_AGE_MS, data: 'x' }), () => null],
      [
        JSON.stringify({ v: 1, savedAt: DRAFT_MAX_AGE_MS, data: 'x' }),
        () => {
          throw new Error('bad');
        },
      ],
      ['null', asIs],
    ];
    for (const [stored, decode] of cases) {
      window.localStorage.setItem(key, stored);
      expect(readDraft('form', 1, decode, DRAFT_MAX_AGE_MS + 1), stored).toBeNull();
      expect(window.localStorage.getItem(key), stored).toBeNull();
    }
  });
});
