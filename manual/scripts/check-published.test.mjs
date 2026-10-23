// Tests for check-published.mjs, run by `npm test` (node:test) in the Manual Build workflow.
//
// The scanner removes <script> and <style> bodies before it scans page text. If it does not
// recognise an end tag the browser accepts (`</script >`), the lazy body runs on to the next
// script and the page text between them is never scanned. These tests build a tiny site in a
// temporary directory and run the real script against it.
//
// The forbidden text is assembled at run time: this file is itself published, and the source
// scan (backend/tests/unit/test_manual_publish_allowlist.py) would refuse it as a literal.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const script = join(dirname(fileURLToPath(import.meta.url)), 'check-published.mjs');
const FORBIDDEN = ['AGENT', 'md'].join('.'); // the planning-tree rule

function scan(html) {
  const dir = mkdtempSync(join(tmpdir(), 'check-published-'));
  try {
    writeFileSync(join(dir, 'index.html'), html);
    return spawnSync(process.execPath, [script, dir], {encoding: 'utf8'});
  } finally {
    rmSync(dir, {recursive: true, force: true});
  }
}

test('a clean page passes', () => {
  const run = scan('<html><body><p>Nothing to see.</p><script>var x = 1;</script></body></html>');
  assert.equal(run.status, 0, run.stderr);
});

test('forbidden text in page content fails the build', () => {
  const run = scan(`<html><body><p>See ${FORBIDDEN}.</p></body></html>`);
  assert.equal(run.status, 1);
  assert.match(run.stderr, /planning-tree/);
});

test('script bodies are not scanned', () => {
  const run = scan(`<html><body><script>var s = "${FORBIDDEN}";</script></body></html>`);
  assert.equal(run.status, 0, run.stderr);
});

for (const end of ['</script >', '</SCRIPT>', '</script\n>', '</script foo="bar">']) {
  test(`text after a script closed by ${JSON.stringify(end)} is still scanned`, () => {
    const html =
      `<html><body><script>var a = 1;${end}<p>See ${FORBIDDEN}.</p>` +
      '<script>var b = 2;</script></body></html>';
    const run = scan(html);
    assert.equal(run.status, 1, `the page text after ${JSON.stringify(end)} was not scanned`);
    assert.match(run.stderr, /planning-tree/);
  });
}

for (const end of ['</style >', '</STYLE>']) {
  test(`text after a style closed by ${JSON.stringify(end)} is still scanned`, () => {
    const html =
      `<html><head><style>p {}${end}</head><body><p>See ${FORBIDDEN}.</p>` +
      '<style>a {}</style></body></html>';
    const run = scan(html);
    assert.equal(run.status, 1, `the page text after ${JSON.stringify(end)} was not scanned`);
  });
}
