// Scans the BUILT site for content that must never be published, using publish-rules.json.
//
// Runs as `postbuild`, so every `npm run build` (local, the CI build check, and the Pages
// deploy on the mirror) fails when a forbidden pattern reaches a page. The Python test
// (backend/tests/unit/test_manual_publish_allowlist.py) checks the SOURCES against the same
// rules; this checks what is actually served, including the generated MCP tool page.
//
// Only page text is scanned: <script> and <style> bodies are removed first, because the
// bundled JavaScript is third-party code whose literals say nothing about our content.
// The end tag is matched as HTML parses it (`</script >`, `</script foo>` and any case all
// close the element). Matching only `</script>` would let the lazy body run on to the NEXT
// script's end tag and remove the page text in between from the scan: a fail-open scanner.
// scripts/check-published.test.mjs pins this.
import {readFileSync, readdirSync, statSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const buildDir = process.argv[2] ? resolve(process.argv[2]) : join(root, 'build');
const rules = JSON.parse(readFileSync(join(root, 'publish-rules.json'), 'utf8')).forbidden;

function htmlFiles(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) out.push(...htmlFiles(path));
    else if (name.endsWith('.html')) out.push(path);
  }
  return out;
}

let files;
try {
  files = htmlFiles(buildDir);
} catch (error) {
  console.error(`check-published: cannot read ${buildDir}: ${error.message}`);
  process.exit(1);
}
// A scan over nothing passes vacuously; the home page must be there.
if (!files.some((f) => relative(buildDir, f) === 'index.html')) {
  console.error(`check-published: no index.html under ${buildDir}; refusing to pass an empty scan`);
  process.exit(1);
}

const compiled = rules.map((rule) => ({...rule, re: new RegExp(rule.pattern, `g${rule.flags || ''}`)}));
const failures = [];
for (const file of files) {
  const text = readFileSync(file, 'utf8')
    .replace(/<script\b[\s\S]*?<\/script\b[^>]*>/gi, ' ')
    .replace(/<style\b[\s\S]*?<\/style\b[^>]*>/gi, ' ');
  for (const rule of compiled) {
    rule.re.lastIndex = 0;
    const match = rule.re.exec(text);
    if (match) {
      failures.push(`${relative(buildDir, file)}: [${rule.id}] "${match[0]}" - ${rule.why}`);
    }
  }
}

if (failures.length > 0) {
  console.error('check-published: the built site contains content that must not be published:');
  for (const line of failures) console.error(`  ${line}`);
  process.exit(1);
}
console.log(`check-published: ${files.length} pages clean against ${rules.length} rules`);
