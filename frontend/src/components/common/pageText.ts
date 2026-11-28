/**
 * A rendered screen as plain text, for its "Copy page" button (operator request 2026-11-22).
 *
 * It reads what is rendered, so a panel added later is copied without being listed here:
 * - a `<pre>` is copied exactly as written, collapsed `<details>` included;
 * - each block is a line, each heading starts a paragraph, a `dt`/`dd` pair is `label: value`;
 * - inline elements side by side keep a space between them (their gap is CSS, not text);
 * - controls (buttons), decoration (`aria-hidden`) and what is marked `data-page-text="omit"`
 *   (a page head's Help links) are left out.
 */
const BLOCKS = new Set([
  'ARTICLE',
  'DETAILS',
  'DIV',
  'DL',
  'H1',
  'H2',
  'H3',
  'HEADER',
  'LI',
  'OL',
  'P',
  'SECTION',
  'SUMMARY',
  'TABLE',
  'TR',
  'UL',
]);

export function pageText(root: Element): string {
  const lines: string[] = [];
  let line = '';

  const flush = (): void => {
    const text = line.replace(/\s+/g, ' ').trim();
    if (text !== '') lines.push(text);
    line = '';
  };
  const paragraph = (): void => {
    flush();
    if (lines.length > 0 && lines[lines.length - 1] !== '') lines.push('');
  };

  const visit = (element: Element): void => {
    const tag = element.tagName;
    if (tag === 'BUTTON' || element.getAttribute('aria-hidden') === 'true') return;
    if (element.getAttribute('data-page-text') === 'omit') return;
    if (tag === 'PRE') {
      flush();
      lines.push(element.textContent ?? '');
      return;
    }
    const block = BLOCKS.has(tag);
    // A step, a card's heading: each starts a paragraph of its own.
    if (tag === 'H1' || tag === 'H2' || (tag === 'LI' && element.querySelector('pre') !== null)) {
      paragraph();
    } else if (block || tag === 'DT') {
      flush();
    }

    let afterElement = false;
    for (const child of element.childNodes) {
      if (child instanceof Element) {
        if (afterElement) line += ' ';
        visit(child);
        afterElement = true;
      } else if (child.nodeType === Node.TEXT_NODE) {
        line += child.textContent ?? '';
        afterElement = false;
      }
    }

    if (tag === 'DT') line = `${line.trimEnd()}: `;
    else if (block || tag === 'DD') flush();
  };

  visit(root);
  flush();
  return lines.join('\n');
}
