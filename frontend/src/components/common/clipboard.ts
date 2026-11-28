/**
 * Put text on the clipboard (operator request 2026-11-22).
 *
 * **miForge is served over plain HTTP on the LAN, and there `navigator.clipboard` does not exist**
 * (browsers offer it only in a secure context: HTTPS or localhost). So where the API is absent or
 * refuses, the text is selected in an off-screen textarea and copied with the older command,
 * which works in both. Call it from a click: the fallback needs the user's gesture.
 */
export async function copyText(text: string): Promise<boolean> {
  const api = (navigator as { clipboard?: Clipboard }).clipboard;
  if (api !== undefined) {
    try {
      await api.writeText(text);
      return true;
    } catch {
      // Refused (no permission, the page not focused): the fallback may still work.
    }
  }
  return copyBySelection(text);
}

function copyBySelection(text: string): boolean {
  const focused = document.activeElement;
  const area = document.createElement('textarea');
  area.value = text;
  area.readOnly = true;
  area.style.position = 'fixed';
  area.style.top = '0';
  area.style.opacity = '0';
  document.body.appendChild(area);
  try {
    // `preventScroll`: the copy must not move the page the operator is reading.
    area.focus({ preventScroll: true });
    area.select();
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    area.remove();
    if (focused instanceof HTMLElement) focused.focus({ preventScroll: true });
  }
}
