import { afterEach, describe, expect, it, vi } from 'vitest';
import { copyText } from './clipboard';

function allowFallback(result: boolean) {
  const seen: string[] = [];
  const exec = vi.fn(() => {
    const area = document.activeElement as HTMLTextAreaElement;
    seen.push(area.value.slice(area.selectionStart, area.selectionEnd));
    return result;
  });
  Object.defineProperty(document, 'execCommand', { configurable: true, value: exec });
  return { exec, seen };
}

afterEach(() => {
  Reflect.deleteProperty(navigator, 'clipboard');
  Reflect.deleteProperty(document, 'execCommand');
  document.body.replaceChildren();
});

describe('copyText', () => {
  it('should use the clipboard API where the browser has one', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    const { exec } = allowFallback(true);
    expect(await copyText('abc')).toBe(true);
    expect(writeText).toHaveBeenCalledWith('abc');
    expect(exec).not.toHaveBeenCalled();
  });

  it('should copy over plain HTTP, where there is no clipboard API', async () => {
    const { exec, seen } = allowFallback(true);
    const button = document.body.appendChild(document.createElement('button'));
    button.focus();
    expect(await copyText('two\nlines')).toBe(true);
    expect(exec).toHaveBeenCalledWith('copy');
    expect(seen).toEqual(['two\nlines']);
    expect(document.querySelector('textarea')).toBeNull();
    expect(document.activeElement).toBe(button);
  });

  it('should fall back when the API refuses', async () => {
    const writeText = () => Promise.reject(new Error('NotAllowedError'));
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    const { seen } = allowFallback(true);
    expect(await copyText('x')).toBe(true);
    expect(seen).toEqual(['x']);
  });

  it('should report a copy that did not happen', async () => {
    allowFallback(false);
    expect(await copyText('x')).toBe(false);
    Reflect.deleteProperty(document, 'execCommand');
    expect(await copyText('x')).toBe(false);
    expect(document.querySelector('textarea')).toBeNull();
  });
});
