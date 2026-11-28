/**
 * A button that copies text and says whether it did (operator request 2026-11-22).
 *
 * `getText` is read at the click, so a live screen copies what it holds at that moment. The
 * label returns after two seconds. A failed copy says so rather than staying silent.
 */
import { useEffect, useState, type ReactNode } from 'react';
import { Check, Copy } from 'lucide-react';
import { Button } from './Button';
import { copyText } from './clipboard';

const RESET_MS = 2_000;

export interface CopyButtonProps {
  readonly getText: () => string;
  /** The idle label. */
  readonly label?: string;
  /** The accessible name, where several buttons share one label. */
  readonly name?: string;
  /** `small` sits beside a text box; `action` is a page action. */
  readonly size?: 'small' | 'action';
  readonly disabled?: boolean;
  readonly title?: string | undefined;
  readonly 'data-testid'?: string;
}

export function CopyButton({
  getText,
  label = 'Copy',
  name,
  size = 'small',
  disabled = false,
  title,
  'data-testid': testId = 'copy-button',
}: CopyButtonProps): ReactNode {
  const [result, setResult] = useState<{ readonly copied: boolean } | null>(null);
  useEffect(() => {
    if (result === null) return;
    const timer = setTimeout(() => setResult(null), RESET_MS);
    return () => clearTimeout(timer);
  }, [result]);

  const copy = async (): Promise<void> => {
    setResult({ copied: await copyText(getText()) });
  };
  const shown = result === null ? label : result.copied ? 'Copied' : 'Copy failed';
  const common = {
    onClick: () => void copy(),
    // While it reports, the visible report is the name.
    'aria-label': result === null ? name : undefined,
    'data-testid': testId,
    disabled,
    title,
  };
  const icon = result?.copied === true ? Check : Copy;

  if (size === 'action') {
    return (
      <Button icon={icon} {...common}>
        <span aria-live="polite">{shown}</span>
      </Button>
    );
  }
  const Icon = icon;
  return (
    <button
      type="button"
      className="inline-flex shrink-0 items-center gap-1 text-hint text-dim hover:underline"
      {...common}
    >
      <Icon size={12} aria-hidden="true" />
      <span aria-live="polite">{shown}</span>
    </button>
  );
}
