/**
 * A moment on an approvals surface (ENH-008 §2.7): the browser's local time as the text, the
 * exact instant in UTC as the tooltip and the `dateTime`. "unknown" for a value that cannot be
 * read - never "Invalid Date".
 */
import type { ReactNode } from 'react';
import { formatLocal } from './time';

export interface LocalTimeProps {
  readonly iso: string | null | undefined;
  /** Text before the time, inside the same element ("asked "). */
  readonly prefix?: string;
  readonly className?: string;
  readonly 'data-testid'?: string;
}

export function LocalTimeText({ iso, prefix = '', className, ...rest }: LocalTimeProps): ReactNode {
  const { text, title } = formatLocal(iso);
  return (
    <time
      dateTime={title === '' ? undefined : title}
      title={title === '' ? undefined : title}
      className={className}
      {...rest}
    >
      {prefix}
      {text}
    </time>
  );
}
