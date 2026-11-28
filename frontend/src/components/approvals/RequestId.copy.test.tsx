import { afterEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { RequestId } from './ApprovalCard';

afterEach(() => Reflect.deleteProperty(document, 'execCommand'));

it('should copy a request id over plain HTTP, where there is no clipboard API', async () => {
  const exec = vi.fn(() => true);
  Object.defineProperty(document, 'execCommand', { configurable: true, value: exec });
  render(<RequestId id="apr_123" />);
  fireEvent.click(screen.getByRole('button', { name: 'Copy request id' }));
  expect(await screen.findByText('Copied')).toBeInTheDocument();
  expect(exec).toHaveBeenCalledWith('copy');
});
