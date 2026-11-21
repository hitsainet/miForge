/**
 * Who asked, in words (feature 015, FR-015.3 UI; FTID §6.1).
 *
 * miStudio's MCP server becomes miWorkbench's. Its stored actor id `agent:mistudio-mcp` is
 * frozen (register entry DR-17: history rows keep it), and the new one is
 * `agent:miworkbench-mcp`; an operator reads both as the same thing.
 */
import { describe, expect, it } from 'vitest';
import { askedBy } from './ApprovalCard';

describe('askedBy', () => {
  it("askedBy maps both MCP ids to miWorkbench's MCP server", () => {
    expect(askedBy('agent:mistudio-mcp')).toBe("asked by miWorkbench's MCP server");
    expect(askedBy('agent:miworkbench-mcp')).toBe("asked by miWorkbench's MCP server");
  });

  it('other ids keep asked by agent <name>', () => {
    expect(askedBy('agent:forge-mcp')).toBe('asked by agent forge-mcp');
    expect(askedBy(null)).toBe('asked by an agent');
    expect(askedBy('')).toBe('asked by an agent');
  });
});
