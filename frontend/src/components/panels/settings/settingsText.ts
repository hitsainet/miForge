/**
 * How the Settings screen words a value and a writer. Shared by the rows and the change history,
 * so the two can never show the same fact two ways.
 */

/** A value as the operator reads it. `null` is "not set", never an empty cell. */
export function showSetting(value: unknown): string {
  if (value === null || value === undefined) return 'not set';
  if (Array.isArray(value)) return value.length === 0 ? 'none' : value.map(showSetting).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

/** Who made a change: the operator, or an agent with the approval that let it run. */
export function showActor(actor: string | null | undefined, approvalId?: string | null): string {
  if (actor === null || actor === undefined || actor === '') return 'unknown';
  if (actor === 'operator') return 'you (the operator)';
  return approvalId ? `${actor}, approved as ${approvalId}` : actor;
}
