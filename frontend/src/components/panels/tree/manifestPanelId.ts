/** The id `aria-controls` points at; a node name may hold `/` and `:`, so it is encoded. */
export function manifestPanelId(name: string): string {
  return `tree-manifest-${encodeURIComponent(name)}`;
}
