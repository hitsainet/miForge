/**
 * Plain sentences for the unavailable-reason codes the signal-source and selector registries
 * return. The registries send a code only; the code is still shown, after the sentence, so
 * it can be searched for in the logs.
 */
const SENTENCES: Readonly<Record<string, string>> = {
  inference_provider_unavailable: 'no generation provider is available',
  judge_provider_not_configured: 'no judge endpoint is configured',
  midataworks_not_configured: 'miDataworks is not configured',
};

/** `"<sentence> (<code>)"` for a known code, the code alone for an unknown one. */
export function unavailableSentence(code: string | null | undefined): string {
  if (code === null || code === undefined || code.length === 0) return 'no reason given';
  const sentence = SENTENCES[code];
  return sentence === undefined ? code : `${sentence} (${code})`;
}
