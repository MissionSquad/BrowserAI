/**
 * Remove `<think>…</think>` blocks from model output. Some chat templates emit an empty think pair
 * even when thinking is disabled (enable_thinking:false), and reasoning models emit filled ones;
 * neither belongs in a visible answer — or in text fed to TTS. Also hides an unclosed block while
 * it is still streaming.
 */
export function stripThinkTags(text: string): string {
  let out = text.replace(/<think>[\s\S]*?<\/think>/gi, "");
  const openIndex = out.search(/<think>/i);
  if (openIndex !== -1) out = out.slice(0, openIndex);
  return out.replace(/^\s+/, "");
}
