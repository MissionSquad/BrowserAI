export type JsonParseResult =
  | { ok: true; value: unknown; rawJson: string; warnings: string[] }
  | { ok: false; error: string; rawJson: string; warnings: string[] };

type BalancedJsonCandidate = {
  rawJson: string;
  hadPrefixOrSuffix: boolean;
};

/**
 * Some models wrap JSON in Markdown or thinking tags despite instructions. WebLLM JSON mode should
 * not do this, but these helpers make output handling robust across different models/settings.
 * (Schema validation lives in schema-validator.ts — one validator for the whole SDK.)
 */
export function normalizeModelText(text: string): string {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const withoutFence = fenced?.[1] ? fenced[1].trim() : trimmed;
  return withoutFence.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
}

export function extractJsonCandidate(text: string): string {
  const cleaned = normalizeModelText(text);
  return findFirstBalancedJson(cleaned)?.rawJson ?? cleaned;
}

export function parseJsonFromModel(text: string): JsonParseResult {
  const cleaned = normalizeModelText(text);
  const warnings: string[] = [];

  try {
    return { ok: true, value: JSON.parse(cleaned), rawJson: cleaned, warnings };
  } catch {
    // Continue below. A common small-model failure is valid JSON followed by repeated
    // text. In that case we parse the first complete JSON value and warn the caller.
  }

  const balanced = findFirstBalancedJson(cleaned);
  if (balanced) {
    if (balanced.hadPrefixOrSuffix) {
      warnings.push(
        "Parsed the first complete JSON value, but the model also produced prefix/suffix text. Tighten generation parameters or lower max tokens if this repeats.",
      );
    }

    try {
      return {
        ok: true,
        value: JSON.parse(balanced.rawJson),
        rawJson: balanced.rawJson,
        warnings,
      };
    } catch (error) {
      return {
        ok: false,
        rawJson: balanced.rawJson,
        warnings,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  return {
    ok: false,
    rawJson: cleaned,
    warnings,
    error: "No complete balanced JSON object or array was found.",
  };
}

export function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export function detectLikelyRepetition(text: string): string[] {
  const warnings: string[] = [];
  const normalized = text.toLowerCase().replace(/\s+/g, " ").trim();
  if (!normalized) return warnings;

  const words = normalized.split(" ").filter((word) => word.length > 2);
  const ngramCounts = new Map<string, number>();
  for (let i = 0; i <= words.length - 3; i += 1) {
    const ngram = words.slice(i, i + 3).join(" ");
    ngramCounts.set(ngram, (ngramCounts.get(ngram) ?? 0) + 1);
  }

  const repeated = Array.from(ngramCounts.entries()).find(([, count]) => count >= 3);
  if (repeated) {
    warnings.push(`Likely repetition detected: "${repeated[0]}" appeared ${repeated[1]} times.`);
  }

  const fragments = normalized
    .split(/[{}\[\],]/)
    .map((line) => line.trim())
    .filter((line) => line.length > 20);
  const fragmentCounts = new Map<string, number>();
  for (const fragment of fragments) {
    fragmentCounts.set(fragment, (fragmentCounts.get(fragment) ?? 0) + 1);
  }
  const repeatedFragment = Array.from(fragmentCounts.entries()).find(([, count]) => count >= 3);
  if (repeatedFragment) {
    warnings.push("Repeated JSON fragments detected in the raw output.");
  }

  return warnings;
}

function findFirstBalancedJson(text: string): BalancedJsonCandidate | null {
  const start = findJsonStart(text);
  if (start < 0) return null;

  const stack: string[] = [];
  let inString = false;
  let escaped = false;

  for (let i = start; i < text.length; i += 1) {
    const char = text[i];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char === "{") {
      stack.push("}");
      continue;
    }

    if (char === "[") {
      stack.push("]");
      continue;
    }

    if (char === "}" || char === "]") {
      const expected = stack.pop();
      if (expected !== char) return null;
      if (stack.length === 0) {
        const rawJson = text.slice(start, i + 1).trim();
        const suffix = text.slice(i + 1).trim();
        return {
          rawJson,
          hadPrefixOrSuffix: start > 0 || suffix.length > 0,
        };
      }
    }
  }

  return null;
}

function findJsonStart(text: string): number {
  const objectStart = text.indexOf("{");
  const arrayStart = text.indexOf("[");

  if (objectStart === -1) return arrayStart;
  if (arrayStart === -1) return objectStart;
  return Math.min(objectStart, arrayStart);
}
