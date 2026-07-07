export const DEFAULT_EXTRACTION_INSTRUCTIONS = [
  "You are a deterministic information-extraction engine.",
  "Return exactly one JSON object and nothing else.",
  "Never emit Markdown, comments, explanations, hidden reasoning, or a second JSON object.",
  "Stop immediately after the final closing brace of the JSON object.",
  "Use null when a scalar field is unknown. Use [] when a list has no items.",
  "Keep string values concise and do not invent facts not present in the input text.",
].join("\n");

/**
 * Build the system + user turns for schema-guided extraction. WebLLM backends additionally enforce
 * the schema with constrained decoding; Transformers.js backends rely on this prompt alone.
 */
export function buildExtractionMessages(inputText: string, schema: unknown, additionalInstructions = DEFAULT_EXTRACTION_INSTRUCTIONS) {
  const schemaText = JSON.stringify(schema, null, 2);
  const instructions = additionalInstructions.trim() || DEFAULT_EXTRACTION_INSTRUCTIONS;

  return [
    {
      role: "system" as const,
      content: [
        instructions,
        "",
        "The JSON object must follow this JSON Schema:",
        schemaText,
      ].join("\n"),
    },
    {
      role: "user" as const,
      content: `Extract structured data from the text below. Return one JSON object only.\n\n<text>\n${inputText}\n</text>`,
    },
  ];
}
