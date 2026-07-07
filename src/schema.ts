/**
 * Example JSON Schema for structured extraction. Useful as a starting point for schema-constrained
 * generation; hosts supply their own schema for real workloads.
 */
export const DEFAULT_EXTRACTION_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    summary: {
      type: "string",
      maxLength: 240,
      description: "One concise sentence summarizing the input text.",
    },
    language: {
      type: "string",
      maxLength: 40,
      description: "The primary language of the input text, e.g. English.",
    },
    sentiment: {
      type: "string",
      enum: ["positive", "neutral", "negative", "mixed"],
    },
    entities: {
      type: "array",
      maxItems: 8,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          text: { type: "string", maxLength: 120 },
          type: {
            type: "string",
            enum: [
              "person",
              "organization",
              "product_feature",
              "integration",
              "competitor",
              "date",
              "money",
              "other",
            ],
          },
        },
        required: ["text", "type"],
      },
    },
    intents: {
      type: "array",
      maxItems: 5,
      items: { type: "string", maxLength: 80 },
      description: "Likely user intents, such as request_feature, ask_pricing, report_blocker, or schedule_followup.",
    },
    action_items: {
      type: "array",
      maxItems: 5,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          owner: { type: ["string", "null"], maxLength: 80 },
          task: { type: "string", maxLength: 160 },
          due_date: { type: ["string", "null"], maxLength: 80 },
        },
        required: ["owner", "task", "due_date"],
      },
    },
    risk_flags: {
      type: "array",
      maxItems: 8,
      items: { type: "string", maxLength: 120 },
      description: "Adoption blockers, competitive risks, deadlines, or security/compliance concerns present in the text.",
    },
    confidence: {
      type: "number",
      minimum: 0,
      maximum: 1,
      description: "Model confidence from 0 to 1.",
    },
  },
  required: [
    "summary",
    "language",
    "sentiment",
    "entities",
    "intents",
    "action_items",
    "risk_flags",
    "confidence",
  ],
} as const;

export function stringifySchema(schema: unknown): string {
  return JSON.stringify(schema, null, 2);
}

/**
 * The schema sent to the constrained decoder does not need human-only metadata.
 * Removing descriptions keeps the prompt and grammar-construction work smaller.
 */
export function stripSchemaMetadata(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(stripSchemaMetadata);
  }

  if (!value || typeof value !== "object") {
    return value;
  }

  const result: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    if (["description", "examples", "title", "$comment"].includes(key)) {
      continue;
    }
    result[key] = stripSchemaMetadata(child);
  }
  return result;
}
