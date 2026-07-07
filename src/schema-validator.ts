type JsonSchema = Record<string, unknown>;

/**
 * Lightweight JSON Schema validator for the subset used by this demo. It intentionally avoids
 * pulling in a large validation dependency so npm install stays fast.
 */
export function validateAgainstJsonSchema(value: unknown, schema: unknown): string[] {
  if (!isObject(schema)) {
    return ["Schema must be a JSON object."];
  }
  return validateNode(value, schema, "$", []);
}

function validateNode(value: unknown, schema: JsonSchema, path: string, errors: string[]): string[] {
  const allowedTypes = normalizeTypeList(schema.type);
  if (allowedTypes.length > 0 && !allowedTypes.some((type) => matchesJsonType(value, type))) {
    errors.push(`${path} must be ${allowedTypes.join(" or ")}.`);
    return errors;
  }

  const enumValues = schema.enum;
  if (Array.isArray(enumValues) && !enumValues.some((candidate) => deepEqual(candidate, value))) {
    errors.push(`${path} must be one of ${enumValues.map(String).join(", ")}.`);
  }

  if (typeof value === "string") {
    if (typeof schema.minLength === "number" && value.length < schema.minLength) {
      errors.push(`${path} must contain at least ${schema.minLength} characters.`);
    }
    if (typeof schema.maxLength === "number" && value.length > schema.maxLength) {
      errors.push(`${path} must contain no more than ${schema.maxLength} characters.`);
    }
  }

  if (typeof value === "number") {
    if (typeof schema.minimum === "number" && value < schema.minimum) {
      errors.push(`${path} must be >= ${schema.minimum}.`);
    }
    if (typeof schema.maximum === "number" && value > schema.maximum) {
      errors.push(`${path} must be <= ${schema.maximum}.`);
    }
  }

  if (Array.isArray(value)) {
    if (typeof schema.minItems === "number" && value.length < schema.minItems) {
      errors.push(`${path} must contain at least ${schema.minItems} items.`);
    }
    if (typeof schema.maxItems === "number" && value.length > schema.maxItems) {
      errors.push(`${path} must contain no more than ${schema.maxItems} items.`);
    }
    if (isObject(schema.items)) {
      value.forEach((item, index) => validateNode(item, schema.items as JsonSchema, `${path}[${index}]`, errors));
    }
  }

  if (isObject(value)) {
    const properties = isObject(schema.properties) ? (schema.properties as Record<string, unknown>) : {};
    const required = Array.isArray(schema.required) ? schema.required.filter((item): item is string => typeof item === "string") : [];

    for (const key of required) {
      if (!(key in value)) {
        errors.push(`${path}.${key} is required.`);
      }
    }

    for (const [key, propertySchema] of Object.entries(properties)) {
      if (key in value && isObject(propertySchema)) {
        validateNode(value[key], propertySchema, `${path}.${key}`, errors);
      }
    }

    if (schema.additionalProperties === false) {
      for (const key of Object.keys(value)) {
        if (!(key in properties)) {
          errors.push(`${path}.${key} is not allowed by additionalProperties:false.`);
        }
      }
    }
  }

  return errors;
}

function normalizeTypeList(type: unknown): string[] {
  if (typeof type === "string") return [type];
  if (Array.isArray(type)) return type.filter((item): item is string => typeof item === "string");
  return [];
}

function matchesJsonType(value: unknown, type: string): boolean {
  switch (type) {
    case "object":
      return isObject(value);
    case "array":
      return Array.isArray(value);
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "integer":
      return Number.isInteger(value);
    case "boolean":
      return typeof value === "boolean";
    case "null":
      return value === null;
    default:
      return true;
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function deepEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
