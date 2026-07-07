import { describe, expect, it } from "vitest";
import type { AppConfig } from "@mlc-ai/web-llm";
import {
  applyContextOverride,
  buildProxiedAppConfig,
  buildWebLLMAppConfig,
  proxyHuggingFaceUrl,
  proxyRawGithubUrl,
} from "../src/model-source.js";
import { CUSTOM_MODEL_RECORDS } from "../src/models.js";
import { WEBLLM_CACHE_SCOPES } from "../src/model-cache.js";
import { normalizeContextLength } from "../src/client.js";
import { DEFAULT_CONTEXT_LENGTH } from "../src/config.js";

const ORIGIN = "https://proxy.example.com";

describe("proxy URL rewriting", () => {
  it("rewrites Hugging Face model URLs to /hf/ and raw GitHub wasm URLs to /gh-raw/", () => {
    const config = buildProxiedAppConfig(
      {
        model_list: [
          {
            model: "https://huggingface.co/mlc-ai/TestModel-MLC",
            model_id: "TestModel-MLC",
            model_lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/test.wasm",
          },
        ],
      } as AppConfig,
      ORIGIN,
    );

    expect(config.model_list[0].model).toBe(`${ORIGIN}/hf/mlc-ai/TestModel-MLC`);
    expect(config.model_list[0].model_lib).toBe(`${ORIGIN}/gh-raw/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/test.wasm`);
  });

  it("proxies custom Gemma model libraries hosted on Hugging Face", () => {
    const config = buildProxiedAppConfig({ model_list: [CUSTOM_MODEL_RECORDS[0]] } as AppConfig, ORIGIN);

    expect(config.model_list[0].model).toBe(`${ORIGIN}/hf/welcoma/gemma-4-E2B-it-q4f16_1-MLC`);
    expect(config.model_list[0].model_lib).toBe(
      `${ORIGIN}/hf/welcoma/gemma-4-E2B-it-q4f16_1-MLC/resolve/main/libs/gemma-4-E2B-it-q4f16_1-MLC-webgpu.wasm`,
    );
  });

  it("passes through non-HF/non-GitHub and unparseable URLs unchanged", () => {
    expect(proxyHuggingFaceUrl("https://example.com/model", ORIGIN)).toBe("https://example.com/model");
    expect(proxyHuggingFaceUrl("not a url", ORIGIN)).toBe("not a url");
    expect(proxyRawGithubUrl("https://example.com/lib.wasm", ORIGIN)).toBe("https://example.com/lib.wasm");
  });

  it("applies cacheBackend and appends custom model records in the WebLLM app config", () => {
    const config = buildWebLLMAppConfig(
      {
        prebuiltAppConfig: {
          cacheBackend: "cache",
          model_list: [
            {
              model: "https://huggingface.co/mlc-ai/TestModel-MLC",
              model_id: "TestModel-MLC",
              model_lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/test.wasm",
            },
          ],
        },
      } as unknown as typeof import("@mlc-ai/web-llm"),
      "same-origin-proxy",
      ORIGIN,
      "indexeddb",
    );

    expect(config.cacheBackend).toBe("indexeddb");
    expect(config.model_list[0].model).toContain("/hf/");
    expect(config.model_list[0].model_lib).toContain("/gh-raw/");
    expect(config.model_list.some((record) => record.model_id === "gemma-4-E2B-it-q4f16_1-MLC")).toBe(true);
  });

  it("keeps direct URLs in direct mode while still merging custom records", () => {
    const config = buildWebLLMAppConfig(
      {
        prebuiltAppConfig: { cacheBackend: "cache", model_list: [] },
      } as unknown as typeof import("@mlc-ai/web-llm"),
      "direct",
      ORIGIN,
      "indexeddb",
    );
    const gemma = config.model_list.find((record) => record.model_id === "gemma-4-E2B-it-q4f16_1-MLC");
    expect(gemma?.model).toContain("https://huggingface.co/welcoma/");
  });

  it("targets only WebLLM artifact cache scopes when deleting downloaded models", () => {
    expect(WEBLLM_CACHE_SCOPES).toEqual(["webllm/model", "webllm/config", "webllm/wasm"]);
  });
});

describe("context-length override", () => {
  const config: AppConfig = {
    model_list: [
      { model: "m", model_id: "a", model_lib: "l" },
      { model: "m", model_id: "b", model_lib: "l", overrides: { sliding_window_size: -1 } },
    ],
  } as AppConfig;

  it("merges the override into only the selected record, preserving curated overrides", () => {
    const next = applyContextOverride(config, "b", 8192);
    expect(next.model_list[0].overrides).toBeUndefined();
    expect(next.model_list[1].overrides).toEqual({ sliding_window_size: -1, context_window_size: 8192 });
  });

  it("returns the config unchanged for a null override", () => {
    expect(applyContextOverride(config, "b", null)).toBe(config);
  });

  it("normalizes: default/invalid → null, out-of-range clamped", () => {
    expect(normalizeContextLength(undefined)).toBeNull();
    expect(normalizeContextLength(null)).toBeNull();
    expect(normalizeContextLength(0)).toBeNull();
    expect(normalizeContextLength(-5)).toBeNull();
    expect(normalizeContextLength(DEFAULT_CONTEXT_LENGTH)).toBeNull();
    expect(normalizeContextLength(100)).toBe(512);
    expect(normalizeContextLength(1_000_000)).toBe(32768);
    expect(normalizeContextLength(8192)).toBe(8192);
  });
});
