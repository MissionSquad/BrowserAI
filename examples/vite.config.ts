import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Resolve the SDK to its local TypeScript source (../src/index.ts) instead of a published npm
// package. This mirrors ../vitest.config.ts, so the examples exercise the exact code in this repo.
// Vite compiles the .ts on the fly and rewrites the SDK's `.js` relative imports to their `.ts`
// files automatically, so no `npm run build` in the parent is required to run these examples.
const sdkEntry = fileURLToPath(new URL("../src/index.ts", import.meta.url));
const repoRoot = fileURLToPath(new URL("..", import.meta.url));

const page = (name: string): string => fileURLToPath(new URL(`./${name}`, import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      "@missionsquad/browserai": sdkEntry,
    },
  },
  server: {
    // The SDK source and its node_modules live one level up (../src, ../node_modules). Vite's dev
    // server refuses to serve files outside its root by default — allow the repo root explicitly.
    fs: { allow: [repoRoot] },
  },
  // @mlc-ai/web-llm and @huggingface/transformers are large and ship their own runtime/wasm loaders.
  // Excluding them from Vite's dependency pre-bundling avoids esbuild choking on their conditional
  // Node-only imports; they are dynamically imported by the SDK only when a model actually loads.
  optimizeDeps: {
    exclude: ["@mlc-ai/web-llm", "@huggingface/transformers"],
  },
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        index: page("index.html"),
        text: page("text-generation.html"),
        json: page("json-extraction.html"),
        speech: page("speech.html"),
        voice: page("voice-assistant.html"),
        vision: page("vision.html"),
      },
    },
  },
});
