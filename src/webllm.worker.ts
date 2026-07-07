import { WebWorkerMLCEngineHandler } from "@mlc-ai/web-llm";

// Route messages from the UI thread to WebLLM's worker-side engine.
// This keeps model loading and token generation off the main thread.
const handler = new WebWorkerMLCEngineHandler();

self.onmessage = (message: MessageEvent) => {
  handler.onmessage(message);
};
