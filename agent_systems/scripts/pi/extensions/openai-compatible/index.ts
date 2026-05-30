/**
 * pi extension: register a generic OpenAI-compatible API server as a provider.
 *
 * Provider-agnostic template. Point it at ANY server that speaks the OpenAI
 * /v1/chat/completions API (vLLM, llama.cpp, LM Studio, OpenRouter, a cloud
 * OpenAI-compatible gateway, ...) using environment variables only -- no
 * provider-specific values are baked in.
 *
 * Environment variables (exported by run_pi_openai_compatible.sh and forwarded
 * into the pi runtime):
 *   PI_OPENAI_BASE_URL        Base URL, e.g. http://localhost:8000/v1   (default)
 *   PI_OPENAI_MODEL           Model id served by the endpoint           (default gpt-4o-mini)
 *   PI_OPENAI_CONTEXT_WINDOW  Context window in tokens                  (default 128000)
 *   PI_OPENAI_MAX_TOKENS      Max output tokens                         (default 8192)
 *   OPENAI_COMPATIBLE_API_KEY Bearer token (loaded by run_pi.sh from
 *                             api_keys/.openai_compatible_key if not set)
 *
 * Select with: pi --provider openai-compatible --model openai-compatible/<PI_OPENAI_MODEL>
 *
 * Note: this registers the model as a reasoning model so `--thinking` is honored.
 * If your endpoint does not support reasoning, run with AGSYS_PI_THINKING=off.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BASE_URL = process.env.PI_OPENAI_BASE_URL ?? "http://localhost:8000/v1";
const MODEL_ID = process.env.PI_OPENAI_MODEL ?? "gpt-4o-mini";
const CONTEXT_WINDOW = Number(process.env.PI_OPENAI_CONTEXT_WINDOW ?? "128000");
const MAX_TOKENS = Number(process.env.PI_OPENAI_MAX_TOKENS ?? "8192");

export default function (pi: ExtensionAPI) {
  pi.registerProvider("openai-compatible", {
    name: "OpenAI-compatible API server",
    baseUrl: BASE_URL,
    apiKey: "OPENAI_COMPATIBLE_API_KEY",
    api: "openai",
    authHeader: true,
    models: [
      {
        id: MODEL_ID,
        name: MODEL_ID,
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: CONTEXT_WINDOW,
        maxTokens: MAX_TOKENS,
      },
    ],
  });
}
