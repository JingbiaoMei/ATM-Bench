# Oracle Script Notes

These wrapper scripts run the Oracle baseline with different answerer models.

For the Claude and Gemini variants, we intentionally route requests through an OpenAI-compatible endpoint (`--provider vllm` with `VLLM_ENDPOINT`) instead of adding separate Anthropic or Google-specific inference code to the Oracle pipeline.

This keeps the Oracle inference path unified:
- one request format
- one baseline implementation
- one evaluation flow

In practice, we use the CLIProxy to convert Anthropic/Gemini API calls to OpenAI-compatible calls, allowing us to reuse the same Oracle code for all models. You may also use openrouter which provides native openai-compatible endpoints for multiple models.

Scripts in this folder that follow this pattern include:
- `run_oracle_opus.sh`
- `run_oracle_sonnet.sh`
- `run_oracle_gemini25.sh`
