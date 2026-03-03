# HippoRAG 2 Changelog

## 2026-01-21 - Strict Reproduction Update

### ✅ Major Changes
- Switched to the **original HippoRAG 2 implementation** for OpenIE, graph/PPR retrieval, and recognition memory.
- Added support for raw media paths (`--media-source raw`) while keeping QA text-only.
- Standardized outputs to `hipporag2_answers.jsonl` + retrieval recall details/summary.
- Updated run scripts and docs to match the new CLI and evaluation flow.

### 🔧 Cache Behavior
- Cache key now hashes embedding model, LLM model, media source, augmentation level, and corpus signature.
- HippoRAG internal caches live under:
  `output/QA_Agent/HippoRag2/index_cache/{cache_key}/{llm_label}_{embedding_label}/`

### ⚠️ Notes
- Qwen embeddings require `--embedding-endpoint` (OpenAI-compatible `/v1/embeddings`).
- The baseline requires the HippoRAG repo (default: `../HippoRAG`).
