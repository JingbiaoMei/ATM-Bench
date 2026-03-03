# HippoRAG 2 Implementation Summary

**Date**: 2026-01-21  
**Status**: ✅ Updated for strict reproduction  
**Location**: `memqa/qa_agent_baselines/HippoRag2/`

## Overview

The HippoRAG 2 baseline now **wraps the original HippoRAG 2 codebase** for strict
reproduction (OpenIE → KG → recognition memory → PPR retrieval → QA). The adapter
converts PersonalMemoryQA evidence into text passages and aligns outputs with the
baseline conventions in `memqa/qa_agent_baselines/AGENTS.md`.

## Core Components

### 1) Wrapper Pipeline (`hipporag2_baseline.py`)
- Imports the original HippoRAG repo via `--hipporag-repo` (default: `../HippoRAG`).
- Builds a HippoRAG index from converted passages.
- Runs retrieval + QA using HippoRAG prompts.
- Emits JSONL predictions and retrieval recall details.

### 2) Data Adapter (`data_adapter.py`)
- Converts emails/images/videos into text passages.
- Ensures **id / timestamp / location** are always present.
- Supports `--media-source raw` to resolve raw media paths.

### 3) Config (`config.py`)
- Paper hyperparameters (Table 5) set as defaults.
- Embedding choices: NV-Embed-v2, all-MiniLM, Qwen3-Embedding.
- Adds optional `hipporag_repo` and embedding endpoint settings.

## Output Artifacts

```
output/QA_Agent/HippoRag2/{method_name}/
├── hipporag2_answers.jsonl
├── retrieval_recall_details.json
├── retrieval_recall_summary.json
└── eval/
```

HippoRAG internal caches live under:
```
output/QA_Agent/HippoRag2/index_cache/{cache_key}/{llm_label}_{embedding_label}/
```

## Alignment Notes

✅ Uses original HippoRAG OpenIE prompts and DSPy-based recognition memory.  
✅ Uses original PPR graph search.  
✅ QA prompts match HippoRAG templates (text-only).  
✅ Evidence ID mapping matches `atm-20260121.json` (stem IDs).  

## Known Constraints

- HippoRAG 2 is text-only; raw media is resolved for indexing metadata only.
- Qwen embeddings require `--embedding-endpoint` (OpenAI-compatible embeddings API).
