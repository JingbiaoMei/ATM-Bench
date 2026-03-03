# HippoRAG 2 Baseline

HippoRAG 2 is a neurobiologically inspired memory framework that builds a knowledge graph
from text passages and retrieves with Personalized PageRank (PPR) plus recognition memory.
This baseline wraps the **original HippoRAG 2 implementation** for strict reproduction on
PersonalMemoryQA.

## Quickstart

```bash
# 1) Clone HippoRAG (required)
# git clone https://github.com/OSU-NLP-Group/HippoRAG ../HippoRAG

# 2) Install HippoRAG dependencies
# pip install python-igraph

# 3) Run a script (example)
bash scripts/QA_Agent/HippoRag2/run_hipporag2_allminilm_qwen3vl2b.sh
```

If HippoRAG is not cloned under `../HippoRAG`, pass `--hipporag-repo /path/to/HippoRAG`.

## What This Baseline Does

1. Converts multimodal evidence (emails, images, videos) into **text passages**.
2. Uses HippoRAG 2 for:
   - OpenIE (NER + triples)
   - Knowledge graph construction
   - Recognition memory (LLM triple filtering)
   - PPR-based retrieval
3. Answers questions using HippoRAG 2 QA prompts (text-only).

## Alignment With Original HippoRAG 2

| Aspect | Original HippoRAG 2 | Our Implementation | Aligned? |
|--------|---------------------|-------------------|----------|
| OpenIE prompts | NER + triple extraction | Original prompts via HippoRAG | ✅ |
| Graph + PPR | Entity + passage nodes | Original HippoRAG graph + PPR | ✅ |
| Recognition memory | DSPy filtering | Original DSPy filtering | ✅ |
| Embeddings | NV-Embed-v2 (paper) | NV-Embed-v2 / all-MiniLM / Qwen3 | ✅ |
| QA prompting | HippoRAG templates | Original HippoRAG prompts | ✅ |
| Multimodality | Text-only | Text-only (passage conversion) | ✅ |

## Raw Mode (Text-Only Memory)

Use `--media-source raw` to resolve raw media paths (images/videos) from `--image-root`
and `--video-root`. HippoRAG 2 remains **text-only**, so QA never consumes raw images.

## Key CLI Arguments

- `--hipporag-repo`: path to the cloned HippoRAG repo (if not in `../HippoRAG`)
- `--embedding-model`: `all-MiniLM-L6-v2`, `nvidia/NV-Embed-v2`, `Qwen/Qwen3-Embedding-0.6B`
- `--embedding-mode`: `auto` | `local` | `api` (see below)
- `--embedding-endpoint`: required for Qwen embeddings in API mode (OpenAI-compatible `/v1/embeddings`)
- `--embedding-device`: device for local embeddings (`cuda`, `cpu`, or auto-detect)
- `--provider`: `openai` | `vllm` | `vllm_local`
- `--model` / `--vllm-endpoint`: OpenIE + QA model and endpoint
- `--answerer-model` / `--answerer-endpoint`: override QA + rerank LLM while reusing the index cache from `--model`
- `--retrieval-top-k`, `--linking-top-k`, `--qa-top-k`: HippoRAG 2 hyperparameters
- `--openie-workers`: thread pool size for OpenIE (online mode)

## Embedding Modes

HippoRAG 2 supports three embedding modes:

| Mode | Description | When to Use |
|------|-------------|-------------|
| `auto` (default) | Auto-detect based on model and endpoint | General use |
| `local` | Run embeddings locally using HuggingFace | **No embedding endpoint required** |
| `api` | Use VLLM/OpenAI-compatible endpoint | When you have an embedding server |

### Local Embedding Mode (Recommended for Qwen)

Use `--embedding-mode local` to run Qwen embeddings directly on your device:

```bash
python memqa/qa_agent_baselines/HippoRag2/hipporag2_baseline.py \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-mode local \
  --embedding-device cuda \
  ...
```

This is equivalent to how MMRAG baselines run embeddings locally. No embedding endpoint is required.

### API Embedding Mode

Use `--embedding-mode api` with an endpoint:

```bash
python memqa/qa_agent_baselines/HippoRag2/hipporag2_baseline.py \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-mode api \
  --embedding-endpoint http://localhost:8000/v1/embeddings \
  ...
```

## Output Structure

```
output/QA_Agent/HippoRag2/{method_name}/
├── hipporag2_answers.jsonl
├── retrieval_recall_details.json
├── retrieval_recall_summary.json
└── eval/

output/QA_Agent/HippoRag2/index_cache/{cache_key}/{llm_label}_{embedding_label}/
├── graph.pickle
├── chunk_embeddings/
├── entity_embeddings/
├── fact_embeddings/
└── llm_cache/
```

## Evaluation

```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth memqa/utils/final_data_processing/atm-20260121.json \
  --predictions output/QA_Agent/HippoRag2/<method>/hipporag2_answers.jsonl \
  --output-dir output/QA_Agent/HippoRag2/<method>/eval \
  --metrics atm \
  --judge-provider vllm \
  --judge-model "GLM-4.7" \
  --judge-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --max-workers 2
```

## References

- **Paper**: https://arxiv.org/abs/2502.14802
- **Code**: https://github.com/OSU-NLP-Group/HippoRAG
