# MMRag Baselines

This directory contains the MMRag retrieval baselines:
- **Retrieve → Answer**: `mmrag_retrieve_answer.py`
- **Retrieve → Rerank → Answer**: `mmrag_retrieve_rerank_answer.py`

Both baselines share the retrieval/reranking modules in `memqa/retrieve/` and emit recall metrics alongside QA outputs.

**For general QA baseline conventions** (I/O formats, CLI argument naming, evaluation standards), see `memqa/qa_agent_baselines/AGENTS.md`.

## Supported Retrievers

| Retriever | Status | Notes |
|-----------|--------|-------|
| `qwen3_vl_embedding` | **Active** | Qwen3-VL-Embedding-2B, unified text+vision encoder |
| `text` | **Active** | Qwen3-Embedding (0.6B/4B), text-only |
| `vista` | **Active** | BGE-M3 + visual adapter |
| `sentence_transformer` | **Active** | all-MiniLM-L6-v2 (mean pooling) |
| `clip` | **Abandoned** | See below |

### Why CLIP was abandoned

CLIP retrieval has a fundamental architectural limitation for this task:

1. **Query encoding issue**: Queries are text-only, but CLIP uses separate text/image encoders producing different embedding spaces. Our implementation concatenates `[text_embed, text_embed]` for queries, but documents use `[text_embed, image_embed]`.

2. **Cross-modal dot product is meaningless**: When computing `query · image_doc`, the second component becomes `text_embed · image_embed`, which compares embeddings from different spaces (text encoder vs image encoder).

3. **Result**: CLIP retrieval degenerates to text-only matching, strongly preferring emails (99.6%) over images (0.4%) regardless of query content. With `--no-vl-text-augment`, CLIP achieves R@100=1.2% vs Qwen3-VL's 12.6%.

For fair multimodal retrieval, use `qwen3_vl_embedding` which processes text+image through a unified encoder, producing comparable embeddings for both queries and documents.

## Architecture & Design Decisions

### Two-Stage Retrieval+Rerank Workflow

**Problem**: Loading both retriever and reranker models simultaneously causes GPU OOM on single-GPU machines (e.g., Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B require >24GB VRAM combined).

**Solution**: `mmrag_retrieve_rerank_answer.py` supports staged execution via `--stage` argument:

```bash
# Stage 1: Retrieval (loads retriever, saves intermediate results, unloads)
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_rerank_answer.py \
    --stage retrieve \
    --qa-file memqa/utils/final_data_processing/atm-20260121.json \
    --retriever qwen3_vl_embedding \
    --method-name my_experiment \
    --media-source raw \
    --image-root data/raw_memory/image \
    --video-root data/raw_memory/video \
    --email-file data/raw_memory/email/merged_emails.json

# Stage 2: Reranking (loads intermediate results, reranks, generates answers)
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_rerank_answer.py \
    --stage rerank \
    --qa-file memqa/utils/final_data_processing/atm-20260121.json \
    --reranker qwen3_vl_reranker \
    --method-name my_experiment \
    --provider vllm \
    --vllm-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
    --model "Qwen/Qwen3-VL-8B-Instruct" \
    --max-workers 128

# Or run both stages sequentially (backward compatible, default behavior)
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_rerank_answer.py \
    --stage all \
    --qa-file ... \
    --method-name my_experiment
```

**Intermediate files** (saved to `{method_dir}/intermediate/`):
- `retrieval_results.json`: QA IDs → retrieved item IDs + scores (lightweight)
- `retrieval_items.json`: Full `RetrievalItem` objects for rerank stage reconstruction (includes all metadata)
- `retrieval_latency.json`: Retrieval timing breakdown (model load, index build, per-query time)
- `rerank_latency.json`: Reranking timing breakdown (model load, per-query rerank time)

**Key implementation details**:
1. **GPU cleanup**: Use `del model; gc.collect(); torch.cuda.empty_cache()` between stages
2. **Item serialization**: `RetrievalItem.to_dict()` and `RetrievalItem.from_dict()` for intermediate storage
3. **Timing**: Use `time.perf_counter()` for high-precision latency measurement

**Research value**: 
- Enables latency decomposition for ablation studies (retrieval vs rerank contribution)
- Fair comparison across methods with different memory footprints
- Supports single-GPU workflows for large models

**See**: Shell scripts in `scripts/QA_Agent/MMRag/qwen3_vl_embedding/*_answerer_rerank.sh` and `scripts/QA_Agent/MMRag/qwen3_embedding/*_answerer_rerank.sh` for production usage examples.

---

### Text Augmentation Strategies

MMRag baselines support two augmentation modes depending on retrieval method:

#### Text-Only Retrieval (Per-Field Control)
**Retrievers**: `text` (Qwen3-Embedding), `sentence_transformer` (all-MiniLM)  
**Mode**: `--media-source batch_results`

**Per-field toggles** (default: all enabled):
- `--include-id` / `--include-type` / `--include-timestamp` / `--include-location` (essential metadata)
- `--include-short-caption` / `--include-caption` (image/video captions)
- `--include-ocr-text` (extracted text from images/videos)
- `--include-tags` (semantic tags)
- `--include-email-summary` / `--include-email-detail` (email content levels)

**4-Level Incremental Hierarchy** (for ablation studies):

| Level | Fields | Use Case |
|-------|--------|----------|
| **short_caption_only** | id, timestamp, location, short_caption | Minimal metadata + brief caption |
| **short_caption_caption** | + caption | Add full detailed caption |
| **short_caption_caption_tag** | + tags | Add semantic tags |
| **short_caption_caption_tag_ocr** | + ocr_text | Full augmentation (all available text) |

**Example** (Level 3: short_caption_caption_tag):
```bash
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
  --retriever text \
  --media-source batch_results \
  --include-id --include-timestamp --include-location \
  --include-short-caption --include-caption --include-tags \
  --no-include-ocr-text  # Explicitly disable OCR for this ablation
```

**Design rationale**:
- **Incremental**: Each level adds exactly one component, isolating its contribution
- **Essential metadata always included**: `id`, `timestamp`, `location` present at all levels
- **Research-friendly**: Clean ablation studies (e.g., "Does OCR help?" = compare level 3 vs level 4)

#### Vision-Language Retrieval (All-or-Nothing)
**Retrievers**: `qwen3_vl_embedding` (unified text+vision encoder), `vista`, `clip`  
**Mode**: `--media-source raw` (direct image/video processing)

**Augmentation control**:
- `--vl-text-augment`: Include ALL text metadata fields (id, timestamp, location, caption, tags, OCR) alongside image
- `--no-vl-text-augment`: Image-only (no text metadata), essential fields still included internally

**Note**: VL retrievers use all-or-nothing augmentation because they process text+image through a unified encoder. Per-field control would require re-encoding the entire corpus for each ablation, which is computationally prohibitive.

**Example** (VL with text augmentation):
```bash
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
  --retriever qwen3_vl_embedding \
  --media-source raw \
  --vl-text-augment  # Include all text metadata with images
```

---

### Output Directory Structure

MMRag baselines organize outputs by retriever family, then by specific configuration:

```
output/QA_Agent/MMRag/{retriever_family}/{retriever_variant}[_{reranker_variant}]/{answerer}/{method}/
```

**Path components**:
- `{retriever_family}`: Top-level category (e.g., `qwen3_vl_embedding`, `qwen3_embedding`, `allMiniLM_embedding`)
- `{retriever_variant}`: Specific model version (e.g., `qwen3_vl_embedding_2b`, `qwen3_embedding_0.6b`)
- `{reranker_variant}`: Optional reranker suffix (e.g., `_qwen3_vl_reranker_2b`, `_qwen3_reranker_4b`)
- `{answerer}`: LLM used for answer generation (e.g., `qwen3vl2b_answerer`, `glm4v_answerer`)
- `{method}`: Experiment/ablation method name (e.g., `short_caption_only`, `short_caption_caption_tag_ocr`)

**Examples**:

```bash
# VL Embedding + VL Reranker (full augmentation)
output/QA_Agent/MMRag/qwen3_vl_embedding/qwen3_vl_embedding_2b_qwen3_vl_reranker_2b/qwen3vl2b_answerer/short_caption_caption_tag_ocr/
├── mmrag_answers.jsonl
├── retrieval_recall_summary.json
├── retrieval_recall_details.json
├── intermediate/
│   ├── retrieval_results.json
│   ├── retrieval_items.json
│   ├── retrieval_latency.json
│   └── rerank_latency.json
└── eval/
    ├── em_results.json
    └── llm_judge_results.json

# Text Embedding (0.6B) + Text Reranker (4B)
output/QA_Agent/MMRag/qwen3_embedding/qwen3_embedding_0.6b_qwen3_reranker_4b/qwen3vl2b_answerer/short_caption_caption_tag/
├── mmrag_answers.jsonl
├── retrieval_recall_summary.json
├── retrieval_recall_details.json
├── intermediate/
│   └── ...
└── eval/

# Text-only retrieval (all-MiniLM, no reranker)
output/QA_Agent/MMRag/allMiniLM_embedding/allMiniLM_L6v2/qwen3vl2b_answerer/short_caption_only/
├── mmrag_answers.jsonl
├── retrieval_recall_summary.json
├── retrieval_recall_details.json
├── retrieval_latency.json
└── eval/
```

**File descriptions**:
- `mmrag_answers.jsonl`: QA predictions (`{"id": "...", "answer": "..."}`)
- `retrieval_recall_summary.json`: Aggregated R@K metrics (R@1, R@5, R@10, R@25, R@50, R@100)
- `retrieval_recall_details.json`: Per-question recall breakdown for error analysis
- `retrieval_latency.json`: Timing breakdown (model load, index build, per-query retrieval)
- `rerank_latency.json`: Reranker timing (only for retrieve+rerank workflows)
- `eval/`: Evaluation results (EM and LLM judge scores)

**Path configuration in shell scripts**:
```bash
# Retrieve-only workflow
OUTPUT_DIR="output/QA_Agent/MMRag/${RETRIEVER_FAMILY}/${RETRIEVER_VARIANT}/${ANSWERER_NAME}/${METHOD_NAME}"

# Retrieve+rerank workflow
OUTPUT_DIR="output/QA_Agent/MMRag/${RETRIEVER_FAMILY}/${RETRIEVER_VARIANT}_${RERANKER_VARIANT}/${ANSWERER_NAME}/${METHOD_NAME}"
```

**See**: `scripts/QA_Agent/MMRag/qwen3_vl_embedding/qwen3vl_2b_answerer.sh` for complete path construction examples.

---

### Retrieval Module Organization

**Module location**: Retrieval and reranking components live in `memqa/retrieve/` for reuse across different QA agents.

**Core modules**:
- `memqa/retrieve/retrievers.py`: Retriever implementations (`Qwen3VLRetriever`, `TextRetriever`, `VistaRetriever`, `CLIPRetriever`)
- `memqa/retrieve/rerankers.py`: Reranker implementations (`Qwen3VLReranker`, `Qwen3Reranker`)
- `memqa/retrieve/retrieval_item.py`: `RetrievalItem` dataclass (unified representation for emails, images, videos)

**Supported models**:

| Component | Models | Status |
|-----------|--------|--------|
| **Text Retrievers** | Qwen3-Embedding (0.6B/4B), all-MiniLM-L6-v2 | Active |
| **VL Retrievers** | Qwen3-VL-Embedding-2B, VISTA (BGE-M3 + visual adapter) | Active |
| **Abandoned** | CLIP (architectural mismatch) | Deprecated |
| **Text Rerankers** | Qwen3-Reranker-2B/4B | Active |
| **VL Rerankers** | Qwen3-VL-Reranker-2B | Active |

**CLI arguments** (for retrieval/reranking):
- `--retriever`: `qwen3_vl_embedding`, `vista`, `clip`, `text`, `sentence_transformer`
- `--reranker`: `qwen3_vl_reranker`, `qwen3_reranker`, `text`, `noop`
- `--text-embedding-model`: Model ID (e.g., `Qwen/Qwen3-Embedding-0.6B`, `all-MiniLM-L6-v2`)
- `--vl-embedding-model`: Model ID (e.g., `Qwen/Qwen3-VL-Embedding-2B`)
- `--clip-model`: Model ID (e.g., `openai/clip-vit-large-patch14`)
- `--vista-model-name`: Base model (e.g., `BAAI/bge-m3`)
- `--vista-weights`: Path to VISTA adapter weights (`.pth` file)
- `--text-reranker-model`: Model ID (e.g., `Qwen/Qwen3-Reranker-4B`)
- `--vl-reranker-model`: Model ID (e.g., `Qwen/Qwen3-VL-Reranker-2B`)

**Index caching**:
- `--index-cache`: Cache directory (default: `output/retrieval/index_cache`)
- `--force-rebuild`: Re-embed all items and overwrite cache
- Cache keys: `{model_name}_{text_config_hash}_{data_mtime}` (automatically invalidates on data changes)

**Retrieval results cache**:
- Enabled by default with `--reuse-retrieval-results` (disable with `--no-reuse-retrieval-results`).
- Stores per-QA retrieval outputs and item snapshots so you can reuse retrieval for new answerers.
- Cache keys include retriever config, `retrieval_max_k`, and QA file mtime.
- Paths (under `--index-cache`):
  - `retrieval_results/<cache_key>.json`
  - `retrieval_items/<cache_key>.json`
- On hit, retrieval is skipped and cached results are used for answer generation and reranking.

**Recall logging**:
- Automatic R@K computation using ground-truth `evidence_ids` from QA JSON
- Summary (`retrieval_recall_summary.json`) + per-question details (`retrieval_recall_details.json`)
- Supports K ∈ {1, 5, 10, 25, 50, 100}

---

### Performance Metrics

#### Latency Tracking

Both `mmrag_retrieve_answer.py` and `mmrag_retrieve_rerank_answer.py` emit detailed latency metrics for reproducible performance analysis.

**Retrieval latency** (`retrieval_latency.json`):
```json
{
  "model_load_ms": 1234.5,
  "index_build_ms": 567.8,
  "retrieval_total_ms": 45000.0,
  "retrieval_per_query_avg_ms": 150.0,
  "num_queries": 300
}
```

**Rerank latency** (`rerank_latency.json`, only for retrieve+rerank workflows):
```json
{
  "model_load_ms": 2345.6,
  "rerank_total_ms": 30000.0,
  "per_query_avg_ms": 100.0,
  "num_queries": 300
}
```

**Measurement points**:
1. **Model load**: Time from import to model ready (includes weight loading, device transfer)
2. **Index build**: Time to encode all items (or load from cache if available)
3. **Per-query retrieval**: Individual query encoding + similarity computation (averaged across all queries)
4. **Per-query rerank**: Candidate reranking on top-K from retrieval (averaged)

**Research guidance**:
- Report **cold start** (first run, no cache) vs **warm start** (cache hit) separately
- For paper benchmarks, use **warm start** to isolate model performance from I/O
- Include hardware specs in experiment logs: GPU model (e.g., A100 80GB), VRAM, CPU
- Latency enables fair cross-method comparison: "Method A is 2x slower but achieves 10% higher R@10"

**Location**: Latency files are saved in:
- Retrieve-only: `{method_dir}/retrieval_latency.json`
- Retrieve+rerank: `{method_dir}/intermediate/retrieval_latency.json` and `{method_dir}/intermediate/rerank_latency.json`

#### Recall Metrics

All MMRag baselines emit structured recall logs for reproducible retrieval evaluation.

**Output files**:
- `retrieval_recall_summary.json`: Aggregated R@K metrics across all QAs
- `retrieval_recall_details.json`: Per-question breakdown (QA ID, ground-truth IDs, retrieved IDs, R@K values)

**Recall@K definition**:
```
R@K = (# ground-truth evidence items in top-K retrieved) / (# total ground-truth items for this QA)
```

Reported for K ∈ {1, 5, 10, 25, 50, 100} to capture retrieval behavior at different cutoffs.

**Example summary**:
```json
{
  "R@1": 0.123,
  "R@5": 0.456,
  "R@10": 0.589,
  "R@25": 0.712,
  "R@50": 0.834,
  "R@100": 0.901,
  "num_questions": 300,
  "total_ground_truth_items": 450
}
```

**Example per-question detail**:
```json
{
  "qa_id": "qa_001",
  "ground_truth_ids": ["IMG_1234.jpg", "IMG_5678.jpg"],
  "retrieved_ids_top_10": ["IMG_1234.jpg", "IMG_9999.jpg", "email_42", ...],
  "R@1": 0.5,
  "R@5": 0.5,
  "R@10": 0.5,
  "found_at_ranks": [1]  // IMG_1234 found at rank 1, IMG_5678 not in top-10
}
```

**Research best practices**:
- **Always report recall alongside answer accuracy**: Answer accuracy alone is insufficient (model might hallucinate correct answers without using retrieved evidence)
- **Analyze failure modes**: Use `retrieval_recall_details.json` to identify patterns:
  - Temporal queries (e.g., "photos from last summer") vs spatial queries (e.g., "photos in Paris")
  - Modality biases (e.g., preferring emails over images)
  - Query types that fail retrieval but succeed in QA (indicates memorization/hallucination)
- **Staged evaluation**: Compare retrieval → rerank improvement:
  - If R@10 improves significantly after reranking: reranker is effective
  - If answer accuracy improves but R@K unchanged: reranker helps select better evidence from retrieved candidates
- **Cross-method comparison**: Normalize by R@K for fair comparison:
  - Example: "Method A achieves 85% QA accuracy at R@10=0.6, Method B achieves 82% at R@10=0.8"
  - This reveals whether gains come from better retrieval or better reasoning

**See**: General best practices in `memqa/qa_agent_baselines/AGENTS.md` under "Performance Metrics & Best Practices".

---

## Dependencies
- `transformers`, `torch`, `qwen-vl-utils`, `scipy`
- **VISTA**: install FlagEmbedding visual_bge (`pip install -e FlagEmbedding/research/visual_bge`) plus `timm`, `einops`, `ftfy`, `torchvision`.

## Outputs
- **Predictions**: JSONL at `--output-dir-base/<retriever>/<method>/mmrag_answers.jsonl` with `{"id","answer"}`.
- **Recall logs**: written next to predictions as:
  - `retrieval_recall_summary.json`
  - `retrieval_recall_details.json`

Recall is reported as `R@1/5/10/25/50/100` using gold evidence IDs.

## Example Commands
### Retrieve → Answer (text-only, batch_results)
```bash
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --media-source batch_results \
  --image-batch-results output/image/qwen3vl2b/batch_results.json \
  --video-batch-results output/video/qwen3vl2b/batch_results.json \
  --email-file data/raw_memory/email/merged_emails.json \
  --retriever text \
  --text-embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --provider vllm \
  --vllm-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --max-workers 128 \
  --timeout 120 \
  --output-dir-base output/QA_Agent/MMRag \
  --method-name text_batch_minimal
```

### Retrieve → Answer (text-only, all-MiniLM)
```bash
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --media-source batch_results \
  --image-batch-results output/image/qwen3vl2b/batch_results.json \
  --video-batch-results output/video/qwen3vl2b/batch_results.json \
  --email-file data/raw_memory/email/merged_emails.json \
  --retriever sentence_transformer \
  --text-embedding-model "all-MiniLM-L6-v2" \
  --provider vllm \
  --vllm-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --max-workers 128 \
  --timeout 120 \
  --output-dir-base output/QA_Agent/MMRag \
  --method-name allMiniLM_text_batch_rich
```

### Retrieve → Answer (Qwen3-VL, raw)
```bash
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --media-source raw \
  --image-root data/raw_memory/image \
  --video-root data/raw_memory/video \
  --email-file data/raw_memory/email/merged_emails.json \
  --retriever qwen3_vl_embedding \
  --vl-embedding-model "Qwen/Qwen3-VL-Embedding-2B" \
  --provider vllm \
  --vllm-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --max-workers 128 \
  --timeout 120 \
  --output-dir-base output/QA_Agent/MMRag \
  --method-name qwen3vl_raw_with_augment
```

### Retrieve → Rerank → Answer (Qwen3-VL, raw)
```bash
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_rerank_answer.py \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --media-source raw \
  --image-root data/raw_memory/image \
  --video-root data/raw_memory/video \
  --email-file data/raw_memory/email/merged_emails.json \
  --retriever qwen3_vl_embedding \
  --vl-embedding-model "Qwen/Qwen3-VL-Embedding-2B" \
  --reranker qwen3_vl_reranker \
  --vl-reranker-model "Qwen/Qwen3-VL-Reranker-2B" \
  --provider vllm \
  --vllm-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --max-workers 128 \
  --timeout 120 \
  --output-dir-base output/QA_Agent/MMRag \
  --method-name qwen3vl_rerank_rich
```

### Retrieve → Answer (CLIP, raw)
```bash
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --media-source raw \
  --image-root data/raw_memory/image \
  --video-root data/raw_memory/video \
  --email-file data/raw_memory/email/merged_emails.json \
  --retriever clip \
  --clip-model "openai/clip-vit-large-patch14" \
  --provider vllm \
  --vllm-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --max-workers 128 \
  --timeout 120 \
  --output-dir-base output/QA_Agent/MMRag \
  --method-name clip_retrieve_rich
```

### Retrieve → Rerank → Answer (CLIP + Qwen3-VL-Reranker, raw)
```bash
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_rerank_answer.py \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --media-source raw \
  --image-root data/raw_memory/image \
  --video-root data/raw_memory/video \
  --email-file data/raw_memory/email/merged_emails.json \
  --retriever clip \
  --clip-model "openai/clip-vit-large-patch14" \
  --reranker qwen3_vl_reranker \
  --vl-reranker-model "Qwen/Qwen3-VL-Reranker-2B" \
  --provider vllm \
  --vllm-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --max-workers 128 \
  --timeout 120 \
  --output-dir-base output/QA_Agent/MMRag \
  --method-name clip_rerank_rich
```

### Retrieve → Answer (VISTA, raw)
```bash
VISTA_WEIGHTS=data/Visualized_base_en_v1.5.pth \
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --media-source raw \
  --image-root data/raw_memory/image \
  --video-root data/raw_memory/video \
  --email-file data/raw_memory/email/merged_emails.json \
  --retriever vista \
  --vista-model-name "BAAI/bge-m3" \
  --vista-weights "${VISTA_WEIGHTS}" \
  --provider vllm \
  --vllm-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --max-workers 128 \
  --timeout 120 \
  --output-dir-base output/QA_Agent/MMRag \
  --method-name vista_retrieve_rich
```

### Retrieve → Rerank → Answer (text-only, batch_results)
```bash
python memqa/qa_agent_baselines/MMRag/mmrag_retrieve_rerank_answer.py \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --media-source batch_results \
  --image-batch-results output/image/qwen3vl2b/batch_results.json \
  --video-batch-results output/video/qwen3vl2b/batch_results.json \
  --email-file data/raw_memory/email/merged_emails.json \
  --retriever text \
  --text-embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --reranker qwen3_reranker \
  --text-reranker-model "Qwen/Qwen3-Reranker-2B" \
  --provider vllm \
  --vllm-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --max-workers 128 \
  --timeout 120 \
  --output-dir-base output/QA_Agent/MMRag \
  --method-name text_batch_rerank_rich
```

## CLI Flags
### Shared QA + evidence
- `--qa-file`: QA JSON input.
- `--media-source`: `batch_results` or `raw`.
- `--image-batch-results`: image batch results JSON (batch mode).
- `--video-batch-results`: video batch results JSON (batch mode).
- `--image-root`: raw image root directory.
- `--video-root`: raw video root directory.
- `--email-file`: merged email JSON.
- `--no-evidence`: run without evidence.
- `--max-evidence-items`: cap evidence items per question.
- `--num-frames`: number of frames per video (raw mode, Qwen3‑VL).
- `--frame-strategy`: frame sampling strategy for answer generation.
- `--output-dir-base`: base output directory for `<retriever>/<method>/`.
- `--method-name`: subfolder name under the retriever output.

### LLM provider
- `--provider`: `openai`, `vllm`, or `vllm_local`.
- `--model`: model name override.
- `--api-key`: API key override.
- `--vllm-endpoint`: OpenAI-compatible endpoint for vLLM.
- `--max-tokens`: max tokens for completion.
- `--temperature`: decoding temperature.
- `--timeout`: request timeout in seconds.
- `--max-workers`: concurrency for API calls (vllm_local is single-threaded).

### Retrieval
- `--retriever`: `qwen3_vl_embedding` (default), `vista`, `clip`, `text`.
- `--retriever-batch-size`: batch size for embedding encoding.
- `--retrieval-top-k`: top‑K evidence for retrieve→answer only.
- `--retrieval-max-k`: max candidates for recall logging.
- `--index-cache`: cache directory (default `output/retrieval/index_cache`).
- `--force-rebuild`: re-embed and overwrite cache.
- `--text-embedding-model`: text embedding model ID (e.g., `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-4B`).
- `--vl-embedding-model`: Qwen3‑VL embedding model ID.
- `--clip-model`: CLIP model ID.
- `--vista-model-name`: base BGE model name (e.g., `BAAI/bge-m3`).
- `--vista-weights`: required path to VISTA weight file (`.pth`, default in scripts: `data/Visualized_base_en_v1.5.pth`).

### Reranking (retrieve‑rerank)
- `--reranker`: `qwen3_vl_reranker` (default), `qwen3_reranker`, `text`, `noop`.
- `--reranker-batch-size`: reranker batch size.
- `--rerank-input-k`: candidates passed to reranker.
- `--rerank-top-k`: final evidence after rerank.
- `--text-reranker-model`: text reranker model ID.
- `--vl-reranker-model`: Qwen3‑VL reranker model ID.

### Text augmentation
- `--vl-text-augment` / `--no-vl-text-augment`: include text metadata with VL embeddings.
- `--include-id/--include-type/--include-timestamp/--include-location`
- `--include-short-caption/--include-caption/--include-ocr-text/--include-tags`
- `--include-email-summary/--include-email-detail`
- `--critic-answerer`: enable draft + critic verification (text-only critic; raw multimodal skips critic). (Deprecated aliases: `--agentic-answer`, `--agentic-answser`)

## Evaluation
Use the evaluator after inference for EM and vLLM judge scoring:
```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth memqa/utils/final_data_processing/atm-20260121.json \
  --predictions output/QA_Agent/MMRag/<retriever>/<method>/mmrag_answers.jsonl \
  --output-dir output/QA_Agent/MMRag/<retriever>/<method>/eval \
  --metrics em

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth memqa/utils/final_data_processing/atm-20260121.json \
  --predictions output/QA_Agent/MMRag/<retriever>/<method>/mmrag_answers.jsonl \
  --output-dir output/QA_Agent/MMRag/<retriever>/<method>/eval \
  --metrics llm \
  --judge-provider vllm \
  --judge-model "Qwen/Qwen3-VL-8B-Instruct" \
  --judge-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --max-workers 128
```

## Cache Notes
Embedding caches are keyed by model + text config + data mtimes. Delete files under `output/retrieval/index_cache` or pass `--force-rebuild` to regenerate.

Retrieval results are cached separately under `output/retrieval/index_cache/retrieval_results/` and `output/retrieval/index_cache/retrieval_items/`. These caches are keyed by retrieval config + QA file mtime, so you can reuse retrieval results across different answerers (e.g., `qwen3vl2b_answerer` → `qwen3vl8b_answerer`) without re-encoding.

## Ablation Study Experiments

### V3 Ablation (Current)

Scripts location: `scripts/QA_Agent/MMRag/v3_ablation/`

This ablation compares retrieval methods with standardized settings:
- Text augmentation: `id + timestamp + location + short_caption` (minimal)
- Top-K configurable via `TOP_K` environment variable (5 or 10)
- Dual judge evaluation: GLM-4.7 + GPT-5-mini
- Answerer models: Qwen3-VL-2B-Instruct and Qwen3-VL-8B-Instruct-FP8

#### Configuration Matrix

| Category | Embedding | Reranker | 2B Answerer | 8B Answerer |
|----------|-----------|----------|:-----------:|:-----------:|
| **Text (no rerank)** | all-MiniLM-L6-v2 | - | ✓ | ✓ |
| | Qwen3-Embedding-0.6B | - | ✓ | ✓ |
| | Qwen3-Embedding-4B | - | - | ✓ |
| **Text + Rerank** | all-MiniLM-L6-v2 | Qwen3-Reranker-0.6B | ✓ | ✓ |
| | Qwen3-Embedding-0.6B | Qwen3-Reranker-0.6B | ✓ | ✓ |
| | Qwen3-Embedding-0.6B | Qwen3-Reranker-4B | - | ✓ |
| **VL Embed** | Qwen3-VL-Embedding-2B | withtext | ✓ | ✓ |
| | Qwen3-VL-Embedding-2B | notext | ✓ | ✓ |
| | Qwen3-VL-Embedding-8B | withtext | - | ✓ |
| | Qwen3-VL-Embedding-8B | notext | - | ✓ |

#### Top-K Settings

| TOP_K | Retrieve-only | Retrieve+Rerank |
|-------|---------------|-----------------|
| 5 | `--retrieval-top-k 5` | `--rerank-input-k 10 --rerank-top-k 5` |
| 10 | `--retrieval-top-k 10` | `--rerank-input-k 20 --rerank-top-k 10` |

#### Usage

```bash
# Run all 2B answerer experiments with topk=10 (default)
bash scripts/QA_Agent/MMRag/v3_ablation/job_qwen3vl2b.sh

# Run all 8B answerer experiments with topk=10
bash scripts/QA_Agent/MMRag/v3_ablation/job_qwen3vl8b.sh

# Run with topk=5
TOP_K=5 bash scripts/QA_Agent/MMRag/v3_ablation/job_qwen3vl2b.sh
TOP_K=5 bash scripts/QA_Agent/MMRag/v3_ablation/job_qwen3vl8b.sh

# Run individual script
TOP_K=5 bash scripts/QA_Agent/MMRag/v3_ablation/text_embed/allminilm_l6_qwen3vl2b.sh
```

#### Script Structure

```
scripts/QA_Agent/MMRag/v3_ablation/
├── common.sh                    # Shared configs (TOP_K, endpoints, models)
├── job_qwen3vl2b.sh             # Run all 2B answerer configs
├── job_qwen3vl8b.sh             # Run all 8B answerer configs
├── text_embed/                  # Text embedding without rerank
│   ├── allminilm_l6_qwen3vl2b.sh
│   ├── allminilm_l6_qwen3vl8b.sh
│   ├── qwen3emb_0.6b_qwen3vl2b.sh
│   ├── qwen3emb_0.6b_qwen3vl8b.sh
│   └── qwen3emb_4b_qwen3vl8b.sh
├── text_embed_rerank/           # Text embedding with rerank
│   ├── allminilm_l6_qwen3rerank0.6b_qwen3vl2b.sh
│   ├── allminilm_l6_qwen3rerank0.6b_qwen3vl8b.sh
│   ├── qwen3emb_0.6b_qwen3rerank0.6b_qwen3vl2b.sh
│   ├── qwen3emb_0.6b_qwen3rerank0.6b_qwen3vl8b.sh
│   └── qwen3emb_0.6b_qwen3rerank4b_qwen3vl8b.sh
└── vl_embed/                    # VL embedding (raw media)
    ├── qwen3vlemb2b_withtext_qwen3vl2b.sh
    ├── qwen3vlemb2b_notext_qwen3vl2b.sh
    ├── qwen3vlemb2b_withtext_qwen3vl8b.sh
    ├── qwen3vlemb2b_notext_qwen3vl8b.sh
    ├── qwen3vlemb8b_withtext_qwen3vl8b.sh
    └── qwen3vlemb8b_notext_qwen3vl8b.sh
```

#### Output Structure

```
output/QA_Agent/MMRag/v3_ablation/
├── topk5/
│   ├── text_embed/
│   │   ├── allminilm_l6/{answerer}/short_caption/
│   │   ├── qwen3emb_0.6b/{answerer}/short_caption/
│   │   └── qwen3emb_4b/{answerer}/short_caption/
│   ├── text_embed_rerank/
│   │   ├── allminilm_l6_qwen3rerank_0.6b/{answerer}/short_caption/
│   │   ├── qwen3emb_0.6b_qwen3rerank_0.6b/{answerer}/short_caption/
│   │   └── qwen3emb_0.6b_qwen3rerank_4b/{answerer}/short_caption/
│   └── vl_embed/
│       ├── qwen3vlemb_2b/{answerer}/{withtext,notext}/
│       └── qwen3vlemb_8b/{answerer}/{withtext,notext}/
└── topk10/
    └── ... (same structure as topk5)
```

Each experiment folder contains:
- `mmrag_answers.jsonl` - model predictions
- `retrieval_recall_summary.json` - aggregated R@K metrics
- `retrieval_recall_details.json` - per-question recall breakdown
- `eval/` - ATM evaluation results (GLM-4.7 + GPT-5-mini judges)

---

### Legacy Ablation (Archived)

Previous ablation scripts are in `scripts/QA_Agent/MMRag/archieved_v2/`. These compared 4 augmentation levels with different embedding models.

#### Script Overview

| Script | Retriever | Reranker | Mode | Purpose |
|--------|-----------|----------|------|---------|
| `retrieve_answer_qwen3vl_raw.sh` | Qwen3-VL-Embedding-2B | - | raw | Text augmentation ablation |
| `retrieve_answer_text_batch.sh` | Qwen3-Embedding (0.6B/4B) | - | batch | Text embedding size + field ablation |
| `retrieve_answer_clip_raw.sh` | CLIP | - | raw | CLIP baseline (abandoned) |
| `retrieve_answer_vista_raw.sh` | VISTA (BGE-base + visual) | - | raw | VISTA baseline |
| `retrieve_rerank_qwen3vl_raw.sh` | Qwen3-VL-Embedding-2B | Qwen3-VL-Reranker-2B | raw | VL retrieve + rerank |
| `retrieve_rerank_text_batch.sh` | Qwen3-Embedding (0.6B/4B) | Qwen3-Reranker-4B | batch | Text retrieve + rerank |
| `retrieve_rerank_clip_raw.sh` | CLIP | Qwen3-VL-Reranker-2B | raw | CLIP + VL rerank |

### Qwen3-VL Text Augmentation Ablation (Legacy)

**Script**: `archieved_v2/qwen3_vl_embedding/`

Tests incremental text field additions to VL embedding. Essential metadata (ID, timestamp, location) always included.

| Method | Caption | Tags | OCR |
|--------|:-------:|:----:|:---:|
| `short_caption_only` | ✓ | - | - |
| `short_caption_caption` | ✓ | ✓ | - |
| `short_caption_caption_tag` | ✓ | ✓ | ✓ |
| `short_caption_caption_tag_ocr` | ✓ | ✓ | ✓ |

### Legacy Output Directory Structure

Output from `archieved_v2/` scripts (2026-01-16 path refactoring):

```
output/QA_Agent/MMRag/
├── qwen3_vl_embedding/
│   └── qwen3_vl_embedding_2b/
├── qwen3_embedding/
│   ├── qwen3_embedding_0.6b/
│   └── qwen3_embedding_4b/
├── allMiniLM_embedding/
│   └── allMiniLM_L6v2/
└── ...
```
