# MemoryOS Baseline for PersonalMemoryQA

Implementation of MemoryOS as a baseline for the PersonalMemoryQA benchmark.

## Paper

**Memory OS of AI Agent**  
Jiazheng Kang, Mingming Ji, Zhe Zhao, Ting Bai  
EMNLP 2025 (Oral)  
[Paper](https://arxiv.org/abs/2506.06326) | [Code](https://github.com/BAI-LAB/MemoryOS)

## Overview

MemoryOS is a hierarchical memory management system inspired by operating system memory management principles. It organizes memory into three tiers:

1. **Short-Term Memory (STM)**: Recent dialogue pages stored in FIFO queue
2. **Mid-Term Memory (MTM)**: Topic-based segments with heat-based eviction
3. **Long-Term Personal Memory (LPM)**: User/agent profiles and knowledge base

### Key Features

- **Hierarchical Storage**: Three-tier architecture for different memory granularity
- **Dynamic Updates**: STM to MTM via FIFO, MTM to LPM via heat-based promotion
- **Semantic Retrieval**: Parallel search across all memory tiers
- **Personalization**: Evolving user profiles and knowledge bases

### Heat Score Formula

For segment management in MTM:
```
Heat = alpha*N_visit + beta*L_interaction + gamma*R_recency
```
- N_visit: Number of times retrieved
- L_interaction: Number of dialogue pages
- R_recency: Time decay (exp(-Delta_t / tau))

## Installation

```bash
# Install from GitHub (latest version)
pip install git+https://github.com/BAI-LAB/MemoryOS.git#subdirectory=memoryos-pypi

# Or install from requirements
pip install -r requirements.txt
```

## Adaptation for Benchmarking

MemoryOS was originally designed for conversational AI with evolving memory. For single-QA benchmarking, we adapt it as follows:

### 1. Memory Item Formatting

Each memory item (email/image/video) is formatted as a dialogue pair:

```python
# Email example:
user_input = "Help me remember this email [email123]: From john@example.com at 2023-05-01..."
agent_response = "I've stored this email in my memory."

# Image example:
user_input = "Help me remember this image [IMG_5678.jpg]: Captured at Paris on 2023-06-15..."
agent_response = "I've stored this image memory."
```

All modalities are merged and sorted chronologically before indexing to preserve temporal order.
Missing timestamp/location/subject metadata is reported as warnings so you can fix the source data.

### 2. Instance Isolation (Critical!)

**Problem**: MemoryOS's `get_response()` automatically adds answered questions to memory, causing contamination.

**Solution**: Create a separate MemoryOS instance for each question and restore a checkpointed storage snapshot before each QA. This avoids contamination while preserving the original MemoryOS behavior.

```python
# Each QA gets an isolated instance with the same user_id.
# The storage folder is restored from a checkpoint before each QA.
memoryos = Memoryos(user_id="benchmark_user", data_storage_path=qa_storage_path, ...)
```

By default this is **sequential**; when `--max-workers > 1`, the baseline copies the checkpoint
into per-QA storage folders to enable parallel execution.

For benchmark speed/evaluation stability, you can disable post-answer memory writes:
```bash
--memoryos-eval-no-update
```
This skips the `get_response()` tail update (`add_memory` + profile/knowledge refresh)
while keeping retrieval and answer generation unchanged.

### 3. Configuration Tuning

Defaults follow the paper (STM=10, MTM=2000, heat threshold=5.0). Override if you want to suppress profile updates during ablations.

### 3.1 Full-History Mode (Benchmark Variant)

For large PersonalMemoryQA runs, this implementation provides:

```bash
--memoryos-full-history-mode
```

When enabled, it applies these minimums:
- `memoryos_mid_term_capacity >= memoryos_full_history_mid_term_capacity` (default 20000)
- `memoryos_top_k_sessions >= memoryos_full_history_top_k_sessions` (default 200)
- `memoryos_heat_threshold >= memoryos_full_history_heat_threshold` (default 1e9)

Important:
- This is a **benchmark scaling variant**, not paper-default MemoryOS settings.
- Checkpoint keys include full-history fields, so full-history runs use a separate checkpoint namespace.

### 4. Model Flexibility

MemoryOS supports any OpenAI-compatible LLM and custom embeddings (override to bge-m3 if you want to match other baselines):

```python
Memoryos(
    openai_base_url="http://localhost:8000/v1",  # vLLM endpoint
    llm_model="Qwen/Qwen3-VL-8B-Instruct",       # Your answering model
    embedding_model_name="all-MiniLM-L6-v2",     # Default embedding
)
```

You can also separate the indexing LLM from the answerer LLM:
```bash
--memoryos-index-llm-model Qwen/Qwen3-VL-2B-Instruct \
--memoryos-answer-llm-model Qwen/Qwen3-VL-8B-Instruct
```
Checkpoint reuse is keyed on the **indexing** LLM, so you can reuse the same memory
checkpoint for multiple answerer models as long as the indexing settings are unchanged.

### Implementation Notes

- MemoryOS `get_response()` always appends to memory; per-question isolation is required
  for unbiased evaluation. This baseline restores a checkpoint before each QA to avoid
  contamination, and can optionally skip post-answer updates via `--memoryos-eval-no-update`.
- Metadata is stored as plain text in `user_input`, so IDs/timestamps/locations are
  embedded directly in the memory text. Missing metadata is logged as warnings.
- Embeddings support SentenceTransformer + FlagEmbedding models; pass kwargs with
  `--memoryos-embedding-kwargs` when needed (for example, bge-m3 `use_fp16`).

## Usage

### Basic Usage

```bash
python memoryos_baseline.py \
    --qa-file /path/to/qa_annotations.json \
    --media-source batch_results \
    --image-batch-results /path/to/image_batch_results.json \
    --video-batch-results /path/to/video_batch_results.json \
    --email-file /path/to/merged_emails.json \
    --provider vllm \
    --vllm-endpoint http://localhost:8000/v1 \
    --memoryos-llm-model Qwen/Qwen3-VL-8B-Instruct \
    --memoryos-embedding-model all-MiniLM-L6-v2 \
    --output-dir-base ./output/QA_Agent/MemoryOS \
    --method-name memoryos_qwen3vl_bge
```

## Outputs

Each run writes to `output/QA_Agent/MemoryOS/<method>/`:

- `memoryos_answers.jsonl`
- `retrieval_recall_summary.json`
- `retrieval_recall_details.json`
- `retrieval_latency.json`
- `config.json`
- `index_progress_<hash>.jsonl`

### Key Arguments

#### MemoryOS-Specific

- `--memoryos-llm-model`: Default LLM for indexing + answering
- `--memoryos-index-llm-model`: LLM for memory indexing (default: memoryos-llm-model)
- `--memoryos-answer-llm-model`: LLM for answering (default: memoryos-llm-model)
- `--memoryos-answer-prompt-style`: `baseline` (aligned with A-Mem/HippoRAG/Oracle) or `memoryos` (native MemoryOS prompt)
- `--memoryos-embedding-model`: Embedding model (default: all-MiniLM-L6-v2)
- `--memoryos-embedding-kwargs`: JSON string of embedding kwargs (default: null -> MemoryOS defaults)
- `--memoryos-short-term-capacity`: STM capacity (default: 10, paper: 10)
- `--memoryos-mid-term-capacity`: MTM capacity (default: 2000, paper: 2000)
- `--memoryos-retrieval-queue-capacity`: Top-k pages (default: 50, paper: 7)
- `--memoryos-top-k-sessions`: Top-k mid-term segments searched before page ranking (default: 50, paper: 5)
- `--memoryos-heat-threshold`: Heat for MTM->LPM (default: 5, paper: 5)
- `--memoryos-use-per-qa-instance`: Isolation strategy (default: True)
- `--memoryos-user-id`: User ID for storage (default: benchmark_user)
- `--memoryos-checkpoint-dir`: Directory to store checkpoints
- `--memoryos-reuse-checkpoint`: Reuse checkpoint if available (default: true)
- `--memoryos-save-checkpoint`: Save checkpoint after indexing (default: true)
- `--memoryos-force-rebuild`: Ignore checkpoint and rebuild
- `--memoryos-quiet`: Suppress MemoryOS internal stdout/stderr (default: true)
- `--memoryos-parallel-llm`: Parallelize keyword extraction LLM calls (default: true)
- `--memoryos-parallel-llm-workers`: Parallel workers for keyword extraction (default: 4)
- `--memoryos-resume-indexing`: Resume from partial indexing progress (default: true)
- `--memoryos-full-history-mode`: Enable benchmark full-history variant (default: false)
- `--memoryos-full-history-mid-term-capacity`: MTM capacity floor in full-history mode (default: 20000)
- `--memoryos-full-history-top-k-sessions`: session-search floor in full-history mode (default: 200)
- `--memoryos-full-history-heat-threshold`: heat-threshold floor in full-history mode (default: 1e9)

#### Standard Baseline Arguments

- `--qa-file`: Path to QA JSON file
- `--media-source`: `batch_results` only (text-only, matches original MemoryOS usage)
- `--provider`: `openai`, `vllm`, or `vllm_local`
- `--vllm-endpoint`: vLLM server URL
- `--timeout`: Request timeout in seconds (propagated to the OpenAI client when supported)
- `--temperature` / `--max-tokens`: LLM decoding controls (used by MemoryOS)
- `--max-workers`: Parallel workers (default: 128)
- `--model`: Alias for `--memoryos-llm-model`

#### Text Augmentation Flags

Control what metadata to include in memory items:
- `--include-id/--no-include-id`: Include item IDs
- `--include-timestamp/--no-include-timestamp`: Include timestamps
- `--include-location/--no-include-location`: Include locations
- `--include-caption/--no-include-caption`: Include image captions
- `--include-email-summary/--no-include-email-summary`: Include email summaries

### Isolation Strategies

#### Per-Question Instances (Recommended)

```bash
python memoryos_baseline.py \
    --memoryos-use-per-qa-instance \
    ...
```

**Pros**: Complete isolation, no contamination  
**Cons**: Higher indexing overhead without checkpoints

When `--memoryos-reuse-checkpoint` is enabled, you can set `--max-workers > 1` to
parallelize per-QA runs (each worker copies the checkpoint into its own storage dir).

#### Shared Instance (Not Recommended)

```bash
python memoryos_baseline.py \
    --no-memoryos-use-per-qa-instance \
    ...
```

**Pros**: Single indexing, faster  
**Cons**: Answered questions contaminate memory

### Example Run Script

```bash
#!/bin/bash

# MemoryOS with Qwen3VL-8B + all-MiniLM embeddings

python memqa/qa_agent_baselines/MemoryOS/memoryos_baseline.py \
    --qa-file memqa/utils/final_data_processing/atm-20260121.json \
    --media-source batch_results \
    --image-batch-results memqa/qa_agent_baselines/example_qas/image_2026_01_06_qwen3vl8b_batch_results.json \
    --video-batch-results memqa/qa_agent_baselines/example_qas/video_2026_01_06_qwen3vl8b_batch_results.json \
    --email-file memqa/qa_agent_baselines/example_qas/merged_emails.json \
    --provider vllm \
    --vllm-endpoint http://localhost:8000/v1 \
    --memoryos-llm-model Qwen/Qwen3-VL-8B-Instruct \
    --memoryos-embedding-model all-MiniLM-L6-v2 \
    --memoryos-use-per-qa-instance \
    --max-workers 32 \
    --output-dir-base output/QA_Agent/MemoryOS \
    --method-name memoryos_qwen3vl_bge_isolated
```

## Configuration

Default configurations are in `config.py`. Key settings:

```python
MEMORYOS_CONFIG = {
    # Memory capacities (paper defaults)
    "memoryos_short_term_capacity": 10,
    "memoryos_mid_term_capacity": 2000,
    "memoryos_long_term_knowledge_capacity": 100,
    
    # Retrieval settings (benchmark defaults; paper defaults were 7/5)
    "memoryos_retrieval_queue_capacity": 50,  # Top-k pages
    "memoryos_top_k_sessions": 50,            # Top-k segments
    "memoryos_top_k_knowledge": 20,          # Top-k LPM entries
    
    # Heat and similarity (paper defaults)
    "memoryos_mid_term_heat_threshold": 5.0,
    "memoryos_mid_term_similarity_threshold": 0.6,
    
    # Models
    "memoryos_llm_model": "Qwen/Qwen3-VL-8B-Instruct",
    "memoryos_embedding_model_name": "all-MiniLM-L6-v2",
}
```

## Performance Considerations

### Checkpointing (Recommended)

MemoryOS persists storage to disk. The baseline uses a checkpoint directory to avoid re-indexing:

- `--memoryos-checkpoint-dir`: base directory for saved checkpoints
- `--memoryos-reuse-checkpoint`: reuse if available (default: true)
- `--memoryos-save-checkpoint`: save after indexing (default: true)
- `--memoryos-force-rebuild`: ignore existing checkpoint

When per-question isolation is enabled, the checkpoint is restored before each QA to prevent contamination.
Checkpoint keys are derived only from indexing-relevant settings (data sources, text config,
embedding model, indexing LLM, memory capacities). Retrieval-only knobs like
`memoryos_retrieval_queue_capacity` and `memoryos_top_k_sessions` do **not** affect the
checkpoint key, so you can change top-k at answer time without rebuilding the index.

### Memory Usage

Each MemoryOS instance requires:
- Embedding model in memory (shared if using same model)
- FAISS index for MTM (grows with memory items)
- JSON storage on disk

For large benchmarks (>1000 QAs), consider:
- Reducing `max_workers` to control memory
- Using GPU for embeddings if available
- Reusing checkpoints to avoid repeated indexing

### Storage Cleanup

By default, MemoryOS saves data to disk under the method directory:

- `output/QA_Agent/MemoryOS/<method>/memoryos_data/`

Delete this folder if you want a fresh run and do not need checkpoints.

## Paper Results (LoCoMo Benchmark)

With GPT-4o-mini:
- **F1 Score**: +49.11% improvement over baselines
- **BLEU-1**: +46.18% improvement
- **Context coherence**: Superior multi-turn dialogue

## Known Limitations

1. **Auto-memory-add**: `get_response()` adds QA to memory (requires isolation)
2. **No read-only mode**: Cannot retrieve without potential state changes
3. **Metadata preservation**: Must embed IDs/timestamps in text, not separate fields
4. **Disk I/O overhead**: Each instance saves to disk (can be slow)

## Benchmark Performance & Issue History (ATM-bench)

This section records deviations and issues observed during ATM-bench runs so results are
interpretable and reproducible.

### Benchmark-default hyperparameters (non-paper)

We intentionally override paper defaults to match ATM-bench scale:
- `memoryos_retrieval_queue_capacity = 50` (paper: 7)
- `memoryos_top_k_sessions = 50` (paper: 5)
- Optional full-history mode:
  - `memoryos_mid_term_capacity >= 20000`
  - `memoryos_top_k_sessions >= 200`
  - `memoryos_heat_threshold >= 1e9`

These are documented above in **Full-History Mode (Benchmark Variant)** and are required
to prevent aggressive evictions on large corpora.

### Historical failure mode: partial resume mismatch

**Symptoms (broken runs):**
- Mid-term memory contained only the most recent ~1k items (mostly 2025).
- Retrieval recall was very low (example: `R@50 ≈ 0.11`).
- ATM list-recall accuracy collapsed into the 0.0x range.

**Root cause:**
`index_progress_*.jsonl` could indicate “complete” while on-disk memory only contained
a suffix of items. Resume logic trusted progress instead of storage, so rebuilds would
silently skip most items.

**Fix (implemented):**
Resume now reconciles against actual stored item IDs (short + mid-term memory). If
progress is stale, it is archived and rewritten from the storage snapshot before resuming.

### Post-fix example (healthy run)

On `memoryos_qwen3vl2b_memory_qwen3vl2b_answerer_fullhistory`:
- Retrieval recall improved significantly (`R@50 ≈ 0.79`).
- Overall ATM accuracy improved to ~0.13 (list_recall still low; see below).

These numbers depend on model/setting and are provided as a sanity reference only.

## Citation

```bibtex
@misc{kang2025memoryosaiagent,
    title={Memory OS of AI Agent}, 
    author={Jiazheng Kang and Mingming Ji and Zhe Zhao and Ting Bai},
    year={2025},
    eprint={2506.06326},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2506.06326}, 
}
```

## Troubleshooting

### Import Error: `memoryos` not found

```bash
pip install git+https://github.com/BAI-LAB/MemoryOS.git#subdirectory=memoryos-pypi
```

### FAISS not found

```bash
pip install faiss-cpu
# or for GPU:
pip install faiss-gpu
```

### Embedding model not found

MemoryOS will download models from HuggingFace on first use. Ensure:
- Internet connection available
- Sufficient disk space (~1-2GB per embedding model)
- Or provide local path: `--memoryos-embedding-model /path/to/local/model`

### Out of memory

Reduce `--max-workers`:
```bash
--max-workers 16  # Lower from default 128
```

Or use shared instance (with contamination caveat):
```bash
--no-memoryos-use-per-qa-instance
```

---

## Baseline Comparison: MemoryOS vs A-Mem vs MIRIX

This section provides a comprehensive comparison of the three memory baselines for PersonalMemoryQA, covering architecture, scalability, and retrieval characteristics.

### Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    MEMORYOS ARCHITECTURE                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  STM (Short-Term Memory)                                           │
│  ┌──────────────────────────────────────┐                         │
│  │ FIFO Queue (Capacity: 10 items)      │                         │
│  │ Last 10 memory pages                 │                         │
│  └──────────┬───────────────────────────┘                         │
│             │ (FIFO eviction)                                      │
│             ▼                                                      │
│  MTM (Mid-Term Memory)                                             │
│  ┌──────────────────────────────────────┐                         │
│  │ Segment-based Storage                │                         │
│  │ • 200 segments max                   │                         │
│  │ • 2000 pages total                   │                         │
│  │ • FAISS vector index                 │                         │
│  │ • Heat-based eviction                │                         │
│  └──────────┬───────────────────────────┘                         │
│             │ (Heat threshold > 5.0)                               │
│             ▼                                                      │
│  LTM (Long-Term Personal Memory)                                   │
│  ┌──────────────────────────────────────┐                         │
│  │ User Profiles + Knowledge Base       │                         │
│  │ • 100 items capacity                 │                         │
│  │ • Summarized information             │                         │
│  └──────────────────────────────────────┘                         │
│                                                                    │
│  Retrieval: Parallel search across all tiers                      │
│  Cost: O(200) segment comparisons after segment limit reached     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                       A-MEM ARCHITECTURE                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Memory Store (Flat Structure)                                     │
│  ┌──────────────────────────────────────┐                         │
│  │ MemoryNote Objects                   │                         │
│  │ • ID, content, timestamp             │                         │
│  │ • Keywords (LLM-extracted)           │                         │
│  │ • Context (1-sentence summary)       │                         │
│  │ • Tags (categorical labels)          │                         │
│  │ • Links (bidirectional connections)  │                         │
│  └──────────────────────────────────────┘                         │
│             │                                                      │
│             ▼                                                      │
│  Retrieval Layer                                                   │
│  ┌──────────────────────────────────────┐                         │
│  │ Embedding-only (default)             │                         │
│  │ • Annoy/FAISS ANN index              │                         │
│  │ • O(log n) search                    │                         │
│  │                                      │                         │
│  │ Hybrid (optional)                    │                         │
│  │ • BM25 + Embeddings (alpha=0.5)      │                         │
│  │ • Link traversal                     │                         │
│  └──────────────────────────────────────┘                         │
│                                                                    │
│  Retrieval: Direct ANN search (no eviction, no summaries)         │
│  Cost: O(log n) per query                                         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                      MIRIX ARCHITECTURE                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Meta-Agent (Orchestrator)                                         │
│  ┌──────────────────────────────────────┐                         │
│  │ • Topic extraction (LLM)             │                         │
│  │ • Temporal parsing                   │                         │
│  │ • Agent routing                      │                         │
│  └──────────┬───────────────────────────┘                         │
│             │                                                      │
│             ▼                                                      │
│  6 Specialized Memory Agents                                       │
│  ┌──────────────────────────────────────┐                         │
│  │ 1. Episodic (personal events)        │                         │
│  │    • occurred_at, filter_tags        │                         │
│  │ 2. Semantic (general knowledge)      │                         │
│  │ 3. Procedural (how-to)               │                         │
│  │ 4. Resource (documents)              │                         │
│  │ 5. Knowledge (key-value pairs)       │                         │
│  │ 6. Core (user identity)              │                         │
│  └──────────┬───────────────────────────┘                         │
│             │                                                      │
│             ▼                                                      │
│  PostgreSQL + pgvector (Persistent)                                │
│  ┌──────────────────────────────────────┐                         │
│  │ • BM25 full-text search (default)    │                         │
│  │ • HNSW vector index                  │                         │
│  │ • Temporal filtering                 │                         │
│  │ • filter_tags provenance             │                         │
│  └──────────────────────────────────────┘                         │
│                                                                    │
│  Retrieval: Dual-stream (recent + relevant) per memory type       │
│  Cost: O(log n) per memory type (database B-tree + HNSW)          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

### Memory Organization and Retention

#### Full Item Retention

| Baseline | Retains All Items? | Storage Mechanism | QA After 11K Items |
|----------|-------------------|-------------------|-------------------|
| **A-Mem** | ✅ **YES** | Flat memory store with embeddings | All 11K items searchable |
| **MIRIX** | ✅ **YES** | PostgreSQL database (persistent disk) | All 11K items searchable |
| **MemoryOS** | ⚠️ **DEPENDS** | Tiered (STM→MTM→LTM) with eviction | Only items in MTM/LTM searchable |

#### MemoryOS Retention Problem

**With default settings (MTM capacity=2000, 200 segments):**

1. **Items 1-10**: Held in STM
2. **Items 11-2000**: Pushed to MTM as segments
3. **Items 2001+**: **Cold segments evicted to LTM**
   - Original content **lost**
   - Only **summaries** remain in LTM

**Example Loss:**
- **Original**: "Photo of me at Golden Gate Bridge on June 15, 2023"
- **After eviction**: "User visited San Francisco in June 2023"
- **Question**: "What bridge was I at on June 15?" → **CANNOT ANSWER** (bridge name lost)

**Solution for Full Retention:**
```bash
--memoryos-mid-term-capacity 15000    # Hold all 11K items
--memoryos-heat-threshold 999999      # Disable LTM eviction
```

---

### Scalability Analysis

#### Computational Complexity

| Baseline | Per-Item Insertion | Total for 10K Items | Bottleneck |
|----------|-------------------|---------------------|------------|
| **A-Mem** | O(log n) | O(n log n) | ANN index insert |
| **MIRIX** | O(log n) | O(n log n) | PostgreSQL B-tree + HNSW |
| **MemoryOS (default)** | O(200) after segment limit | O(n²) | Segment matching + heat calculations |

#### MemoryOS Scalability Breakdown

**Phase 1: Items 1-200 (Linear Growth)**
- Each new item creates a segment
- Cost per item: O(current_segment_count)
- Average cost: ~100 segment comparisons

**Phase 2: Items 201-10,000 (Constant Overhead)**
- Segment limit (200) reached
- Every insert requires:
  - **O(200) segment matching**: Compare against all 200 segments
  - **O(200) heat calculation**: When eviction triggered (periodic)
- Total segment comparisons for 10K items: **~2,000,000**

**Expected Timing (Observed):**
- Items 1-100: **~0.5s per item** (STM + early MTM)
- Items 100-2000: **~1-2s per item** (growing segment count)
- Items 2000-10,000: **~6s per item** (200 segment overhead)
- **Total for 10K items: ~8 hours**

**Root Cause:**
- MemoryOS designed for **conversational paradigm** (~300 turns)
- PersonalMemoryQA is **batch ingestion** (10,000+ items)
- Scale mismatch: **33x more items**, **278x more tokens**

---

### Optimization Strategies for MemoryOS

#### Phase 1: No-Code Configuration (Immediate)

Increase capacities to eliminate eviction overhead:

```bash
--memoryos-short-term-capacity 500        # Batch 500 items before MTM
--memoryos-mid-term-capacity 15000        # No evictions for 10K items
--memoryos-heat-threshold 999999          # Disable LPM promotion
```

**Expected speedup**: 10-20x (8 hours → 30-50 minutes)

**Trade-offs**:
- LTM layer unused (acceptable for batch QA)
- Higher memory usage (FAISS index grows)

#### Phase 2: Batched Updates (Requires Library Modification)

Reduce frequency of expensive operations:

```python
# In memoryos/memoryos.py
class Memoryos:
    def __init__(
        self,
        batch_update_interval: int = 500,      # Update heat every 500 items
        faiss_rebuild_interval: int = 2000,    # Rebuild FAISS every 2000 items
        disable_heat_during_batch: bool = True,
    ):
```

**Expected speedup**: 50-100x (8 hours → 5-10 minutes)

**Implementation needed**:
- Modify `Memoryos.__init__()` in library
- Add CLI arguments to `memoryos_baseline.py`

#### Phase 3: Checkpoint-First Strategy

Build checkpoint once with optimized settings, reuse for all QA evaluations.

**Recommended workflow**:
1. Index once with `--memoryos-mid-term-capacity 15000`
2. Save checkpoint
3. Reuse checkpoint for all experiments (different answer models, etc.)

---

### Retrieval Characteristics

#### Search Methods

| Baseline | Default Search | Optional Methods | Complexity |
|----------|---------------|------------------|------------|
| **A-Mem** | Embedding-only | Hybrid BM25+Embedding (alpha=0.5), Link traversal | O(log n) |
| **MIRIX** | BM25 full-text | Vector similarity, Temporal filtering | O(log n) per memory type |
| **MemoryOS** | Vector similarity | Parallel across STM/MTM/LTM | O(k) where k = retrieval queue capacity |

#### Temporal Anchoring

| Baseline | Temporal Support | Mechanism |
|----------|-----------------|-----------|
| **A-Mem** | ✅ Timestamp stored | Optional in retrieval logic |
| **MIRIX** | ✅ **Native** | `occurred_at` field, automatic temporal parsing |
| **MemoryOS** | ⚠️ **Text-embedded** | Timestamp in page text (no structured field) |

**MIRIX advantage**: Explicit `occurred_at` enables queries like "What did I do last Saturday?" with automatic date parsing.

#### Multi-Modal Support

| Baseline | Text-Only | Multimodal | Image Handling |
|----------|-----------|------------|----------------|
| **A-Mem** | Default (batch captions) | Optional (`--construct-from-raw`, `--answer-from-raw`) | Base64 in messages |
| **MIRIX** | Default (batch captions) | Optional (`--memory-mode multimodal`) | Base64 in messages |
| **MemoryOS** | **Only** (batch captions) | Not supported | N/A |

---

### Baseline Recommendations

#### When to Use Each Baseline

**A-Mem**:
- ✅ Need full item retention with minimal overhead
- ✅ Want LLM-enhanced memory construction (keywords, tags, context)
- ✅ Flexible model backends (OpenAI, vLLM, local)
- ⚠️ No native temporal reasoning (timestamp is metadata only)

**MIRIX**:
- ✅ Need persistent storage (PostgreSQL)
- ✅ Temporal queries critical ("What did I do last week?")
- ✅ Multi-agent categorization (episodic, semantic, procedural, etc.)
- ✅ Production-ready (database backend, Redis caching)
- ⚠️ Requires running MIRIX server

**MemoryOS**:
- ✅ Testing tiered memory paradigm (STM→MTM→LTM)
- ✅ Conversational memory evolution (heat-based promotion)
- ⚠️ **NOT RECOMMENDED** for large-scale batch ingestion without optimization
- ⚠️ Requires capacity tuning for PersonalMemoryQA (15K+ items)

---

### Performance Comparison (10K Items)

| Metric | A-Mem | MIRIX | MemoryOS (default) | MemoryOS (optimized) |
|--------|-------|-------|-------------------|---------------------|
| **Indexing time** | ~2.8 hours | ~2.2 hours | ~8 hours | ~30-50 minutes |
| **Per-item cost** | ~1.0s | ~0.8s | 0.5s → 6s (degrades) | ~0.2s (constant) |
| **Memory retention** | All items | All items | ~2000 items (rest summarized) | All items (with high capacity) |
| **Scalability** | O(n log n) | O(n log n) | O(n²) | O(n log n) |
| **Storage** | In-memory (pickle cache) | Persistent (PostgreSQL) | In-memory (JSON + FAISS) | In-memory (JSON + FAISS) |

---

### Configuration Alignment for Fair Comparison

To ensure fair comparison across all baselines:

#### 1. Embedding Model
```bash
# A-Mem (default)
--embedding-model all-MiniLM-L6-v2

# MIRIX (OpenAI provider)
--mirix-provider openai  # Uses text-embedding-3-small (1536 dim)

# MemoryOS (override to match A-Mem)
--memoryos-embedding-model all-MiniLM-L6-v2
```

#### 2. Memory LLM
```bash
# A-Mem (paper default)
--memory-model gpt-4o-mini

# MIRIX (meta-agent)
--mirix-model gpt-4o-mini

# MemoryOS (indexing LLM)
--memoryos-index-llm-model gpt-4o-mini
```

#### 3. Answer LLM (External)
```bash
# All baselines use same answerer for fair comparison
--provider vllm
--model Qwen/Qwen3-VL-8B-Instruct
--vllm-endpoint http://localhost:8000/v1
```

#### 4. Retrieval Settings
```bash
# A-Mem
--retrieve-k 10
--follow-links

# MIRIX
--retrieval-limit 10

# MemoryOS
--memoryos-retrieval-queue-capacity 10
```

---

### Design Philosophy Comparison

| Aspect | A-Mem | MIRIX | MemoryOS |
|--------|-------|-------|----------|
| **Paradigm** | Self-organizing memory graph | Multi-agent categorization | OS-inspired tiered memory |
| **Evolution** | LLM-driven link creation | Agent-based extraction | Heat-based promotion |
| **Focus** | Semantic connections | Functional categorization | Temporal relevance |
| **Inspiration** | Knowledge graphs | Human cognition | Operating systems (STM/LTM) |
| **Best for** | Relational queries | Multi-modal categorized data | Conversational agents |

---

### References

- **MemoryOS Paper**: [Memory OS of AI Agent (EMNLP 2025)](https://arxiv.org/abs/2506.06326)
- **A-Mem Paper**: [A-MEM: Agentic Memory for LLM Agents (NeurIPS 2025)](https://arxiv.org/abs/2502.12110)
- **MIRIX Paper**: [MIRIX: Multi-Agent Personal Assistant with an Advanced Memory System](https://arxiv.org/abs/2507.07957)

---

## Contact

For issues specific to this baseline implementation:
- Open an issue in the PersonalMemoryQA repository

For MemoryOS core issues:
- See [MemoryOS GitHub](https://github.com/BAI-LAB/MemoryOS)
