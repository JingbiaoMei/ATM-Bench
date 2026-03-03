# A-Mem Baseline

A-Mem is an agentic memory baseline adapted from [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem). It builds a memory index from batch results (captions) and emails, then answers QA queries using A-Mem retrieval + a standard LLM answer stage.

## Paper Reference

> **A-MEM: Agentic Memory for LLM Agents**
> Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang
> *NeurIPS 2025*
>
> - **arXiv**: [https://arxiv.org/abs/2502.12110](https://arxiv.org/abs/2502.12110)
> - **GitHub (Benchmark)**: [https://github.com/WujiangXu/A-mem](https://github.com/WujiangXu/A-mem)
> - **GitHub (System)**: [https://github.com/WujiangXu/A-mem-sys](https://github.com/WujiangXu/A-mem-sys)

## Original Paper's Default Configuration

| Component | Model / Setting |
|-----------|-----------------|
| **LLM (Memory Analysis)** | `gpt-4o-mini` (also tested: GPT-4o, Claude 3.0/3.5 Haiku, Qwen-2.5, Llama 3.2) |
| **Embedding Model** | `all-MiniLM-L6-v2` (sentence-transformers) |
| **Retrieval** | Embedding-only (SimpleEmbeddingRetriever; paper describes hybrid BM25+embedding) + Link Traversal |
| **Query Processing** | Direct query (no query-time LLM enhancement) |
| **Memory Enhancement** | LLM extracts keywords/context/tags at storage time |

---

## Method Overview

### What is A-Mem?

A-Mem (Agentic Memory) is a framework for building **self-organizing, LLM-augmented memory systems**. Unlike traditional RAG that stores and retrieves raw documents, A-Mem treats each piece of information as a **MemoryNote** — a structured unit enriched with:

- **Keywords**: Salient terms extracted via LLM analysis
- **Context**: A one-sentence semantic summary
- **Tags**: Categorical labels for grouping related memories
- **Links**: Connections to other related memories (via evolution)
- **Timestamp**: Temporal anchor for the memory

The core insight is that memories should be **actively processed** at insertion time (not just embedded) and can **evolve** over time as new memories arrive, strengthening connections and updating context.

### Original A-Mem Design (Paper)

The original A-Mem paper proposes:

1. **Memory Insertion**: Each new piece of content is analyzed by an LLM to extract keywords, context, and tags. The content is embedded and stored as a `MemoryNote`.

2. **Memory Evolution**: When a new memory is added, the system finds its nearest neighbors and asks an LLM whether to:
   - **Strengthen**: Create bidirectional links between related memories
   - **Update Neighbors**: Refine the context/tags of existing memories based on new information
   
3. **Retrieval**: At query time, the system uses a hybrid of embedding similarity and BM25 keyword matching to find relevant memories, then traverses links to gather additional context.

4. **Activation-based Decay**: Memories have activation scores that decay over time; frequently accessed memories remain more accessible.

---

## Our Adaptation for PersonalMemoryQA

We adapt A-Mem for the **personal memory QA task**, where the goal is to answer questions about a user's photos, videos, and emails. Our implementation aligns closely with the original paper to ensure fair baseline comparison.

### Alignment with Original

| Aspect | Original A-Mem | Our Implementation | Aligned? |
|--------|----------------|-------------------|----------|
| **Embedding Model** | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | Yes |
| **Retrieval** | Embedding-only (SimpleEmbeddingRetriever) | Embedding-only (SimpleEmbeddingRetriever) | Yes |
| **Link Traversal** | Enabled | Enabled (`--follow-links`) | Yes |
| **Query Processing** | Direct query embedding | Direct query embedding | Yes |
| **Memory Analysis** | LLM extracts keywords/context/tags | LLM extracts keywords/context/tags | Yes |
| **Memory Evolution** | LLM-driven neighbor updates | LLM-driven neighbor updates | Yes |
| **Activation Decay** | Implemented | Removed (batch QA, no temporal patterns) | No |

### Intentional Differences

1. **No Activation Decay**: The original uses activation scores that decay over time. We remove this because our batch QA setting has no temporal access patterns to exploit.

2. **Media-Centric Content**: Instead of free-form text notes, we ingest VLM-generated captions from images/videos. Use `--caption-only` for clean baseline comparison.

3. **Flexible LLM Backends**: We support OpenAI, vLLM (remote), and vLLM (local) for memory analysis and answer generation.

4. **Caching**: We cache memories and embeddings to disk to avoid re-running expensive LLM analysis on every run.

### LoCoMo: Multimodal Dataset vs. A-Mem Text-Only Implementation

The **LoCoMo dataset** (arXiv:2402.17753) is explicitly **multimodal** and includes image-sharing and image-reaction behaviors. However, the **A‑Mem paper’s implementation is text‑only**, as stated in its limitations.

**LoCoMo (multimodal evidence):**
- *Abstract*: “...we equip each agent with the capability of sharing and reacting to images... multi‑modal dialogue generation tasks.”
- *Section 1 (Figure 1 caption)*: “Multimodal dialog is enabled with image‑sharing and image‑response behaviors.”
- *Section 3.3*: “The image sharing & image reaction functions are integrated to add a multi‑modal dimension...”
- *Section 4.3*: “The model is required to generate a response ... with the provided image.”

**A‑Mem paper (text-only limitation):**
- *Section 6, Limitations*: “...our current implementation focuses on text‑based interactions...”

**Implication:** The A‑Mem evaluation on LoCoMo uses the **text‑only channel**, even though LoCoMo supports images. Our baseline mirrors that text‑only setup unless `--construct-from-raw` or `--answer-from-raw` is enabled.

### Swapping VLM Captioners

To compare different captioning models (e.g., Qwen3-VL 8B vs 4B vs 2B), point to different batch result files:

```bash
# Qwen3-VL 8B (default)
--image-batch-results path/to/image_qwen3vl8b_batch_results.json

# Qwen3-VL 4B
--image-batch-results path/to/image_qwen3vl4b_batch_results.json

# Qwen3-VL 2B
--image-batch-results path/to/image_qwen3vl2b_batch_results.json
```

---

## Prompt Alignment & Implementation Analysis

**Status**: ✅ **100% Aligned with Original A-Mem Implementation**

This section documents our rigorous alignment process with the original A-Mem paper and codebase, including critical findings about discrepancies between the paper's description and its actual implementation.

### Prompt Implementation Status

| Component | Our Implementation | Paper (Appendix B) | Original Code | Status |
|-----------|-------------------|-------------------|---------------|--------|
| **Memory Analysis** | Paper-aligned | Ps1 (B.1) | Exact match | ✅ 100% |
| **Memory Evolution** | Paper-aligned | Ps3 (B.3) | Exact match | ✅ 100% |
| **Link Generation** | N/A | Ps2 (B.2) described | **NOT in code** | ✅ Matches code |
| **Evolution Actions** | strengthen, update_neighbor | merge, prune mentioned | **Only strengthen, update_neighbor** | ✅ Matches code |

### Critical Finding: Paper vs. Code Discrepancies

After thorough analysis of both the [paper](https://arxiv.org/pdf/2502.12110) and the [original implementation](https://github.com/WujiangXu/A-mem), we discovered **significant discrepancies** between what the paper describes and what the code actually implements:

#### 1. **Two-Step Evolution (Described but Not Implemented)**

**Paper Description (Appendix B.2 + B.3):**
```
Step 1: Link Generation (Ps2) - Decide IF to evolve
Step 2: Memory Evolution (Ps3) - Decide HOW to evolve
```

**Actual Implementation:**
```python
# Only ONE evolution call in process_memory()
response = llm.get_completion(evolution_prompt, ...)  # Combines both steps
```

**Finding:** The original code does **NOT** have a separate "Link Generation" step. Both the decision ("should_evolve?") and action planning ("strengthen" or "update_neighbor"?) happen in a **single LLM call** using only Ps3.

**Our Alignment:** ✅ We implement the single-step evolution matching the actual code, not the paper description.

#### 2. **Merge and Prune Actions (Mentioned but Not Implemented)**

**Paper Prompt Template (B.3):**
```json
{
  "actions": ["strengthen", "merge", "prune"]
}
```

**Actual Implementation:**
```python
# Only these two actions are implemented:
if action == "strengthen":
    # Create bidirectional links
    note.links.extend(connection_ids)
    note.tags = tags_to_update
elif action == "update_neighbor":
    # Update neighbor contexts and tags
    neighbor.context = new_contexts[i]
    neighbor.tags = new_tags[i]
# NO implementation for "merge" or "prune"!
```

**Finding:** Despite being mentioned in the paper's prompt template, **"merge" and "prune" actions are never implemented** in the original codebase. Only "strengthen" and "update_neighbor" actually function.

**Our Alignment:** ✅ We implement only the two functional actions, matching the actual code behavior.

### Exact Paper Prompts (from Appendix B)

All prompts are stored in `config.py` under `AMEM_PROMPTS`:

#### **Ps1: Note Construction (Memory Analysis)**

<details>
<summary>Click to expand full prompt</summary>

```
Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object:
{
    "keywords": [
        // several specific, distinct keywords that capture key concepts and terminology
        // Order from most to least important
        // Don't include keywords that are the name of the speaker or time
        // At least three keywords, but don't be too redundant.
    ],
    "context": 
        // one sentence summarizing:
        // - Main topic/domain
        // - Key arguments/points
        // - Intended audience/purpose
    ,
    "tags": [
        // several broad categories/themes for classification
        // Include domain, format, and type tags
        // At least three tags, but don't be too redundant.
    ]
}

Content for analysis:
{content}
```

</details>

**Key Features:**
- ✅ Inline JSON comments providing detailed guidance
- ✅ Explicit keyword constraints: "focus on nouns, verbs, and key concepts"
- ✅ Ordering requirement: "Order from most to least important"
- ✅ Exclusion rule: "Don't include keywords that are the name of the speaker or time"
- ✅ Quantity guideline: "At least three keywords, but don't be too redundant"
- ✅ Context structure: Main topic/domain, Key arguments/points, Intended audience/purpose
- ✅ Tag categories: "Include domain, format, and type tags"

#### **Ps3: Memory Evolution**

<details>
<summary>Click to expand full prompt</summary>

```
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.
Return your decision in JSON format with the following structure:
{
    "should_evolve": True or False,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["neighbor_memory_ids"],
    "tags_to_update": ["tag_1",..."tag_n"], 
    "new_context_neighborhood": ["new context",...,"new context"],
    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
}
```

</details>

**Key Features:**
- ✅ Full role description: "responsible for managing and evolving a knowledge base"
- ✅ Detailed action explanations (2.1, 2.2)
- ✅ Length constraints: "length of new_tags_neighborhood must equal the number of input neighbors"
- ✅ Purpose explanation: "can be used to retrieve them later and categorize them"
- ✅ Preservation rule: "If the context and the tags are not updated, the new context and tags should be the same as the original ones"

### Multimodal Extension (Our Addition for PersonalMemoryQA)

We extended the paper's text-only prompt to support multimodal memory construction:

**Ps1-Multimodal: Note Construction with Images**

```
Analyze the provided image and its caption to generate a structured memory note.
1. Identifying the most salient keywords (focus on nouns, verbs, and key visual concepts)
2. Extracting core themes and contextual elements from both visual and textual clues
3. Creating relevant categorical tags

[Same JSON structure as Ps1]

Caption/Text: {content}
```

**Usage:** Enabled with `--construct-from-raw` flag, this prompt is used when the memory construction LLM receives both the caption text and the raw image.

### Comparison: Before vs. After Alignment

#### **Memory Analysis Prompt**

**Before (Our Custom):**
```
Generate a structured analysis of the following content by:
1. Identifying the most salient keywords.
2. Extracting core themes and contextual elements.
3. Creating relevant categorical tags.

Format the response as a JSON object:
{
  "keywords": ["keyword1", "keyword2"],
  "context": "one sentence summary",
  "tags": ["tag1", "tag2"]
}

Content for analysis:
{content}
```

**After (Paper-Aligned):**
- ✅ Added inline JSON comments with detailed guidelines
- ✅ Added keyword focus: "nouns, verbs, and key concepts"
- ✅ Added ordering requirement
- ✅ Added exclusion rules (no speaker names/times)
- ✅ Added quantity guidelines
- ✅ Added context structure framework
- ✅ Added tag category requirements

**Impact:** Better keyword quality, more structured contexts, more diverse tag categories.

#### **Evolution Prompt**

**Before (Our Custom):**
```
You are an AI memory evolution agent.
New memory context: {note.context}
content: {note.content}
keywords: {note.keywords}
neighbors: {neighbor_text}
neighbor_number: {len(neighbor_indices)}
Return JSON with fields should_evolve, actions, suggested_connections, 
tags_to_update, new_context_neighborhood, new_tags_neighborhood.
```

**After (Paper-Aligned):**
- ✅ Added full role description
- ✅ Added decision framework ("Should this memory be evolved?")
- ✅ Added detailed action explanations (2.1, 2.2)
- ✅ Added preservation rules
- ✅ Added length constraints
- ✅ Added purpose explanations

**Impact:** LLM better understands when to evolve, clearer action selection, correct neighborhood lengths.

### Verification Methodology

Our alignment verification involved:

1. **Paper Analysis**: Read Appendix B of [arxiv.org/pdf/2502.12110](https://arxiv.org/pdf/2502.12110)
2. **Code Analysis**: Cloned and analyzed the upstream `memory_layer.py` implementation
3. **Discrepancy Detection**: Identified paper vs. code mismatches
4. **Implementation Alignment**: Matched actual code behavior, not paper description
5. **Prompt Extraction**: Copied exact prompt text from original implementation
6. **Functional Testing**: Verified JSON schema compatibility

### References

- **Paper**: [A-MEM: Agentic Memory for LLM Agents (NeurIPS 2025)](https://arxiv.org/pdf/2502.12110) - Appendix B
- **Original Codebase**: [github.com/WujiangXu/A-mem](https://github.com/WujiangXu/A-mem)
- **Our Implementation**: `memqa/qa_agent_baselines/A-Mem/config.py` (`AMEM_PROMPTS`)

---


## Answer Generation (QA Stage)

**Summary:** A-Mem’s QA stage is a standard retrieval‑augmented generation step (retrieve → answer). The **agentic behavior lives in memory construction/evolution**, not in the QA loop.

### Paper References

- **Section 2.2 (Retrieval‑Augmented Generation)**: The paper contrasts agentic RAG with A‑Mem and positions A‑Mem’s agency in **memory evolution**, not in query‑time retrieval.
- **Section 3.4 (Retrieve Relative Memory)**: The QA step is described as computing a query embedding, retrieving top‑k memories, and passing the retrieved context to an LLM for response—no iterative or tool‑using agent loop is described.

### Our Implementation

- **Retrieval**: Embedding-only (SimpleEmbeddingRetriever) by default; optional hybrid BM25+embedding via `--use-hybrid` and link traversal via `--follow-links`.
- **Answering**: A single LLM call with fixed prompts.

**Prompt (from** `memqa/qa_agent_baselines/A-Mem/config.py`**):**
```
SYSTEM: You are a memory QA assistant. Use ONLY the provided evidence to answer.
If the evidence is insufficient, answer 'Unknown'. Respond with only the answer.

USER: Question: {question}

Evidence:
{evidence}

Provide the answer based solely on the evidence.
```

## Two-Stage Pipeline Architecture

**NEW**: A-Mem supports a **two-stage pipeline** to enable flexible model swapping without re-running expensive memory construction.

### Why Two Stages?

**Problem**: On single-server setups, you can only run one model at a time. To compare different answer models (e.g., Qwen3-VL 2B vs 4B vs 8B), you traditionally had to:
1. Build memories with Model A → Answer with Model A
2. Rebuild memories with Model B (expensive!) → Answer with Model B
3. Rebuild memories with Model C (expensive!) → Answer with Model C

**Solution**: Separate memory construction from answer generation:
1. **Stage 1 (build)**: Construct memories once with a memory LLM → cache to disk
2. **Stage 2 (answer)**: Load cached memories, answer with different LLMs (no rebuild)

### Pipeline Stages

| Stage | Command | Purpose | Required Args |
|-------|---------|---------|---------------|
| **build** | `--stage build` | Memory construction only | Memory LLM args, cache-affecting args |
| **answer** | `--stage answer` | Answer generation only | Answer LLM args, same cache-affecting args |
| **all** | `--stage all` (default) | Both stages sequentially | All args (backward compatible) |

### Cache-Affecting Arguments

These arguments **determine the cache key**. Stage 2 must use the **exact same values** as Stage 1 to load the correct cache.

**Core Arguments:**
- `--image-batch-results`, `--video-batch-results`, `--email-file`
- `--embedding-model`, `--memory-provider`, `--memory-model`
- `--disable-evolution`, `--evo-threshold`, `--use-hybrid`, `--alpha`
- `--caption-only`, `--use-short-caption`, `--construct-from-raw`

**Text Configuration:**
- `--include-type`, `--include-caption`, `--include-short-caption`
- `--include-ocr-text`, `--include-tags`

**Important**: Changing **any** of these in Stage 2 will cause cache miss error!

### Stage-Specific Arguments

| Argument | Stage 1 (build) | Stage 2 (answer) | Notes |
|----------|-----------------|------------------|-------|
| `--memory-provider` | ✓ Required | (ignored) | LLM for memory construction |
| `--memory-model` | ✓ Required | (ignored) | Model for memory analysis |
| `--memory-api-key` | ✓ Required | (ignored) | API key for memory LLM |
| `--memory-vllm-endpoint` | ✓ Required | (ignored) | Endpoint for memory LLM |
| `--memory-workers` | ✓ Optional | (ignored) | Parallel workers for construction (default: 1) |
| `--checkpoint-interval` | ✓ Optional | (ignored) | Save partial memory cache every N ingests (default: 100) |
| `--resume` | ✓ Optional | (ignored) | Resume from partial cache if present (default: enabled) |
| `--provider` | (ignored) | ✓ Required | LLM for answer generation |
| `--model` | (ignored) | ✓ Required | Model for answering |
| `--vllm-endpoint` | (ignored) | ✓ Required | Endpoint for answer LLM |
| `--api-key` | (ignored) | ✓ Required | API key for answer LLM |
| `--max-workers` | (ignored) | ✓ Optional | Parallel workers for QA |
| `--retrieve-k` | (ignored) | ✓ Required | Number of memories to retrieve |
| `--follow-links` | (ignored) | ✓ Optional | Traverse memory links |
| `--answer-from-raw` | (ignored) | ✓ Optional | Pass raw images to answerer |
| `--output-dir-base` | (ignored) | ✓ Required | Output directory for results |
| `--method-name` | (ignored) | ✓ Required | Method name for outputs |

### Shell Scripts

Pre-configured two-stage scripts are provided in `scripts/QA_Agent/A-Mem/`:

**1. Raw Image Memory + Multiple Answerers** (`run_amem_qwen3vlMemory_raw.sh`)
- **Stage 1**: Qwen3-VL 8B memory with raw images (ID, timestamp, location only)
- **Stage 2**: Test with Qwen3-VL 2B, 4B, 8B, and QwQ-32B answerers
- Each answerer runs independently, reusing the same memory cache

**2. Batch Caption Memory + Multiple Answerers** (`run_amem_batch.sh`)
- **Stage 1**: OpenAI gpt-4o-mini memory with short captions (batch results)
- **Stage 2**: Test with Qwen3-VL 2B, 4B, 8B, and QwQ-32B answerers
- Each answerer runs independently, reusing the same memory cache

### Example: Stage 1 (Build Memories)

```bash
python memqa/qa_agent_baselines/A-Mem/amem_baseline.py \
  --stage build \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --image-batch-results memqa/qa_agent_baselines/example_qas/image_2026_01_06_qwen3vl8b_batch_results.json \
  --video-batch-results memqa/qa_agent_baselines/example_qas/video_2026_01_06_qwen3vl8b_batch_results.json \
  --email-file memqa/qa_agent_baselines/example_qas/merged_emails.json \
  --embedding-model all-MiniLM-L6-v2 \
  --memory-provider vllm \
  --memory-model Qwen/Qwen3-VL-8B-Instruct \
  --memory-vllm-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --memory-workers 8 \
  --checkpoint-interval 100 \
  --resume \
  --construct-from-raw \
  --image-root data/image/Filtered_images/Selected_Photos_2025_08_05_downsampled \
  --no-include-caption \
  --no-include-short-caption \
  --no-include-ocr-text \
  --no-include-tags \
  --disable-evolution

# Output:
# ==============================================================================
# ✓ MEMORY CONSTRUCTION COMPLETE
# ==============================================================================
# Cache key:      abc123def456...
# Cache location: output/QA_Agent/A-Mem/index_cache/
# Files saved:
#   - abc123def456_memories.pkl
#   - abc123def456_retriever.pkl
#   - abc123def456_retriever.npy
# Total memories: 1234
```


### Example: Stage 2 (Answer with Different Models)

```bash
# Experiment 1: Qwen3-VL 2B Answerer
python memqa/qa_agent_baselines/A-Mem/amem_baseline.py \
  --stage answer \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --image-batch-results memqa/qa_agent_baselines/example_qas/image_2026_01_06_qwen3vl8b_batch_results.json \
  --video-batch-results memqa/qa_agent_baselines/example_qas/video_2026_01_06_qwen3vl8b_batch_results.json \
  --email-file memqa/qa_agent_baselines/example_qas/merged_emails.json \
  --embedding-model all-MiniLM-L6-v2 \
  --construct-from-raw \
  --image-root data/image/Filtered_images/Selected_Photos_2025_08_05_downsampled \
  --no-include-caption \
  --no-include-short-caption \
  --no-include-ocr-text \
  --no-include-tags \
  --disable-evolution \
  --provider vllm \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --vllm-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --answer-from-raw \
  --retrieve-k 10 \
  --follow-links \
  --max-workers 128 \
  --output-dir-base output/QA_Agent/A-Mem \
  --method-name qwen3vl8b_memory_qwen3vl2b_answerer

# Experiment 2: Qwen3-VL 4B Answerer (same cache, different model!)
python memqa/qa_agent_baselines/A-Mem/amem_baseline.py \
  --stage answer \
  [same cache-affecting args as above] \
  --provider vllm \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --vllm-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --answer-from-raw \
  --retrieve-k 10 \
  --follow-links \
  --max-workers 128 \
  --output-dir-base output/QA_Agent/A-Mem \
  --method-name qwen3vl8b_memory_qwen3vl4b_answerer
```

### Retrieval Logging (Top-100)

A‑Mem now logs detailed retrieval outputs (mirroring the MMRag format):

- `retrieval_recall_summary.json`: aggregated recall metrics (R@1/5/10/25/50/100)
- `retrieval_recall_details.json`: per‑QA retrieval IDs, scores, and recall values

By default, **top‑100** items are logged (`--retrieval-log-k 100`), while **top‑k** (default 10) are used as context for answering.

### Cache Validation & Error Messages

Stage 2 automatically validates cache existence before proceeding. If cache is missing, you'll see:

```
================================================================================
ERROR: Memory cache not found!
================================================================================
Expected cache key:  abc123def456...
Expected location:   output/QA_Agent/A-Mem/index_cache/
Required files:
  - abc123def456_memories.pkl ✗ MISSING
  - abc123def456_retriever.pkl ✗ MISSING
  - abc123def456_retriever.npy ✗ MISSING

You must run Stage 1 (memory construction) first:
  python amem_baseline.py --stage build \
    [all cache-affecting args]

Cache-affecting arguments:
  --image-batch-results, --video-batch-results, --email-file
  --embedding-model, --memory-provider, --memory-model
  --disable-evolution, --evo-threshold, --use-hybrid
  --caption-only, --use-short-caption, --construct-from-raw
  --include-type, --include-caption, --include-short-caption
  --include-ocr-text, --include-tags
================================================================================
```

### Helper: Print Cache Key

Preview the cache key without running the pipeline:

```bash
python memqa/qa_agent_baselines/A-Mem/amem_baseline.py \
  --print-cache-key \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --image-batch-results memqa/qa_agent_baselines/example_qas/image_2026_01_06_qwen3vl8b_batch_results.json \
  --embedding-model all-MiniLM-L6-v2 \
  --disable-evolution

# Output:
# Cache key: abc123def456...
# Cache dir: output/QA_Agent/A-Mem/index_cache
```

---

## Architecture & Code Structure

```
A-Mem/
├── amem_baseline.py     # Main CLI entrypoint
├── memory_layer.py      # Core A-Mem classes (MemoryNote, AgenticMemorySystem)
├── config.py            # Defaults and prompts
└── README.md            # This file
```

### Memory Construction Pipeline

Our implementation mirrors the original A-Mem paper's flow, but it's important to distinguish between **Data Preprocessing** and **Memory Construction**:

1. **Preprocessing (Multimodal Pipeline)**: 
   - Images and videos are processed *before* A-Mem runs.
   - VLM (e.g., Qwen3-VL) generates captions, OCR, and tags.
   - Output: `batch_results.json`. This is the *raw input* for A-Mem.

2. **Memory Construction (A-Mem "Note Taking")**:
   - `amem_baseline.py` reads `batch_results.json`.
   - `AgenticMemorySystem.add_note()` is called for each item.
   - **LLM Analysis**: `MemoryNote.analyze_content` calls the LLM (e.g., `gpt-4o-mini`) to "Generate a structured analysis...".
   - This transforms raw captions into a **structured Note** (keywords, context, tags). This matches the paper's "Memory Creation" phase.

3. **Memory Evolution (A-Mem "Linking")**:
   - Immediately after a note is created (inside `add_note`), `_process_memory` is called.
   - **LLM Evolution**: The system retrieves neighbors and asks the LLM "You are an AI memory evolution agent...".
   - It determines `should_evolve`, `actions` (strengthen/update), and creates links or updates tags.
   - This sequential "Insert → Construct → Evolve" flow is consistent with the original paper.

### Core Classes (memory_layer.py)

#### `MemoryNote`
A dataclass representing a single memory unit:
```python
@dataclass
class MemoryNote:
    content: str           # Original text (caption, email body, etc.)
    id: str                # Unique identifier (e.g., "IMG_1234", "email_001")
    keywords: List[str]    # LLM-extracted salient terms
    context: str           # One-sentence summary
    tags: List[str]        # Categorical labels
    links: List[str]       # IDs of connected memories (via evolution)
    timestamp: str         # Temporal anchor
```

**Content Analysis**: When a note is created, `MemoryNote.analyze_content()` calls the LLM with:
```
Generate a structured analysis of the following content by:
1. Identifying the most salient keywords.
2. Extracting core themes and contextual elements.
3. Creating relevant categorical tags.

Format as JSON: {"keywords": [...], "context": "...", "tags": [...]}
```

#### `HybridRetriever`
A retrieval layer matching the original A-Mem paper:
- Combines **BM25** (lexical) and **embedding similarity** (semantic) with configurable weight `alpha`
- Formula: `score = alpha * BM25_score + (1 - alpha) * semantic_score`
- Default `alpha=0.5` (equal weight, matching original paper)
- `add_documents(docs)`: Tokenize for BM25, encode for embeddings
- `search(query, k)`: Return top-k indices by hybrid score
- `save/load`: Persist BM25 corpus + embeddings to disk

#### `SimpleEmbeddingRetriever`
A lightweight embedding-only retrieval layer (for ablation studies):
- `add_documents(docs)`: Encode and store documents
- `search(query, k)`: Return top-k indices by cosine similarity
- `save/load`: Persist embeddings to disk for caching

#### `AgenticMemorySystem`
The main orchestrator:
- `add_note(content, timestamp, note_id, disable_evolution)`: Create a MemoryNote, optionally trigger evolution
- `_process_memory(note)`: The evolution step — find neighbors, ask LLM if/how to evolve
- `find_related_memories(query, k, follow_links)`: Retrieve top-k memories, optionally traverse links

### Main Pipeline (amem_baseline.py)

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY BUILD PHASE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load Data Sources                                           │
│     ├── image_batch_results.json  (captions, OCR, tags)         │
│     ├── video_batch_results.json  (captions, OCR, tags)         │
│     └── merged_emails.json        (subject, body, summary)      │
│                                                                 │
│  2. Build Retrieval Items                                       │
│     └── build_retrieval_items() → List[RetrievalItem]           │
│         Each item has: item_id, text, metadata                  │
│                                                                 │
│  3. Convert to Memory Rows                                      │
│     └── memory_from_items() → List[(id, content, timestamp)]    │
│                                                                 │
│  4. Build or Load Memory System                                 │
│     ├── Check cache: {cache_key}_memories.pkl, *_retriever.pkl  │
│     │                                                           │
│     │   If cached: Load memories + embeddings                   │
│     │   Else: For each row:                                     │
│     │         ├── LLM: Extract keywords/context/tags            │
│     │         ├── Create MemoryNote                             │
│     │         ├── (Optional) Evolution: Update neighbors        │
│     │         └── Embed and store                               │
│     │                                                           │
│     └── Save cache for future runs                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QA PHASE (per question)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Memory Retrieval (Aligned with Original)                    │
│     └── find_related_memories_raw(question, k=10)               │
│         ├── Hybrid search: BM25 + Embedding (alpha=0.5)         │
│         ├── Link traversal: Gather connected memories           │
│         └── Returns: "20240315: Sushi dinner at Tsukiji..."     │
│                                                                 │
│  2. Answer Generation                                           │
│     └── LLM: System + User prompt with evidence                 │
│         "Use ONLY the provided evidence to answer..."           │
│                                                                 │
│  3. Output                                                      │
│     └── {"id": "q_001", "answer": "Tsukiji Sushi Restaurant"}   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Evolution (Optional)

When `--disable-evolution` is **not** set, each new memory triggers:

```python
def _process_memory(self, note: MemoryNote):
    # 1. Find 5 nearest neighbors
    neighbors = self.find_related_memories(note.content, k=5)
    
    # 2. Ask LLM: Should we evolve?
    prompt = f"""
    You are an AI memory evolution agent.
    New memory context: {note.context}
    Content: {note.content}
    Keywords: {note.keywords}
    Neighbors: {neighbors}
    
    Return JSON:
    - should_evolve: bool
    - actions: ["strengthen", "update_neighbor", ...]
    - suggested_connections: [indices of neighbors to link]
    - tags_to_update: new tags for this memory
    - new_context_neighborhood: updated contexts for neighbors
    - new_tags_neighborhood: updated tags for neighbors
    """
    
    # 3. Apply evolution actions
    if response["should_evolve"]:
        if "strengthen" in actions:
            note.links.extend(connection_ids)
            note.tags = tags_to_update
        if "update_neighbor" in actions:
            for neighbor in neighbors:
                neighbor.context = new_context
                neighbor.tags = new_tags
```

**Trade-off**: Evolution creates richer memory structures but requires O(N) LLM calls during indexing. For large datasets, use `--disable-evolution`.

---

## Differences from Original Implementation

### Aligned Components

The following components match the original A-Mem implementation:

| Component | Original | Our Implementation |
|-----------|----------|-------------------|
| Embedding Model | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` |
| Hybrid Retrieval | BM25 + Embedding (alpha=0.5) | BM25 + Embedding (alpha=0.5) |
| Link Traversal | Enabled | Enabled (via `--follow-links`) |
| Query Processing | Direct query embedding | Direct query embedding |
| Memory Analysis | LLM extracts keywords/context/tags | LLM extracts keywords/context/tags |
| Evolution Logic | LLM-driven strengthen/update | LLM-driven strengthen/update |

### Intentional Differences

#### 1. No Activation Decay
- **Original**: Memories have activation scores that decay over time
- **Ours**: All memories are equally weighted
- **Reason**: Batch QA has no temporal access patterns to exploit

#### 2. Flexible LLM Backends
- **Original**: OpenAI API only
- **Ours**: OpenAI, vLLM (OpenAI-compatible), vLLM local
- **Reason**: Support for local/self-hosted models

#### 3. Media-Centric Memory Construction
- **Original**: Generic text notes
- **Ours**: VLM-generated captions from images/videos (use `--caption-only` for clean comparison)
- **Reason**: Tailored for PersonalMemoryQA data format

#### 4. Caching
- **Original**: No caching
- **Ours**: Pickle-based cache for memories + numpy cache for embeddings + BM25 corpus
- **Reason**: Avoid re-running expensive LLM analysis on every run

---

## Configuration

### CLI Arguments

#### Retrieval Options (Aligned with Original)

| Argument | Default | Description |
|----------|---------|-------------|
| `--alpha` | `0.5` | Hybrid weight: `alpha*BM25 + (1-alpha)*embedding` (paper default). |
| `--use-hybrid` | `False` | Enable hybrid BM25+embedding retrieval (paper mentions hybrid, code is embedding-only) |
| `--no-use-hybrid` | - | Use embedding-only retrieval (aligned with original code) |
| `--follow-links` | `True` | Traverse memory links during retrieval (aligned with original) |
| `--no-follow-links` | - | Disable link traversal (for ablation) |

#### Text Augmentation Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--caption-only` | `False` | Use minimal text mode (Caption + ID/Time/Location only). Aligned with paper. |
| `--use-short-caption` | `False` | Use `short_caption` instead of full `caption` (requires `--caption-only`). |
| `--include-ocr-text` | `True` | Include OCR text (disabled if `--caption-only` is set) |
| `--include-tags` | `True` | Include tags (disabled if `--caption-only` is set) |
| `--include-email-summary` | `True` | Include email summary |
| `--include-email-detail` | `True` | Include email detailed body |

#### Multimodal Options

**A-Mem supports two modes for using visual information:**

1. **Text-only (default)**: Memory construction LLM sees only VLM-generated captions (from `--image-batch-results`)
2. **Multimodal**: Memory construction LLM sees both caption text AND raw images

| Argument | Default | Description |
|----------|---------|-------------|
| `--construct-from-raw` | `False` | **Memory Construction**: Pass raw images to the memory LLM during `MemoryNote.analyze_content()`. Requires multimodal `--memory-model` (e.g., `gpt-4o`, `Qwen3-VL`). When enabled, the LLM analyzes both the caption text and visual content to extract keywords/context/tags. |
| `--answer-from-raw` | `False` | **Answer Generation**: Pass raw images to the answer LLM during QA. Requires multimodal `--model`. Retrieved memories include their associated images in the prompt. |

**Important distinctions:**

- `--construct-from-raw`: Whether the **memory construction** LLM sees images (in addition to captions)
- `--answer-from-raw`: Whether the **answer generation** LLM sees images (in addition to retrieved text)

**Example: Multimodal memory construction**
```bash
python memqa/qa_agent_baselines/A-Mem/amem_baseline.py \
  --construct-from-raw \
  --image-root data/image/Filtered_images/Selected_Photos_2025_08_05_downsampled \
  --memory-provider vllm \
  --memory-model Qwen/Qwen3-VL-8B-Instruct \
  --memory-vllm-endpoint http://127.0.0.1:8000/v1/chat/completions \
  ...
```

**Example: Multimodal answer generation**
```bash
python memqa/qa_agent_baselines/A-Mem/amem_baseline.py \
  --answer-from-raw \
  --image-root data/image/Filtered_images/Selected_Photos_2025_08_05_downsampled \
  --provider vllm \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --vllm-endpoint http://127.0.0.1:8000/v1/chat/completions \
  ...
```

**Note**: Both modes require VLM-capable models. Text-only models like `gpt-4o-mini` will fail if you enable these flags.

#### Memory LLM Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--memory-provider` | Same as `--provider` | Provider for memory operations (openai, vllm, vllm_local) |
| `--memory-model` | Same as `--model` | Model for memory operations. Original paper uses `gpt-4o-mini`. |
| `--memory-api-key` | Same as `--api-key` | API key for memory operations |
| `--memory-vllm-endpoint` | Same as `--vllm-endpoint` | Endpoint for memory operations |

#### Memory Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence-transformer model (matches original) |
| `--retrieve-k` | `10` | Number of memories to retrieve per query |
| `--retrieval-log-k` | `100` | Log top-K retrieved items for analysis (default 100) |
| `--evo-threshold` | `100` | Evolution frequency (lower = more frequent) |
| `--disable-evolution` | `False` | Skip memory evolution during indexing |

#### Content Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--caption-only` | `False` | Use only caption for memory content (recommended for baseline) |
| `--index-cache` | `output/QA_Agent/A-Mem/index_cache` | Cache directory for memories/embeddings |
| `--force-rebuild` | `False` | Ignore cache and rebuild memory index |

### Cache Key Components
The cache is keyed by:
- `embedding_model`
- `evo_threshold`
- `disable_evolution`
- `alpha` (hybrid retrieval weight)
- `use_hybrid` (retrieval mode)
- `caption_only` (content mode)
- `use_short_caption` (content detail)
- `construct_from_raw` (multimodal build)
- `memory_provider`
- `memory_model`
- Granular text flags (`include_ocr_text`, `include_tags`, etc.)
- Input file paths (image/video batch results, email file)

Changing any of these triggers a cache miss and full rebuild.

---

## Dependencies
Install A-Mem dependencies in addition to core requirements:

```bash
pip install sentence-transformers rank-bm25 scikit-learn nltk litellm
```

## Output
Outputs are written under:
```
output/QA_Agent/A-Mem/<method>/amem_answers.jsonl
```

## Example Run (Aligned with Original - Recommended)
```bash
python memqa/qa_agent_baselines/A-Mem/amem_baseline.py \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --image-batch-results memqa/qa_agent_baselines/example_qas/image_2026_01_06_qwen3vl8b_batch_results.json \
  --video-batch-results memqa/qa_agent_baselines/example_qas/video_2026_01_06_qwen3vl8b_batch_results.json \
  --email-file memqa/qa_agent_baselines/example_qas/merged_emails.json \
  --provider vllm \
  --vllm-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --embedding-model all-MiniLM-L6-v2 \
  --follow-links \
  --caption-only \
  --use-short-caption \
  --retrieve-k 10 \
  --output-dir-base output/QA_Agent/A-Mem \
  --method-name amem_hybrid_llm
```

## Evaluation
```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth memqa/utils/final_data_processing/atm-20260121.json \
  --predictions output/QA_Agent/A-Mem/amem_aligned/amem_answers.jsonl \
  --output-dir output/QA_Agent/A-Mem/amem_aligned/eval \
  --metrics em

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth memqa/utils/final_data_processing/atm-20260121.json \
  --predictions output/QA_Agent/A-Mem/amem_aligned/amem_answers.jsonl \
  --output-dir output/QA_Agent/A-Mem/amem_aligned/eval \
  --metrics llm \
  --judge-provider vllm \
  --judge-model "GLM-4.7" \
  --judge-endpoint "http://127.0.0.1:8000/v1/chat/completions" \
  --max-workers 2
```
