# Baselines

This repo ships multiple **reference baselines** for ATMBench under:

- Run scripts: `scripts/QA_Agent/<Baseline>/`
- Baseline implementations: `memqa/qa_agent_baselines/<Baseline>/`

This document explains:
1) what each baseline is measuring,  
2) why it is implemented the way it is (design rationale), and  
3) the important knobs (CLI flags and environment variables) you may want to change.

If you are looking for the **expected local data layout**, start with `docs/data.md`.

---

## Baseline Principles (Why they look similar)

### 1) Evidence-grounded answering

ATMBench is an **evidence-grounded** benchmark. Most baselines use prompts that:
- instruct the model to answer **only from retrieved evidence**, and
- return `Unknown` when evidence is insufficient.

This reduces “hallucinated correctness” and makes retrieval quality measurable.

### 2) Text-only vs raw-media modes

Many baselines accept:
- `--media-source batch_results`: treat images/videos as **text** via precomputed `batch_results.json` (captions/OCR/tags).
- `--media-source raw`: load **raw** images/videos and insert them into the prompt (multimodal inference).

Rationale:
- `batch_results` is cheaper, faster, and easier to reproduce.
- `raw` is closer to “true multimodal QA”, but it is slower and depends on a multimodal LLM endpoint.

### 3) “Memory/index model” vs “answerer model”

Several baselines are two-stage or two-model:
- a **memory/index model** for ingestion / extraction / note building
- an **answerer model** for final QA

Rationale:
- ingestion is often the expensive “O(#memories)” stage; you can use a smaller model here,
- QA is “O(#questions)”; you can use a larger model for higher answer quality.

In this repo’s canonical run scripts, the defaults follow:
- **2B** model for memory/indexing
- **8B** model for answering

You can override models via environment variables in the `run.sh` scripts.

### 4) Avoiding evaluation contamination

Some memory systems update memory after answering a question (e.g., appending the QA itself).
That contaminates evaluation because later questions can “cheat”.

Where applicable we implement:
- **per-question isolation** (fresh instance restored from a checkpoint), and/or
- **no-update evaluation mode** (skip post-answer writes).

### 5) Reproducibility and caching

Baselines write only to `output/` and are designed to be restartable:
- most runs are keyed by a `--method-name` (output subdirectory),
- long stages cache indexes/checkpoints,
- many scripts will skip work when outputs already exist.

---

## Setup (One-time)

From repo root:

```bash
conda create -n atmbench python=3.11 -y
conda activate atmbench

pip install -r requirements.txt
pip install -e .
```

Baseline-specific dependencies are intentionally **not** all in root `requirements.txt`.
Install what you need:

```bash
pip install -r memqa/qa_agent_baselines/HippoRag2/requirements.txt
pip install -r memqa/qa_agent_baselines/MemoryOS/requirements.txt
pip install -r memqa/qa_agent_baselines/A-Mem/requirements.txt
pip install -r memqa/qa_agent_baselines/mem0/requirements.txt
```

Notes:
- Some baselines have heavy/native deps (e.g., `python-igraph`, `faiss-*`). Install them per your platform.
- Some baselines vendor upstream code under `third_party/` (see `third_party/README.md`).

## Environment Recommendations

- Core baselines (`MMRAG`, `Oracle`, `NIAH`) are tested in the main `atmbench` environment.
- Third-party memory-system baselines in this repo are:
  - `A-Mem`
  - `HippoRAG2`
  - `mem0`
  - `MemoryOS`
- `MemoryOS` is strongly recommended to run in a separate conda environment.
- `A-Mem`, `HippoRAG2`, and `mem0` are tested to be compatible with the core baseline environment, but separate environments are still safer for dependency isolation.
- Setup for these four baselines is documented under:
  - `third_party/A-mem/`
  - `third_party/HippoRAG/`
  - `third_party/mem0/`
  - `third_party/MemoryOS/`
- OpenClaw, OpenCode, and Codex baselines are compatible with this repo’s evaluation workflow, but each requires its own third-party software installation.

## Vendored upstream code (third_party)

For reproducibility, some baselines vendor upstream code directly into this repo under `third_party/`.

What to expect in a vendored subtree:
- `LICENSE` (upstream license)
- `UPSTREAM.md` (source URL + pinned commit/tag)
- optional `ATMBench.patch` (local changes applied on top of upstream snapshot)

Baselines that rely on vendored code:
- HippoRAG 2: `third_party/HippoRAG/`
- MemoryOS: `third_party/MemoryOS/` (baseline can also use a pip-installed `memoryos` package)
- A‑Mem: `third_party/A-mem/`
- Mem0: `third_party/mem0/` (baseline can also use a pip-installed `mem0` / `mem0ai`)

---

## Common Runtime Knobs

### Endpoints and keys

Most scripts assume an OpenAI-compatible Chat Completions endpoint:

- `VLLM_ENDPOINT` (default: `http://127.0.0.1:8000/v1/chat/completions`)
- `VLLM_API_KEY` (optional; picked up by code if your endpoint requires auth)
- `OPENAI_API_KEY` (required only for baselines using `--provider openai` or for LLM-judge eval)

This repo supports providers:
- `--provider vllm`: OpenAI-compatible HTTP endpoint (vLLM, etc.)
- `--provider openai`: OpenAI API (Responses API preferred when available)
- `--provider vllm_local`: local vLLM Python runtime (used by some baselines)

### Data paths

All canonical scripts assume the `docs/data.md` layout:

- QA files:
  - `data/atm-bench/atm-bench.json`
  - `data/atm-bench/atm-bench-hard.json`
- Text evidence for media:
  - `output/image/qwen3vl2b/batch_results.json`
  - `output/video/qwen3vl2b/batch_results.json`
- Optional emails:
  - `data/raw_memory/email/emails.json`

### Generating `batch_results.json` (required for `--media-source batch_results`)

Most baselines are text-first and expect `batch_results.json` to exist. In this repo, the default
baseline scripts read directly from:

- `output/image/qwen3vl2b/batch_results.json`
- `output/video/qwen3vl2b/batch_results.json`

Note on GPS / reverse-geocoding:
- By default, the processors reverse-geocode GPS coordinates via a public provider (OpenStreetMap Nominatim).
- Public geocoding endpoints are rate-limited (often strict per-IP requests/minute) and do not tolerate high concurrency.
- If you have a pre-extracted GPS cache bundle, place the `*_location_name.json` entries under
  `data/raw_memory/geocoding_cache/image` and `data/raw_memory/geocoding_cache/video`, then copy them into your
  processor cache directory **before** running the memory processors so geocoding is skipped and runs don’t stall.

Example (Qwen3-VL-2B captioner via vLLM):

```bash
# Optional (recommended): copy pre-extracted GPS cache to skip geocoding calls
python memqa/utils/copy_gps_info.py data/raw_memory/geocoding_cache/image output/image/qwen3vl2b/cache
python memqa/utils/copy_gps_info.py data/raw_memory/geocoding_cache/video output/video/qwen3vl2b/cache

# Generate batch results
bash scripts/memory_processor/image/memory_itemize/run_qwen3vl2b.sh
bash scripts/memory_processor/video/memory_itemize/run_qwen3vl2b.sh
```

The `run_qwen3vl2b.sh` wrappers write these files directly under `output/image/qwen3vl2b/`
and `output/video/qwen3vl2b/`, which is what the QA baselines read by default.

### Output directories

Outputs live under `output/QA_Agent/<Baseline>/...`.
If you want to re-run with different settings, prefer changing:
- `--method-name` (creates a new subdir), or
- `--output-dir-base` (separates experiment families).

---

## Baselines

### Oracle (Upper Bound Answering)

**What it is:** An *upper bound* baseline that uses **ground-truth evidence IDs** from the dataset and asks the model to answer.

**What it measures:** Answer synthesis quality given perfect retrieval (not a retrieval baseline).

**Default script:**
- `bash scripts/QA_Agent/Oracle/run_oracle_qwen3vl8b_raw.sh`

**Related scripts:**
- `scripts/QA_Agent/Oracle/run_oracle_no_evidence_qwen3vl8b.sh` (answers with no evidence; sanity check)
- `scripts/QA_Agent/Oracle/run_oracle_gpt5.sh` (OpenAI answerer variant)

**Implementation:**
- `memqa/qa_agent_baselines/oracle/oracle_baseline.py`

**Key design choices:**
- Uses dataset-provided evidence IDs, so retrieval recall is not meaningful here.
- Typically uses `--media-source raw` so the model sees the true pixels for image/video evidence.

**Important knobs (script-level):**
- `VLLM_ENDPOINT`: answerer endpoint
- `--model`: answerer model
- `--max-workers`: QA concurrency

**Common CLI flags (implementation):**
- `--qa-file`: QA JSON input
- `--output-file`: where to write answers JSONL
- `--use-niah-pools`: used by NIAH wrapper to swap in fixed evidence pools
- `--media-source {raw,batch_results}`: raw pixels vs text-only evidence
- `--provider {vllm,openai,vllm_local}` / `--vllm-endpoint` / `--model`: answerer backend
- `--timeout` / `--max-workers`: reliability + throughput

**Outputs (typical):**
- `<output>/oracle_*.jsonl` (answers)

---

### MMRAG (Retrieve → Answer Baseline)

**What it is:** A strong “standard” baseline: retrieve top-k evidence with an embedding retriever, then answer with an evidence-only prompt.

**What it measures:** End-to-end retrieval + answering, with minimal “memory system” behavior.

**Default script:**
- `bash scripts/QA_Agent/MMRAG/run.sh`

**Key design choices:**
- Default mode is `batch_results` (text-only) to keep runs cheap and reproducible.
- Retrieval is pluggable (`--retriever`) to support text-only and VL embedding retrieval.
- Uses resume/checkpoint JSONL files so partial runs can be resumed.

**Important knobs:**
- `TOP_K` (script env var): number of retrieved items used for answering
- `TEXT_EMBED_MODEL` (script env var): text embedding model for `sentence_transformer` retriever
- `--retriever`: `sentence_transformer`, `text`, `qwen3_vl_embedding`, `clip`, `vista`
- `--reuse-retrieval-results/--no-reuse-retrieval-results`: cache retrieval outputs
- `--insert-raw-images`: optionally attach raw images even when retrieval is text-only
- `--critic-answerer`: draft + critic verification (slower; can reduce unsupported answers)

**Common CLI flags (most used):**
- Data:
  - `--qa-file`
  - `--media-source {batch_results,raw}`
  - `--image-batch-results`, `--video-batch-results`, `--email-file`
- Retrieval:
  - `--retriever`, `--retrieval-top-k`
  - `--text-embedding-model` (text retrievers)
  - `--vl-embedding-model` / `--clip-model` / `--vista-*` (VL retrievers)
  - `--force-rebuild`, `--index-cache`, `--reuse-retrieval-results`
- Answering:
  - `--provider`, `--vllm-endpoint`, `--model`
  - `--no-evidence` (answer without evidence; debug)
  - `--max-workers`, `--timeout`

**Where to look for full CLI:** `memqa/qa_agent_baselines/MMRag/mmrag_retrieve_answer.py`

**Baseline README:** `memqa/qa_agent_baselines/MMRag/README.md`

**Outputs (typical):**
- `<output>/mmrag_answers.jsonl`
- `<output>/retrieval_recall_details.json`
- `<output>/retrieval_recall_summary.json`

---

### HippoRAG 2 (Graph Memory Baseline)

**What it is:** A graph-based memory baseline (HippoRAG 2) that builds a knowledge graph via OpenIE and retrieves via Personalized PageRank + reranking.

**What it measures:** Whether structured graph memory improves retrieval compared to pure embedding baselines.

**Default script (2B index/OpenIE, 8B answerer):**
- `bash scripts/QA_Agent/HippoRag2/run.sh`

**Key design choices:**
- We vendor upstream under `third_party/HippoRAG` and default `--hipporag-repo` points there.
- Two-stage pipeline:
  - `--stage build`: build HippoRAG index (OpenIE, graph construction)
  - `--stage answer`: run QA using the cached index
- Index cache is keyed on embedding model + OpenIE model + augmentation level + corpus hash.

**Important knobs:**
- `MODEL` (script env var): index/OpenIE model (default 2B)
- `ANSWERER_MODEL` (script env var): QA/rerank model (default 8B)
- `AUGMENTATION_LEVEL`: how much text is included from `batch_results` (caption-only vs caption+tags+ocr)
- `--openie-workers`: parallelism for OpenIE calls
- `--qa-top-k`: number of retrieved passages provided to the answerer

**Common CLI flags (most used):**
- Pipeline:
  - `--stage {build,answer,all}`
  - `--force-rebuild` (rebuild index)
  - `--force-openie` (re-run OpenIE extraction)
- Upstream integration:
  - `--hipporag-repo` (defaults to `third_party/HippoRAG`)
- Indexing/retrieval:
  - `--embedding-model`
  - `--augmentation-level`
  - `--retrieval-top-k`, `--linking-top-k`, `--qa-top-k`
  - `--openie-mode`, `--openie-workers`
- Answering:
  - `--provider`, `--vllm-endpoint`, `--model` (index/OpenIE model)
  - `--answerer-model`, `--answerer-endpoint` (QA/rerank model)
- Output:
  - `--index-cache` (index cache root)
  - `--output-dir-base`, `--method-name`

**Dependencies:**
- Requires `python-igraph` (and other deps in `memqa/qa_agent_baselines/HippoRag2/requirements.txt`).

**Where to look for full CLI:** `memqa/qa_agent_baselines/HippoRag2/hipporag2_baseline.py`

**Baseline README:** `memqa/qa_agent_baselines/HippoRag2/README.md`

**Outputs (typical):**
- `<output>/hipporag2_answers.jsonl`
- `<output>/retrieval_recall_details.json`

---

### MemoryOS (Tiered Memory System Baseline)

**What it is:** An OS-inspired tiered memory system (STM/MTM/LPM) adapted for batch QA.

**What it measures:** Whether tiered memory management + session/page retrieval helps on long-horizon memory QA.

**Default script (2B index, 8B answerer):**
- `bash scripts/QA_Agent/MemoryOS/run.sh`

**Key design choices:**

1) **Chronological indexing**
- We merge emails/images/videos and sort by timestamp before ingestion.

2) **Per-question isolation (prevents contamination)**
- By default we restore from a checkpoint for each QA so the QA itself is not added to memory for later QAs.

3) **Full-history mode (benchmark variant)**
- Paper-default capacities (e.g., MTM≈2000) can aggressively summarize/evict historical items.
- For large memory banks, this can produce very poor retention and can degrade ingestion/runtime behavior.
- We therefore provide `--memoryos-full-history-mode` which enforces floors on MTM capacity and session search.

In this repo’s canonical `run.sh`, full-history mode is enabled by default:
- `FULL_HISTORY_MODE=1` (script-level)
- method name is suffixed as `memoryos_fullhistory` to make the variant explicit.

**Paper defaults vs full-history variant (what changes):**

- Paper-ish defaults (see `memqa/qa_agent_baselines/MemoryOS/config.py`):
  - `memoryos_short_term_capacity=10`
  - `memoryos_mid_term_capacity=2000`
  - small retrieval/search depths (paper uses small queue/session counts)
- Full-history variant (this repo’s benchmark setting):
  - `memoryos_mid_term_capacity >= 20000`
  - `memoryos_top_k_sessions >= 200`
  - `memoryos_heat_threshold >= 1e9` (suppresses MTM→LPM promotion churn)

Practical implication on large memory banks:
- paper-ish defaults can retain only ~MTM capacity worth of page-level history and can become slow/unstable;
- full-history mode retains substantially more historical items and keeps ingestion behavior stable.

**Important knobs:**
- `TOP_K`: retrieval queue capacity (pages) used during evaluation
- `TOP_K_SESSIONS`: MTM session search depth
- `EVAL_NO_UPDATE`: if `1`, passes `--memoryos-eval-no-update` to skip post-answer writes
- `FULL_HISTORY_MODE`: enable/disable the benchmark full-history variant

**Common CLI flags (most used):**
- Models:
  - `--provider`, `--vllm-endpoint`
  - `--memoryos-index-llm-model` (indexing LLM)
  - `--memoryos-answer-llm-model` (answering LLM)
  - `--memoryos-embedding-model`
- Contamination control:
  - `--memoryos-use-per-qa-instance` (isolate each QA)
  - `--memoryos-eval-no-update` (skip post-answer writes)
- Checkpointing:
  - `--memoryos-checkpoint-dir`
  - `--memoryos-reuse-checkpoint/--no-memoryos-reuse-checkpoint`
  - `--memoryos-save-checkpoint/--no-memoryos-save-checkpoint`
  - `--memoryos-force-rebuild`
- Retrieval/capacity:
  - `--memoryos-retrieval-queue-capacity`
  - `--memoryos-top-k-sessions`
  - `--memoryos-full-history-mode` (+ its `--memoryos-full-history-*` floors)

**Where to look for full CLI:** `memqa/qa_agent_baselines/MemoryOS/memoryos_baseline.py`

**Baseline README:** `memqa/qa_agent_baselines/MemoryOS/README.md`

**Outputs (typical):**
- `<output>/memoryos_answers.jsonl`
- `<output>/retrieval_recall_details.json`

---

### A‑Mem (Agentic Memory Baseline)

**What it is:** An “agentic memory” system that analyzes each memory item with an LLM to create structured notes (keywords/context/tags) and optionally evolves links.

**What it measures:** Whether LLM-augmented memory construction + link traversal improves retrieval/QA.

**Default script (2B memory, 8B answerer, text-only construction):**
- `bash scripts/QA_Agent/A-Mem/run.sh`

**Key design choices:**

1) **Two-stage pipeline with explicit caching**
- `--stage build` constructs memories and caches them to disk.
- `--stage answer` loads the cache and answers questions.

2) **Batch-results (text-only) by default in this repo**
- The default run script builds from `batch_results.json` for speed/reproducibility.
- You can switch to raw-media construction/answering with:
  - `--construct-from-raw` (Stage 1)
  - `--answer-from-raw` (Stage 2)

**What we used in development experiments:**
- The primary A‑Mem scripts used **raw image construction** (`--construct-from-raw`) with minimal text
  fields (ID/timestamp/location only), and then answered with a larger model.
- An older archived variant built from `batch_results` (`--caption-only --use-short-caption`).
This repo keeps raw mode as an optional variant because it is slower and more endpoint-dependent.

**Important knobs:**
- `MEMORY_MODEL`: memory/index model (default 2B)
- `ANSWER_MODEL`: answerer model (default 8B)
- `--index-cache`: cache directory (must be shared between Stage 1 and Stage 2)
- `--disable-evolution`: disable link evolution (faster, simpler)
- `--follow-links/--no-follow-links`: whether retrieval traverses memory links
- `--caption-only`: minimal text mode (useful when comparing captioners)
- `--construct-from-raw` / `--answer-from-raw`: raw image variants (slower; endpoint-dependent)

**Correctness pitfall (common):**
- Stage 2 must reuse the **same cache key** as Stage 1. That means cache-affecting args must match:
  `--image-batch-results`, `--video-batch-results`, `--email-file`, `--embedding-model`,
  and memory-LLM settings like `--memory-model` / `--memory-provider`.

**Common CLI flags (most used):**
- Pipeline:
  - `--stage {build,answer,all}`
  - `--index-cache` (cache dir)
  - `--resume`, `--force-rebuild`, `--checkpoint-interval`
- Memory construction:
  - `--memory-provider`, `--memory-vllm-endpoint`, `--memory-model`
  - `--memory-workers` (parallel note construction)
  - `--disable-evolution`, `--evo-threshold`
- Retrieval/answer:
  - `--retrieve-k`, `--follow-links`
  - `--provider`, `--vllm-endpoint`, `--model` (answerer)
  - `--max-workers` (QA concurrency)
- Text mode:
  - `--caption-only`, `--use-short-caption`
  - fine-grained `--include-*` toggles

**Where to look for full CLI:** `memqa/qa_agent_baselines/A-Mem/amem_baseline.py`

**Baseline README:** `memqa/qa_agent_baselines/A-Mem/README.md`

**Outputs (typical):**
- `<output>/amem_answers.jsonl`
- `<output>/retrieval_recall_details.json`

---

### Mem0 (Memory Extraction + Vector Store Baseline)

**What it is:** A mem0-backed baseline that indexes memories into a vector store and retrieves top-k “memories” for QA.

**What it measures:** Whether mem0’s memory representation helps retrieval/QA on ATMBench-style evidence.

**Important note about mem0 extraction:**

mem0 supports an LLM-based extraction/update pipeline (`--mem0-infer`) that is optimized for
**conversational personal facts**. For structured batch-result style items, the upstream extraction prompt
often returns empty `facts`, producing a near-empty index.

Therefore, the canonical script defaults to:
- `MEM0_INFER=0` → `--no-mem0-infer` (store memories directly as embeddings)

If you want to try extraction, set:
- `MEM0_INFER=1`
- `USE_CUSTOM_EXTRACTION_PROMPT=1` (uses an ATMBench-adapted extraction prompt)

**Default script (8B answerer, local ST embedder, no-extraction indexing):**
- `bash scripts/QA_Agent/mem0/run.sh`

**Key design choices:**

1) **Stable indexing by explicit paths**
- The script pins:
  - `--mem0-vector-path` (Chroma dir)
  - `--mem0-history-db-path`
  - `--mem0-progress-path`
  - `--mem0-collection-name`
  so that `atmbench` and `hard` do not accidentally double-index into the same collection.

2) **Local embedding by default**
- Uses `--mem0-embedder-provider local` + `--mem0-local-retriever sentence_transformer`
  with `all-MiniLM-L6-v2` for reproducibility.

**Important knobs:**
- `TEXT_EMBEDDING_MODEL`: sentence-transformers model used for indexing/retrieval
- `MEM0_LOCAL_DEVICE`: `cpu` or `cuda` (if you want local embeddings on GPU)
- `MEM0_LLM_MODEL` + `MEM0_LLM_BASE_URL`: used only when `MEM0_INFER=1` (extraction)

**Common CLI flags (most used):**
- Indexing storage (avoid accidental double-indexing):
  - `--mem0-collection-name`
  - `--mem0-vector-path`
  - `--mem0-history-db-path`
  - `--mem0-progress-path`
  - `--mem0-user-id`
- Extraction behavior:
  - `--mem0-infer/--no-mem0-infer`
  - `--use-custom-extraction-prompt`
  - `--mem0-llm-model`, `--mem0-llm-base-url`
- Embedding behavior:
  - `--mem0-embedder-provider local`
  - `--mem0-local-retriever sentence_transformer`
  - `--text-embedding-model`
- QA:
  - `--mem0-top-k`
  - `--provider`, `--vllm-endpoint`, `--model`
  - `--max-workers`, `--timeout`

**Where to look for full CLI:** `memqa/qa_agent_baselines/mem0/mem0_baseline.py`

**Baseline README:** `memqa/qa_agent_baselines/mem0/README.md`

**Outputs (typical):**
- `<output>/mem0_answers.jsonl`
- `<output>/retrieval_recall_details.json`

---

### NIAH (Needle-In-A-Haystack; Generation-only)

**What it is:** A generation-only protocol for the hard split where each question includes a fixed evidence pool
(`niah_evidence_ids`) guaranteed to contain the ground truth.

**What it measures:** Answer synthesis under a fixed pool size *k* (not retrieval).

**Default script:**
- `bash scripts/QA_Agent/NIAH/run_niah_qwen3vl8b.sh`

**How it is implemented here:**
- `memqa/qa_agent_baselines/NIAH/niah_evaluate.py` is a wrapper around the Oracle baseline:
  it swaps `niah_evidence_ids -> evidence_ids` and then calls `oracle_baseline.py`.

**Common CLI flags (wrapper):**
- `--qa-file` / `--niah-qa-file`: NIAH pool QA file (must contain `niah_evidence_ids`)
- `--niah-field`: field name for pools (default: `niah_evidence_ids`)
- All other flags are passed through to `oracle_baseline.py` (provider/model/media flags, output path, etc.)

See `docs/niah.md` for schema and usage.
