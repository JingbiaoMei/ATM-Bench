# Mem0 Baseline

This directory contains the [Mem0](https://github.com/mem0ai/mem0) baseline adapted for the PersonalMemoryQA task. It leverages the Mem0 library to manage a personal memory store consisting of emails, image captions, and video summaries, enabling semantic retrieval for Question Answering.

## Library Reference

> **Mem0: The Memory Layer for AI Applications**
>
> - **GitHub**: [https://github.com/mem0ai/mem0](https://github.com/mem0ai/mem0)
> - **Docs**: [https://docs.mem0.ai/](https://docs.mem0.ai/)

## Method Overview

### What is Mem0?

Mem0 is a memory layer designed to enhance AI assistants by providing intelligent memory management. It typically handles:
- **Storage**: Storing user interactions and facts in a hybrid vector/graph store.
- **Retrieval**: Semantic search to recall relevant context.
- **Personalization**: Remembering user preferences over time.

### Our Adaptation for PersonalMemoryQA

We adapt Mem0 from a "conversational memory" tool into a **static retrieval-augmented generation (RAG) engine** for personal archives. Instead of remembering a chat history, we use Mem0 to index a massive static dataset of personal digital life (emails, photos, videos).

#### Key Adaptations

1. **Multimodal-to-Text Ingestion**:
   - Since Mem0 is primarily text-based, we convert multimodal evidence into rich text representations before indexing.
   - **Images/Videos**: Represented by VLM-generated captions, OCR text, and tags (from batch processing results).
   - **Emails**: Represented by subject, body summaries, and sender details.

2. **Custom Local Embedders (The "LangChain Adapter")**:
   - The native Mem0 library supports providers like OpenAI, but to support our specific research embedding models (e.g., **Qwen3-VL**, **VISTA**), we implemented a custom adapter.
   - We wrap our internal `memqa.retrieve` retrievers in a `langchain.embeddings.base.Embeddings` class.
   - This tricks Mem0 into using our specialized multimodal encoders for vectorization, ensuring that the "memory" of an image is semantically aligned with the questions.

3. **Batch Indexing Workflow**:
   - Unlike a chatbot that adds memories incrementally, we perform a **batch ingestion** phase where all evidence items are indexed into Mem0 at the start of the run (or loaded from a persisted vector store).
   - We index the **raw evidence text** (no Mem0 extraction/update) to preserve evidence IDs, timestamps, and locations for QA.

## Architecture & Code Structure

```
mem0/
├── mem0_baseline.py     # Main CLI entrypoint
├── config.py            # Configuration and Prompt templates
└── README.md            # This file
```

### The Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY BUILD PHASE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load Data Sources                                           │
│     ├── Batch Results (Images/Videos): Captions, OCR, Tags      │
│     └── Emails: Content & Metadata                              │
│                                                                 │
│  2. Build Text Payloads                                         │
│     └── Convert each item into a text chunk:                    │
│         "[Image] A dog playing in the park... (Date: 2023-01-01)"│
│                                                                 │
│  3. Initialize Mem0                                             │
│     ├── Vector Store: Chroma (Persisted locally)                │
│     └── Embedder: Custom Local Adapter (Qwen3-VL / Text / etc.) │
│                                                                 │
│  4. Indexing                                                    │
│     └── mem0.add(messages=[...])                                │
│         Vectors are computed via the custom adapter and stored. │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QA PHASE (per question)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Query Processing                                            │
│     └── Question: "Where did I go for dinner on Friday?"        │
│                                                                 │
│  2. Mem0 Retrieval                                              │
│     └── mem0.search(query, user_id="dataset", limit=k)          │
│         Retrieves top-k semantically relevant text chunks.      │
│                                                                 │
│  3. Answer Generation                                           │
│     └── LLM: System + User prompt with retrieved memories       │
│         "Based ONLY on these memories, answer the question..."  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Differences from Original Implementation

| Aspect | Standard Mem0 Usage | Our Baseline Implementation |
|--------|---------------------|-----------------------------|
| **Use Case** | Conversational Memory (User chats) | Static Knowledge Base QA (Personal Archive) |
| **Data Source** | Real-time chat logs | Pre-processed Batch JSONs (Captions, OCR, Emails) |
| **Embeddings** | OpenAI / HuggingFace (Standard) | **Custom Qwen3-VL / VISTA** via LangChain Adapter |
| **Persistence** | Often cloud-hosted or ephemeral | Local ChromaDB persistence for reproducibility |
| **Graph Memory**| Often enabled for entity relations | Mostly relying on Vector Search for this baseline |
| **Extraction/Update** | LLM-driven memory refinement | Disabled; raw evidence text is stored as memories |

## Configuration

The baseline supports extensive configuration via CLI arguments, inheriting defaults from `memqa/global_config.py`.

### Key CLI Arguments

- **Data Sources**:
  - `--qa-file`: Path to questions.
  - `--image-batch-results`, `--video-batch-results`: Paths to pre-computed VLM outputs.
  - `--email-file`: Path to email corpus.

- **Mem0 Configuration**:
  - `--mem0-vector-path`: Where to store/load the ChromaDB index.
  - `--mem0-top-k`: Number of memories to retrieve (default: 10).
  - `--mem0-log-k`: Max memories to log for recall (capped at 100).
  - `--mem0-infer`: Enable Mem0 extraction/update during indexing (default: true).
  - `--mem0-resume`: Skip items already indexed in the current cache (default: true).
  - `--mem0-progress-path`: Optional JSONL progress file path (defaults under the output dir).
  - `--mem0-local-retriever`: Which local embedding model to use (`text`, `sentence_transformer`, `clip`, `qwen3_vl`, `vista`).
  - Evidence formatting flags control whether IDs, timestamps, and locations are embedded into the memory text (e.g., `--include-id`, `--include-timestamp`, `--include-location`).

- **LLM Provider**:
  - `--provider`: `vllm`, `openai`, or `vllm_local`.
  - `--model`: Model name (e.g., `Qwen/Qwen3-VL-8B-Instruct`).

### Example Command

```bash
python memqa/qa_agent_baselines/mem0/mem0_baseline.py \
    --qa-file memqa/utils/final_data_processing/atm-20260121.json \
    --media-source batch_results \
    --image-batch-results memqa/qa_agent_baselines/example_qas/image_2026_01_06_qwen3vl8b_batch_results.json \
    --video-batch-results memqa/qa_agent_baselines/example_qas/video_2026_01_06_qwen3vl8b_batch_results.json \
    --email-file memqa/qa_agent_baselines/example_qas/merged_emails.json \
    --provider vllm \
    --vllm-endpoint http://127.0.0.1:8000/v1/chat/completions \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --output-dir-base output/QA_Agent/Mem0 \
    --method-name mem0_base \
    --mem0-embedder-provider local \
    --mem0-local-retriever text \
    --mem0-top-k 10 \
    --vl-text-augment
```

## Performance Notes: `infer=True` and Model Capability

### The `infer` Parameter

When `--mem0-infer` is enabled (the default), Mem0 uses the configured LLM to **extract facts** from each document before storing them. This triggers a multi-step pipeline per document:

```
For each document:
1. LLM Call #1: Fact extraction → returns N facts (0 to many)
2. For EACH extracted fact:
   - Embed the fact
   - Vector search against existing memories (limit=5)
3. LLM Call #2: Memory update decision (ADD/UPDATE/DELETE/NONE)
4. Execute the memory operations
```

When `--no-mem0-infer` is used, documents are stored directly as raw memories (1 embedding per document, no LLM calls during indexing).

### Speed Variation: Model Capability Affects Indexing Performance

We observed significant speed differences when using different LLM sizes for fact extraction:

| Model | Observed Behavior | Pattern |
|-------|-------------------|---------|
| **8B (Qwen3-VL-8B)** | Variable speed: sometimes very fast, then slows, then fast again | Oscillating |
| **2B (Qwen3-VL-2B)** | Consistently slow throughout the entire run | Uniform slow |

#### Root Cause: Fact Extraction Volume

The speed difference is **not** primarily due to network latency or model inference speed. It stems from **how many facts each model extracts** from the same input:

| Model | Extraction Behavior | Consequence |
|-------|---------------------|-------------|
| **8B (more capable)** | Better instruction following. Correctly identifies when there's nothing meaningful to extract (returns `{"facts": []}`). More selective about what constitutes a "fact". | Fast on simple docs (0 facts → skips expensive loop), slow on rich docs (many facts) |
| **2B (less capable)** | Worse at instruction following. Over-extracts "facts" from everything, including noise. May fragment text into many small facts. | Always extracts many facts → always does N embeddings + N searches + 2nd LLM call |

#### Example

For a simple image caption like `"A sunset over the ocean"`:

- **8B** correctly returns `{"facts": []}` → **Fast path**: No embeddings, no searches, no 2nd LLM call
- **2B** might return `{"facts": ["sunset", "ocean", "image shows sunset", "scene is over ocean"]}` → **Slow path**: 4 embeddings, 4 vector searches, 1 more LLM call

#### Code Evidence (from `mem0/memory/main.py`)

```python
# Fast path (8B hits this more often)
if not new_retrieved_facts:
    logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")
    return returned_memories  # Returns immediately

# Slow path (2B hits this more often)
for new_mem in new_retrieved_facts:  # More facts = more iterations
    messages_embeddings = self.embedding_model.embed(new_mem, "add")
    existing_memories = self.vector_store.search(query=new_mem, vectors=messages_embeddings, limit=5, ...)
```

#### Recommendations

1. **For faster indexing**: Use `--no-mem0-infer` to bypass fact extraction entirely (stores raw text as memories).
2. **For quality with speed**: Use a more capable model (8B+) for the Mem0 LLM to reduce over-extraction.
3. **For debugging**: Add logging to count extracted facts per document to diagnose performance issues.

## Conversational Format Conversion

### Why Conversational Format?

Mem0's fact extraction (`--mem0-infer`) is designed for conversational data like chat messages. When we feed it structured metadata (e.g., `"ID: IMG_1234\nTimestamp: 2022-05-07\nCaption: A dog..."`), the LLM often returns `{"facts": []}` because it doesn't recognize field-value pairs as extractable facts.

**The solution**: Convert structured batch results into natural first-person diary entries that Mem0's extraction can understand.

### Conversion Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT (Structured)              OUTPUT (Conversational)        │
│  ─────────────────               ─────────────────────          │
│                                                                 │
│  Email:                          "Memory email202201010001:     │
│    id: email202201010001          On 2022-01-01, I received an  │
│    subject: "Meeting Notes"  →    email about 'Meeting Notes'   │
│    short_summary: "..."           from John. The team discussed │
│                                   project timelines."           │
│                                                                 │
│  Image:                          "Memory 20220507_150929:       │
│    image_path: 20220507.jpg       On a sunny afternoon on May   │
│    timestamp: 2022-05-07     →    7th, 2022, I was walking      │
│    location: Oxford               through Oxford when I spotted │
│    [+ raw image]                  a wild rabbit near the path." │
│                                                                 │
│  Video:                          "Memory VID_20220815:          │
│    video_path: VID_20220815       On August 15th, 2022, I was   │
│    timestamp: 2022-08-15     →    at the beach watching the     │
│    location: Brighton             sunset. The waves were calm   │
│    [+ extracted frames]           and children played nearby."  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Conversion Methods by Modality

| Modality | Method | LLM Required | Processing |
|----------|--------|--------------|------------|
| **Email** | Template-based | No | String formatting from fields |
| **Image** | VLM generation | Yes | Raw image + metadata → VLM prompt |
| **Video** | VLM generation | Yes | Extracted frames + metadata → VLM prompt |

### Prompts and Templates

#### Email Template (No LLM)

Emails use a deterministic template that reformats existing fields:

```
Memory {id}: On {date}, I received an email about '{subject}'{sender_part}. {summary_sentence}
```

**Example output**:
```
Memory email202201010001: On 2022-01-01, I received an email about 'Can a Baby Be Born Before Water Breaks?' from BabyCenter. The email discusses early labor signs and what to expect.
```

#### Image Prompt (VLM Required)

**System prompt**:
```
You are helping me create a personal memory diary. Write natural first-person 
entries that capture experiences vividly. Always include the Memory ID exactly 
as provided.
```

**User prompt**:
```
Create a personal memory diary entry for this photo.

Memory ID: {id}
Date/Time: {timestamp}
Location: {location}

Write a natural first-person diary entry (2-3 sentences) describing what I 
experienced or saw. Start with 'Memory {id}: ' and include the date, location, 
and what's visible in the photo. Write as natural prose, not structured data.
```

**Example output**:
```
Memory 20220507_150929: On a sunny afternoon on May 7th, 2022, I was walking 
through OMFC Patch in Wolvercote, Oxford when I spotted a wild rabbit nibbling 
grass near the footpath. The little creature seemed completely unbothered by 
my presence.
```

#### Video Prompt (VLM Required)

**System prompt**: Same as image.

**User prompt**:
```
Create a personal memory diary entry for this video clip.

Memory ID: {id}
Date/Time: {timestamp}
Location: {location}

Write a natural first-person diary entry (2-3 sentences) describing what 
happened in this moment. Start with 'Memory {id}: ' and include the date, 
location, and what's happening in the video. Write as natural prose.
```

### Output Schema

All modalities produce entries with a `conversational_text` field:

```json
{
  "id": "email202201010001",
  "timestamp": "2022-01-01 10:30:00",
  "conversational_text": "Memory email202201010001: On 2022-01-01, I received an email about 'Meeting Notes' from John. The team discussed project timelines."
}
```

For images/videos, the schema includes the original path:

```json
{
  "image_path": "20220507_150929.jpg",
  "timestamp": "2022-05-07 15:09:29",
  "location_name": "Oxford, UK",
  "conversational_text": "Memory 20220507_150929: On a sunny afternoon..."
}
```

### Usage

#### Step 1: Convert Data

```bash
# Email-only (no VLM needed, instant)
python scripts/QA_Agent/mem0/convert_to_conversational.py \
  --skip-images --skip-videos

# Full conversion (requires VLM endpoint)
python scripts/QA_Agent/mem0/convert_to_conversational.py \
  --provider vllm \
  --vllm-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --max-workers 64

# Or use the wrapper script
VLLM_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions \
MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8 \
./scripts/QA_Agent/mem0/run_convert_to_conversational.sh
```

Output files are saved to `scripts/QA_Agent/mem0/data/`:
- `merged_emails_conversational.json`
- `image_*_conversational.json`
- `video_*_conversational.json`

#### Step 2: Run Baseline with Conversational Format

```bash
python memqa/qa_agent_baselines/mem0/mem0_baseline.py \
  --qa-file memqa/utils/final_data_processing/atm-20260121.json \
  --email-file scripts/QA_Agent/mem0/data/merged_emails_conversational.json \
  --image-batch-results scripts/QA_Agent/mem0/data/images_conversational.json \
  --video-batch-results scripts/QA_Agent/mem0/data/videos_conversational.json \
  --use-conversational-format \
  --mem0-infer \
  ...
```

The `--use-conversational-format` flag tells the baseline to read `conversational_text` instead of building structured payloads.

### CLI Arguments for Conversion Script

| Argument | Default | Description |
|----------|---------|-------------|
| `--image-batch-results` | config default | Input image batch results |
| `--video-batch-results` | config default | Input video batch results |
| `--email-file` | config default | Input email file |
| `--image-root` | config default | Root directory for raw images |
| `--video-root` | config default | Root directory for raw videos |
| `--provider` | `vllm` | LLM provider (`vllm` or `openai`) |
| `--model` | None | Model name for VLM |
| `--vllm-endpoint` | None | VLLM endpoint URL |
| `--max-workers` | 64 | Concurrent workers for VLM calls |
| `--num-frames` | 8 | Frames to extract per video |
| `--resume` | True | Resume from existing output |
| `--skip-images` | False | Skip image conversion |
| `--skip-videos` | False | Skip video conversion |
| `--skip-emails` | False | Skip email conversion |
| `--limit` | None | Limit items for testing |

## Answer Modes and Prompt Customization

### Standard Mode (Default)

Uses `PROMPTS["SYSTEM"]` from `config.py`. This prompt is fully customizable and includes the list_recall instruction for consistent evaluation with other baselines.

### Chat API Mode (`--mem0-chat-api`)

Uses the mem0 library's built-in `MEMORY_ANSWER_PROMPT` (imported from `mem0.configs.prompts`). 

**Limitation**: This prompt is defined in the third-party mem0 library and **cannot be modified** without forking the library. It does NOT include the list_recall instruction:

```
"If the question asks to recall or list items (photos/emails/videos), respond with 
the corresponding evidence IDs only, comma-separated, with no extra text."
```

**Impact on Evaluation**: For fair comparison on list_recall questions (qtype=`list_recall`), use the standard mode instead of Chat API mode. Chat API mode may produce verbose answers for list questions instead of comma-separated evidence IDs.

**Recommendation**: Use standard mode (`--no-mem0-chat-api`, which is the default) for benchmark experiments requiring consistent prompt formatting across all baselines.

## Evaluation

Outputs are written to `output/QA_Agent/Mem0/<method_name>/mem0_answers.jsonl`.
Retrieval logs are saved to:
- `output/QA_Agent/Mem0/<method_name>/retrieval_recall_details.json`
- `output/QA_Agent/Mem0/<method_name>/retrieval_recall_summary.json`
Evaluate using the standard evaluator:

```bash
python memqa/utils/evaluator/evaluate_qa.py \
    --ground-truth memqa/utils/final_data_processing/atm-20260121.json \
    --predictions output/QA_Agent/Mem0/mem0_base/mem0_answers.jsonl \
    --output-dir output/QA_Agent/Mem0/mem0_base/eval \
    --metrics em llm
```
