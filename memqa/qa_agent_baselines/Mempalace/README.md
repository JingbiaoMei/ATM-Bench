# MemPalace Baseline

MemPalace is a hybrid drawer + closet retrieval system that combines BM25 keyword
search with vector similarity over MiniLM embeddings, then enriches closet-boosted
hits with neighboring chunks ("drawer-grep"). This baseline wraps the upstream
[`mempalace`](https://pypi.org/project/mempalace/) PyPI package (`3.3.5`) for
strict reproduction on ATM-Bench.

## Paper / Upstream

> **MemPalace** — hybrid local memory store for LLM agents (PyPI package
> [`mempalace`](https://pypi.org/project/mempalace/), version pinned to 3.3.5 in
> `requirements.txt`).

Pinned upstream API surface used by this baseline (all imports are upstream;
nothing is re-implemented):

- `mempalace.miner.chunk_text` — paragraph-aware splitter (CHUNK_SIZE=800,
  CHUNK_OVERLAP=100, MIN_CHUNK_SIZE=50)
- `mempalace.miner._build_drawer_metadata` — drawer metadata (wing/room/
  source_file/chunk_index/added_by/filed_at/normalize_version/hall/entities)
- `mempalace.miner._extract_entities_for_metadata` — entity extraction for
  closet metadata
- `mempalace.palace.get_collection` / `get_closets_collection` — Chroma palace
  collections (`mempalace_drawers`, `mempalace_closets`)
- `mempalace.palace.build_closet_lines` — regex topic / entity / quote
  extraction packed into pointer lines (`topic|entities|→drawer_ids`)
- `mempalace.palace.upsert_closet_lines` / `purge_file_closets` — closet
  upsert with purge-before-write for re-mines
- `mempalace.searcher.search_memories` — hybrid BM25+cosine rerank with closet
  boost and drawer-grep neighborhood expansion

## Alignment With Upstream MemPalace

| Aspect | Upstream MemPalace 3.3.5 | This baseline | Aligned? |
|---|---|---|---|
| Embedding | `all-MiniLM-L6-v2` (ONNX) | Same | ✅ |
| Chunking | 800 chars, 100 overlap, min 50 | `chunk_text()` | ✅ |
| Drawer storage | Chroma `mempalace_drawers` | Same | ✅ |
| Closet index | Regex closet lines | `build_closet_lines()` | ✅ |
| Hybrid retrieval | BM25 (k1=1.5, b=0.75) + cosine | Same (`search_memories`) | ✅ |
| Vector / BM25 weights | 0.6 / 0.4 | Same (upstream internal) | ✅ |
| Closet boost ranks | `[0.40, 0.25, 0.15, 0.08, 0.04]` | Same (upstream internal) | ✅ |
| Drawer-grep enrichment | Best chunk + 1 neighbor each side | Same (upstream internal) | ✅ |
| Candidate strategy | `vector` (default) | `vector` (default), `union` available | ✅ |
| Upsert batch size | `DRAWER_UPSERT_BATCH_SIZE=1000` | Same | ✅ |
| `n_results` default | 5 | 100 (over-fetch for retrieval recall) | Intentional ablation |
| Wing / room source | Directory + room_detector | Modality + sanitized item ID | Intentional adaptation |
| Source file | Real path on disk | Virtual `atmbench://<item_id>` | Intentional adaptation |

### Intentional Adaptations

1. **No filesystem mining.** ATM-Bench items are not files on disk; instead of
   calling `mempalace.miner.process_file()`, this baseline inlines the same
   pipeline (`chunk_text` → `_build_drawer_metadata` → batched
   `collection.upsert` → `build_closet_lines` → `purge_file_closets` +
   `upsert_closet_lines`) so the resulting Chroma collection is byte-identical
   to what `process_file` would write for an equivalent on-disk corpus.
2. **Wing = modality, room = sanitized item ID.** The upstream
   `detect_room` walks the project directory tree, which is meaningless here.
   We replace it with modality (`email`/`image`/`video`) for the wing and the
   item ID (slashes/dots replaced with `_`) for the room. Closet boost and
   wing/room filtering still work; the directory-routing heuristics are gone.
3. **`n_results=100`** for the retrieval call. Upstream defaults to 5 — useful
   for an interactive agent but too small for benchmark recall reporting. We
   over-fetch to 100 and trim to top `--retrieve-k` (default 10) when building
   the LLM evidence pool. The full 100-item ranked list is still persisted in
   `retrieval_recall_details.json` for `R@{1,5,10,25,50,100}` reporting.
4. **Evidence text is the full `RetrievalItem.text`,** not the chunk that the
   hybrid search landed on. ATM-Bench evidence IDs are item-level (one photo /
   one email), so the answerer should see the whole item; the chunk hit is
   only used as a fallback when the item lookup fails.

## Setup

MemPalace ships its own ONNX MiniLM model and ChromaDB binding, both of which
can conflict with other baselines' embedding stacks. A dedicated conda env is
strongly recommended.

```bash
conda create -n mempalace python=3.11 -y
conda activate mempalace
pip install -r memqa/qa_agent_baselines/Mempalace/requirements.txt
# CPU torch is enough — MemPalace embeds locally via ONNX.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

API keys follow the repo convention:

- Answerer (vLLM): `VLLM_API_KEY` env var, or `api_keys/.vllm_key`
- Answerer (OpenAI) and GPT-5-mini judge: `OPENAI_API_KEY`, or
  `api_keys/.openai_key`

## Quick Start

After preparing memory files (see the top-level
[`README.md`](../../../README.md)), run from the repository root:

```bash
conda activate mempalace
bash scripts/QA_Agent/Mempalace/run.sh
```

The script will:

1. Build one shared MemPalace palace from the full image + video + email
   corpus (drawers + regex closets).
2. Answer `data/atm-bench/atm-bench.json` and
   `data/atm-bench/atm-bench-hard.json` using the configured vLLM answerer
   (default: `Qwen/Qwen3-VL-8B-Instruct-FP8`).
3. Run the shared `run_eval_bundle` to score EM + ATM with `gpt-5-mini`,
   compute comprehensive recall, and compute joint answer + retrieval
   accuracy.

### Key environment overrides

| Variable | Default | Meaning |
|---|---|---|
| `TOP_K` | `10` | Evidence items passed to the answerer |
| `N_RESULTS` | `100` | MemPalace search candidates (over-fetch for recall) |
| `VLLM_ENDPOINT` | `http://127.0.0.1:8000/v1/chat/completions` | vLLM endpoint |
| `ANSWERER_MODEL` | `Qwen/Qwen3-VL-8B-Instruct-FP8` | Answer model |
| `REBUILD_INDEX` | `0` | Set `1` to force-rebuild the palace cache |
| `OVERWRITE` | `0` | Set `1` to regenerate existing answer files |
| `MAX_WORKERS` | `8` | Answer-generation thread workers |
| `JUDGE_MODEL_GPT` | `gpt-5-mini` | ATM open-end judge (set in `common_eval.sh`) |
| `RETRIEVAL_K_VALUES` | `1,5,10,25,50,100` | Comprehensive retrieval summary k values |

## Manual Commands

Build only (one-time per memory corpus):

```bash
python memqa/qa_agent_baselines/Mempalace/mempalace_baseline.py \
  --stage build \
  --qa-file data/atm-bench/atm-bench.json \
  --image-batch-results output/image/qwen3vl2b/batch_results.json \
  --video-batch-results output/video/qwen3vl2b/batch_results.json \
  --email-file data/raw_memory/email/emails.json \
  --image-root data/raw_memory/image \
  --video-root data/raw_memory/video \
  --index-cache output/QA_Agent/Mempalace/index_cache \
  --force-rebuild
```

Answer a single QA file:

```bash
python memqa/qa_agent_baselines/Mempalace/mempalace_baseline.py \
  --stage answer \
  --qa-file data/atm-bench/atm-bench.json \
  --provider vllm \
  --vllm-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --retrieve-k 10 \
  --n-results 100 \
  --index-cache output/QA_Agent/Mempalace/index_cache \
  --output-dir-base output/QA_Agent/Mempalace \
  --method-name atmbench/mempalace
```

Evaluate with `gpt-5-mini`:

```bash
source scripts/QA_Agent/common_eval.sh
run_eval_bundle \
  data/atm-bench/atm-bench.json \
  output/QA_Agent/Mempalace/atmbench/mempalace/mempalace_answers.jsonl \
  output/QA_Agent/Mempalace/atmbench/mempalace/eval \
  output/QA_Agent/Mempalace/atmbench/mempalace/retrieval_recall_details.json
```

## Outputs

Per method directory (`<output_dir_base>/<method_name>/`):

- `mempalace_answers.jsonl` — predictions, one `{"id", "answer", ...}` per line
- `mempalace_answers_run_stats.json` — answer-stage token totals
- `retrieval_recall_details.json` — per-question retrieved IDs + scores + recall
- `retrieval_recall_summary.json` — built-in recall summary
- `retrieval_recall_comprehensive_summary.json` — `R@{1..100}`, hit@1, recall@gt
- `retrieval_recall_joint_accuracy_summary.json` — joint strict/partial accuracy
- `eval/atm_gpt-5-mini.json` and `eval/atm_gpt-5-mini_summary.json` — ATM judge

## Notes / Gotchas

- **WSL / NTFS HNSW quarantine.** ChromaDB occasionally quarantines the HNSW
  index on first query when the palace lives on NTFS. The answer stage runs
  three warmup `search_memories` calls before the parallel QA loop precisely
  for this reason — do not remove them.
- **Re-running with the same cache key reuses the palace.** Pass
  `--force-rebuild` (or `REBUILD_INDEX=1` to the run script) when changing
  any field that affects the indexed text (e.g. toggling `--include-caption`,
  swapping batch-results files).
- **`vector_weight=0.6 / bm25_weight=0.4`** are upstream constants inside
  `_hybrid_rank`. They appear in `retrieval_recall_summary.json` for
  documentation; this baseline does not expose them as flags because changing
  them would require monkey-patching upstream.
