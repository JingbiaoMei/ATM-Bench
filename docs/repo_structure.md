# Repo Structure

ATMBench is organized around:

- `memqa/`: core library (processors, retrieval, baselines, evaluation)
- `scripts/`: runnable workflows (bash + small utilities)
- `data/`: local inputs (gitignored)
- `output/`: generated artifacts/results (gitignored)

## Key Directories

### `memqa/mem_processor/`

Preprocessing pipelines for:
- `image/`
- `video/`
- `email/`

These scripts turn raw artifacts into normalized metadata (`batch_results.json`)
used by QA baselines.

### `memqa/retrieve/`

Retrieval + reranking utilities used by MMRAG and other baselines.

### `memqa/qa_agent_baselines/`

Baselines shipped with this repo:

- `MMRag/`: retrieval + evidence-grounded answering
- `oracle/`: upper bound (answers using GT evidence IDs)
- `NIAH/`: generation-only evaluation using fixed evidence pools
- `HippoRag2/`: HippoRAG 2 graph memory baseline
- `MemoryOS/`: tiered memory baseline (STM/MTM/LPM)
- `A-Mem/`: agentic memory baseline (two-stage cache)
- `mem0/`: mem0-backed memory baseline

### `memqa/utils/evaluator/`

Evaluation tools for:
- QA metrics (`em`, `atm`, judge-based scoring)
- retrieval metrics
- joint metrics

### `scripts/QA_Agent/`

Public runnable scripts (repo-root execution):
- `MMRAG/`: main baseline scripts
- `Oracle/`: oracle + no-evidence scripts
- `NIAH/`: NIAH runs + utilities
- `HippoRag2/`: HippoRAG 2 runs
- `MemoryOS/`: MemoryOS runs
- `A-Mem/`: A‑Mem runs
- `mem0/`: Mem0 runs

### `data/`

Local-only inputs (see `docs/data.md`):
- `data/atm-bench/` (benchmark)
- `data/raw_memory/` (your artifacts + batch results)
- `data/processed_memory/` (optional)
