# Reproducibility

This doc describes how to reproduce ATMBench runs in a paper-compatible way.

## Environment

Recommended setup:

```bash
conda create -n atmbench python=3.11 -y
conda activate atmbench
pip install -r requirements.txt
pip install -e .
```

## Data

See `docs/data.md` for the expected local layout:

- `data/atm-bench/` (benchmark files)
- `data/raw_memory/` (your raw memory)
- `output/image/qwen3vl2b/batch_results.json`
- `output/video/qwen3vl2b/batch_results.json`

## Model Endpoints / Keys

- OpenAI judge/models: set `OPENAI_API_KEY` (or use `api_keys/.openai_key`)
- vLLM answerers: set `VLLM_ENDPOINT` (default: `http://127.0.0.1:8000/v1/chat/completions`)

## What To Run

### Precompute Memory Itemization

Run this first so the baseline wrappers can read `output/{image,video}/qwen3vl2b/batch_results.json`:

```bash
bash scripts/memory_processor/image/copy_gps_cache.sh output/image/qwen3vl2b/cache
bash scripts/memory_processor/video/copy_gps_cache.sh output/video/qwen3vl2b/cache
bash scripts/memory_processor/image/memory_itemize/run_qwen3vl2b.sh
bash scripts/memory_processor/video/memory_itemize/run_qwen3vl2b.sh
```

### MMRAG (retrieval + answering)

Runs both ATM-bench and ATM-bench-hard:

```bash
bash scripts/QA_Agent/MMRAG/run.sh
```

### Oracle (upper bound using GT evidence IDs)

```bash
bash scripts/QA_Agent/Oracle/run_oracle_qwen3vl8b.sh
bash scripts/QA_Agent/Oracle/run_oracle_gpt5.sh
```

### NIAH (generation-only; fixed evidence pools)

```bash
bash scripts/QA_Agent/NIAH/run_niah_qwen3vl8b.sh
bash scripts/QA_Agent/NIAH/run_niah_gpt5.sh
```

For additional memory-agent baselines (HippoRAG 2, MemoryOS, A‑Mem, Mem0), see `docs/baseline.md`.

Environment note:
- `MemoryOS` is strongly recommended to run in a separate conda environment.
- `HippoRAG 2`, `A‑Mem`, and `Mem0` are tested to be compatible with the core baseline environment, but separate environments are still safer for reproducibility.

Agent-system note:
- OpenClaw, OpenCode, and Codex baselines are compatible with this repo’s evaluation workflow, but each requires its own third-party software installation.

## Evaluation Defaults (Paper)

- `open_end` judge model: `gpt-5-mini` (default in this repo)

To summarize NIAH runs into a Markdown table:

```bash
python scripts/QA_Agent/NIAH/summarize_niah_results.py \
  --output-root output/QA_Agent/NIAH/hard
```

## Notes

- API-based evaluation is not perfectly deterministic.
- Prefer keeping the same model versions, judge model, and decoding settings for comparisons.
