# NIAH Run Scripts

These scripts run the NIAH (Needle In A Haystack) evaluation on the ATMBench hard
set using the pool files under `data/atm-bench/niah/`.

Run from repo root.

## Quick Start

### GPT-5 (answerer) + GPT-5-mini (judge)

```bash
bash scripts/QA_Agent/NIAH/run_niah_gpt5.sh
```

### Qwen3-VL-8B (answerer via vLLM) + GPT-5-mini (judge)

```bash
bash scripts/QA_Agent/NIAH/run_niah_qwen3vl8b.sh
```

## Prerequisites

- Hard set exists:
  - `data/atm-bench/atm-bench-hard.json`
- NIAH pool files exist:
  - `data/atm-bench/niah/atm-bench-hard-niah25.json`
  - `data/atm-bench/niah/atm-bench-hard-niah50.json`
  - `data/atm-bench/niah/atm-bench-hard-niah100.json`
  - `data/atm-bench/niah/atm-bench-hard-niah200.json`

- For OpenAI judge/models: set `OPENAI_API_KEY` (preferred), or create
  `api_keys/.openai_key`.

- For vLLM answerers: set `VLLM_ENDPOINT` to your OpenAI-compatible endpoint.

## Build/Validate Pools

```bash
python scripts/QA_Agent/NIAH/build_niah_pools.py \
  --qa-file data/atm-bench/atm-bench-hard.json \
  --retrieval-details <PATH_TO>/retrieval_recall_details.json \
  --pool-sizes 25 50 100 200
```

Validate existing pools:

```bash
python scripts/QA_Agent/NIAH/build_niah_pools.py \
  --qa-file data/atm-bench/atm-bench-hard.json \
  --pool-sizes 25 50 100 200 \
  --validate-only
```

## Summarize Results

```bash
python scripts/QA_Agent/NIAH/summarize_niah_results.py \
  --output-root output/QA_Agent/NIAH/hard
```
