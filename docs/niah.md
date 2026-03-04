# NIAH (Needle In A Haystack)

NIAH is a **generation-only** evaluation setup for ATM-Bench-Hard.
Each question is paired with a fixed evidence pool that is guaranteed to include all
ground-truth items, so the evaluation emphasizes answer synthesis/reasoning rather than
retrieval.

## Why NIAH

- Standard ATM-Bench evaluation measures retrieval + answering jointly.
- NIAH decouples retrieval quality from answer generation quality.
- Varying pool size controls distractor difficulty.

## Protocol Definition

For each question:

- Ground truth remains `evidence_ids`.
- NIAH adds `niah_evidence_ids` with fixed pool size `k` (typically `25, 50, 100, 200`).
- `evidence_ids` must be a subset of `niah_evidence_ids`.

Interpretation:
- Given a fixed pool of `k` memory items that contains all needles, can the model
  identify relevant evidence and produce the correct answer?

## Data Files

- Hard set source:
  - `data/atm-bench/atm-bench-hard.json`
- NIAH pool files:
  - `data/atm-bench/niah/atm-bench-hard-niah25.json`
  - `data/atm-bench/niah/atm-bench-hard-niah50.json`
  - `data/atm-bench/niah/atm-bench-hard-niah100.json`
  - `data/atm-bench/niah/atm-bench-hard-niah200.json`

## Prerequisites

- Run commands from repo root.
- Required data files listed above must exist.
- For OpenAI judge/models: set `OPENAI_API_KEY` (preferred) or use `api_keys/.openai_key`.
- For vLLM answerers: set `VLLM_ENDPOINT` to your OpenAI-compatible endpoint.

## Build or Validate Pools

Build from retrieval outputs:

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

Construction rule (per question, per `k`):

1. Start from top-`k` retrieved IDs.
2. If GT IDs are missing, replace lowest-ranked non-GT IDs.
3. Shuffle deterministically to avoid position bias.

## Run NIAH (Quick Start)

```bash
bash scripts/QA_Agent/NIAH/run_niah_gpt5.sh
bash scripts/QA_Agent/NIAH/run_niah_qwen3vl8b.sh
```

## Run NIAH (Direct CLI)

Implementation wrapper:
- `memqa/qa_agent_baselines/NIAH/niah_evaluate.py`

The wrapper delegates to:
- `memqa/qa_agent_baselines/oracle/oracle_baseline.py`

Example:

```bash
python memqa/qa_agent_baselines/NIAH/niah_evaluate.py \
  --qa-file data/atm-bench/niah/atm-bench-hard-niah50.json \
  --media-source batch_results \
  --image-batch-results <PATH_TO>/image_batch_results.json \
  --video-batch-results <PATH_TO>/video_batch_results.json \
  --email-file <PATH_TO>/merged_emails.json \
  --provider openai --model gpt-5 \
  --max-workers 8 --timeout 120 \
  --output-file output/QA_Agent/NIAH/hard/gpt5/batch_results/niah50/niah_answers.jsonl
```

## Evaluate and Summarize

Evaluate NIAH predictions with standard ATM:

```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth data/atm-bench/atm-bench-hard.json \
  --predictions <NIAH_ANSWERS_JSONL> \
  --output-dir <EVAL_DIR> \
  --metrics atm
```

Summarize multi-`k` runs:

```bash
python scripts/QA_Agent/NIAH/summarize_niah_results.py \
  --output-root output/QA_Agent/NIAH/hard
```

## Related Docs

- `docs/baseline.md`
- `docs/metrics.md`
