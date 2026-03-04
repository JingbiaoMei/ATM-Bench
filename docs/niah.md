# NIAH (Needle In A Haystack)

NIAH is a **generation-focused** evaluation setup for ATM-Bench-Hard.

Instead of testing end-to-end retrieval, each question is paired with a fixed
evidence pool that is guaranteed to include all ground-truth evidence items.
This lets us evaluate answer synthesis and reasoning under controlled evidence.

## Why NIAH

- Standard ATM-Bench evaluation measures retrieval + answering jointly.
- NIAH decouples retrieval from generation.
- By varying pool size, NIAH controls distractor difficulty.

## Core Definition

For each question:

- Ground truth remains `evidence_ids`.
- NIAH adds `niah_evidence_ids` with fixed size `k` (typically `k in {25, 50, 100, 200}`).
- `evidence_ids` must be a subset of `niah_evidence_ids`.

Interpretation:
- Given a fixed pool of `k` memory items that contains all needles, can the
  model identify the relevant evidence and produce the correct answer?

## Pool Construction

NIAH pools are built from retrieval outputs (`retrieval_recall_details.json`):

1. Start from top-`k` retrieved IDs.
2. Ensure all GT IDs are present; if missing, replace lowest-ranked non-GT IDs.
3. Shuffle deterministically to avoid position bias.

Builder:

```bash
python scripts/QA_Agent/NIAH/build_niah_pools.py \
  --qa-file data/atm-bench/atm-bench-hard.json \
  --retrieval-details <PATH_TO>/retrieval_recall_details.json \
  --pool-sizes 25 50 100 200
```

Validate only:

```bash
python scripts/QA_Agent/NIAH/build_niah_pools.py \
  --qa-file data/atm-bench/atm-bench-hard.json \
  --pool-sizes 25 50 100 200 \
  --validate-only
```

## Data Files

- Hard set source:
  - `data/atm-bench/atm-bench-hard.json`
- NIAH pool files:
  - `data/atm-bench/niah/atm-bench-hard-niah25.json`
  - `data/atm-bench/niah/atm-bench-hard-niah50.json`
  - `data/atm-bench/niah/atm-bench-hard-niah100.json`
  - `data/atm-bench/niah/atm-bench-hard-niah200.json`

## Running NIAH

Main scripts:

```bash
bash scripts/QA_Agent/NIAH/run_niah_gpt5.sh
bash scripts/QA_Agent/NIAH/run_niah_qwen3vl8b.sh
```

Implementation:

- Wrapper: `memqa/qa_agent_baselines/NIAH/niah_evaluate.py`
- The wrapper delegates to the Oracle pipeline with NIAH pools enabled.

## Evaluation

NIAH predictions are evaluated with the same ATM evaluator used elsewhere:

```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth data/atm-bench/atm-bench-hard.json \
  --predictions <NIAH_ANSWERS_JSONL> \
  --output-dir <EVAL_DIR> \
  --metrics atm
```

## Related Docs

- `memqa/qa_agent_baselines/NIAH/README.md`
- `scripts/QA_Agent/NIAH/README.md`
- `docs/metrics.md`
