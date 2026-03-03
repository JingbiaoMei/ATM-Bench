# NIAH (Needle In A Haystack) — ATMBench

NIAH is a **generation-only** evaluation protocol: for each question, we provide
a **fixed evidence pool** of size *k* that is guaranteed to contain all
ground-truth evidence items. This isolates answer synthesis from retrieval.

This repo implements NIAH by:
1. Providing (or generating) pool files that extend the standard QA JSON schema
   with `niah_evidence_ids`.
2. Running a thin wrapper (`niah_evaluate.py`) that swaps
   `niah_evidence_ids -> evidence_ids` inside the oracle baseline.

## Files

**Hard set source (25 questions):**
- `data/atm-bench/atm-bench-hard.json`

**NIAH pool files (same QA list + new `niah_evidence_ids`):**
- `data/atm-bench/niah/atm-bench-hard-niah25.json`
- `data/atm-bench/niah/atm-bench-hard-niah50.json`
- `data/atm-bench/niah/atm-bench-hard-niah100.json`
- `data/atm-bench/niah/atm-bench-hard-niah200.json`

**Evaluation wrapper (delegates to `oracle_baseline.py`):**
- `memqa/qa_agent_baselines/NIAH/niah_evaluate.py`

**Pool builder (optional):**
- `scripts/QA_Agent/NIAH/build_niah_pools.py`

## Pool File Schema

Each entry extends the usual ATM QA schema with:
- `niah_evidence_ids`: list of evidence IDs of length *k* (a superset of `evidence_ids`)

`evidence_ids` remains the ground truth and is unchanged.

## Run NIAH Evaluation

`niah_evaluate.py` forwards all other arguments to:
`memqa/qa_agent_baselines/oracle/oracle_baseline.py`.

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
