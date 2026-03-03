# Metrics

ATMBench evaluates both:

- **Answer quality** (did the model answer correctly?)
- **Retrieval quality** (did the system retrieve the right evidence?)

The main entrypoint is:

```bash
python memqa/utils/evaluator/evaluate_qa.py --metrics em atm ...
```

For deeper implementation details, see `memqa/utils/evaluator/README.md`.

## Answer Metrics

### EM (Exact Match)

Deterministic string match after normalization (dates/times/numbers/punctuation).

### ATM / QS (Question-Type Score)

Per-question scoring depends on question type:

- `number`: Exact Match after normalization.
- `list_recall`: Jaccard similarity between predicted list and ground-truth list.
- `open_end`: LLM judge (`gpt-5-mini` by default in this repo).

ATM aggregates these into:

- overall accuracy
- per-type accuracies (`number`, `list_recall`, `open_end`)

## Retrieval Metrics

For retrieval-based methods (MMRAG), we compute:

- `Recall@k`: fraction of questions where at least one GT evidence is retrieved in top-k
- `Recall@kGT`: fraction of questions where **all** GT evidences are retrieved in top-k

The retrieval evaluator consumes a `retrieval_recall_details.json` produced by
MMRAG runs.

## Joint Diagnostic Metric

We also report:

- `Joint@k = QS * Recall@k`

This is a diagnostic for retrieval dependence:
- It drops when answers are correct but unsupported by retrieved evidence, or when retrieval fails.

## Judge Configuration

`open_end` questions use an LLM judge via `evaluate_qa.py`.

Defaults:
- judge model: `gpt-5-mini`
- judge reasoning effort: `minimal` (scripts)

You can override per-run with:

```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal
```

