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

## Token Accounting

Each Oracle / NIAH run saves a `*_run_stats.json` file next to the predictions
file. The key field is `avg_prompt_tokens` (average input tokens per sample),
which is used for the "Avg. Context Tokens" column in results tables.

### How tokens are counted

Token counts come directly from the **inference API's `usage` response**:

| Provider | Source field | Vision tokens included? |
|----------|-------------|------------------------|
| `openai` (Responses API) | `response.usage.input_tokens` | Yes — OpenAI counts vision tokens in input |
| `openai` (Chat Completions) | `response.usage.prompt_tokens` | Yes |
| `vllm` (remote, OpenAI-compat) | `response.usage.prompt_tokens` | Yes — vLLM includes image tile tokens for VL models |
| `vllm_local` | `len(output.prompt_token_ids)` | N/A — multimodal not supported |

Implementation: `memqa/qa_agent_baselines/oracle/oracle_baseline.py`, lines
around `chat_with_usage()` / `_chat_openai_with_usage()` /
`_chat_vllm_with_usage()`.

### SGM vs Raw token differences

- **SGM** (`--media-source batch_results`): Text-only prompts (captions, OCR,
  tags). No vision tokens.
- **Raw** (`--media-source raw`): Images sent as base64 `image_url` content
  parts. Vision tokens are included in `prompt_tokens` by the serving backend.

The gap between Raw and SGM context tokens at the same NIAH pool size reflects
the vision token overhead from sending actual images.

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

