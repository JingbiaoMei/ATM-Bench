# Evaluation Metrics Guide

Comprehensive documentation for PersonalMemoryQA evaluation system.

---

## Table of Contents

- [Evaluation Metrics Guide](#evaluation-metrics-guide)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Answer Evaluation Metrics](#answer-evaluation-metrics)
    - [EM (Exact Match)](#em-exact-match)
    - [LLM Judge](#llm-judge)
    - [ATM (Answer Type Metric)](#atm-answer-type-metric)
      - [Jaccard Index (for `list_recall`)](#jaccard-index-for-list_recall)
      - [ATM Workflow](#atm-workflow)
  - [Retrieval Evaluation Metrics](#retrieval-evaluation-metrics)
    - [Comprehensive Metrics](#comprehensive-metrics)
    - [Joint Accuracy](#joint-accuracy)
      - [1. Joint Strict (`joint_strict@k`)](#1-joint-strict-joint_strictk)
      - [2. Joint Partial (`joint_partial@k`)](#2-joint-partial-joint_partialk)
  - [Question Types (Qtype)](#question-types-qtype)
  - [Usage Examples](#usage-examples)
    - [Example 1: Dual Judge Evaluation](#example-1-dual-judge-evaluation)
    - [Example 2: Complete Evaluation Pipeline](#example-2-complete-evaluation-pipeline)
    - [Example 3: EM-only (no LLM costs)](#example-3-em-only-no-llm-costs)
  - [Files and Outputs](#files-and-outputs)
  - [References](#references)

---

## Overview

The evaluation system provides multiple metrics to assess both **answer quality** and **retrieval performance**:

- **Answer Metrics**: `em`, `llm`, `atm` - evaluate whether generated answers match ground truth
- **Retrieval Metrics**: `recall@k`, `hit@1`, `recall@gt` - evaluate whether the correct evidence was retrieved
- **Joint Metrics**: `joint_strict@k`, `joint_partial@k` - evaluate both answer correctness AND retrieval success simultaneously

All metrics support **caching and resumption** to avoid re-evaluating items when runs are interrupted.

---

## Answer Evaluation Metrics

### EM (Exact Match)

**Description**: Deterministic exact match with extensive normalization.

**How it works**:
1. Normalizes both ground truth and prediction (lowercase, strip punctuation, etc.)
2. Extracts and compares structured elements (dates, times, numbers, currency)
3. Falls back to token-based matching if exact match fails

**Normalization pipeline**:
- Date/time extraction and normalization (e.g., "March 1" → "0301", "4pm" → "16:00")
- Number normalization (removes commas: "1,000" → "1000")
- Currency extraction (£12.85, $100 USD)
- List item splitting (comma, semicolon, "and", "/")
- Parenthetical detail removal ("£12.85 (with discount)" → "£12.85")
- Leading article stripping ("The Star Wars event" → "Star Wars event")

**When to use**:
- Quick deterministic evaluation without API costs
- Baseline metric for number/date/factual questions

**Usage**:
```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth merged_qa.json \
  --predictions predictions.jsonl \
  --output-dir eval/ \
  --metrics em
```

**Output**: `eval/deterministic_accuracy.json` with per-item results and summary.

---

### LLM Judge

**Description**: LLM-based evaluation using a judge model to assess answer correctness.

**How it works**:
1. Constructs a prompt with question, ground truth answer, and predicted answer
2. Calls judge model (OpenAI or VLLM) to determine if prediction is correct
3. Caches results per-item in `llm_judge_<model>.json` for reuse

**Features**:
- **Automatic caching**: Skips already-evaluated items on re-runs
- **Retry logic**: Exponential backoff for rate limits (up to 10 retries)
- **Parallel execution**: Supports multi-threaded API calls
- **Incremental saves**: Writes results immediately to avoid losing progress

**Judge prompt structure**:
```
Question: {{question}}
Ground Truth Answer: {{answer}}
Predicted Answer: {{prediction}}

Is the predicted answer correct? Respond with JSON: {"accuracy": true/false, "explanation": "..."}
```

**When to use**:
- Open-ended questions where exact match is too strict
- Questions requiring semantic understanding
- When EM gives too many false negatives

**Usage**:
```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth merged_qa.json \
  --predictions predictions.jsonl \
  --output-dir eval/ \
  --metrics llm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --max-workers 2 \
  --request-delay 10.0
```

**Output**: `eval/llm_judge_<model>.json` with per-item judgments and explanations.

---

### ATM (Answer Type Metric)

**Description**: **Qtype-aware** evaluation that uses the most appropriate metric for each question type.

**How it works**:

ATM automatically selects the evaluation method based on the detected or annotated question type:

| Question Type | Metric Used | Description |
|---------------|-------------|-------------|
| `number` | **Exact Match (EM)** | Deterministic matching for dates, times, numbers, currency |
| `list_recall` | **Jaccard Index** | Partial credit for list-style answers |
| `open_end` | **LLM Judge** | Semantic evaluation for open-ended questions |

**Qtype Detection**:
- Questions are automatically classified by analyzing the answer structure
- Can be overridden with explicit `qtype` field in ground truth JSON
- See [Question Types](#question-types-qtype) section for detection rules

#### Jaccard Index (for `list_recall`)

For list-style answers, ATM scores partial credit using **Jaccard similarity**:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Where:
- $A$ = normalized ground-truth item set
- $B$ = normalized prediction item set

**Item normalization**:
1. Lowercase and strip punctuation
2. Split on commas, semicolons, `and`, or `/`
3. Remove empty items

**Examples**:

| Ground Truth | Prediction | Score | Explanation |
|--------------|------------|-------|-------------|
| "a, b, c" | "a, b, c" | 1.0 | Perfect match: 3/3 items |
| "a, b, c" | "a, b" | 0.67 | Partial: &#124;{a,b} ∩ {a,b,c}&#124; / &#124;{a,b} ∪ {a,b,c}&#124; = 2/3 |
| "a, b, c" | "a, b, d" | 0.5 | Partial: &#124;{a,b} ∩ {a,b,c,d}&#124; / &#124;{a,b,c,d}&#124; = 2/4 |
| "a, b, c" | "d, e, f" | 0.0 | No overlap |

**Code reference**: `list_jaccard_score()` in `evaluate_qa.py` (line 275), `split_list_items()` in `normalizer.py` (line 597)

#### ATM Workflow

```
For each question:
  1. Detect or load qtype (number/list_recall/open_end)
  2. If qtype = number:
       → Run deterministic EM
       → Cache result in atm_<model>.json
  3. If qtype = list_recall:
       → Compute Jaccard similarity
       → Cache result in atm_<model>.json
  4. If qtype = open_end:
       → Check llm_judge_<model>.json cache
       → If cached: reuse result
       → If not cached: call LLM judge
       → Write to both caches
```

**Caching strategy**:
- `atm_<model>.json`: Stores all ATM results (EM + Jaccard + LLM)
- `llm_judge_<model>.json`: Stores LLM judgments (shared with `llm` metric)
- ATM reuses LLM cache to avoid redundant API calls

**When to use**:
- **Default choice** for comprehensive evaluation
- When dataset has mixed question types
- When you want partial credit for list questions

**Usage**:
```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth merged_qa.json \
  --predictions predictions.jsonl \
  --output-dir eval/ \
  --metrics atm \
  --judge-provider openai \
  --judge-model gpt-5-mini
```

**Output**:
- `eval/atm_<model>.json`: Per-item results with `qtype`, `metric`, and `accuracy`
- `eval/atm_<model>_summary.json`: Overall and per-qtype accuracy breakdown

**Summary structure**:
```json
{
  "count": 1013,
  "correct": 456.7,
  "accuracy": 0.451,
  "by_qtype": {
    "number": {"count": 300, "correct": 240, "accuracy": 0.800},
    "list_recall": {"count": 200, "correct": 156.2, "accuracy": 0.781},
    "open_end": {"count": 513, "correct": 60.5, "accuracy": 0.118}
  }
}
```

---

## Retrieval Evaluation Metrics

Retrieval metrics evaluate whether the QA system retrieved the correct evidence documents, regardless of answer correctness.

**Data source**: All retrieval metrics read from `retrieval_recall_details.json`, which must be generated during the QA run.

**Format of `retrieval_recall_details.json`**:
```json
[
  {
    "id": "qa_123",
    "gt_evidence_ids": ["doc_5", "doc_12"],
    "retrieval_ids": ["doc_5", "doc_8", "doc_12", "doc_3", ...]
  },
  ...
]
```

### Comprehensive Metrics

**Script**: `memqa/utils/evaluator/evaluate_retrieval/comprehensive_eval.py`

**Description**: Evaluates retrieval performance independently of answer correctness.

**Metrics**:

| Metric | Formula | Description |
|--------|---------|-------------|
| **recall@k** | $\frac{&#124;\text{GT} \cap \text{Top-k}&#124;}{&#124;\text{GT}&#124;}$ | Fraction of ground truth evidence found in top-k results |
| **recall@gt** | $\frac{&#124;\text{GT} \cap \text{Top-}&#124;\text{GT}&#124;&#124;}{&#124;\text{GT}&#124;}$ | R-precision: recall when k = number of GT items |
| **hit@1** | $1$ if top-1 ∈ GT, else $0$ | Whether the top-ranked item is a ground truth evidence |

**Example calculation**:

Given:
- GT evidence: `["doc_5", "doc_12"]` (2 items)
- Retrieved: `["doc_5", "doc_8", "doc_12", "doc_3", "doc_99", ...]`

Results:
- `recall@1` = 1/2 = 0.5 (only doc_5 in top-1)
- `recall@5` = 2/2 = 1.0 (both doc_5 and doc_12 in top-5)
- `recall@10` = 2/2 = 1.0 (both in top-10)
- `recall@gt` = 2/2 = 1.0 (both in top-2, since |GT|=2)
- `hit@1` = 1.0 (doc_5 is a GT item)

**When to use**:
- Debugging retrieval systems
- Comparing different retrieval strategies
- Understanding upper-bound performance (if retrieval was perfect, how good could answers be?)

**Usage**:
```bash
python memqa/utils/evaluator/evaluate_retrieval/comprehensive_eval.py \
  --details output/QA_Agent/method/retrieval_recall_details.json \
  --k-values 1,5,10
```

**Output**: `retrieval_recall_comprehensive_summary.json`
```json
{
  "count": 1013,
  "metrics": {
    "recall@1": 0.456,
    "recall@5": 0.678,
    "recall@10": 0.723,
    "recall@gt": 0.689,
    "hit@1": 0.456
  },
  "gt_stats": {
    "count_with_gt": 1013,
    "count_empty_gt": 0,
    "gt_count_avg": 2.3,
    "gt_count_min": 1,
    "gt_count_max": 5
  }
}
```

---

### Joint Accuracy

**Script**: `memqa/utils/evaluator/evaluate_retrieval/joint_accuracy.py`

**Description**: **Compound metric** that measures simultaneous success of both retrieval AND answer correctness.

**Purpose**: Ensures models are "right for the right reasons" rather than generating lucky guesses or failing to use retrieved evidence.

**Two variants**:

#### 1. Joint Strict (`joint_strict@k`)

**Binary metric**: $1.0$ only when BOTH conditions are met:
- ✅ Answer is correct (from ATM evaluation)
- ✅ **ALL** ground truth evidence IDs are in top-k retrieved items

**Formula**:
```python
strict = 1.0 if (answer_correct AND gt_set ⊆ topk) else 0.0
```

**Interpretation**:
- Most demanding metric
- Requires 100% retrieval recall AND correct answer
- Penalizes systems that answer correctly but didn't retrieve all evidence
- Measures "provable correctness" - the system must have had access to complete evidence

**Example**:

| GT Evidence | Retrieved Top-5 | Answer Correct? | `joint_strict@5` |
|-------------|-----------------|-----------------|------------------|
| [A, B] | [A, B, C, D, E] | ✅ Yes | 1.0 (all GT in top-5) |
| [A, B] | [A, C, D, E, F] | ✅ Yes | 0.0 (missing B) |
| [A, B] | [A, B, C, D, E] | ❌ No | 0.0 (wrong answer) |

#### 2. Joint Partial (`joint_partial@k`)

**Continuous metric**: Combines answer correctness with retrieval recall

**Formula**:
```python
partial = answer_correctness × recall@k
```

Where:
- `answer_correctness` = 1.0 if correct, 0.0 if incorrect (from ATM)
- `recall@k` = (# of GT items in top-k) / (total # of GT items)

**Interpretation**:
- If answer is wrong → score is 0 (no credit)
- If answer is correct → score = percentage of evidence retrieved
- Gives partial credit for retrieving some (but not all) evidence

**Example**:

| GT Evidence | Retrieved Top-5 | Answer Correct? | Recall@5 | `joint_partial@5` |
|-------------|-----------------|-----------------|----------|-------------------|
| [A, B] | [A, B, C, D, E] | ✅ Yes | 2/2 = 1.0 | 1.0 × 1.0 = 1.0 |
| [A, B] | [A, C, D, E, F] | ✅ Yes | 1/2 = 0.5 | 1.0 × 0.5 = 0.5 |
| [A, B] | [A, B, C, D, E] | ❌ No | 2/2 = 1.0 | 0.0 × 1.0 = 0.0 |

**When to use joint metrics**:

Use joint accuracy when you need to:
- Verify that correct answers are based on correct evidence (not lucky guesses)
- Measure end-to-end system reliability
- Debug "hallucination" issues (answering correctly without evidence)
- Compare retrieval-augmented vs parametric knowledge usage

**Interpretation guide**:

| Observation | Implication |
|-------------|-------------|
| `joint_strict` ≪ ATM accuracy | System answers correctly without retrieving all evidence (may use parametric knowledge or partial evidence) |
| `joint_partial` ≈ `joint_strict` | When correct, system usually retrieves complete evidence (systematic reasoning) |
| `joint_partial` > `joint_strict` significantly | System is robust to incomplete retrieval (can answer with partial evidence) |
| `joint_strict@5` < `joint_strict@10` | Increasing context window helps (more evidence → better answers) |

**Usage**:
```bash
python memqa/utils/evaluator/evaluate_retrieval/joint_accuracy.py \
  --retrieval-details output/QA_Agent/method/retrieval_recall_details.json \
  --atm-details output/QA_Agent/method/eval/atm_gpt-5-mini.json \
  --k-values 5,10
```

**Output**: `retrieval_recall_joint_accuracy_summary.json`
```json
{
  "count": 1013,
  "metrics": {
    "joint_strict@5": 0.316,
    "joint_strict@10": 0.333,
    "joint_partial@5": 0.335,
    "joint_partial@10": 0.347
  },
  "counts": {
    "count_with_gt": 1013,
    "count_empty_gt": 0,
    "missing_answer": 0
  },
  "source_retrieval_details": "...",
  "source_atm_details": "...",
  "k_values": [5, 10]
}
```

**Real-world example interpretation**:

From HippoRag2 run:
- `joint_strict@5`: 0.316 → 31.6% of questions: correct answer + 100% evidence in top-5
- `joint_strict@10`: 0.333 → 33.3% with top-10 (slight improvement)
- `joint_partial@5`: 0.335 → When considering partial evidence, score is 33.5%
- `joint_partial@10`: 0.347 → 34.7% with top-10

**Analysis**: `joint_partial` ≈ `joint_strict` suggests that when the system answers correctly, it usually has retrieved most/all evidence. The small gap indicates robustness to minor retrieval failures.

---

## Question Types (Qtype)

The evaluation system automatically classifies questions into three types based on answer structure.

**Types**:

| Qtype | Detection Logic | Example Answers |
|-------|----------------|-----------------|
| `number` | Contains dates, times, numbers, or currency; minimal non-numeric text | "March 15, 2022", "£12.85", "3 sessions", "4:00 PM" |
| `list_recall` | Evidence ID list (e.g., `email202201120830`) | "email202201120830, email202201150945" |
| `open_end` | Everything else; prose answers | "The event was Star Wars themed", "John and Sarah" |

**Detection implementation**: `memqa/utils/evaluator/qtype_utils.py`

**Key functions**:
- `is_number_answer(answer)`: Checks if answer is number-type
- `is_list_answer(answer)`: Checks if answer is evidence ID list
- `detect_qtype(answer)`: Returns `"number"`, `"list_recall"`, or `"open_end"`

**Detection algorithm** (simplified):

```python
def detect_qtype(answer: str) -> str:
    # Preprocess: strip parentheticals, currency breakdowns, etc.
    cleaned = preprocess(answer)
    
    # Check for dates, times, numbers, currency
    has_dates = extract_dates(cleaned)
    has_times = extract_times(cleaned)
    numbers, currencies = extract_numbers(cleaned)
    
    if has_dates or has_times or numbers or currencies:
        # Remove all numeric elements and connectors
        remainder = remove_date_time_text(cleaned)
        remainder = remove_number_connectors(remainder)
        
        # If nothing left → it's a number answer
        if not remainder:
            return "number"
    
    # Check for evidence ID patterns
    if is_evidence_id_list(answer):
        return "list_recall"
    
    # Default: open-ended
    return "open_end"
```

**Manual annotation**:

You can override automatic detection by adding `qtype` field to ground truth JSON:

```json
{
  "id": "qa_123",
  "question": "How many sessions?",
  "answer": "3",
  "qtype": "number"
}
```

**Qtype annotation tool**: `memqa/utils/final_data_processing/add_qtype.py`

```bash
python memqa/utils/final_data_processing/add_qtype.py \
  --input merged_qa.json
```

This adds/updates `qtype` field for all questions based on automatic detection.

---

## Usage Examples

### Example 1: Dual Judge Evaluation

Run two LLM judges in parallel (GLM-4.7 + GPT-5-mini):

```bash
run_eval_dual() {
  local predictions="$1"
  local eval_dir="$2"

  python memqa/utils/evaluator/evaluate_qa.py \
    --ground-truth "${QA_FILE}" \
    --predictions "${predictions}" \
    --output-dir "${eval_dir}/eval" \
    --judge-provider vllm \
    --judge-model "${JUDGE_MODEL_GLM}" \
    --judge-endpoint "${JUDGE_ENDPOINT}" \
    --judge-thinking "${JUDGE_THINKING}" \
    --request-delay 1.0 \
    --max-workers 1 &
  pid_glm=$!

  python memqa/utils/evaluator/evaluate_qa.py \
    --ground-truth "${QA_FILE}" \
    --predictions "${predictions}" \
    --output-dir "${eval_dir}/eval" \
    --judge-provider openai \
    --judge-model "${JUDGE_MODEL_GPT5_MINI}" \
    --judge-reasoning-effort minimal \
    --max-workers 2 &
  pid_gpt=$!

  wait "${pid_glm}"
  wait "${pid_gpt}"
}
```

### Example 2: Complete Evaluation Pipeline

```bash
# Step 1: Add qtype labels to ground truth
python memqa/utils/final_data_processing/add_qtype.py \
  --input memqa/utils/final_data_processing/atm-20260121.json

# Step 2: Run ATM evaluation
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth memqa/utils/final_data_processing/atm-20260121.json \
  --predictions output/QA_Agent/method/mmrag_answers.jsonl \
  --output-dir output/QA_Agent/method/eval \
  --metrics atm \
  --judge-provider openai \
  --judge-model gpt-5-mini

# Step 3: Post-hoc retrieval evaluation
python memqa/utils/evaluator/evaluate_retrieval/comprehensive_eval.py \
  --details output/QA_Agent/method/retrieval_recall_details.json

# Step 4: Joint answer + retrieval accuracy
python memqa/utils/evaluator/evaluate_retrieval/joint_accuracy.py \
  --retrieval-details output/QA_Agent/method/retrieval_recall_details.json \
  --atm-details output/QA_Agent/method/eval/atm_gpt-5-mini.json
```

### Example 3: EM-only (no LLM costs)

```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth merged_qa.json \
  --predictions predictions.jsonl \
  --output-dir eval/ \
  --metrics em
```

---

## Files and Outputs

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `evaluate_qa.py` | Ground truth + predictions | `{em,llm,atm}_*.json` | Answer evaluation |
| `comprehensive_eval.py` | `retrieval_recall_details.json` | `retrieval_recall_comprehensive_summary.json` | Retrieval metrics |
| `joint_accuracy.py` | Retrieval details + ATM results | `retrieval_recall_joint_accuracy_summary.json` | Joint metrics |
| `add_qtype.py` | Ground truth JSON | Updated JSON with `qtype` | Qtype annotation |

---

## References

- **Normalization**: `memqa/utils/evaluator/normalizer.py`
- **Qtype detection**: `memqa/utils/evaluator/qtype_utils.py`
- **Main evaluator**: `memqa/utils/evaluator/evaluate_qa.py`
- **Config**: `memqa/utils/evaluator/config.py`
