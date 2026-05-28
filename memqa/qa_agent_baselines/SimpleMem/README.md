# SimpleMem Baseline

SimpleMem is an efficient lifelong-memory framework for LLM agents. It builds
a compact memory bank by sliding a window over the dialogue stream, distilling
each window into structured memory units, and indexing them across three
complementary views (semantic vectors, lexical / FTS, and symbolic metadata).
Retrieval uses an intent-aware planner that selects which views to consult.

This baseline wraps the **upstream SimpleMem implementation** for strict
reproduction on ATMBench.

## Version Pin

This baseline is verified against upstream SimpleMem commit
**`094027eca4c890dc9912be8cee1da04428de8076`**. The run script aborts if the
checked-out `SIMPLEMEM_DIR` does not match this commit; override the pin by
exporting `SIMPLEMEM_EXPECTED_COMMIT=<sha>` (or empty string to disable the
check entirely) if you intentionally want to run against a different version.

## Quickstart

```bash
# 1) Clone SimpleMem and check out the pinned commit.
git clone https://github.com/aiming-lab/SimpleMem.git ../SimpleMem
git -C ../SimpleMem checkout 094027eca4c890dc9912be8cee1da04428de8076
export SIMPLEMEM_DIR=../SimpleMem

# 2) Install SimpleMem dependencies + this baseline's extras.
pip install -r "${SIMPLEMEM_DIR}/requirements.txt"
pip install -r memqa/qa_agent_baselines/SimpleMem/requirements.txt

# 3) Create the SimpleMem config (copy ${SIMPLEMEM_DIR}/config.py.example
#    to ${SIMPLEMEM_DIR}/config.py and fill in API key + endpoint).

# 4) Run the baseline (build + answer + eval on both splits).
bash scripts/QA_Agent/SimpleMem/run.sh
```

## What This Baseline Does

1. Converts multimodal evidence (emails, images, videos) into **text dialogue
   pairs** (one "user" turn per memory item plus a fixed "agent" ack).
2. Slides a window of size 40 (overlap 2) across the dialogue stream and asks
   the **build LLM** to emit `MemoryEntry` records for each window, with a
   `[SOURCE_ID: ...]` prefix attributing the entry to the source item IDs.
3. Persists those records in a **LanceDB** store with three indexes:
   - dense semantic embeddings,
   - lexical (Tantivy FTS) keywords,
   - symbolic metadata (entities, time/location tags).
4. At question time, plans which view(s) to query, retrieves the top-K candidates
   from each, and concatenates them into the answer prompt for the QA LLM.

## Alignment With Upstream SimpleMem

| Aspect | Upstream SimpleMem | This baseline | Aligned? |
|---|---|---|---|
| Memory schema (`MemoryEntry`, restatements, keywords) | pydantic schema | unchanged | yes |
| Sliding window (size, overlap) | size=40, overlap=2 | same | yes |
| Three-view retrieval (semantic + lexical + symbolic) | upstream LanceDB indexes | unchanged | yes |
| Planning + reflection (max 2 rounds) | upstream defaults | same | yes |
| Source-ID attribution prompt | n/a (no per-source IDs upstream) | added — see below | **augmented** |
| Answer prompt | upstream native | "atm" (HippoRAG2/A-Mem/MemoryOS-aligned), with `native` and `external` modes selectable | **augmented** |

### Source-ID Attribution

ATMBench requires each memory entry to be attributable back to the source item
IDs that produced it (so retrieval recall can be computed against gold
evidence). SimpleMem's pydantic schema does not natively carry per-source IDs.

We address this in two complementary ways:

1. **Prompt-side:** the build prompt is extended to instruct the LLM to prefix
   each `restatement` with `[SOURCE_ID: id_a, id_b, ...]`, and the example
   payload in the prompt uses an ATM-Bench item (e.g. `email_042`, `image_017`)
   to demonstrate the expected format.
2. **Deterministic fallback:** the post-processor parses any LLM-emitted
   source IDs, then deterministically attributes any remaining entries using
   the window's known `dialogue_id → item_id` map (substring match + window
   fallback). The build raises only if **direct LLM coverage** falls below
   `--simplemem-min-source-id-coverage` (default 0.9), and entries that are
   filled deterministically are tracked in `_atm_entry_to_item_ids.json`.

### Answer Modes

- `--simplemem-answer-mode atm` (default): use ATMBench's QA prompt
  (`QA_SYSTEM` / `QA_USER`), via SimpleMem's own LLM client. This aligns the
  answer prompt with HippoRAG2, A-Mem, MemoryOS.
- `--simplemem-answer-mode native`: use SimpleMem's upstream
  `answer_generator.generate_answer`.
- `--simplemem-answer-mode external`: route through the repo's `LLMClient`
  (matches the MMRag-style answer path; useful when the answer model lives on
  a different endpoint than the build model).

### Build / Resume

- Per-window checkpoints are written to the LanceDB folder during indexing.
- `--no-simplemem-force-rebuild` (script default `SIMPLEMEM_FORCE_REBUILD=0`)
  resumes from the last checkpointed window. Set `SIMPLEMEM_FORCE_REBUILD=1`
  to wipe the LanceDB store and rebuild from scratch.
- The build cache key is derived from the batch-result + email file **paths
  plus their mtimes** (same convention as the MemoryOS / HippoRAG2 baselines),
  along with the build LLM, endpoint, window/overlap, and the source-tracking
  prompt version. Renaming or rewriting any of the source files invalidates
  the cache.

## Key CLI Arguments

| Flag | Meaning |
|---|---|
| `--qa-file` | QA annotations JSON (`atm-bench.json` or `atm-bench-hard.json`) |
| `--stage build\|answer\|all` | Pipeline stage |
| `--simplemem-dir` | Path to the cloned SimpleMem repo |
| `--build-model` / `--build-endpoint` | Indexing LLM (default Qwen3-VL-2B) |
| `--model` / `--vllm-endpoint` | Answer LLM (default Qwen3-VL-8B-FP8) |
| `--simplemem-window-size`, `--simplemem-overlap-size` | Sliding window |
| `--simplemem-semantic-top-k` / `--simplemem-keyword-top-k` / `--simplemem-structured-top-k` | Top-K per retrieval view |
| `--simplemem-answer-mode atm\|native\|external` | Answer prompt + LLM routing |
| `--simplemem-force-rebuild` / `--no-simplemem-force-rebuild` | Wipe vs. resume |
| `--simplemem-min-source-id-coverage` | Min LLM-emitted source-ID coverage (default 0.9) |
| `--simplemem-enable-planning`, `--simplemem-enable-reflection`, `--simplemem-max-reflection-rounds` | SimpleMem retrieval planner |

## Output Structure

```
output/QA_Agent/SimpleMem/main_table/topk35/atmbench/simplemem/
├── simplemem_answers.jsonl
├── simplemem_answers_run_stats.json
├── retrieval_recall_details.json
├── retrieval_recall_comprehensive_summary.json
├── retrieval_recall_joint_accuracy_summary.json
└── eval/
    └── atm_<judge>.json

output/QA_Agent/SimpleMem/simplemem_lancedb/
├── <table_name>.lance/
├── _atm_entry_to_item_ids.json   # deterministic + LLM-emitted attribution map
├── _atm_build_checkpoint.json    # per-window resume state
└── _atm_build_meta.json          # cache key / corpus fingerprint
```

## Evaluation

The shipped `run.sh` calls `run_eval_bundle` (from `common_eval.sh`) which runs
`evaluate_qa.py` (ATM + EM judges) followed by `comprehensive_eval.py` and
`joint_accuracy.py`. To re-run only the evaluation step on already-produced
predictions:

```bash
source scripts/QA_Agent/common_eval.sh
run_eval_bundle \
  ./data/atm-bench/atm-bench.json \
  output/QA_Agent/SimpleMem/main_table/topk35/atmbench/simplemem/simplemem_answers.jsonl \
  output/QA_Agent/SimpleMem/main_table/topk35/atmbench/simplemem/eval \
  output/QA_Agent/SimpleMem/main_table/topk35/atmbench/simplemem/retrieval_recall_details.json
```

## References

- **Paper:** https://arxiv.org/abs/2601.02553
- **Code:**  https://github.com/aiming-lab/SimpleMem
- Pinned commit (verified by `run.sh`): `094027eca4c890dc9912be8cee1da04428de8076`
