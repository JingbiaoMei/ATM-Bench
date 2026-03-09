# Oracle QA Baseline

The Oracle baseline answers each question using the ground-truth evidence IDs associated with it. It provides an upper-bound reference for Memory QA and a sanity-check baseline when run without evidence.

## Functionality
- **Evidence routing**: Uses `evidence_ids` to load emails, images, and videos.
- **Evidence modes**:
  - `batch_results`: text-only evidence from batch metadata (captions, OCR, summaries).
  - `raw`: multimodal inputs (images and video frames) with location and timestamp metadata from batch results.
- **No-evidence mode**: `--no-evidence` disables all evidence to measure how much memory is required.
- **Providers**: `openai`, `vllm` (OpenAI-compatible endpoint), `vllm_local` (text-only, single-threaded).
- **Concurrency**: `--max-workers` controls parallel requests.
- **Progress**: `tqdm` progress bar for inference.

## Usage
Run from repo root:

```bash
bash scripts/QA_Agent/Oracle/run_oracle_gpt5.sh
bash scripts/QA_Agent/Oracle/run_oracle_qwen3vl8b.sh
bash scripts/QA_Agent/Oracle/run_oracle_no_evidence_qwen3vl8b.sh
```

These scripts evaluate with `gpt-5-mini` as the judge (paper default).

### Direct CLI (Example)

```bash
python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file data/atm-bench/atm-bench.json \
  --media-source raw \
  --image-batch-results output/image/qwen3vl2b/batch_results.json \
  --video-batch-results output/video/qwen3vl2b/batch_results.json \
  --image-root data/raw_memory/image \
  --video-root data/raw_memory/video \
  --email-file data/raw_memory/email/emails.json \
  --provider vllm \
  --vllm-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --max-workers 8 \
  --timeout 120 \
  --output-file output/QA_Agent/Oracle/qwen3vl8b/atmbench/oracle_qwen3vl8b.jsonl
```

### CLI Arguments
- `--qa-file`: QA JSON input.
- `--use-niah-pools`: treat `niah_evidence_ids` (or `--niah-field`) as `evidence_ids` (for NIAH evaluation).
- `--niah-field`: field name for the NIAH evidence pool (default: `niah_evidence_ids`).
- `--niah-strict`: error if any entry is missing the NIAH evidence field.
- `--media-source`: `batch_results` or `raw`.
- `--image-batch-results`: image batch results JSON (batch and raw mode, provides metadata).
- `--video-batch-results`: video batch results JSON (batch and raw mode, provides metadata).
- `--image-root`: raw image root directory (raw mode).
- `--video-root`: raw video root directory (raw mode).
- `--email-file`: merged email JSON.
- `--provider`: `openai`, `vllm`, or `vllm_local`.
- `--model`: model name override.
- `--api-key`: API key override.
- `--vllm-endpoint`: OpenAI-compatible endpoint for vLLM.
- `--max-tokens`: max tokens for completion.
- `--temperature`: decoding temperature.
- `--timeout`: request timeout in seconds.
- `--max-evidence-items`: cap evidence items per question.
- `--batch-fields`: batch_results image/video fields to include (comma/space-separated). Use `all` or `none`. ID is always included.
- `--no-evidence`: run without evidence.
- `--num-frames`: number of frames per video (raw mode).
- `--frame-strategy`: frame sampling strategy.
- `--max-workers`: concurrency for API calls.
- `--output-file`: JSONL output path.

## Configuration
Defaults and prompts live in `memqa/qa_agent_baselines/oracle/config.py` and reuse `memqa/global_config.py` for model and API settings.
