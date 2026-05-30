# Agent Systems — General-Purpose Agent Benchmark Harness

This harness runs general-purpose agent CLIs — **Claude Code**, **Codex**,
**pi**, **opencode**, and **OpenClaw** — as QA agents on ATM-Bench. Each
benchmark question is isolated into its own sandboxed workspace; the agent reads
the memory store, answers the question, and the harness collects uniform
`answer.json` + `usage.json` outputs and scores them with the repo's standard
evaluator.

Run all commands from the repo root.

## Prerequisites

Install and authenticate the agent CLI(s) you want to benchmark, locally:

- `claude` (Claude Code), `codex`, `pi` (`npm i -g @earendil-works/pi-coding-agent`),
  `opencode`, `openclaw`
- `node`/`npm`, `bwrap` (Bubblewrap — the default filesystem sandbox), `rg` (ripgrep)
- OpenClaw additionally needs `rsync`, `curl`, and `sudo` / a dedicated Linux
  user (see `agent_systems/openclaw/`)

Authentication (per agent, on the host):
- Claude Code: `~/.claude` credentials, or `ANTHROPIC_API_KEY`
- Codex: `~/.codex/auth.json`
- pi / opencode generic endpoint: `api_keys/.openai_compatible_key` (or the
  matching `*_API_KEY` env var)

See `agent_systems/runner_setup_public.md` for install/auth details and
`agent_systems/runner_versions.md` for the CLI versions used.

## Configuration

Defaults live in `agent_systems/config.py`, overridable via `AGSYS_*` env vars:

```bash
python3 agent_systems/config.py --print-json
```

Common knobs:
- `AGSYS_QA_SOURCE`: benchmark ground-truth JSON, used only for sandbox prep +
  evaluation, never exposed to agents (default `data/atm-bench/atm-bench-hard.json`;
  set to `data/atm-bench/atm-bench.json` for the full set)
- `AGSYS_MEMORY_MODE`: memory representation — `sgm` (default), `raw`, or `descriptive`
- `AGSYS_EVAL_ROOT`: sandbox root (mode-specific default: `agent_systems/eval_root_sgm|_raw|_dm`)
- `AGSYS_IMAGE_METADATA`, `AGSYS_VIDEO_METADATA`, `AGSYS_EMAILS`: memory inputs
  (default `output/{image,video}/qwen3vl2b/batch_results.json`, `data/raw_memory/email/emails.json`)
- `AGSYS_RAW_IMAGE_DIR`, `AGSYS_RAW_VIDEO_DIR`: raw media dirs for `raw` mode
  (default `data/raw_memory/{image,video}`)
- `AGSYS_RUN_TAG`: groups runs under `eval_root/runs/<run_tag>/...` (default `atm-bench-hard`)
- `AGSYS_RESULTS_ROOT`: where `answers.jsonl` and eval outputs go (default `output/QA_Agent/AgentSystems`)
- `AGSYS_EVAL_ENABLE_GPT_JUDGE`, `AGSYS_EVAL_GPT_*`: primary ATM judge (default `gpt-5-mini` via `openai`, reasoning effort `minimal`)
- `AGSYS_EVAL_ENABLE_GLM_JUDGE`, `AGSYS_EVAL_GLM_*`: **optional** secondary ATM judge, **off by default** (no endpoint baked in)
- `AGSYS_<AGENT>_MODEL`, `AGSYS_<AGENT>_MODEL_TAG`: per-agent model selection + result dir naming
- pi knobs: `AGSYS_PI_PROVIDER`, `AGSYS_PI_THINKING`, `AGSYS_PI_TOOLS`, `AGSYS_PI_SANDBOX`, `AGSYS_PI_OFFLINE`, `AGSYS_PI_EXTENSION_PATHS`, `AGSYS_PI_TIMEOUT_S`

### Evaluation judges

Evaluation reuses the repo's standard evaluator (`memqa/utils/evaluator/evaluate_qa.py`).
The default is the **GPT judge only** (`gpt-5-mini`, EM + ATM). The GLM judge is
optional and disabled by default; to enable it, point it at your own
OpenAI/vLLM-compatible endpoint:

```bash
export AGSYS_EVAL_ENABLE_GLM_JUDGE=1
export AGSYS_EVAL_GLM_ENDPOINT=http://your-glm-host:8000/v1/chat/completions
```

## 1) Prepare the sandbox (`eval_root/`)

Build the sanitized dataset (questions + memory only; no ground-truth leakage):

```bash
python3 agent_systems/prepare_sandbox.py --force
```

`--force` archives the existing eval root under `agent_systems/eval_root_archive/`
before rebuilding. It creates:

```
agent_systems/eval_root_sgm/
  memory/               # representation-specific JSON files
  qas/<id>/             # question.json (sanitized) + question.txt
  prompts/              # runtime copies of system_prompt.txt + qa_schema.json
  question_ids.txt
```

Memory modes:
- `sgm` (default): full schema-guided image/video JSON plus emails.
- `raw`: trimmed image/video JSON (IDs + raw media paths) with hardlinked media under
  `memory/raw_images/` and `memory/raw_videos/`; emails unchanged.
- `descriptive`: trimmed image/video JSON (`{id, caption}`) only; emails unchanged.

Prepare non-default variants (each gets its own eval root so variants can coexist):

```bash
AGSYS_MEMORY_MODE=raw         python3 agent_systems/prepare_sandbox.py --force
AGSYS_MEMORY_MODE=descriptive python3 agent_systems/prepare_sandbox.py --force
```

## 2) Run an agent (one isolated run dir per question)

Each runner creates per-question workspaces under
`eval_root/runs/<run_tag>/<agent>/<model_tag>/<question_id>/output/` containing
`trace.*`, `answer.json` (strict `{id, question, answer}`), `usage.json`,
`stderr.log`, and `runtime_manifest.json`.

Run all questions with a shipped preset:

```bash
bash agent_systems/scripts/claude_code/run_claude_code_opus47-xhigh.sh      # Claude Opus 4.7 (xhigh)
bash agent_systems/scripts/codex/run_codex_gpt55_m.sh                       # Codex GPT-5.5 (medium)
bash agent_systems/scripts/pi/run_pi_openai_compatible.sh                   # pi via any OpenAI-compatible endpoint
bash agent_systems/scripts/opencode/run_opencode_openai_compatible.sh       # opencode via any OpenAI-compatible endpoint
# OpenClaw: run via the benchmark-user workflow (see below)
```

When you run *all* questions (no `<question_id>`), each runner then (1) collects
`answers.jsonl`, (2) collects `usage.jsonl` + `usage_summary.json`, and (3) runs
ATM evaluation. Skip step 3 with `AGSYS_SKIP_EVAL=1`. Run a single question by
passing its id:

```bash
bash agent_systems/scripts/codex/run_codex_gpt55_m.sh <question_id>
```

The Claude Code and Codex engines (`run_claude_code.sh`, `run_codex.sh`) take
the model via `AGSYS_CLAUDE_MODEL` / `AGSYS_CODEX_MODEL`; the presets above just
set those plus the effort level. Keep `AGSYS_MEMORY_MODE` set to run against a
prepared variant — non-`sgm` modes are suffixed into the result dir (e.g.
`-raw`, `-dm`).

### Generic OpenAI-compatible presets (pi, opencode)

The `pi` and `opencode` presets are **provider-agnostic templates**: point them
at any server that speaks the OpenAI `/v1/chat/completions` API (vLLM,
llama.cpp, LM Studio, OpenRouter, a cloud gateway, ...). Configure entirely via
env vars; nothing provider-specific is baked in.

```bash
# pi
PI_OPENAI_BASE_URL=https://your-host/v1 PI_OPENAI_MODEL=your-model \
  bash agent_systems/scripts/pi/run_pi_openai_compatible.sh

# opencode
OPENCODE_BASE_URL=https://your-host/v1 OPENCODE_MODEL_ID=your-model \
  bash agent_systems/scripts/opencode/run_opencode_openai_compatible.sh
```

The API key is read from `OPENAI_COMPATIBLE_API_KEY` or `api_keys/.openai_compatible_key`
and forwarded into the sandbox at runtime — it is never written into a config
file on disk. The pi extension lives at
`agent_systems/scripts/pi/extensions/openai-compatible/`; the opencode reference
config at `agent_systems/scripts/opencode/configs/openai-compatible.json`.

To benchmark other models/providers, copy a preset and set the appropriate
`AGSYS_*_MODEL` (and provider config). List opencode model ids with
`bash agent_systems/scripts/opencode/opencode_models.sh`.

### OpenClaw (benchmark-user workflow)

OpenClaw runs under a dedicated, isolated Linux user. Do **not** call
`run_openclaw.sh` directly; use the benchmark-user workflow:

1. `python3 agent_systems/prepare_sandbox.py --force`
2. `sudo bash agent_systems/scripts/openclaw/package_openclaw_bundle.sh <bundle_dir>`
3. `sudo chown -R openclaw-bench:openclaw-bench <bundle_dir>`
4. As `openclaw-bench` from the bundle root: `bash agent_systems/scripts/openclaw/smoke_test_openclaw.sh`
5. As `openclaw-bench` from the bundle root: `bash agent_systems/scripts/openclaw/run_openclaw_bench_user.sh`

Full procedure and rationale: `agent_systems/openclaw/README.md`,
`benchmark_quickstart.md`, and `isolation_setup.md`.

## 3) Collect answers and usage

```bash
python3 agent_systems/collect_results.py --all --model-tag <model_tag>
python3 agent_systems/collect_usage.py   --all --model-tag <model_tag>
```

Outputs land under `$AGSYS_RESULTS_ROOT/<run_tag>/<agent>/<model_tag>/`:
`answers.jsonl`, `usage.jsonl`, `usage_summary.json`.

Token totals are **not** apples-to-apples across agents: Claude Code totals are
dominated by cached-context rereads, Codex by replayed shell output + verbose
reasoning, while opencode keeps a tighter cumulative context. Inspect both
`usage_summary.json` and representative `trace.jsonl` files when comparing.

## 4) Evaluate (EM + ATM)

The runners evaluate automatically; to run it manually:

```bash
python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth data/atm-bench/atm-bench-hard.json \
  --predictions "$AGSYS_RESULTS_ROOT/<run_tag>/<agent>/<model_tag>/answers.jsonl" \
  --output-dir  "$AGSYS_RESULTS_ROOT/<run_tag>/<agent>/<model_tag>/eval" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2
```

Summary files written under `eval/`:
- `deterministic_accuracy_summary.json` (EM)
- `atm_gpt-5-mini_summary.json` (ATM, GPT judge)

The evaluator extracts memory-item IDs from answer prose uniformly, so
agent-system predictions are scored the same way as the other baselines.

## See also

- `agent_systems/RUNNER_GUIDE.md` — golden rules / runner contract
- `agent_systems/runner_setup_public.md` — per-agent install + auth
- `agent_systems/runner_versions.md` — recorded CLI versions
- `agent_systems/openclaw/` — OpenClaw isolation + operator guide
