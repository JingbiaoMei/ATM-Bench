# AGENT_SYSTEMS

## OVERVIEW
This folder contains a lightweight harness to benchmark agent-style CLIs on **ATM-hard** using:
- a sanitized `eval_root/` dataset (questions + memory only; no GT leakage)
- per-question isolated run directories
- uniform `answer.json` outputs for ATM evaluation

Assume all commands are run from the repo root.

## GOLDEN RULES (DO NOT BREAK)
- **Never expose ground-truth** files (e.g. `data/atm-bench/atm-bench*.json`) to the agent runtime environment.
- **One question = one isolated run directory**; no cross-question session persistence.
- **Do not commit** generated artifacts: `agent_systems/eval_root/` is gitignored, and results default under `output/` via `AGSYS_RESULTS_ROOT`.
- **Never print / cat credential files** or provider configs that may contain API keys.
- **OpenClaw needs extra suspicion**: its normal environment can inherit broad workspace access, agents, skills, plugins, and local instruction files. Treat it as a special-case runner and document any extra isolation assumptions explicitly.

## LAYOUT
- `agent_systems/config.py`: single source of truth for defaults (override via `AGSYS_*` env vars)
- `agent_systems/prepare_sandbox.py`: builds `agent_systems/eval_root/` with sanitized Qs + copied memory JSONs + runtime prompt copies
- `agent_systems/runtime_artifacts.py`: writes isolated per-run config artifacts for Claude/Codex/OpenClaw/opencode
- `agent_systems/scripts/`: per-agent runners (Claude Code / Codex / pi / opencode / OpenClaw)
- `agent_systems/openclaw/`: OpenClaw-specific benchmark docs (`benchmark_quickstart.md`, `isolation_setup.md`)
- `agent_systems/extract_answer.py`: normalizes tool output into `output/answer.json`
- `agent_systems/extract_usage.py`: best-effort token usage into `output/usage.json`
- `agent_systems/collect_results.py`: aggregates `answer.json` → `answers.jsonl`
- `agent_systems/collect_usage.py`: aggregates `usage.json` → `usage.jsonl` + `usage_summary.json`

## CONFIG CONVENTION
- Defaults live in `agent_systems/config.py` (no `defaults.sh`).
- Bash runners load resolved defaults via:
  - `eval "$(python3 agent_systems/config.py --print-bash)"`
- Keep new knobs as `AGSYS_*` env vars + CLI flags where helpful.

## RUNNER CONTRACT (PER QUESTION)
Each run directory is:
`agent_systems/eval_root/runs/<run_tag>/<agent>/<model_tag>/<question_id>/`

Inputs (symlinked into the run dir):
- `question.json`, `question.txt`
- `memory/` (3 JSON files)

Outputs (created by runner):
- `output/trace.*` (tool-specific raw trace)
- `output/answer.json` (strict schema: `{id, question, answer}`)
- `output/usage.json` (best-effort; may contain nulls if the tool doesn’t expose usage)
- `output/stderr.log`

## ADDING A NEW AGENT TOOL
- Put the runner in `agent_systems/scripts/<agent>/` and source `agent_systems/scripts/common.sh`.
- Use `agsys_setup_run_dir` and write outputs only under the per-question run dir.
- Enforce isolation (ephemeral/no-session-persistence) and isolate tool state (XDG/state dirs) per question.
- Capture a trace and run `extract_answer.py` + `extract_usage.py`.

## EVALUATION NOTE (LEAK-PROOFING)
Local runs from the repo root can still allow tools to read other repo files unless the tool sandbox blocks it.
For strict leakage prevention, run agents inside an external sandbox/container that mounts **only**
`agent_systems/eval_root/` (and no GT files).
