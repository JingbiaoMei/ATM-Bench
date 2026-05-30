# OpenClaw Benchmark Quickstart

This is the canonical operator procedure for running the OpenClaw benchmark.

Use this document if you want the shortest correct path. For the design
rationale and threat-model discussion, see `agent_systems/openclaw/isolation_setup.md`.
Use `agent_systems/openclaw/README.md` as the top-level entry page.

Scope:
- source host prepares the benchmark bundle from the main repo
- benchmark user runs OpenClaw from the copied bundle only
- evaluation happens back on the source host

Do not use `agent_systems/scripts/openclaw/run_openclaw.sh` directly. It is an
internal worker script.

## Supported scripts

Use only these scripts:
- `agent_systems/scripts/openclaw/package_openclaw_bundle.sh`
- `agent_systems/scripts/openclaw/smoke_test_openclaw.sh`
- `agent_systems/scripts/openclaw/run_openclaw_bench_user.sh`

## Phase 1: Source host setup

Run these on the source host from the main repo checkout.

### 1. Create the benchmark user and directories

```bash
sudo adduser --disabled-password --gecos "" openclaw-bench
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/atmbench
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/.openclaw-atmbench
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/.openclaw-state
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/.config/openclaw
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/.cache/openclaw
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/.local/openclaw-cli
```

### 2. Prepare the benchmark sandbox

```bash
python3 agent_systems/prepare_sandbox.py --force
```

Warning: `--force` archives the existing `agent_systems/eval_root_sgm/` tree before rebuilding it. Previous run artifacts are moved under `agent_systems/eval_root_archive/`.

### 3. Package the benchmark bundle

```bash
sudo bash agent_systems/scripts/openclaw/package_openclaw_bundle.sh /home/openclaw-bench/atmbench
sudo chown -R openclaw-bench:openclaw-bench /home/openclaw-bench/atmbench
```

Why `sudo` is needed here:
- `/home/openclaw-bench/atmbench` is owned by `openclaw-bench`
- the operator usually cannot write there directly
- if you want to avoid `sudo` for packaging, package to a the operator-writable temp dir first, then copy it over with `sudo`

What this package contains:
- `agent_systems/`
- prepared `agent_systems/eval_root_sgm/`

What it excludes:
- `data/atm-bench/atm-bench*.json`
- `agent_systems/eval_root_sgm/runs/`
- hidden runtime residue such as `agent_systems/eval_root_sgm/.codex-runtime`

## Phase 2: Benchmark user setup

Switch to the benchmark user:

```bash
sudo -u openclaw-bench -H bash
cd /home/openclaw-bench/atmbench
```

### 4. Install OpenClaw

```bash
curl -fsSL --proto "=https" --tlsv1.2 https://openclaw.ai/install-cli.sh | bash -s -- \
  --prefix /home/openclaw-bench/.local/openclaw-cli \
  --no-onboard
```

### 5. Create the benchmark config

```bash
cd /home/openclaw-bench
mkdir -p .openclaw
```

Create `/home/openclaw-bench/.openclaw/openclaw.json`:

```json
{
  "agents": {
    "defaults": {
      "workspace": "/home/openclaw-bench/atmbench/agent_systems/eval_root_sgm"
    }
  },
  "tools": {
    "web": {
      "search": {
        "enabled": false
      },
      "fetch": {
        "enabled": false
      }
    },
    "agentToAgent": {
      "enabled": false
    }
  },
  "browser": {
    "enabled": false
  },
  "plugins": {
    "entries": {}
  },
  "skills": {
    "load": {
      "extraDirs": [],
      "watch": false
    },
    "entries": {}
  }
}
```

Add your single benchmark model to this config. Do not add fallback models if
you want a clean single-model benchmark.
The benchmark launcher now defaults to `/home/openclaw-bench/.openclaw/openclaw.json`
and falls back to `/home/openclaw-bench/.config/openclaw/openclaw.json` only if
the former does not exist.
If `agents.defaults.model.primary` is set, the launcher will also derive the
result `model_tag` from that value automatically.

### 6. Configure model credentials

You have two supported options.

Option A: explicit `provider.env` file

Create `/home/openclaw-bench/.config/openclaw/provider.env`:

```bash
cat > /home/openclaw-bench/.config/openclaw/provider.env <<'EOF'
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
EOF
chmod 600 /home/openclaw-bench/.config/openclaw/provider.env
```

Use the provider variables appropriate for your OpenClaw model backend.

Option B: configure model auth via the OpenClaw CLI wizard

```bash
/home/openclaw-bench/.local/openclaw-cli/bin/openclaw config --section model
```

This uses OpenClaw's interactive model setup flow under the benchmark user.
If this path works well on your machine, it is simpler than the lower-level
`models auth ...` commands.

Notes:
- the installed binary is `openclaw`, not `openclaw-cli`
- if you use `openclaw config --section model`, you may skip `provider.env`
- `provider.env` is still the more explicit and reproducible path for scripted setups
- this flow normally writes the active config to `/home/openclaw-bench/.openclaw/openclaw.json`
- the benchmark runner will seed auth from `/home/openclaw-bench/.openclaw/agents/main/agent/`
  into the isolated temp state for each question

Lower-level alternative if you want direct auth-profile commands:

```bash
/home/openclaw-bench/.local/openclaw-cli/bin/openclaw models auth add
/home/openclaw-bench/.local/openclaw-cli/bin/openclaw models auth login
```

## Phase 3: Run OpenClaw

### 7. Run the smoke test first

From the benchmark bundle root:

```bash
cd /home/openclaw-bench/atmbench
bash agent_systems/scripts/openclaw/smoke_test_openclaw.sh
```

If you want to test a specific question:

```bash
bash agent_systems/scripts/openclaw/smoke_test_openclaw.sh <question_id>
```

Inspect these files before running the batch:
- `agent_systems/eval_root_sgm/runs/<run_tag>/openclaw/<model_tag>/<question_id>/output/answer.json`
- `agent_systems/eval_root_sgm/runs/<run_tag>/openclaw/<model_tag>/<question_id>/output/trace.json`
- `agent_systems/eval_root_sgm/runs/<run_tag>/openclaw/<model_tag>/<question_id>/output/runtime_manifest.json`
- `agent_systems/eval_root_sgm/runs/<run_tag>/openclaw/<model_tag>/<question_id>/output/stderr.log`

### 8. Run the full benchmark

```bash
cd /home/openclaw-bench/atmbench
bash agent_systems/scripts/openclaw/run_openclaw_bench_user.sh
```

## Phase 4: Bring results back

The benchmark run should keep `AGSYS_SKIP_EVAL=1`, so evaluation is not done on
the benchmark user side.

On the host machine, copy the OpenClaw outputs from the benchmark bundle back
into the main repo checkout.

Example for the a `<your-model-tag>` run:

```bash
mkdir -p output/QA_Agent/AgentSystems/atm-bench-hard/openclaw
mkdir -p agent_systems/eval_root_sgm/runs/atm-bench-hard/openclaw

sudo cp -a /home/openclaw-bench/atmbench/output/QA_Agent/AgentSystems/atm-bench-hard/openclaw/<your-model-tag> \
  output/QA_Agent/AgentSystems/atm-bench-hard/openclaw/

sudo cp -a /home/openclaw-bench/atmbench/agent_systems/eval_root_sgm/runs/atm-bench-hard/openclaw/<your-model-tag> \
  agent_systems/eval_root_sgm/runs/atm-bench-hard/openclaw/
```

Replace `<your-model-tag>` with your actual OpenClaw `model_tag` if different.

The first copy brings back the aggregate outputs:
- `answers.jsonl`
- `usage.jsonl`
- `usage_summary.json`

The second copy brings back the per-question run artifacts for auditing:
- `answer.json`
- `trace.json`
- `runtime_manifest.json`
- `stderr.log`

Then, on the host machine, rebuild the aggregate outputs and run evaluation
from the source repo checkout, where the ground-truth lives.

Example for the current `<your-model-tag>` run:

```bash
# Run this from the ATMBench repo root, where the ground-truth lives.
cd /path/to/ATMBench

python3 agent_systems/collect_results.py \
  --agent openclaw \
  --run-tag atm-bench-hard \
  --model-tag <your-model-tag>

python3 agent_systems/collect_usage.py \
  --agent openclaw \
  --run-tag atm-bench-hard \
  --model-tag <your-model-tag>

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth data/atm-bench/atm-bench-hard.json \
  --predictions output/QA_Agent/AgentSystems/atm-bench-hard/openclaw/<your-model-tag>/answers.jsonl \
  --output-dir output/QA_Agent/AgentSystems/atm-bench-hard/openclaw/<your-model-tag>/eval \
  --metrics em atm \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-reasoning-effort minimal \
  --max-workers 2
```

If you already copied back the aggregate `answers.jsonl` and `usage_summary.json`
and do not need to refresh them from the per-question run artifacts, you may
skip the `collect_results.py` and `collect_usage.py` steps.

## Files that document this workflow

- top-level doc index: `agent_systems/openclaw/README.md`
- bundle packager: `agent_systems/scripts/openclaw/package_openclaw_bundle.sh`
- smoke test: `agent_systems/scripts/openclaw/smoke_test_openclaw.sh`
- full-run launcher: `agent_systems/scripts/openclaw/run_openclaw_bench_user.sh`
- detailed rationale: `agent_systems/openclaw/isolation_setup.md`

## Common mistakes

- running `run_openclaw.sh` directly
- copying the whole repo instead of only the packaged bundle
- forgetting to run the smoke test first
- evaluating on the benchmark user side
- leaving `AGENTS.md`, `SOUL.md`, `TOOLS.md`, or `BOOTSTRAP.md` inside the benchmark workspace
