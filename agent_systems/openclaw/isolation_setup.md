# OpenClaw Isolation Setup

If you only need the exact operator procedure, read
`agent_systems/openclaw/benchmark_quickstart.md` first. That file is the
canonical step-by-step runbook. This document explains why the setup looks the
way it does and what to audit.

This guide is the recommended operator setup for running OpenClaw on `agent_systems` without inheriting the operator's normal `$HOME/.openclaw`, workspace files, skills, plugins, or session state.

Goal:
- prepare the benchmark from the main repo as the operator
- run OpenClaw under a separate Linux user with a fresh OpenClaw home/state/config
- keep evaluation on the operator's side after answers are collected

This is the best practical setup for now. It is simpler than full Docker, aligns with OpenClaw's official config model, and avoids sharing the operator's normal OpenClaw environment.

## Why This Layout

OpenClaw's docs make four things clear:
- `OPENCLAW_HOME`, `OPENCLAW_STATE_DIR`, and `OPENCLAW_CONFIG_PATH` are first-class isolation knobs
- `--profile` is an official way to run separate gateways on the same host
- the agent workspace is only the default working directory, not a hard sandbox
- Docker sandboxing isolates tool execution, but the gateway still runs on the host

That means "install another OpenClaw binary" is not the main boundary. The real boundary is:
- separate Linux user
- separate home/state/config
- separate workspace
- no inherited workspace bootstrap files

## Recommended Topology

Use two sides:

1. the operator side: full repo, sandbox preparation, answer collection, evaluation
2. `openclaw-bench` side: OpenClaw execution only, with a copied benchmark bundle

Do not run the OpenClaw benchmark from the full repo root if your goal is strict isolation.

## What To Copy Into The Benchmark Bundle

Copy only a minimal runtime bundle for the benchmark user:

- `agent_systems/`
- inside it, the generated `agent_systems/eval_root_sgm/`

Do not copy:
- `data/atm-bench/atm-bench*.json`
- the rest of the repo
- the operator's `$HOME/.openclaw`
- the operator's workspace bootstrap files

The benchmark run should set `AGSYS_SKIP_EVAL=1` so OpenClaw never needs ground-truth files on its side.

## One-Time Host Setup

Run these as the operator on the host. Commands using `sudo` are for you to execute manually.

Helper scripts in this repo:
- top-level doc index: `agent_systems/openclaw/README.md`
- benchmark-user launcher: `agent_systems/scripts/openclaw/run_openclaw_bench_user.sh`
- bundle packager: `agent_systems/scripts/openclaw/package_openclaw_bundle.sh`
- one-question smoke test: `agent_systems/scripts/openclaw/smoke_test_openclaw.sh`
- canonical quickstart: `agent_systems/openclaw/benchmark_quickstart.md`

### 1. Create a dedicated benchmark user

```bash
sudo adduser --disabled-password --gecos "" openclaw-bench
```

### 2. Create dedicated directories

```bash
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/atmbench
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/.openclaw-atmbench
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/.openclaw-state
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/.config/openclaw
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/.cache/openclaw
sudo install -d -o openclaw-bench -g openclaw-bench /home/openclaw-bench/.local/openclaw-cli
```

### 3. Prepare the benchmark sandbox in the main repo

From the repo root:

```bash
python3 agent_systems/prepare_sandbox.py --force
```

Warning: `--force` archives the current `agent_systems/eval_root_sgm/` tree, including previous run artifacts stored under `agent_systems/eval_root_sgm/runs/`, into `agent_systems/eval_root_archive/`.

### 4. Package only the benchmark bundle and transfer it to the benchmark user

Use the packaging helper instead of direct `rsync`. It excludes `eval_root/runs/`,
hidden runtime residue such as `.codex-runtime`, and aborts if the packaged bundle
still contains symlinks.

```bash
sudo bash agent_systems/scripts/openclaw/package_openclaw_bundle.sh /home/openclaw-bench/atmbench
sudo chown -R openclaw-bench:openclaw-bench /home/openclaw-bench/atmbench
```

Packaging directly into `/home/openclaw-bench/...` usually requires `sudo`,
because those directories are intentionally owned by `openclaw-bench`, not by the operator.

### 5. Install OpenClaw under the benchmark user

Use OpenClaw's official CLI-only installer under that user, with onboarding disabled and a separate install prefix. Do not run the normal onboarding flow for the benchmark user.

Example:

```bash
sudo -u openclaw-bench -H bash -lc \
  'curl -fsSL --proto "=https" --tlsv1.2 https://openclaw.ai/install-cli.sh | bash -s -- \
    --prefix /home/openclaw-bench/.local/openclaw-cli \
    --no-onboard'
```

This keeps the binary install separate from the runtime state/config paths used below.

If you already have a shared system binary and want to reuse it, that is acceptable, but do not reuse the operator's home/state/config and do not run onboarding against the operator's environment.

## Benchmark User Runtime Layout

Under `/home/openclaw-bench`, keep these paths separate:

- CLI install prefix:
  - `/home/openclaw-bench/.local/openclaw-cli`
- `OPENCLAW_HOME=/home/openclaw-bench/.openclaw-atmbench`
- `OPENCLAW_STATE_DIR=/home/openclaw-bench/.openclaw-state`
- `OPENCLAW_CONFIG_PATH=/home/openclaw-bench/.openclaw/openclaw.json`
- workspace root inside the copied benchmark tree:
  - `/home/openclaw-bench/atmbench/agent_systems/eval_root_sgm/`

Do not use:
- `/home/openclaw-bench/.openclaw/workspace`
- the operator's repo checkout as the OpenClaw workspace

Clarification:
- the benchmark user's `/home/openclaw-bench/.openclaw/openclaw.json` is acceptable as the config source
- what you must avoid is reusing the operator's `$HOME/.openclaw` or using the default `/home/openclaw-bench/.openclaw/workspace`

## Minimal Base OpenClaw Config

The current runner builds a sanitized per-question config from `AGSYS_OPENCLAW_CONFIG_SOURCE`, so the benchmark user still needs a base config file to exist.

Create `/home/openclaw-bench/.openclaw/openclaw.json` with a minimal baseline like this:

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

Add your own model/provider configuration on top of this if needed. Keep it benchmark-specific.
If you set `agents.defaults.model.primary`, the benchmark launcher will derive
`AGSYS_OPENCLAW_MODEL_TAG` from that primary model automatically unless you
override it explicitly.

## Keep The Benchmark User Clean

Before running anything substantial, verify these are empty or benchmark-specific:

- `/home/openclaw-bench/.openclaw-atmbench`
- `/home/openclaw-bench/.openclaw-state`
- `/home/openclaw-bench/.openclaw/openclaw.json`
- `/home/openclaw-bench/atmbench/agent_systems/eval_root_sgm`

Do not place any of these files in the benchmark workspace unless you deliberately want them exposed:

- `AGENTS.md`
- `SOUL.md`
- `TOOLS.md`
- `BOOTSTRAP.md`

OpenClaw treats workspace bootstrap files as part of its normal runtime model, so keeping them absent is intentional here.

## Provider Credentials

`openclaw agent --local` still needs provider credentials. Do not rely on the operator's normal OpenClaw auth state.

Recommended pattern for scripted reproducibility:

1. create a benchmark-only env file owned by `openclaw-bench`
2. keep it outside the workspace
3. source it only for benchmark runs

Example:

```bash
sudo -u openclaw-bench -H bash -lc 'umask 077 && cat > /home/openclaw-bench/.config/openclaw/provider.env <<EOF
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
EOF'
```

Adjust the variables for your actual provider. Keep this file out of the copied benchmark tree.

Alternative:

If the env-file route is awkward on your machine, configure the benchmark
user's model auth interactively instead:

```bash
/home/openclaw-bench/.local/openclaw-cli/bin/openclaw config --section model
```

That path is simpler for manual setup. In that case, `provider.env` can be absent
and the benchmark launcher will assume auth is coming from OpenClaw-managed config/state.
For the current benchmark harness, that means seeding auth from
`/home/openclaw-bench/.openclaw/agents/main/agent/` into the temp per-question state.

## Running OpenClaw As The Benchmark User

Switch to the benchmark user:

```bash
sudo -u openclaw-bench -H bash
```

Then run from the copied benchmark root:

```bash
cd /home/openclaw-bench/atmbench

export OPENCLAW_HOME=/home/openclaw-bench/.openclaw-atmbench
export OPENCLAW_STATE_DIR=/home/openclaw-bench/.openclaw-state
export OPENCLAW_CONFIG_PATH=/home/openclaw-bench/.openclaw/openclaw.json

# Optional if you are using the env-file path:
if [[ -f /home/openclaw-bench/.config/openclaw/provider.env ]]; then
  source /home/openclaw-bench/.config/openclaw/provider.env
fi

export AGSYS_OPENCLAW_BIN=/home/openclaw-bench/.local/openclaw-cli/bin/openclaw
export AGSYS_OPENCLAW_CONFIG_SOURCE=/home/openclaw-bench/.openclaw/openclaw.json
export AGSYS_SKIP_EVAL=1

bash agent_systems/scripts/openclaw/smoke_test_openclaw.sh
bash agent_systems/scripts/openclaw/run_openclaw_bench_user.sh
```

If you want a separate OpenClaw profile on top of the separate user, use one consistently:

```bash
export OPENCLAW_PROFILE=atmbench
```

That is optional here because the separate user plus explicit env vars is already the main isolation boundary.

## Collect And Bring Results Back

For a normal full-batch run, the runner already performs:
- `collect_results.py`
- `collect_usage.py`

So you usually only need to copy the generated outputs back to the operator's side.

Manual collection is only needed if you ran a one-question smoke test, stopped a
batch early, or want to rebuild aggregates after an interrupted run.

Still as `openclaw-bench`, manual collection is:

```bash
cd /home/openclaw-bench/atmbench
python3 agent_systems/collect_results.py --agent openclaw --run-tag atm-bench-hard --model-tag <your-model-tag>
python3 agent_systems/collect_usage.py --agent openclaw --run-tag atm-bench-hard --model-tag <your-model-tag>
```

If you changed the primary model or overrode `AGSYS_OPENCLAW_MODEL_TAG`, use that
model tag instead of `<your-model-tag>`.

Then as the operator, copy only the produced result files back into the main repo or evaluate directly from that copied location. Keep evaluation on the operator's side where the ground-truth lives.

## Auditing Checklist

Before trusting a run, check:

1. `output/runtime_manifest.json` exists for each question
2. `output/openclaw-logs/` exists when OpenClaw emitted logs
3. `output/stderr.log` does not show fallback prompts, browser startup, or plugin loading
4. the benchmark user's config file still has web/browser/plugin/skill helpers disabled
5. the benchmark workspace does not contain unexpected `AGENTS.md`, `SOUL.md`, or copied secrets

## Optional Later Hardening: Docker

If you need a stronger boundary later, the next step is full Docker-backed OpenClaw, not just another host install.

Use Docker only if you are prepared to control mounts carefully:
- mount only the copied benchmark tree
- keep `workspaceAccess` conservative
- keep Docker network disabled unless required
- avoid broad bind mounts

Do not assume the built-in OpenClaw sandbox alone is enough. OpenClaw's docs are explicit that the gateway remains on the host unless you run the whole gateway in Docker.

## Practical Recommendation

For this project, the recommended order is:

1. separate Linux user
2. copied benchmark bundle under that user
3. fresh OpenClaw home/state/config
4. no workspace bootstrap files
5. `AGSYS_SKIP_EVAL=1`
6. only if needed later, full Docker gateway isolation

## Sources

- [OpenClaw Install](https://docs.openclaw.ai/install/index)
- [OpenClaw Installer Internals](https://docs.openclaw.ai/install/installer)
- [OpenClaw Getting Started](https://docs.openclaw.ai/start/getting-started)
- [OpenClaw Multiple Gateways](https://docs.openclaw.ai/gateway/multiple-gateways)
- [OpenClaw Agent Workspace](https://docs.openclaw.ai/agent-workspace)
- [OpenClaw Agent Bootstrapping](https://docs.openclaw.ai/start/bootstrapping)
- [OpenClaw Sandboxing](https://docs.openclaw.ai/gateway/sandboxing)
- [OpenClaw Docker Install](https://docs.openclaw.ai/install/docker)
- [OpenClaw Configuration](https://docs.openclaw.ai/gateway/configuration)
