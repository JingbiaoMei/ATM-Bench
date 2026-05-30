# Runner Setup Notes For Public Use

This note covers the four non-OpenClaw runners:
- `claude_code`
- `codex`
- `opencode`
- `pi`

These runners isolate each benchmark question into a fresh temp workspace and
temp runtime home. They still need a valid local CLI install and local
authentication on the operator machine.

This document answers three practical questions:
- what must be installed
- where auth is expected to live
- what the runner copies into its isolated runtime

## Common expectations

Run from the repo root.

Prepare the benchmark sandbox first:

```bash
python3 agent_systems/prepare_sandbox.py --force
```

Warning: `--force` archives the existing `agent_systems/eval_root/` tree under `agent_systems/eval_root_archive/`. It no longer deletes prior run artifacts outright.

Each runner then:
- copies the current question and memory JSONs into a temp workspace
- creates a temp runtime home/config area
- runs one process per question
- writes:
  - `output/answer.json`
  - `output/usage.json`
  - `output/trace.*`
  - `output/runtime_manifest.json`

## Claude Code

### Install expectation

Default binary:
- `claude` on `PATH`

Override if needed:

```bash
export AGSYS_CLAUDE_BIN=/path/to/claude
```

### Auth expectation

The runner accepts either:
- Claude auth env vars already exported in the shell
- local Claude auth files already present on the host

Current host-side auth locations used by the runner:
- `~/.claude/.credentials.json`
- `~/.claude.json`
- `~/.claude/settings.json` for env passthrough

Supported passthrough env vars:
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_AUTH_TOKEN`
- `CLAUDE_API_KEY`
- `CLAUDE_CODE_OAUTH_TOKEN`

### Important behavior

The runner copies local Claude auth into the isolated temp runtime home for the
duration of the run. This is a local filesystem copy only. It is not a separate
upload step by the harness.

If auth is missing, the run fails early instead of hanging in a login loop.

### Recommended preflight

```bash
which claude
claude --version
```

Then run one question first:

```bash
bash agent_systems/scripts/claude_code/run_claude_code_opus47-xhigh.sh <question_id>
```

### Common failure signature

If you see a Claude trace with repeated entries like:
- `error":"authentication_failed"`
- `Failed to authenticate. API Error: 401 ...`
- synthetic user messages saying `Stop hook feedback: You MUST call the StructuredOutput tool`

then the root problem is usually Claude authentication, not the benchmark
question itself.

Typical fixes:
- refresh Claude login on the host
- or prefer API-key env auth for stability

Example:

```bash
export ANTHROPIC_API_KEY=...
```

Important rerun note:
- if the failed run already wrote a fallback `output/answer.json` containing the
  auth error text, delete that per-question run directory or at least its
  `output/answer.json` before rerunning
- otherwise the harness may treat the bad fallback answer as a completed run

## Codex

### Install expectation

Default binary:
- `codex` on `PATH`

Override if needed:

```bash
export AGSYS_CODEX_BIN=/path/to/codex
```

The current harness expects a Codex CLI that supports:
- `codex exec --ephemeral`
- `--output-schema`
- `--output-last-message`

### Auth expectation

Current host-side auth location used by the runner:
- `~/.codex/auth.json`

### Important behavior

The runner copies local Codex auth into a fresh temp `CODEX_HOME` for the run.
This is a local filesystem copy only.

The runner also writes an isolated `config.toml` into that temp home so the run
does not depend on the user's normal Codex config.

### Recommended preflight

```bash
which codex
codex --version
codex exec --help
```

Then run one question first:

```bash
bash agent_systems/scripts/codex/run_codex_gpt55_m.sh <question_id>
```

## opencode

### Install expectation

Default binary:
- `~/.opencode/bin/opencode`

Override if needed:

```bash
export AGSYS_OPENCODE_BIN=/path/to/opencode
```

Default config source:
- `~/.config/opencode/opencode.json`

Override source config:

```bash
export AGSYS_OPENCODE_CONFIG_SOURCE=/path/to/opencode.json
```

Or override with a full config directory:

```bash
export AGSYS_OPENCODE_CONFIG_DIR=/path/to/config_dir
```

### Auth expectation

For providers authenticated through `opencode auth`, the runner copies:
- `~/.local/share/opencode/auth.json`

into the isolated temp runtime home.

This is a local filesystem copy only.

### Important behavior

The runner generates or copies an isolated opencode config into the temp home,
copies auth when needed, and by default requires `bwrap`.

Default sandbox policy:
- `AGSYS_OPENCODE_SANDBOX=bwrap`

Fallbacks:
- `AGSYS_OPENCODE_SANDBOX=auto`
- `AGSYS_OPENCODE_SANDBOX=off`

Use fallback modes only if you intentionally accept weaker isolation.

### Recommended preflight

```bash
${AGSYS_OPENCODE_BIN:-$HOME/.opencode/bin/opencode} --version
bash agent_systems/scripts/opencode/opencode_models.sh
```

The shipped opencode preset is provider-agnostic — point it at any OpenAI
`/v1/chat/completions` server via env vars. It renders an isolated config and
forwards the key into the sandbox; the static reference config lives at
`agent_systems/scripts/opencode/configs/openai-compatible.json`:

```bash
export OPENCODE_BASE_URL=https://your-host/v1   # default http://localhost:8000/v1
export OPENCODE_MODEL_ID=your-model             # default gpt-4o-mini
# key: api_keys/.openai_compatible_key, or:
export OPENAI_COMPATIBLE_API_KEY=...
```

Then run one question first:

```bash
bash agent_systems/scripts/opencode/run_opencode_openai_compatible.sh <question_id>
```

## pi

### Install expectation

Default binary:
- `pi` on `PATH` (install via `npm i -g @earendil-works/pi-coding-agent`)

Override if needed:

```bash
export AGSYS_PI_BIN=/path/to/pi
```

### Generic OpenAI-compatible endpoint

The shipped pi preset is provider-agnostic. Point it at any OpenAI
`/v1/chat/completions` server via environment variables:

```bash
export PI_OPENAI_BASE_URL=https://your-host/v1   # default http://localhost:8000/v1
export PI_OPENAI_MODEL=your-model                # default gpt-4o-mini
# key: api_keys/.openai_compatible_key, or:
export OPENAI_COMPATIBLE_API_KEY=...
```

The provider is registered by the generic extension at
`agent_systems/scripts/pi/extensions/openai-compatible/`. The runner forwards the
endpoint config + key into the (offline, bwrap) pi runtime, so no provider value
is hardcoded. The model is registered as a reasoning model so `--thinking` is
honored; for non-reasoning endpoints run with `AGSYS_PI_THINKING=off`.

### Recommended preflight

```bash
pi --version
bash agent_systems/scripts/pi/run_pi_openai_compatible.sh <question_id>
```

## Optional GLM judge

The GLM judge is optional.

If you do not want to use the GLM judge, disable it:

```bash
export AGSYS_EVAL_ENABLE_GLM_JUDGE=0
```

If you do want to use it, set the endpoint explicitly for your environment:

```bash
export AGSYS_EVAL_GLM_ENDPOINT=http://your-glm-host:8000/v1/chat/completions
```

The GPT judge can also be disabled independently:

```bash
export AGSYS_EVAL_ENABLE_GPT_JUDGE=0
```

## Safety note on auth copying

For all three runners, the harness may copy local auth artifacts into a temp
runtime directory so the isolated process can authenticate without reading the
user's normal home directory directly.

That copy is:
- local only
- per-run
- deleted with the temp runtime directory after the run

What still matters operationally:
- log in to the CLI on the host first
- keep the host auth files valid
- inspect `output/runtime_manifest.json` if you want to confirm which auth path
  or config source was used
