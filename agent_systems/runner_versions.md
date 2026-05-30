# Agent Runner Versions

CLI versions recorded for the current `agent_systems` benchmark environment.

Captured on: 2026-03-12

## Version Table

| Runner | Binary | Version Command | Recorded Version |
|-------|--------|-----------------|------------------|
| OpenClaw | `openclaw` | `openclaw --version` | `OpenClaw 2026.3.11 (29dc654)` |
| Codex | `codex` | `codex --version` | `codex-cli 0.113.0` |
| Claude Code | `claude` | `claude --version` | `2.1.74 (Claude Code)` |
| OpenCode | `opencode` | `opencode --version` | `1.2.24` |
| Pi | `pi` | `pi --version` | `pi 0.75.4` |

## Notes

- `opencode --version` may print a startup warning before the final version line if
  `models.dev` is temporarily unreachable. The final version line is still the
  value to record.
- The benchmark-user OpenClaw setup is configured around an OpenAI-compatible
  coding model of your choice:
  - primary model: `<your-openai-compatible-model>`
  - alias: `<your-model-alias>`
  - config path written by the CLI wizard: `/home/openclaw-bench/.openclaw/openclaw.json`

## Commands Used

```bash
openclaw --version
codex --version
claude --version
opencode --version
pi --version
```
