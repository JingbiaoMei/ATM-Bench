#!/usr/bin/env python3
"""
Central configuration defaults for `agent_systems/`.

This repo generally keeps workflow defaults in Python (see `memqa/*/config.py`),
so this module is the single source of truth for:
  - dataset / memory input paths
  - eval_root layout
  - runner defaults (run tags, model tags, binaries)
  - ATM evaluation defaults (public default: GPT judge only; LAN GLM judge off)

Bash runners load these defaults via:
  eval "$(python3 agent_systems/config.py --print-bash)"

Environment variables (AGSYS_*) always override defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    value = os.environ.get(key, "")
    return value if value else default


def _repo_path_str(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    # Keep root-relative strings stable for scripts.
    return str(p)


def _shell_export(key: str, value: str) -> str:
    return f"export {key}={shlex.quote(value)}"


def resolve_repo_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


@dataclass(frozen=True)
class AgentSystemsConfig:
    # Dataset (GT is used ONLY by prepare_sandbox.py; never expose to agents directly).
    qa_source: str
    eval_root: str

    # Memory sources (copied into eval_root/memory by default).
    memory_mode: str
    image_metadata: str
    video_metadata: str
    emails: str
    raw_image_dir: str
    raw_video_dir: str

    # Prompt/schema source of truth in the repo, plus eval_root runtime copies.
    prompts_source_dir: str
    system_prompt_source: str
    qa_schema_source: str
    prompts_dir: str
    system_prompt: str
    qa_schema: str

    # Run organization.
    run_tag: str
    runs_root: str
    results_root: str

    # Evaluation defaults.
    eval_extra_flags: str
    eval_glm_enabled: str
    eval_glm_provider: str
    eval_glm_model: str
    eval_glm_endpoint: str
    eval_glm_thinking: str
    eval_glm_reasoning_effort: str
    eval_glm_max_workers: str
    eval_glm_request_delay: str
    eval_glm_max_retries: str
    eval_glm_flags: str
    eval_gpt_enabled: str
    eval_gpt_provider: str
    eval_gpt_model: str
    eval_gpt_endpoint: str
    eval_gpt_thinking: str
    eval_gpt_reasoning_effort: str
    eval_gpt_max_workers: str
    eval_gpt_request_delay: str
    eval_gpt_max_retries: str
    eval_gpt_flags: str

    # CLI binaries.
    claude_bin: str
    codex_bin: str
    openclaw_bin: str
    opencode_bin: str

    # Per-agent defaults.
    claude_budget_usd: str
    claude_model: str
    claude_model_tag: str
    claude_tools: str
    claude_allowed_tools: str
    claude_disallowed_tools: str
    claude_effort: str
    claude_sandbox: str
    claude_debug: str
    claude_timeout_s: str
    claude_max_assistant_events: str

    codex_model: str
    codex_model_tag: str
    codex_reasoning_effort: str

    openclaw_config_source: str
    openclaw_timeout_s: str
    openclaw_thinking: str
    openclaw_model_tag: str
    openclaw_verbose: str
    openclaw_agent: str

    opencode_model: str
    opencode_variant: str
    opencode_config_source: str
    opencode_config_dir: str
    opencode_sandbox: str

    pi_bin: str
    pi_model: str
    pi_model_tag: str
    pi_provider: str
    pi_thinking: str
    pi_tools: str
    pi_timeout_s: str
    pi_sandbox: str
    pi_offline: str
    pi_config_source_dir: str
    pi_extension_paths: str
    pi_api_keys_dir: str

    def as_env(self) -> Dict[str, str]:
        return {
            "AGSYS_QA_SOURCE": self.qa_source,
            "AGSYS_EVAL_ROOT": self.eval_root,
            "AGSYS_MEMORY_MODE": self.memory_mode,
            "AGSYS_IMAGE_METADATA": self.image_metadata,
            "AGSYS_VIDEO_METADATA": self.video_metadata,
            "AGSYS_EMAILS": self.emails,
            "AGSYS_RAW_IMAGE_DIR": self.raw_image_dir,
            "AGSYS_RAW_VIDEO_DIR": self.raw_video_dir,
            "AGSYS_PROMPTS_SOURCE_DIR": self.prompts_source_dir,
            "AGSYS_SYSTEM_PROMPT_SOURCE": self.system_prompt_source,
            "AGSYS_QA_SCHEMA_SOURCE": self.qa_schema_source,
            "AGSYS_PROMPTS_DIR": self.prompts_dir,
            "AGSYS_SYSTEM_PROMPT": self.system_prompt,
            "AGSYS_QA_SCHEMA": self.qa_schema,
            "AGSYS_RUN_TAG": self.run_tag,
            "AGSYS_RUNS_ROOT": self.runs_root,
            "AGSYS_RESULTS_ROOT": self.results_root,
            "AGSYS_EVAL_EXTRA_FLAGS": self.eval_extra_flags,
            "AGSYS_EVAL_ENABLE_GLM_JUDGE": self.eval_glm_enabled,
            "AGSYS_EVAL_GLM_PROVIDER": self.eval_glm_provider,
            "AGSYS_EVAL_GLM_MODEL": self.eval_glm_model,
            "AGSYS_EVAL_GLM_ENDPOINT": self.eval_glm_endpoint,
            "AGSYS_EVAL_GLM_THINKING": self.eval_glm_thinking,
            "AGSYS_EVAL_GLM_REASONING_EFFORT": self.eval_glm_reasoning_effort,
            "AGSYS_EVAL_GLM_MAX_WORKERS": self.eval_glm_max_workers,
            "AGSYS_EVAL_GLM_REQUEST_DELAY": self.eval_glm_request_delay,
            "AGSYS_EVAL_GLM_MAX_RETRIES": self.eval_glm_max_retries,
            "AGSYS_EVAL_GLM_FLAGS": self.eval_glm_flags,
            "AGSYS_EVAL_ENABLE_GPT_JUDGE": self.eval_gpt_enabled,
            "AGSYS_EVAL_GPT_PROVIDER": self.eval_gpt_provider,
            "AGSYS_EVAL_GPT_MODEL": self.eval_gpt_model,
            "AGSYS_EVAL_GPT_ENDPOINT": self.eval_gpt_endpoint,
            "AGSYS_EVAL_GPT_THINKING": self.eval_gpt_thinking,
            "AGSYS_EVAL_GPT_REASONING_EFFORT": self.eval_gpt_reasoning_effort,
            "AGSYS_EVAL_GPT_MAX_WORKERS": self.eval_gpt_max_workers,
            "AGSYS_EVAL_GPT_REQUEST_DELAY": self.eval_gpt_request_delay,
            "AGSYS_EVAL_GPT_MAX_RETRIES": self.eval_gpt_max_retries,
            "AGSYS_EVAL_GPT_FLAGS": self.eval_gpt_flags,
            "AGSYS_CLAUDE_BIN": self.claude_bin,
            "AGSYS_CODEX_BIN": self.codex_bin,
            "AGSYS_OPENCLAW_BIN": self.openclaw_bin,
            "AGSYS_OPENCODE_BIN": self.opencode_bin,
            "AGSYS_CLAUDE_BUDGET_USD": self.claude_budget_usd,
            "AGSYS_CLAUDE_MODEL": self.claude_model,
            "AGSYS_CLAUDE_MODEL_TAG": self.claude_model_tag,
            "AGSYS_CLAUDE_TOOLS": self.claude_tools,
            "AGSYS_CLAUDE_ALLOWED_TOOLS": self.claude_allowed_tools,
            "AGSYS_CLAUDE_DISALLOWED_TOOLS": self.claude_disallowed_tools,
            "AGSYS_CLAUDE_EFFORT": self.claude_effort,
            "AGSYS_CLAUDE_SANDBOX": self.claude_sandbox,
            "AGSYS_CLAUDE_DEBUG": self.claude_debug,
            "AGSYS_CLAUDE_TIMEOUT_S": self.claude_timeout_s,
            "AGSYS_CLAUDE_MAX_ASSISTANT_EVENTS": self.claude_max_assistant_events,
            "AGSYS_CODEX_MODEL": self.codex_model,
            "AGSYS_CODEX_MODEL_TAG": self.codex_model_tag,
            "AGSYS_CODEX_REASONING_EFFORT": self.codex_reasoning_effort,
            "AGSYS_OPENCLAW_CONFIG_SOURCE": self.openclaw_config_source,
            "AGSYS_OPENCLAW_TIMEOUT_S": self.openclaw_timeout_s,
            "AGSYS_OPENCLAW_THINKING": self.openclaw_thinking,
            "AGSYS_OPENCLAW_MODEL_TAG": self.openclaw_model_tag,
            "AGSYS_OPENCLAW_VERBOSE": self.openclaw_verbose,
            "AGSYS_OPENCLAW_AGENT": self.openclaw_agent,
            "AGSYS_OPENCODE_MODEL": self.opencode_model,
            "AGSYS_OPENCODE_VARIANT": self.opencode_variant,
            "AGSYS_OPENCODE_CONFIG_SOURCE": self.opencode_config_source,
            "AGSYS_OPENCODE_CONFIG_DIR": self.opencode_config_dir,
            "AGSYS_OPENCODE_SANDBOX": self.opencode_sandbox,
            "AGSYS_PI_BIN": self.pi_bin,
            "AGSYS_PI_MODEL": self.pi_model,
            "AGSYS_PI_MODEL_TAG": self.pi_model_tag,
            "AGSYS_PI_PROVIDER": self.pi_provider,
            "AGSYS_PI_THINKING": self.pi_thinking,
            "AGSYS_PI_TOOLS": self.pi_tools,
            "AGSYS_PI_TIMEOUT_S": self.pi_timeout_s,
            "AGSYS_PI_SANDBOX": self.pi_sandbox,
            "AGSYS_PI_OFFLINE": self.pi_offline,
            "AGSYS_PI_CONFIG_SOURCE_DIR": self.pi_config_source_dir,
            "AGSYS_PI_EXTENSION_PATHS": self.pi_extension_paths,
            "AGSYS_PI_API_KEYS_DIR": self.pi_api_keys_dir,
        }


def load_config() -> AgentSystemsConfig:
    home = str(Path.home())

    qa_source = _env("AGSYS_QA_SOURCE", "data/atm-bench/atm-bench-hard.json")
    memory_mode = _env("AGSYS_MEMORY_MODE", "sgm").strip().lower().replace("-", "_")
    memory_mode = {
        "baseline": "sgm",
        "full": "sgm",
        "description": "descriptive",
        "dm": "descriptive",
        "descriptive_memory": "descriptive",
        "raw_entries": "raw",
        "raw_media": "raw",
    }.get(memory_mode, memory_mode)
    default_eval_root = {
        "sgm": "agent_systems/eval_root_sgm",
        "raw": "agent_systems/eval_root_raw",
        "descriptive": "agent_systems/eval_root_dm",
    }.get(memory_mode, "agent_systems/eval_root_sgm")
    eval_root = _env("AGSYS_EVAL_ROOT", default_eval_root)
    image_metadata = _env(
        "AGSYS_IMAGE_METADATA",
        "output/image/qwen3vl2b/batch_results.json",
    )
    video_metadata = _env(
        "AGSYS_VIDEO_METADATA",
        "output/video/qwen3vl2b/batch_results.json",
    )
    emails = _env(
        "AGSYS_EMAILS",
        "data/raw_memory/email/emails.json",
    )
    raw_image_dir = _env(
        "AGSYS_RAW_IMAGE_DIR",
        "data/raw_memory/image",
    )
    raw_video_dir = _env(
        "AGSYS_RAW_VIDEO_DIR",
        "data/raw_memory/video",
    )

    prompts_source_dir = _env("AGSYS_PROMPTS_SOURCE_DIR", "agent_systems/prompts")
    default_system_prompt_name = {
        "raw": "system_prompt_raw.txt",
        "descriptive": "system_prompt_descriptive.txt",
    }.get(memory_mode, "system_prompt.txt")
    system_prompt_source = _env("AGSYS_SYSTEM_PROMPT_SOURCE", f"{prompts_source_dir}/{default_system_prompt_name}")
    qa_schema_source = _env("AGSYS_QA_SCHEMA_SOURCE", f"{prompts_source_dir}/qa_schema.json")

    prompts_dir = _env("AGSYS_PROMPTS_DIR", f"{eval_root}/prompts")
    system_prompt = _env("AGSYS_SYSTEM_PROMPT", f"{prompts_dir}/system_prompt.txt")
    qa_schema = _env("AGSYS_QA_SCHEMA", f"{prompts_dir}/qa_schema.json")

    run_tag = _env("AGSYS_RUN_TAG", "atm-bench-hard")
    runs_root = _env("AGSYS_RUNS_ROOT", f"{eval_root}/runs")
    results_root = _env("AGSYS_RESULTS_ROOT", "output/QA_Agent/AgentSystems")

    eval_extra_flags = _env("AGSYS_EVAL_EXTRA_FLAGS", "")
    eval_glm_enabled = _env("AGSYS_EVAL_ENABLE_GLM_JUDGE", "0")
    eval_glm_provider = _env("AGSYS_EVAL_GLM_PROVIDER", "vllm")
    eval_glm_model = _env("AGSYS_EVAL_GLM_MODEL", "GLM-4.7")
    eval_glm_endpoint = _env(
        "AGSYS_EVAL_GLM_ENDPOINT",
        "",
    )
    eval_glm_thinking = _env("AGSYS_EVAL_GLM_THINKING", "disabled")
    eval_glm_reasoning_effort = _env("AGSYS_EVAL_GLM_REASONING_EFFORT", "")
    eval_glm_max_workers = _env("AGSYS_EVAL_GLM_MAX_WORKERS", "2")
    eval_glm_request_delay = _env("AGSYS_EVAL_GLM_REQUEST_DELAY", "")
    eval_glm_max_retries = _env("AGSYS_EVAL_GLM_MAX_RETRIES", "")
    eval_glm_flags = _env("AGSYS_EVAL_GLM_FLAGS", "")
    eval_gpt_enabled = _env("AGSYS_EVAL_ENABLE_GPT_JUDGE", "1")
    eval_gpt_provider = _env("AGSYS_EVAL_GPT_PROVIDER", "openai")
    eval_gpt_model = _env("AGSYS_EVAL_GPT_MODEL", "gpt-5-mini")
    eval_gpt_endpoint = _env("AGSYS_EVAL_GPT_ENDPOINT", "")
    eval_gpt_thinking = _env("AGSYS_EVAL_GPT_THINKING", "")
    eval_gpt_reasoning_effort = _env("AGSYS_EVAL_GPT_REASONING_EFFORT", "minimal")
    eval_gpt_max_workers = _env("AGSYS_EVAL_GPT_MAX_WORKERS", "2")
    eval_gpt_request_delay = _env("AGSYS_EVAL_GPT_REQUEST_DELAY", "")
    eval_gpt_max_retries = _env("AGSYS_EVAL_GPT_MAX_RETRIES", "")
    eval_gpt_flags = _env("AGSYS_EVAL_GPT_FLAGS", "")

    claude_bin = _env("AGSYS_CLAUDE_BIN", "claude")
    codex_bin = _env("AGSYS_CODEX_BIN", "codex")
    openclaw_bin = _env("AGSYS_OPENCLAW_BIN", f"{home}/.npm-global/bin/openclaw")
    opencode_bin = _env("AGSYS_OPENCODE_BIN", f"{home}/.opencode/bin/opencode")

    claude_budget_usd = _env("AGSYS_CLAUDE_BUDGET_USD", "2.00")
    claude_model = _env("AGSYS_CLAUDE_MODEL", "")
    claude_model_tag = _env("AGSYS_CLAUDE_MODEL_TAG", "")
    claude_tools = _env(
        "AGSYS_CLAUDE_TOOLS",
        "Bash,Glob,Grep,Read,StructuredOutput,ToolSearch",
    )
    claude_allowed_tools = _env(
        "AGSYS_CLAUDE_ALLOWED_TOOLS",
        "Bash,Glob,Grep,Read,StructuredOutput,ToolSearch",
    )
    claude_disallowed_tools = _env(
        "AGSYS_CLAUDE_DISALLOWED_TOOLS",
        "WebFetch,WebSearch,Write,Edit,NotebookEdit",
    )
    claude_effort = _env("AGSYS_CLAUDE_EFFORT", "medium")
    claude_sandbox = _env("AGSYS_CLAUDE_SANDBOX", "bwrap")
    claude_debug = _env("AGSYS_CLAUDE_DEBUG", "1")
    claude_timeout_s = _env("AGSYS_CLAUDE_TIMEOUT_S", "600")
    claude_max_assistant_events = _env("AGSYS_CLAUDE_MAX_ASSISTANT_EVENTS", "50")

    codex_model = _env("AGSYS_CODEX_MODEL", "")
    codex_model_tag = _env("AGSYS_CODEX_MODEL_TAG", "")
    codex_reasoning_effort = _env("AGSYS_CODEX_REASONING_EFFORT", "high")

    openclaw_config_source = _env("AGSYS_OPENCLAW_CONFIG_SOURCE", f"{home}/.openclaw/openclaw.json")
    openclaw_timeout_s = _env("AGSYS_OPENCLAW_TIMEOUT_S", "600")
    openclaw_thinking = _env("AGSYS_OPENCLAW_THINKING", "off")
    openclaw_model_tag = _env("AGSYS_OPENCLAW_MODEL_TAG", "")
    openclaw_verbose = _env("AGSYS_OPENCLAW_VERBOSE", "on")
    openclaw_agent = _env("AGSYS_OPENCLAW_AGENT", "")

    opencode_model = _env("AGSYS_OPENCODE_MODEL", "")
    opencode_variant = _env("AGSYS_OPENCODE_VARIANT", "")
    opencode_config_source = _env("AGSYS_OPENCODE_CONFIG_SOURCE", f"{home}/.config/opencode/opencode.json")
    opencode_config_dir = _env("AGSYS_OPENCODE_CONFIG_DIR", "")
    opencode_sandbox = _env("AGSYS_OPENCODE_SANDBOX", "bwrap")

    pi_bin = _env("AGSYS_PI_BIN", f"{home}/.npm-global/bin/pi")
    pi_model = _env("AGSYS_PI_MODEL", "")
    pi_model_tag = _env("AGSYS_PI_MODEL_TAG", "")
    pi_provider = _env("AGSYS_PI_PROVIDER", "")
    pi_thinking = _env("AGSYS_PI_THINKING", "medium")
    pi_tools = _env("AGSYS_PI_TOOLS", "read,bash,grep,find,ls")
    pi_timeout_s = _env("AGSYS_PI_TIMEOUT_S", "900")
    pi_sandbox = _env("AGSYS_PI_SANDBOX", "bwrap")
    pi_offline = _env("AGSYS_PI_OFFLINE", "1")
    pi_config_source_dir = _env("AGSYS_PI_CONFIG_SOURCE_DIR", f"{home}/.pi/agent")
    pi_extension_paths = _env("AGSYS_PI_EXTENSION_PATHS", "")
    pi_api_keys_dir = _env("AGSYS_PI_API_KEYS_DIR", "api_keys")

    return AgentSystemsConfig(
        qa_source=_repo_path_str(qa_source),
        eval_root=_repo_path_str(eval_root),
        memory_mode=memory_mode,
        image_metadata=_repo_path_str(image_metadata),
        video_metadata=_repo_path_str(video_metadata),
        emails=_repo_path_str(emails),
        raw_image_dir=_repo_path_str(raw_image_dir),
        raw_video_dir=_repo_path_str(raw_video_dir),
        prompts_source_dir=_repo_path_str(prompts_source_dir),
        system_prompt_source=_repo_path_str(system_prompt_source),
        qa_schema_source=_repo_path_str(qa_schema_source),
        prompts_dir=_repo_path_str(prompts_dir),
        system_prompt=_repo_path_str(system_prompt),
        qa_schema=_repo_path_str(qa_schema),
        run_tag=run_tag,
        runs_root=_repo_path_str(runs_root),
        results_root=_repo_path_str(results_root),
        eval_extra_flags=eval_extra_flags,
        eval_glm_enabled=eval_glm_enabled,
        eval_glm_provider=eval_glm_provider,
        eval_glm_model=eval_glm_model,
        eval_glm_endpoint=eval_glm_endpoint,
        eval_glm_thinking=eval_glm_thinking,
        eval_glm_reasoning_effort=eval_glm_reasoning_effort,
        eval_glm_max_workers=eval_glm_max_workers,
        eval_glm_request_delay=eval_glm_request_delay,
        eval_glm_max_retries=eval_glm_max_retries,
        eval_glm_flags=eval_glm_flags,
        eval_gpt_enabled=eval_gpt_enabled,
        eval_gpt_provider=eval_gpt_provider,
        eval_gpt_model=eval_gpt_model,
        eval_gpt_endpoint=eval_gpt_endpoint,
        eval_gpt_thinking=eval_gpt_thinking,
        eval_gpt_reasoning_effort=eval_gpt_reasoning_effort,
        eval_gpt_max_workers=eval_gpt_max_workers,
        eval_gpt_request_delay=eval_gpt_request_delay,
        eval_gpt_max_retries=eval_gpt_max_retries,
        eval_gpt_flags=eval_gpt_flags,
        claude_bin=claude_bin,
        codex_bin=codex_bin,
        openclaw_bin=openclaw_bin,
        opencode_bin=opencode_bin,
        claude_budget_usd=claude_budget_usd,
        claude_model=claude_model,
        claude_model_tag=claude_model_tag,
        claude_tools=claude_tools,
        claude_allowed_tools=claude_allowed_tools,
        claude_disallowed_tools=claude_disallowed_tools,
        claude_effort=claude_effort,
        claude_sandbox=claude_sandbox,
        claude_debug=claude_debug,
        claude_timeout_s=claude_timeout_s,
        claude_max_assistant_events=claude_max_assistant_events,
        codex_model=codex_model,
        codex_model_tag=codex_model_tag,
        codex_reasoning_effort=codex_reasoning_effort,
        openclaw_config_source=_repo_path_str(openclaw_config_source),
        openclaw_timeout_s=openclaw_timeout_s,
        openclaw_thinking=openclaw_thinking,
        openclaw_model_tag=openclaw_model_tag,
        openclaw_verbose=openclaw_verbose,
        openclaw_agent=openclaw_agent,
        opencode_model=opencode_model,
        opencode_variant=opencode_variant,
        opencode_config_source=_repo_path_str(opencode_config_source),
        opencode_config_dir=opencode_config_dir,
        opencode_sandbox=opencode_sandbox,
        pi_bin=pi_bin,
        pi_model=pi_model,
        pi_model_tag=pi_model_tag,
        pi_provider=pi_provider,
        pi_thinking=pi_thinking,
        pi_tools=pi_tools,
        pi_timeout_s=pi_timeout_s,
        pi_sandbox=pi_sandbox,
        pi_offline=pi_offline,
        pi_config_source_dir=_repo_path_str(pi_config_source_dir),
        pi_extension_paths=pi_extension_paths,
        pi_api_keys_dir=_repo_path_str(pi_api_keys_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="agent_systems config helper")
    parser.add_argument(
        "--print-bash",
        action="store_true",
        help="Print `export KEY=VALUE` lines for bash scripts.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print resolved config as JSON (debugging).",
    )
    args = parser.parse_args()

    cfg = load_config()
    env = cfg.as_env()

    if args.print_bash:
        for key in sorted(env.keys()):
            print(_shell_export(key, env[key]))
        return

    if args.print_json:
        print(json.dumps(env, ensure_ascii=False, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
