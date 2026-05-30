#!/usr/bin/env python3
"""
Generate isolated runtime config artifacts for agent_systems runners.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_claude_env(source: Path) -> None:
    env_map: dict[str, str] = {}
    if source.exists():
        data = load_json(source)
        raw_env = data.get("env")
        if isinstance(raw_env, dict):
            for key, value in raw_env.items():
                if not isinstance(key, str):
                    continue
                if value is None:
                    continue
                env_map[key] = str(value)

    # Allow explicit auth env vars from the operator environment to pass through
    # into the isolated Claude runtime when present.
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        value = os.environ.get(key, "")
        if value:
            env_map[key] = value

    env_map.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")

    for key in sorted(env_map):
        value = os.environ.get(key, "") or env_map[key]
        print(f"{key}={value}")


def write_claude_settings(out_path: Path) -> None:
    payload = {
        "env": {},
        "enabledPlugins": {},
        "alwaysThinkingEnabled": False,
    }
    write_json(out_path, payload)


def write_openclaw_config(source: Path, out_path: Path, workspace: str) -> None:
    if not source.exists():
        raise FileNotFoundError(f"OpenClaw config source not found: {source}")

    data = load_json(source)
    if not isinstance(data, dict):
        raise ValueError(f"OpenClaw config is not a JSON object: {source}")

    tools = data.setdefault("tools", {})
    if isinstance(tools, dict):
        tools["profile"] = "coding"
        tools["allow"] = ["read", "exec"]
        existing_deny = tools.get("deny")
        deny = list(existing_deny) if isinstance(existing_deny, list) else []
        for name in (
            "browser",
            "canvas",
            "nodes",
            "cron",
            "message",
            "gateway",
            "agents_list",
            "sessions_list",
            "sessions_history",
            "sessions_send",
            "sessions_spawn",
            "subagents",
            "image",
            "pdf",
            "memory_search",
            "memory_get",
            "write",
            "edit",
            "process",
        ):
            if name not in deny:
                deny.append(name)
        tools["deny"] = deny
        fs = tools.setdefault("fs", {})
        if isinstance(fs, dict):
            fs["workspaceOnly"] = True
        web = tools.setdefault("web", {})
        if isinstance(web, dict):
            search = web.setdefault("search", {})
            if isinstance(search, dict):
                search["enabled"] = False
            fetch = web.setdefault("fetch", {})
            if isinstance(fetch, dict):
                fetch["enabled"] = False
        agent_to_agent = tools.setdefault("agentToAgent", {})
        if isinstance(agent_to_agent, dict):
            agent_to_agent["enabled"] = False

    browser = data.setdefault("browser", {})
    if isinstance(browser, dict):
        browser["enabled"] = False

    plugins = data.setdefault("plugins", {})
    if isinstance(plugins, dict):
        plugins["enabled"] = False
        plugins["entries"] = {}

    skills = data.setdefault("skills", {})
    if isinstance(skills, dict):
        load_cfg = skills.setdefault("load", {})
        if isinstance(load_cfg, dict):
            load_cfg["extraDirs"] = []
            load_cfg["watch"] = False
        skills["allowBundled"] = []
        skills["entries"] = {}

    hooks = data.get("hooks")
    if isinstance(hooks, dict):
        hooks.clear()

    commands = data.get("commands")
    if isinstance(commands, dict):
        commands.clear()

    bindings = data.get("bindings")
    if isinstance(bindings, list):
        data["bindings"] = []

    agents = data.setdefault("agents", {})
    if isinstance(agents, dict):
        defaults = agents.setdefault("defaults", {})
        if isinstance(defaults, dict):
            defaults["workspace"] = workspace
            defaults["repoRoot"] = workspace
            defaults["skipBootstrap"] = True
        agent_list = agents.get("list")
        if isinstance(agent_list, list):
            for agent in agent_list:
                if isinstance(agent, dict):
                    agent["workspace"] = workspace
                    agent["repoRoot"] = workspace
                    agent["skipBootstrap"] = True

    write_json(out_path, data)


def write_codex_config(out_path: Path, reasoning_effort: str, model: str) -> None:
    lines = [
        'approval_policy = "never"',
        'sandbox_mode = "workspace-write"',
        'personality = "pragmatic"',
    ]
    if reasoning_effort:
        lines.append(f"model_reasoning_effort = {json.dumps(reasoning_effort)}")
    if model:
        lines.append(f"model = {json.dumps(model)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_opencode_config(source: Path, out_path: Path, model: str) -> None:
    payload: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "plugin": [],
        "mcp": {},
        "provider": {},
    }

    if source.exists():
        data = load_json(source)
        if not isinstance(data, dict):
            raise ValueError(f"OpenCode config is not a JSON object: {source}")
        source_schema = data.get("$schema")
        if isinstance(source_schema, str) and source_schema:
            payload["$schema"] = source_schema
        source_providers = data.get("provider")
        if isinstance(source_providers, dict):
            if model and "/" in model:
                provider_id = model.split("/", 1)[0]
                provider_cfg = source_providers.get(provider_id)
                if provider_cfg is not None:
                    payload["provider"] = {
                        provider_id: copy.deepcopy(provider_cfg),
                    }
            elif not model:
                payload["provider"] = copy.deepcopy(source_providers)

    write_json(out_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate isolated runtime artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    claude_env = subparsers.add_parser("print-claude-env", help="Print KEY=VALUE lines for Claude auth env")
    claude_env.add_argument("--source", default=str(Path.home() / ".claude" / "settings.json"))

    claude_settings = subparsers.add_parser("write-claude-settings", help="Write isolated Claude settings JSON")
    claude_settings.add_argument("--out", required=True)

    openclaw = subparsers.add_parser("write-openclaw-config", help="Write sanitized OpenClaw config JSON")
    openclaw.add_argument("--source", required=True)
    openclaw.add_argument("--workspace", required=True)
    openclaw.add_argument("--out", required=True)

    codex = subparsers.add_parser("write-codex-config", help="Write isolated Codex config.toml")
    codex.add_argument("--out", required=True)
    codex.add_argument("--reasoning-effort", default="")
    codex.add_argument("--model", default="")

    opencode = subparsers.add_parser("write-opencode-config", help="Write isolated OpenCode config JSON")
    opencode.add_argument("--out", required=True)
    opencode.add_argument("--source", default=str(Path.home() / ".config" / "opencode" / "opencode.json"))
    opencode.add_argument("--model", default="")

    args = parser.parse_args()

    if args.command == "print-claude-env":
        print_claude_env(Path(args.source))
        return

    if args.command == "write-claude-settings":
        write_claude_settings(Path(args.out))
        return

    if args.command == "write-openclaw-config":
        write_openclaw_config(Path(args.source), Path(args.out), args.workspace)
        return

    if args.command == "write-codex-config":
        write_codex_config(Path(args.out), args.reasoning_effort, args.model)
        return

    if args.command == "write-opencode-config":
        write_opencode_config(Path(args.source), Path(args.out), args.model)
        return

    raise AssertionError(f"Unhandled command: {shlex.quote(args.command)}")


if __name__ == "__main__":
    main()
