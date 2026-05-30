#!/usr/bin/env python3
"""
Collect per-question usage.json files into a single usage.jsonl + summary.

Usage (from repo root):
  python3 agent_systems/collect_usage.py --agent codex --run-tag "$AGSYS_RUN_TAG" --model-tag default
  python3 agent_systems/collect_usage.py --all --run-tag "$AGSYS_RUN_TAG" --model-tag default
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from config import load_config, resolve_repo_path


CFG = load_config()
EVAL_ROOT = resolve_repo_path(CFG.eval_root)
RUNS_ROOT = resolve_repo_path(CFG.runs_root)
RESULTS_DIR = resolve_repo_path(CFG.results_root)
DEFAULT_RUN_TAG = CFG.run_tag
AGENTS = ["claude_code", "codex", "openclaw", "opencode", "pi"]


def load_question_ids(eval_root: Path) -> list[str]:
    ids_path = eval_root / "question_ids.txt"
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing question_ids.txt: {ids_path}")
    return [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sum_opt(values: list[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(sum(present))


def collect_agent(agent: str, run_tag: str, model_tag: str) -> Dict[str, Any]:
    runs_dir = RUNS_ROOT / run_tag / agent / model_tag
    if not runs_dir.exists():
        print(f"  ERROR: runs dir not found: {runs_dir}", file=sys.stderr)
        return {"agent": agent, "total": 0, "collected": 0, "missing": 0, "malformed": 0}

    out_dir = RESULTS_DIR / run_tag / agent / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "usage.jsonl"
    out_summary = out_dir / "usage_summary.json"

    question_ids = load_question_ids(EVAL_ROOT)

    collected = 0
    missing = 0
    malformed = 0
    records: list[Dict[str, Any]] = []

    for qid in question_ids:
        usage_file = runs_dir / qid / "output" / "usage.json"
        if not usage_file.exists():
            missing += 1
            continue
        try:
            data = json.loads(usage_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                malformed += 1
                continue
            data = dict(data)
            data["id"] = qid
            data["agent"] = data.get("agent") or agent
            data["model_tag"] = data.get("model_tag") or model_tag
            data["run_tag"] = run_tag
            records.append(data)
            collected += 1
        except Exception as e:
            print(f"  WARN: {qid} parse error: {e}", file=sys.stderr)
            malformed += 1

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    in_tokens = _sum_opt([r.get("input_tokens") for r in records])  # type: ignore[arg-type]
    uncached_in_tokens = _sum_opt([r.get("input_tokens_uncached") for r in records])  # type: ignore[arg-type]
    cache_creation_tokens = _sum_opt([r.get("cache_creation_input_tokens") for r in records])  # type: ignore[arg-type]
    cache_read_tokens = _sum_opt([r.get("cache_read_input_tokens") for r in records])  # type: ignore[arg-type]
    out_tokens = _sum_opt([r.get("output_tokens") for r in records])  # type: ignore[arg-type]
    total_tokens = _sum_opt([r.get("total_tokens") for r in records])  # type: ignore[arg-type]
    cost_usd = _sum_opt([r.get("cost_usd") for r in records])  # type: ignore[arg-type]

    summary = {
        "agent": agent,
        "run_tag": run_tag,
        "model_tag": model_tag,
        "total_questions": len(question_ids),
        "collected": collected,
        "missing": missing,
        "malformed": malformed,
        "sum_input_tokens": int(in_tokens) if in_tokens is not None else None,
        "sum_input_tokens_uncached": int(uncached_in_tokens) if uncached_in_tokens is not None else None,
        "sum_cache_creation_input_tokens": int(cache_creation_tokens) if cache_creation_tokens is not None else None,
        "sum_cache_read_input_tokens": int(cache_read_tokens) if cache_read_tokens is not None else None,
        "sum_output_tokens": int(out_tokens) if out_tokens is not None else None,
        "sum_total_tokens": int(total_tokens) if total_tokens is not None else None,
        "sum_cost_usd": cost_usd,
    }

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary["output_jsonl"] = str(out_jsonl)
    summary["output_summary"] = str(out_summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect usage.json into JSONL + summary")
    parser.add_argument("--agent", help="Agent name (claude_code, codex, openclaw, opencode, pi)")
    parser.add_argument("--all", action="store_true", help="Collect for all agents")
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG, help="Run tag under runs/ (default: env AGSYS_RUN_TAG)")
    parser.add_argument("--model-tag", default="default", help="Model tag under runs/<run_tag>/<agent>/")
    args = parser.parse_args()

    if not args.agent and not args.all:
        parser.print_help()
        sys.exit(1)

    agents = AGENTS if args.all else [args.agent]

    for agent in agents:
        print(f"\n=== Collecting usage: {agent} ===")
        stats = collect_agent(agent, run_tag=args.run_tag, model_tag=args.model_tag)
        print(f"  Run tag:           {stats.get('run_tag', '')}")
        print(f"  Model tag:         {stats.get('model_tag', '')}")
        print(f"  Total questions:   {stats.get('total_questions', stats.get('total', 0))}")
        print(f"  Collected:         {stats['collected']}")
        print(f"  Missing:           {stats['missing']}")
        print(f"  Malformed:         {stats['malformed']}")
        if "sum_total_tokens" in stats:
            print(f"  Sum total tokens:  {stats['sum_total_tokens']}")
        if "sum_cost_usd" in stats:
            print(f"  Sum cost (USD):    {stats['sum_cost_usd']}")
        if stats.get("output_jsonl"):
            print(f"  Output:            {stats['output_jsonl']}")
            print(f"  Summary:           {stats['output_summary']}")


if __name__ == "__main__":
    main()
