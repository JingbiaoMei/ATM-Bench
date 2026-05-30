#!/usr/bin/env python3
"""
Collect per-question answer.json files into a single answers.jsonl for evaluation.

Usage (from repo root):
    python3 agent_systems/collect_results.py --agent claude_code --model-tag default
    python3 agent_systems/collect_results.py --all --model-tag default

Then evaluate with the public evaluator (GPT judge, EM + ATM):
    python memqa/utils/evaluator/evaluate_qa.py \\
        --ground-truth ${AGSYS_QA_SOURCE} \\
        --predictions ${AGSYS_RESULTS_ROOT}/<run_tag>/<agent>/<model_tag>/answers.jsonl \\
        --output-dir ${AGSYS_RESULTS_ROOT}/<run_tag>/<agent>/<model_tag>/eval \\
        --metrics em atm \\
        --judge-provider openai \\
        --judge-model gpt-5-mini \\
        --judge-reasoning-effort minimal \\
        --max-workers 2
"""

import argparse
import json
import sys
from pathlib import Path

from config import load_config, resolve_repo_path

CFG = load_config()
EVAL_ROOT = resolve_repo_path(CFG.eval_root)
RUNS_ROOT = resolve_repo_path(CFG.runs_root)
RESULTS_DIR = resolve_repo_path(CFG.results_root)
DEFAULT_RUN_TAG = CFG.run_tag
GT_QA_FILE = resolve_repo_path(CFG.qa_source)
AGENTS = ["claude_code", "codex", "openclaw", "opencode", "pi"]
RUNTIME_DEFAULT_GLM_ENDPOINT = ""
RUNTIME_DEFAULT_GPT_MODEL = "gpt-5-mini"
LEGACY_GENERIC_GPT_MODEL = "gpt-5-mini"


def load_question_ids(eval_root: Path) -> list[str]:
    ids_path = eval_root / "question_ids.txt"
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing question_ids.txt: {ids_path}")
    return [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def print_eval_command(
    *,
    ground_truth: Path,
    predictions: Path,
    eval_dir: Path,
    judge_provider: str,
    judge_model: str,
    max_workers: str,
    endpoint: str,
    thinking: str,
    reasoning_effort: str,
) -> None:
    print("python memqa/utils/evaluator/evaluate_qa.py \\")
    print(f"  --ground-truth {ground_truth} \\")
    print(f"  --predictions {predictions} \\")
    print(f"  --output-dir {eval_dir} \\")
    print("  --metrics em atm \\")
    print(f"  --judge-provider {judge_provider} \\")
    print(f"  --judge-model {judge_model} \\")
    if endpoint:
        print(f"  --judge-endpoint {endpoint} \\")
    if thinking:
        print(f"  --judge-thinking {thinking} \\")
    if reasoning_effort:
        print(f"  --judge-reasoning-effort {reasoning_effort} \\")
    print(f"  --max-workers {max_workers}")


def runtime_eval_value(value: str, fallback: str, legacy_values: tuple[str, ...] = ()) -> str:
    if not value or value in legacy_values:
        return fallback
    return value


def collect_agent(agent: str, run_tag: str, model_tag: str) -> dict:
    """Collect answers for a single agent. Returns stats dict."""
    runs_dir = RUNS_ROOT / run_tag / agent / model_tag
    if not runs_dir.exists():
        print(f"  ERROR: runs dir not found: {runs_dir}", file=sys.stderr)
        return {"agent": agent, "total": 0, "collected": 0, "missing": 0, "malformed": 0}

    # Output directory
    out_dir = RESULTS_DIR / run_tag / agent / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "answers.jsonl"

    question_ids = load_question_ids(EVAL_ROOT)
    collected = 0
    missing = 0
    malformed = 0
    answers = []

    for qid in question_ids:
        answer_file = runs_dir / qid / "output" / "answer.json"
        q_path = EVAL_ROOT / "qas" / qid / "question.json"
        question_text = ""
        if q_path.exists():
            try:
                question_text = json.loads(q_path.read_text(encoding="utf-8")).get("question", "")
            except Exception:
                question_text = ""

        if not answer_file.exists():
            missing += 1
            continue

        try:
            with open(answer_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate required fields
            if "answer" not in data:
                print(f"  WARN: {qid} missing 'answer' field", file=sys.stderr)
                malformed += 1
                continue

            # Normalize: ensure id is present
            if "id" not in data:
                data["id"] = qid

            question = data.get("question", "") or question_text
            answer = data["answer"]

            answers.append({
                "id": data["id"],
                "question": question,
                "answer": answer,
            })
            collected += 1

        except (json.JSONDecodeError, Exception) as e:
            print(f"  WARN: {qid} parse error: {e}", file=sys.stderr)
            malformed += 1

    # Write JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    stats = {
        "agent": agent,
        "run_tag": run_tag,
        "model_tag": model_tag,
        "total": len(question_ids),
        "collected": collected,
        "missing": missing,
        "malformed": malformed,
        "output": str(out_file),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Collect agent answers into JSONL for evaluation")
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
        print(f"\n=== Collecting: {agent} ===")
        stats = collect_agent(agent, run_tag=args.run_tag, model_tag=args.model_tag)
        print(f"  Run tag:         {stats.get('run_tag', '')}")
        print(f"  Model tag:       {stats.get('model_tag', '')}")
        print(f"  Total questions: {stats['total']}")
        print(f"  Collected:       {stats['collected']}")
        print(f"  Missing:         {stats['missing']}")
        print(f"  Malformed:       {stats['malformed']}")
        if stats.get("collected", 0) > 0:
            print(f"  Output:          {stats['output']}")

    # Print evaluation command hint
    if len(agents) == 1 and agents[0]:
        agent = agents[0]
        pred_path = RESULTS_DIR / args.run_tag / agent / args.model_tag / "answers.jsonl"
        eval_dir = RESULTS_DIR / args.run_tag / agent / args.model_tag / "eval"
        print(f"\n── Evaluate with: ──")
        if CFG.eval_glm_enabled != "0":
            print_eval_command(
                ground_truth=GT_QA_FILE,
                predictions=pred_path,
                eval_dir=eval_dir,
                judge_provider=CFG.eval_glm_provider,
                judge_model=CFG.eval_glm_model,
                max_workers=CFG.eval_glm_max_workers,
                endpoint=runtime_eval_value(
                    CFG.eval_glm_endpoint, RUNTIME_DEFAULT_GLM_ENDPOINT
                ),
                thinking=CFG.eval_glm_thinking,
                reasoning_effort=CFG.eval_glm_reasoning_effort,
            )
            print("")
        if CFG.eval_gpt_enabled != "0":
            print_eval_command(
                ground_truth=GT_QA_FILE,
                predictions=pred_path,
                eval_dir=eval_dir,
                judge_provider=CFG.eval_gpt_provider,
                judge_model=runtime_eval_value(
                    CFG.eval_gpt_model,
                    RUNTIME_DEFAULT_GPT_MODEL,
                    legacy_values=(LEGACY_GENERIC_GPT_MODEL,),
                ),
                max_workers=CFG.eval_gpt_max_workers,
                endpoint=CFG.eval_gpt_endpoint,
                thinking=CFG.eval_gpt_thinking,
                reasoning_effort=CFG.eval_gpt_reasoning_effort,
            )


if __name__ == "__main__":
    main()
