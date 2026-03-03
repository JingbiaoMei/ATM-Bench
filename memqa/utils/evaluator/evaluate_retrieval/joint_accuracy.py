#!/usr/bin/env python3
"""Compute joint answer + retrieval accuracy using ATM and retrieval details."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_atm_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = load_json(path)
    if isinstance(data, list):
        return {str(item.get("id")): item for item in data if "id" in item}
    return {}


def compute_joint_accuracy(
    retrieval_details: List[Dict[str, Any]],
    atm_results: Dict[str, Dict[str, Any]],
    k_values: List[int],
) -> Dict[str, Any]:
    totals = {f"joint_strict@{k}": 0.0 for k in k_values}
    totals.update({f"joint_partial@{k}": 0.0 for k in k_values})

    count_with_gt = 0
    empty_gt = 0
    missing_answer = 0

    for item in retrieval_details:
        qa_id = str(item.get("id"))
        gt_ids = item.get("gt_evidence_ids") or []
        retrieval_ids = item.get("retrieval_ids") or []

        if not gt_ids:
            empty_gt += 1
            continue

        if qa_id not in atm_results:
            missing_answer += 1
            continue

        answer_correct = bool(atm_results[qa_id].get("accuracy"))
        gt_set = set(gt_ids)
        gt_count = len(gt_set)
        if gt_count == 0:
            empty_gt += 1
            continue

        count_with_gt += 1

        for k in k_values:
            topk = set(retrieval_ids[:k])
            recall = len(gt_set & topk) / gt_count
            strict = 1.0 if answer_correct and gt_set.issubset(topk) else 0.0
            partial = float(answer_correct) * recall
            totals[f"joint_strict@{k}"] += strict
            totals[f"joint_partial@{k}"] += partial

    denom = count_with_gt if count_with_gt else 1
    metrics = {key: value / denom for key, value in totals.items()}

    return {
        "count": len(retrieval_details),
        "metrics": metrics,
        "counts": {
            "count_with_gt": count_with_gt,
            "count_empty_gt": empty_gt,
            "missing_answer": missing_answer,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint answer+retrieval accuracy")
    parser.add_argument(
        "--retrieval-details",
        required=True,
        help="Path to retrieval_recall_details.json",
    )
    parser.add_argument(
        "--atm-details",
        required=True,
        help="Path to atm_<model>.json output",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output summary path (default: retrieval_recall_joint_accuracy_summary.json in same folder).",
    )
    parser.add_argument(
        "--k-values",
        default="5,10",
        help="Comma-separated list of k values (default: 5,10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_path = Path(args.retrieval_details)
    atm_path = Path(args.atm_details)
    output_path = (
        Path(args.output)
        if args.output
        else retrieval_path.with_name("retrieval_recall_joint_accuracy_summary.json")
    )
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]

    retrieval_details = load_json(retrieval_path)
    if not isinstance(retrieval_details, list):
        raise ValueError("Expected retrieval details JSON to be a list.")

    atm_results = load_atm_results(atm_path)
    summary = compute_joint_accuracy(retrieval_details, atm_results, k_values)
    summary["source_retrieval_details"] = str(retrieval_path)
    summary["source_atm_details"] = str(atm_path)
    summary["k_values"] = k_values

    write_json(output_path, summary)
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
