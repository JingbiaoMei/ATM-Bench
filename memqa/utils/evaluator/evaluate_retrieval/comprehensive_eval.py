#!/usr/bin/env python3
"""Compute comprehensive retrieval metrics from saved retrieval details."""

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


def compute_metrics(details: List[Dict[str, Any]], k_values: List[int]) -> Dict[str, Any]:
    totals = {f"recall@{k}": 0.0 for k in k_values}
    totals["recall@gt"] = 0.0
    totals["hit@1"] = 0.0

    count_with_gt = 0
    empty_gt = 0
    gt_counts: List[int] = []

    for item in details:
        gt_ids = item.get("gt_evidence_ids") or []
        retrieval_ids = item.get("retrieval_ids") or []

        if not gt_ids:
            empty_gt += 1
            continue

        gt_set = set(gt_ids)
        gt_count = len(gt_set)
        if gt_count == 0:
            empty_gt += 1
            continue

        count_with_gt += 1
        gt_counts.append(gt_count)

        for k in k_values:
            topk = set(retrieval_ids[:k])
            recall = len(gt_set & topk) / gt_count
            totals[f"recall@{k}"] += recall

        topk_gt = set(retrieval_ids[: min(gt_count, len(retrieval_ids))])
        totals["recall@gt"] += len(gt_set & topk_gt) / gt_count

        hit = 1.0 if retrieval_ids and retrieval_ids[0] in gt_set else 0.0
        totals["hit@1"] += hit

    metrics = {}
    denom = count_with_gt if count_with_gt else 1
    for key, value in totals.items():
        metrics[key] = value / denom

    gt_stats = {
        "count_with_gt": count_with_gt,
        "count_empty_gt": empty_gt,
    }
    if gt_counts:
        gt_stats.update(
            {
                "gt_count_avg": sum(gt_counts) / len(gt_counts),
                "gt_count_min": min(gt_counts),
                "gt_count_max": max(gt_counts),
            }
        )

    return {
        "count": len(details),
        "metrics": metrics,
        "gt_stats": gt_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive retrieval evaluation")
    parser.add_argument(
        "--details",
        required=True,
        help="Path to retrieval_recall_details.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output summary path (default: retrieval_recall_comprehensive_summary.json in same folder).",
    )
    parser.add_argument(
        "--k-values",
        default="1,5,10",
        help="Comma-separated list of k values (default: 1,5,10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details_path = Path(args.details)
    output_path = (
        Path(args.output)
        if args.output
        else details_path.with_name("retrieval_recall_comprehensive_summary.json")
    )
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]

    details = load_json(details_path)
    if not isinstance(details, list):
        raise ValueError("Expected details JSON to be a list.")

    summary = compute_metrics(details, k_values)
    summary["source_details"] = str(details_path)
    summary["k_values"] = k_values

    write_json(output_path, summary)
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
