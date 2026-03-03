#!/usr/bin/env python3
"""Summarize NIAH ATM results across pool sizes.

This script scans a NIAH output directory for ATM summary files produced by:
`memqa/utils/evaluator/evaluate_qa.py` (files named: `atm_<judge>_summary.json`),
then prints a compact Markdown table.

Example:
  python scripts/QA_Agent/NIAH/summarize_niah_results.py \
    --output-root output/QA_Agent/NIAH/hard
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


NIAH_DIR_RE = re.compile(r"^niah(?P<k>\d+)$")
ATM_SUMMARY_RE = re.compile(r"^atm_(?P<judge>.+)_summary\.json$")


@dataclass(frozen=True)
class Record:
    model: str
    mode: str
    k: int
    judge_dir: str
    judge_model: str
    accuracy: float
    by_qtype: Dict[str, float]
    path: Path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_atm_summaries(root: Path) -> Iterable[Path]:
    yield from root.rglob("atm_*_summary.json")


def infer_context(
    output_root: Path, summary_path: Path
) -> Optional[Tuple[str, str, int, str]]:
    """Infer (model, mode, k, judge_dir) from:

    <output_root>/<model>/<mode>/niah<k>/<judge_dir>/atm_*_summary.json
    """

    try:
        rel = summary_path.relative_to(output_root)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 5:
        return None

    model = parts[0]
    mode = parts[1]
    niah_dir = parts[2]
    judge_dir = parts[3]

    match = NIAH_DIR_RE.match(niah_dir)
    if not match:
        return None

    k = int(match.group("k"))
    return model, mode, k, judge_dir


def coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract_by_qtype(summary: Dict[str, Any]) -> Dict[str, float]:
    by_qtype: Dict[str, float] = {}
    raw = summary.get("by_qtype")
    if not isinstance(raw, dict):
        return by_qtype

    for qtype, stats in raw.items():
        if not isinstance(stats, dict):
            continue
        by_qtype[str(qtype)] = coerce_float(stats.get("accuracy"))

    return by_qtype


def parse_record(output_root: Path, path: Path) -> Optional[Record]:
    context = infer_context(output_root, path)
    if context is None:
        return None

    model, mode, k, judge_dir = context

    name_match = ATM_SUMMARY_RE.match(path.name)
    if not name_match:
        return None

    judge_model = name_match.group("judge")

    payload = load_json(path)
    if not isinstance(payload, dict):
        return None

    accuracy = coerce_float(payload.get("accuracy"))
    by_qtype = extract_by_qtype(payload)

    return Record(
        model=model,
        mode=mode,
        k=k,
        judge_dir=judge_dir,
        judge_model=judge_model,
        accuracy=accuracy,
        by_qtype=by_qtype,
        path=path,
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}"


def print_markdown_table(records: List[Record]) -> None:
    headers = [
        "model",
        "mode",
        "k",
        "judge_dir",
        "judge_model",
        "ATM",
        "number",
        "list_recall",
        "open_end",
        "path",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for rec in records:
        number = rec.by_qtype.get("number")
        list_recall = rec.by_qtype.get("list_recall")
        open_end = rec.by_qtype.get("open_end")
        print(
            "| "
            + " | ".join(
                [
                    rec.model,
                    rec.mode,
                    str(rec.k),
                    rec.judge_dir,
                    rec.judge_model,
                    format_pct(rec.accuracy),
                    format_pct(number) if number is not None else "",
                    format_pct(list_recall) if list_recall is not None else "",
                    format_pct(open_end) if open_end is not None else "",
                    str(rec.path),
                ]
            )
            + " |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize NIAH ATM results")
    parser.add_argument(
        "--output-root",
        default="output/QA_Agent/NIAH/hard",
        help="Root NIAH output directory (default: output/QA_Agent/NIAH/hard)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.exists():
        raise SystemExit(f"Output root not found: {output_root}")

    records: List[Record] = []
    for path in iter_atm_summaries(output_root):
        rec = parse_record(output_root, path)
        if rec is not None:
            records.append(rec)

    records.sort(key=lambda r: (r.model, r.mode, r.judge_dir, r.judge_model, r.k))
    if not records:
        print(f"No ATM summary files found under: {output_root}")
        return 0

    print_markdown_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
