#!/usr/bin/env python3
"""Build NIAH (Needle In A Haystack) evidence pools for ATMBench.

Given a hard-set QA file and an `retrieval_recall_details.json` produced by an
MMRAG run, this script constructs fixed evidence pools of size k that are
guaranteed to contain all ground-truth evidence items.

Example:
  python scripts/QA_Agent/NIAH/build_niah_pools.py \
    --qa-file data/atm-bench/atm-bench-hard.json \
    --retrieval-details output/QA_Agent/MMRAG/.../retrieval_recall_details.json \
    --pool-sizes 25 50 100 200
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    print(f"  Written: {path}")


def load_qa_list(qa_data: Any) -> List[Dict[str, Any]]:
    if isinstance(qa_data, list):
        return qa_data
    if isinstance(qa_data, dict) and "qas" in qa_data and isinstance(qa_data["qas"], list):
        return qa_data["qas"]
    raise ValueError("Unsupported QA schema. Expected list or dict with 'qas'.")


def build_pool(
    gt_ids: List[str],
    retrieval_ids: List[str],
    pool_size: int,
    rng: random.Random,
) -> List[str]:
    """Build a pool of exactly pool_size items containing all GT items."""

    gt_set: Set[str] = set(gt_ids)

    if len(gt_ids) > pool_size:
        print(
            f"WARNING: GT count ({len(gt_ids)}) > pool_size ({pool_size}). "
            f"Pool will contain all {len(gt_ids)} GT items + no distractors.",
            file=sys.stderr,
        )
        pool = list(gt_ids)
        rng.shuffle(pool)
        return pool

    candidates = retrieval_ids[:pool_size]
    candidates_set = set(candidates)
    missing_gt = [eid for eid in gt_ids if eid not in candidates_set]

    if missing_gt:
        mutable = list(candidates)
        slots_needed = len(missing_gt)
        removed = 0
        for i in range(len(mutable) - 1, -1, -1):
            if removed >= slots_needed:
                break
            if mutable[i] not in gt_set:
                mutable.pop(i)
                removed += 1
        mutable.extend(missing_gt)
        candidates = mutable

    candidates = candidates[:pool_size]
    rng.shuffle(candidates)
    return candidates


def validate_pool(qa_id: str, gt_ids: List[str], pool: List[str], pool_size: int) -> List[str]:
    issues: List[str] = []
    gt_set = set(gt_ids)
    pool_set = set(pool)

    missing = gt_set - pool_set
    if missing:
        issues.append(f"[{qa_id}] Missing GT items: {sorted(missing)}")

    if len(gt_ids) <= pool_size and len(pool) != pool_size:
        issues.append(f"[{qa_id}] Pool size {len(pool)} != expected {pool_size}")

    if len(pool) != len(pool_set):
        issues.append(f"[{qa_id}] Duplicates in pool: {len(pool)} items, {len(pool_set)} unique")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NIAH evidence pools")
    parser.add_argument(
        "--qa-file",
        required=True,
        help="Path to the hard-set QA JSON (e.g., data/atm-bench/atm-bench-hard.json)",
    )
    parser.add_argument(
        "--retrieval-details",
        default=None,
        help="Path to retrieval_recall_details.json from an MMRAG run",
    )
    parser.add_argument(
        "--pool-sizes",
        type=int,
        nargs="+",
        default=[25, 50, 100, 200],
        help="Evidence pool sizes to generate (default: 25 50 100 200)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <qa_dir>/niah)",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix (default: <qa_stem>-niah)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing pool files, do not generate",
    )

    args = parser.parse_args()

    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        print(f"ERROR: QA file not found: {qa_path}", file=sys.stderr)
        return 1

    retrieval_path: Optional[Path] = None
    if args.retrieval_details is not None:
        retrieval_path = Path(args.retrieval_details)
        if not retrieval_path.exists():
            print(f"ERROR: Retrieval details not found: {retrieval_path}", file=sys.stderr)
            return 1

    qa_data = load_qa_list(load_json(qa_path))

    output_dir = Path(args.output_dir) if args.output_dir else (qa_path.parent / "niah")
    output_prefix = args.output_prefix if args.output_prefix else f"{qa_path.stem}-niah"

    retrieval_index: Dict[str, Dict[str, Any]] = {}
    max_retrieval = 0
    if retrieval_path is not None:
        retrieval_raw = load_json(retrieval_path)
        if not isinstance(retrieval_raw, list):
            print("ERROR: retrieval details must be a JSON array", file=sys.stderr)
            return 1
        for entry in retrieval_raw:
            rid = str(entry.get("id", ""))
            if rid:
                retrieval_index[rid] = entry
        max_retrieval = max(
            (
                len(entry.get("retrieval_ids") or [])
                for entry in retrieval_index.values()
                if isinstance(entry.get("retrieval_ids"), list)
            ),
            default=0,
        )

    print(f"QA items:          {len(qa_data)}")
    if retrieval_path is None:
        print("Retrieval entries: (skipped; validate-only)")
    else:
        print(f"Retrieval entries: {len(retrieval_index)}")
    print(f"Pool sizes:        {args.pool_sizes}")
    print(f"Seed:              {args.seed}")
    print()

    all_valid = True

    for k in sorted(args.pool_sizes):
        output_path = output_dir / f"{output_prefix}{k}.json"
        print(f"=== Pool size k={k} ===")

        if args.validate_only:
            if not output_path.exists():
                print(f"File not found: {output_path}")
                all_valid = False
                continue
            pool_data = load_json(output_path)
            issues_total = 0
            for qa_entry in pool_data:
                qa_id = str(qa_entry.get("id", ""))
                gt_ids = qa_entry.get("evidence_ids", [])
                pool = qa_entry.get("niah_evidence_ids", [])
                issues = validate_pool(qa_id, gt_ids, pool, k)
                for issue in issues:
                    print(issue)
                    issues_total += 1
            if issues_total == 0:
                print(f"VALID: {len(pool_data)} entries, all pools OK")
            else:
                print(f"INVALID: {issues_total} issues found")
                all_valid = False
            continue

        if retrieval_path is None:
            print("ERROR: --retrieval-details is required unless --validate-only", file=sys.stderr)
            return 1

        if max_retrieval < k:
            print(
                f"WARNING: Retrieval max_k={max_retrieval} < pool_size={k}. "
                "Some pools may have fewer distractors than expected."
            )

        rng = random.Random(args.seed + k)
        output_entries: List[Dict[str, Any]] = []
        issues_total = 0

        for qa in qa_data:
            qa_id = str(qa.get("id", ""))
            gt_ids_raw = qa.get("evidence_ids", [])
            gt_ids = [str(item) for item in gt_ids_raw if item] if isinstance(gt_ids_raw, list) else []

            retrieval_entry = retrieval_index.get(qa_id)
            retrieval_ids_raw = retrieval_entry.get("retrieval_ids", []) if retrieval_entry else []
            retrieval_ids = (
                [str(item) for item in retrieval_ids_raw if item]
                if isinstance(retrieval_ids_raw, list)
                else []
            )

            if not retrieval_ids:
                print(f"WARNING: No retrieval for {qa_id}, using GT only", file=sys.stderr)
                pool = list(gt_ids)
                rng.shuffle(pool)
            else:
                pool = build_pool(gt_ids, retrieval_ids, k, rng)

            issues = validate_pool(qa_id, gt_ids, pool, k)
            for issue in issues:
                print(issue)
                issues_total += 1

            entry = dict(qa)
            entry["niah_evidence_ids"] = pool
            output_entries.append(entry)

        if issues_total > 0:
            print(f"ISSUES: {issues_total}", file=sys.stderr)
            all_valid = False
        else:
            print("Validation: PASS")

        write_json(output_path, output_entries)
        print()

    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
