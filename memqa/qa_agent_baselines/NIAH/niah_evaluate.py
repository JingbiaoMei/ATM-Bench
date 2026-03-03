#!/usr/bin/env python3
"""NIAH (Needle In A Haystack) evaluation wrapper for ATMBench.

NIAH is a generation-only evaluation: each question provides a fixed evidence
pool (e.g., ``niah_evidence_ids``) that is guaranteed to contain all ground-truth
items. This wrapper delegates to ``oracle_baseline.py`` by passing
``--use-niah-pools``.

Example:
  python memqa/qa_agent_baselines/NIAH/niah_evaluate.py \
    --qa-file data/atm-bench/niah/atm-bench-hard-niah50.json \
    --media-source batch_results \
    --provider openai --model gpt-5 \
    --max-workers 8 --timeout 120 \
    --output-file output/QA_Agent/NIAH/hard/gpt5/batch_results/niah50/niah_answers.jsonl
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NIAH evaluation — wrapper around oracle_baseline.py"
    )
    parser.add_argument(
        "--niah-qa-file",
        "--qa-file",
        required=True,
        help="Path to NIAH QA JSON (with niah_evidence_ids field)",
    )
    parser.add_argument(
        "--niah-field",
        default="niah_evidence_ids",
        help="Field name for NIAH evidence pools (default: niah_evidence_ids)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress wrapper prints",
    )

    # All remaining arguments are passed through to oracle_baseline.py
    args, passthrough = parser.parse_known_args()

    niah_path = Path(args.niah_qa_file)
    if not niah_path.exists():
        print(f"ERROR: NIAH QA file not found: {niah_path}", file=sys.stderr)
        return 1

    oracle_script = str(
        Path(__file__).resolve().parent.parent / "oracle" / "oracle_baseline.py"
    )
    cmd = [
        sys.executable,
        oracle_script,
        "--qa-file",
        str(niah_path),
        "--use-niah-pools",
        "--niah-field",
        args.niah_field,
    ] + passthrough

    if not args.quiet:
        print(f"NIAH QA file: {niah_path}")
        print(f"Invoking oracle_baseline.py with {len(passthrough)} passthrough args...")
        print()

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
