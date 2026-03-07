#!/usr/bin/env python3
"""
Copy reverse-geocoding cache entries (location_name) between processor cache directories.

The image/video processors cache reverse-geocoding results as JSON files named:
  <md5>_location_name.json

If you already have a directory containing these cache files (for example, from a
previous run or from a provided GPS cache bundle stored under
data/raw_memory/geocoding_cache/{image,video}), copying them into the target
cache directory lets the processors skip geocoding API calls.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_gps_cache_files(source_dir: str, target_dir: str) -> int:
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        print(f"Error: source directory does not exist: {source_path}")
        return 1
    if not source_path.is_dir():
        print(f"Error: source is not a directory: {source_path}")
        return 1

    target_path.mkdir(parents=True, exist_ok=True)

    location_files = list(source_path.glob("*_location_name.json"))
    if not location_files:
        print(f"No *_location_name.json files found in: {source_path}")
        return 0

    copied_count = 0
    skipped_count = 0
    for file_path in location_files:
        if not file_path.is_file():
            continue
        target_file = target_path / file_path.name
        if target_file.exists():
            skipped_count += 1
            continue
        try:
            shutil.copy2(file_path, target_file)
            copied_count += 1
        except Exception as exc:
            print(f"Error copying {file_path.name}: {exc}")

    print(
        f"GPS cache copy complete: copied={copied_count} skipped_existing={skipped_count} "
        f"from={source_path} to={target_path}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy reverse-geocoding cache files (*_location_name.json) from source to target."
        )
    )
    parser.add_argument("source", help="Source cache directory")
    parser.add_argument("target", help="Target cache directory")
    args = parser.parse_args()
    return copy_gps_cache_files(args.source, args.target)


if __name__ == "__main__":
    raise SystemExit(main())
