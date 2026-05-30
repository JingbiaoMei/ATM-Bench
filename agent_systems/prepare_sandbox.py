#!/usr/bin/env python3
"""
Prepare the evaluation sandbox for agent-based ATM-hard QA.

Creates eval_root/ with:
  - memory/          representation-specific metadata JSONs
  - qas/<id>/        question.json (sanitized: {id, question} only) + question.txt
  - prompts/         runtime copies of the authoritative prompt/schema files

Usage (from repo root):
    python agent_systems/prepare_sandbox.py [--force]

Memory representation:
    AGSYS_MEMORY_MODE=sgm          full SGM metadata baseline (default)
    AGSYS_MEMORY_MODE=raw          minimal ID-to-raw-media JSON plus hardlinked media
    AGSYS_MEMORY_MODE=descriptive  minimal {id, caption} image/video JSON

WARNING:
    --force archives the existing eval_root directory tree before rebuilding it.
    Existing run artifacts are moved under agent_systems/eval_root_archive/.
"""

import argparse
from datetime import datetime
import json
import os
import shutil
import sys
from pathlib import Path

from config import load_config, resolve_repo_path
from memory_variants import build_memory_variant, normalize_memory_mode


CFG = load_config()

QA_SOURCE = resolve_repo_path(CFG.qa_source)
EVAL_ROOT = resolve_repo_path(CFG.eval_root)
ARCHIVE_ROOT = EVAL_ROOT.parent / "eval_root_archive"
MEMORY_MODE = normalize_memory_mode(CFG.memory_mode)

MEMORY_SOURCES = {
    "image_metadata.json": resolve_repo_path(CFG.image_metadata),
    "video_metadata.json": resolve_repo_path(CFG.video_metadata),
    "emails.json": resolve_repo_path(CFG.emails),
}
RAW_IMAGE_DIR = resolve_repo_path(CFG.raw_image_dir)
RAW_VIDEO_DIR = resolve_repo_path(CFG.raw_video_dir)

PROMPTS_SOURCES = {
    "system_prompt.txt": resolve_repo_path(CFG.system_prompt_source),
    "qa_schema.json": resolve_repo_path(CFG.qa_schema_source),
}


def load_questions(path: Path) -> list[dict]:
    """Load ATM-hard questions and strip GT fields."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    questions = []
    for item in raw:
        questions.append({
            "id": item["id"],
            "question": item["question"],
        })
    return questions


def archive_existing_eval_root() -> Path:
    """Move the current eval_root aside under a timestamped archive directory."""
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = ARCHIVE_ROOT / f"{EVAL_ROOT.name}_{ts}"
    suffix = 1
    while dst.exists():
        dst = ARCHIVE_ROOT / f"{EVAL_ROOT.name}_{ts}_{suffix}"
        suffix += 1

    print(f"Archiving existing {EVAL_ROOT} -> {dst} ...")
    shutil.move(str(EVAL_ROOT), str(dst))
    return dst


def create_eval_root(questions: list[dict], force: bool = False) -> None:
    """Build the eval_root directory tree."""
    archived_to: Path | None = None
    if EVAL_ROOT.exists():
        if not force:
            print(f"ERROR: {EVAL_ROOT} already exists. Use --force to recreate.", file=sys.stderr)
            sys.exit(1)
        archived_to = archive_existing_eval_root()

    print(f"Creating eval_root at {EVAL_ROOT}")

    # ── memory/ ────────────────────────────────────────────────────────
    memory_dir = EVAL_ROOT / "memory"
    memory_dir.mkdir(parents=True)
    for name, src in MEMORY_SOURCES.items():
        if not src.exists():
            print(f"  WARNING: memory source missing: {src}", file=sys.stderr)
            sys.exit(1)

    manifest = build_memory_variant(
        mode=MEMORY_MODE,
        image_source=MEMORY_SOURCES["image_metadata.json"],
        video_source=MEMORY_SOURCES["video_metadata.json"],
        emails_source=MEMORY_SOURCES["emails.json"],
        out_dir=memory_dir,
        raw_image_dir=RAW_IMAGE_DIR,
        raw_video_dir=RAW_VIDEO_DIR,
    )
    print(f"  memory mode: {manifest['memory_mode']}")
    for name in ("image_metadata.json", "video_metadata.json", "emails.json", "memory_variant.json"):
        dst = memory_dir / name
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"  memory/{name} ({size_mb:.1f} MB)")
    if MEMORY_MODE == "raw":
        for media_name in ("raw_images", "raw_videos"):
            media_info = manifest["media"][media_name]
            missing_count = len(media_info["missing"])
            print(f"  memory/{media_name}/ hardlinks: {media_info['linked']} missing: {missing_count}")
            if missing_count:
                examples = ", ".join(media_info["missing"][:5])
                print(f"  WARNING: missing {media_name} examples: {examples}", file=sys.stderr)

    # ── qas/<id>/ ──────────────────────────────────────────────────────
    qas_dir = EVAL_ROOT / "qas"
    for q in questions:
        qid = q["id"]
        qdir = qas_dir / qid
        qdir.mkdir(parents=True)

        # question.json: sanitized {id, question}
        with open(qdir / "question.json", "w", encoding="utf-8") as f:
            json.dump(q, f, indent=2)

        # question.txt: plain text for easy `$(cat ...)`
        with open(qdir / "question.txt", "w", encoding="utf-8") as f:
            f.write(q["question"])

    print(f"  qas/ -> {len(questions)} questions")

    # ── prompts/ ───────────────────────────────────────────────────────
    prompts_dst = EVAL_ROOT / "prompts"
    prompts_dst.mkdir(parents=True)

    for fname, src in PROMPTS_SOURCES.items():
        if src.exists():
            shutil.copy2(src, prompts_dst / fname)
            print(f"  prompts/{fname}")
        else:
            print(f"  WARNING: {src} not found, skipping", file=sys.stderr)

    # ── question_ids.txt (convenience) ─────────────────────────────────
    with open(EVAL_ROOT / "question_ids.txt", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q["id"] + "\n")
    print(f"  question_ids.txt")

    # ── Verification ──────────────────────────────────────────────────
    # Make sure runtime files do not escape eval_root via symlink targets.
    for root, dirs, files in os.walk(EVAL_ROOT):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.is_symlink():
                target = fpath.resolve()
                try:
                    target.relative_to(EVAL_ROOT)
                except ValueError:
                    print(f"  LEAK DETECTED: {fpath} -> {target}", file=sys.stderr)
                    sys.exit(1)

    print("\nDone. eval_root is ready.")
    if archived_to is not None:
        print(f"  Previous eval_root archived to: {archived_to}")
    print(f"  Total dirs:  {sum(1 for _ in EVAL_ROOT.rglob('*') if _.is_dir())}")
    print(f"  Total files: {sum(1 for _ in EVAL_ROOT.rglob('*') if _.is_file())}")


def main():
    parser = argparse.ArgumentParser(description="Prepare agent evaluation sandbox")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Archive the existing eval_root tree under eval_root_archive/ and recreate it from scratch",
    )
    args = parser.parse_args()

    if not QA_SOURCE.exists():
        print(f"ERROR: QA source file not found: {QA_SOURCE}", file=sys.stderr)
        sys.exit(1)

    questions = load_questions(QA_SOURCE)
    print(f"Loaded {len(questions)} questions from {QA_SOURCE}")

    create_eval_root(questions, force=args.force)


if __name__ == "__main__":
    main()
