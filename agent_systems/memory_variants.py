#!/usr/bin/env python3
"""Build memory input variants for agent_systems sandboxes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


VALID_MEMORY_MODES = ("sgm", "raw", "descriptive")


def normalize_memory_mode(mode: str) -> str:
    normalized = (mode or "sgm").strip().lower().replace("-", "_")
    aliases = {
        "baseline": "sgm",
        "full": "sgm",
        "description": "descriptive",
        "dm": "descriptive",
        "descriptive_memory": "descriptive",
        "raw_entries": "raw",
        "raw_media": "raw",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_MEMORY_MODES:
        raise ValueError(f"Unsupported memory mode: {mode!r}. Expected one of: {', '.join(VALID_MEMORY_MODES)}")
    return normalized


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def dump_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def item_id(item: dict[str, Any], path_key: str) -> str:
    path_value = item.get(path_key, "")
    if not path_value:
        raise ValueError(f"Missing {path_key} in metadata item")
    return Path(str(path_value)).stem


def media_name(item: dict[str, Any], path_key: str) -> str:
    path_value = item.get(path_key, "")
    if not path_value:
        raise ValueError(f"Missing {path_key} in metadata item")
    return Path(str(path_value)).name


def raw_entries(records: list[dict[str, Any]], path_key: str, media_dir_name: str) -> list[dict[str, str]]:
    return [
        {
            "id": item_id(item, path_key),
            path_key: f"memory/{media_dir_name}/{media_name(item, path_key)}",
        }
        for item in records
    ]


def descriptive_entries(records: list[dict[str, Any]], path_key: str) -> list[dict[str, str]]:
    return [
        {
            "id": item_id(item, path_key),
            "caption": str(item.get("caption", "")),
        }
        for item in records
    ]


def hardlink_media_files(
    records: list[dict[str, Any]],
    path_key: str,
    source_dir: Path,
    dst_dir: Path,
) -> dict[str, Any]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Raw media source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Raw media source is not a directory: {source_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    missing: list[str] = []
    for item in records:
        name = media_name(item, path_key)
        src = source_dir / name
        dst = dst_dir / name
        if not src.exists():
            missing.append(name)
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError as exc:
            raise OSError(
                f"Could not hardlink raw media {src} -> {dst}. "
                "Set AGSYS_RAW_IMAGE_DIR/AGSYS_RAW_VIDEO_DIR to media on the same filesystem, "
                "or extend memory_variants.py with an explicit copy mode."
            ) from exc
        linked += 1

    return {
        "source_dir": str(source_dir),
        "dst_dir": str(dst_dir),
        "linked": linked,
        "missing": missing,
    }


def build_memory_variant(
    *,
    mode: str,
    image_source: Path,
    video_source: Path,
    emails_source: Path,
    out_dir: Path,
    raw_image_dir: Path | None = None,
    raw_video_dir: Path | None = None,
) -> dict[str, Any]:
    mode = normalize_memory_mode(mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_records = load_json_list(image_source)
    video_records = load_json_list(video_source)

    media_manifest: dict[str, Any] = {}
    if mode == "sgm":
        shutil.copy2(image_source, out_dir / "image_metadata.json")
        shutil.copy2(video_source, out_dir / "video_metadata.json")
    elif mode == "raw":
        dump_json(out_dir / "image_metadata.json", raw_entries(image_records, "image_path", "raw_images"))
        dump_json(out_dir / "video_metadata.json", raw_entries(video_records, "video_path", "raw_videos"))
        if raw_image_dir is None or raw_video_dir is None:
            raise ValueError("raw mode requires raw_image_dir and raw_video_dir")
        media_manifest["raw_images"] = hardlink_media_files(
            image_records,
            "image_path",
            raw_image_dir,
            out_dir / "raw_images",
        )
        media_manifest["raw_videos"] = hardlink_media_files(
            video_records,
            "video_path",
            raw_video_dir,
            out_dir / "raw_videos",
        )
    elif mode == "descriptive":
        dump_json(out_dir / "image_metadata.json", descriptive_entries(image_records, "image_path"))
        dump_json(out_dir / "video_metadata.json", descriptive_entries(video_records, "video_path"))
    else:
        raise AssertionError(f"Unhandled memory mode: {mode}")

    shutil.copy2(emails_source, out_dir / "emails.json")

    manifest = {
        "memory_mode": mode,
        "image_metadata": "image_metadata.json",
        "video_metadata": "video_metadata.json",
        "emails": "emails.json",
        "image_count": len(image_records),
        "video_count": len(video_records),
        "emails_source": str(emails_source),
        "media": media_manifest,
    }
    dump_json(out_dir / "memory_variant.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an agent_systems memory variant")
    parser.add_argument("--mode", default="sgm", choices=VALID_MEMORY_MODES)
    parser.add_argument("--image-source", required=True, type=Path)
    parser.add_argument("--video-source", required=True, type=Path)
    parser.add_argument("--emails-source", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--raw-image-dir", type=Path)
    parser.add_argument("--raw-video-dir", type=Path)
    args = parser.parse_args()

    manifest = build_memory_variant(
        mode=args.mode,
        image_source=args.image_source,
        video_source=args.video_source,
        emails_source=args.emails_source,
        out_dir=args.out_dir,
        raw_image_dir=args.raw_image_dir,
        raw_video_dir=args.raw_video_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
