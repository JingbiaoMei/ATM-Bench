#!/usr/bin/env python3
"""Shared utilities for retrieval and reranking."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from memqa.global_config import (
    DEFAULT_PATHS,
    IMAGE_PROCESSING_CONFIG,
    VIDEO_PROCESSING_CONFIG,
)

PROJECT_ROOT = Path(DEFAULT_PATHS["project_root"])
IMAGE_EXTENSIONS = tuple(
    sorted(
        {
            f".{ext.lstrip('.').lower()}"
            for ext in IMAGE_PROCESSING_CONFIG.get("supported_extensions", [])
        }
    )
)
VIDEO_EXTENSIONS = tuple(
    sorted(
        {
            f".{ext.lstrip('.').lower()}"
            for ext in VIDEO_PROCESSING_CONFIG.get("supported_extensions", [])
        }
    )
)


@dataclass(frozen=True)
class MediaTextConfig:
    include_id: bool = True
    include_type: bool = True
    include_timestamp: bool = True
    include_location: bool = True
    include_short_caption: bool = True
    include_caption: bool = True
    include_ocr_text: bool = True
    include_tags: bool = True


@dataclass(frozen=True)
class EmailTextConfig:
    include_id: bool = True
    include_timestamp: bool = True
    include_summary: bool = True
    include_detail: bool = True


@dataclass
class RetrievalItem:
    item_id: str
    modality: str
    text: str
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "modality": self.modality,
            "text": self.text,
            "image_path": str(self.image_path) if self.image_path else None,
            "video_path": str(self.video_path) if self.video_path else None,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalItem":
        return cls(
            item_id=data["item_id"],
            modality=data["modality"],
            text=data.get("text", ""),
            image_path=Path(data["image_path"]) if data.get("image_path") else None,
            video_path=Path(data["video_path"]) if data.get("video_path") else None,
            metadata=data.get("metadata") or {},
        )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def encode_image_to_base64(path: Path) -> str:
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def load_qa_list(qa_data: Any) -> List[Dict[str, Any]]:
    if isinstance(qa_data, list):
        return qa_data
    if (
        isinstance(qa_data, dict)
        and "qas" in qa_data
        and isinstance(qa_data["qas"], list)
    ):
        return qa_data["qas"]
    raise ValueError("Unsupported QA schema. Expected list or dict with 'qas'.")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_preserve(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def extract_evidence_ids(qa: Dict[str, Any]) -> List[str]:
    evidence_ids = qa.get("evidence_ids", [])
    if not isinstance(evidence_ids, list):
        return []
    return dedupe_preserve([str(item) for item in evidence_ids if item])


def classify_evidence_id(evidence_id: str) -> str:
    lowered = evidence_id.lower()
    if lowered.startswith("email"):
        return "email"
    if lowered.endswith(VIDEO_EXTENSIONS):
        return "video"
    if lowered.endswith(IMAGE_EXTENSIONS):
        return "image"
    return "unknown"


def resolve_batch_path(path_value: str) -> Path:
    raw_path = Path(os.fspath(path_value))
    if raw_path.is_absolute():
        return raw_path
    return PROJECT_ROOT / raw_path


def resolve_media_file(
    root: Path, evidence_id: str, extensions: Tuple[str, ...]
) -> Optional[Path]:
    candidate = root / evidence_id
    if candidate.exists():
        return candidate
    stem = Path(evidence_id).stem
    for ext in extensions:
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def resolve_media_path(
    batch_item: Dict[str, Any],
    path_key: str,
    root: Path,
    extensions: Tuple[str, ...],
) -> Optional[Path]:
    raw_path = batch_item.get(path_key)
    if not raw_path:
        return None
    batch_path = resolve_batch_path(str(raw_path))
    if batch_path.exists():
        return batch_path
    return resolve_media_file(root, Path(str(raw_path)).stem, extensions)


def build_batch_index(
    items: List[Dict[str, Any]], path_key: str
) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in items:
        raw_path = item.get(path_key)
        if not raw_path:
            continue
        base_id = Path(str(raw_path)).stem
        mapping[base_id] = item
    return mapping


def format_media_text(
    item: Dict[str, Any],
    evidence_id: str,
    evidence_type: str,
    config: MediaTextConfig,
) -> str:
    parts: List[str] = []
    if config.include_id:
        parts.append(f"ID: {evidence_id}")
    if config.include_type:
        parts.append(f"Type: {evidence_type}")
    if config.include_timestamp:
        timestamp = item.get("timestamp", "") or ""
        parts.append(f"Timestamp: {timestamp}")
    if config.include_location:
        location = item.get("location_name", "") or ""
        parts.append(f"Location: {location}")
    if config.include_short_caption:
        short_caption = item.get("short_caption", "") or ""
        parts.append(f"Short Caption: {short_caption}")
    if config.include_caption:
        caption = item.get("caption", "") or ""
        parts.append(f"Caption: {caption}")
    if config.include_ocr_text:
        ocr_text = item.get("ocr_text", "") or ""
        parts.append(f"OCR: {ocr_text}")
    if config.include_tags:
        tags = item.get("tags", []) or []
        tags_text = ", ".join(tags) if isinstance(tags, list) else str(tags)
        parts.append(f"Tags: {tags_text}")
    return "\n".join(parts)


def format_email_text(item: Dict[str, Any], config: EmailTextConfig) -> str:
    parts: List[str] = []
    if config.include_id:
        parts.append(f"ID: {item.get('id', '') or ''}")
    if config.include_timestamp:
        parts.append(f"Timestamp: {item.get('timestamp', '') or ''}")
    if config.include_summary:
        parts.append(f"Summary: {item.get('short_summary', '') or ''}")
    if config.include_detail:
        parts.append(f"Detail: {item.get('detail', '') or ''}")
    return "\n".join(parts)


def extract_email_subject(detail: str) -> str:
    if not detail:
        return ""
    match = re.search(r"^Subject:\s*(.+)$", detail, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def minimal_media_text_config() -> MediaTextConfig:
    return MediaTextConfig(
        include_id=True,
        include_type=False,
        include_timestamp=True,
        include_location=True,
        include_short_caption=False,
        include_caption=False,
        include_ocr_text=False,
        include_tags=False,
    )


def minimal_email_text_config() -> EmailTextConfig:
    return EmailTextConfig(
        include_id=True,
        include_timestamp=True,
        include_summary=False,
        include_detail=False,
    )


def build_retrieval_items(
    email_entries: List[Dict[str, Any]],
    image_entries: List[Dict[str, Any]],
    video_entries: List[Dict[str, Any]],
    media_text_config: MediaTextConfig,
    email_text_config: EmailTextConfig,
    image_root: Path,
    video_root: Path,
) -> List[RetrievalItem]:
    items: List[RetrievalItem] = []

    for entry in email_entries:
        item_id = str(entry.get("id", ""))
        if not item_id:
            continue
        text = format_email_text(entry, email_text_config)
        detail = entry.get("detail", "") or ""
        subject = entry.get("subject", "") or extract_email_subject(detail)
        items.append(
            RetrievalItem(
                item_id=item_id,
                modality="email",
                text=text,
                metadata={
                    "source": "email",
                    "timestamp": entry.get("timestamp", "") or "",
                    "subject": subject,
                    "sender": entry.get("sender", "") or "",
                    "location": entry.get("location", "") or "",
                },
            )
        )

    for entry in image_entries:
        raw_path = entry.get("image_path")
        if not raw_path:
            continue
        base_id = Path(str(raw_path)).stem
        text = format_media_text(entry, base_id, "image", media_text_config)
        image_path = resolve_media_path(
            entry, "image_path", image_root, IMAGE_EXTENSIONS
        )
        items.append(
            RetrievalItem(
                item_id=base_id,
                modality="image",
                text=text,
                image_path=image_path,
                metadata={
                    "source": "image",
                    "batch_path": raw_path,
                    "timestamp": entry.get("timestamp", "") or "",
                    "location": entry.get("location_name", "")
                    or entry.get("location", "")
                    or "",
                },
            )
        )

    for entry in video_entries:
        raw_path = entry.get("video_path")
        if not raw_path:
            continue
        base_id = Path(str(raw_path)).stem
        text = format_media_text(entry, base_id, "video", media_text_config)
        video_path = resolve_media_path(
            entry, "video_path", video_root, VIDEO_EXTENSIONS
        )
        items.append(
            RetrievalItem(
                item_id=base_id,
                modality="video",
                text=text,
                video_path=video_path,
                metadata={
                    "source": "video",
                    "batch_path": raw_path,
                    "timestamp": entry.get("timestamp", "") or "",
                    "location": entry.get("location_name", "")
                    or entry.get("location", "")
                    or "",
                },
            )
        )

    return items


def build_cache_key(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def save_index(
    cache_path: Path,
    embeddings: Any,
    items: List[RetrievalItem],
    config: Dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "embeddings": embeddings,
        "items": [item.to_dict() for item in items],
        "config": config,
    }
    with cache_path.open("wb") as handle:
        pickle.dump(data, handle)


def load_index(
    cache_path: Path,
) -> Optional[Tuple[Any, List[RetrievalItem], Dict[str, Any]]]:
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as handle:
        data = pickle.load(handle)
    items = [RetrievalItem.from_dict(item) for item in data.get("items", [])]
    return data.get("embeddings"), items, data.get("config", {})


def get_cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.pkl"


def extract_first_frame(
    video_path: Path, output_dir: Optional[Path] = None
) -> Optional[Path]:
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / f"{video_path.stem}_frame.jpg"
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        "0.1",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-q:v",
        "2",
        str(frame_path),
    ]
    try:
        subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    if frame_path.exists():
        return frame_path
    return None


def ensure_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return Path(value)


def count_intersection(gt_ids: Sequence[str], retrieved_ids: Sequence[str]) -> int:
    gt_set = set(gt_ids)
    return sum(1 for item_id in retrieved_ids if item_id in gt_set)
