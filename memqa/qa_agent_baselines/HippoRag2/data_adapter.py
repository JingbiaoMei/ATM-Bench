"""
Data Adapter for HippoRAG 2

Converts PersonalMemoryQA multimodal evidence (images, videos, emails) to
text passages in HippoRAG 2's expected format for graph construction.

This adapter follows the MMRag text-only retrieval setup with essential
metadata (id, timestamp, location) plus short_caption from batch_results.

Reference: memqa/qa_agent_baselines/MMRag/README.md (Text-Only Retrieval section)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TextPassage:
    """A text passage for HippoRAG 2 indexing.
    
    Follows HippoRAG 2's corpus format:
    {"title": "...", "text": "...", "idx": 0}
    
    We extend with additional metadata for traceability.
    """
    title: str
    text: str
    idx: int
    
    # Extension fields for PersonalMemoryQA
    evidence_id: str = ""
    modality: str = ""  # "email", "image", "video"
    timestamp: str = ""
    location: str = ""
    media_path: str = ""
    
    def to_hipporag_dict(self) -> Dict[str, Any]:
        """Convert to HippoRAG corpus format."""
        return {
            "title": self.title,
            "text": self.text,
            "idx": self.idx,
        }
    
    def to_extended_dict(self) -> Dict[str, Any]:
        """Convert to extended format with all metadata."""
        return asdict(self)


@dataclass
class HippoRAGCorpus:
    """A corpus of text passages for HippoRAG 2."""
    passages: List[TextPassage] = field(default_factory=list)
    id_to_idx: Dict[str, int] = field(default_factory=dict)
    
    def add_passage(self, passage: TextPassage) -> None:
        """Add a passage and update the index mapping."""
        self.passages.append(passage)
        self.id_to_idx[passage.evidence_id] = passage.idx
    
    def get_passage_by_id(self, evidence_id: str) -> Optional[TextPassage]:
        """Get a passage by its evidence ID."""
        idx = self.id_to_idx.get(evidence_id)
        if idx is not None:
            return self.passages[idx]
        return None
    
    def to_hipporag_corpus(self) -> List[Dict[str, Any]]:
        """Convert to HippoRAG corpus format for saving."""
        return [p.to_hipporag_dict() for p in self.passages]
    
    def to_extended_corpus(self) -> List[Dict[str, Any]]:
        """Convert to extended format for debugging."""
        return [p.to_extended_dict() for p in self.passages]
    
    def save(self, path: Path, extended: bool = False) -> None:
        """Save corpus to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_extended_corpus() if extended else self.to_hipporag_corpus()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved corpus with {len(self.passages)} passages to {path}")
    
    @classmethod
    def load(cls, path: Path, extended: bool = False) -> "HippoRAGCorpus":
        """Load corpus from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        corpus = cls()
        for item in data:
            if extended:
                passage = TextPassage(**item)
            else:
                passage = TextPassage(
                    title=item["title"],
                    text=item["text"],
                    idx=item["idx"],
                )
            corpus.add_passage(passage)
        return corpus


# =============================================================================
# Text Conversion Functions
# =============================================================================


def format_email_passage(
    email: Dict[str, Any],
    include_detail: bool = True,
) -> str:
    """Convert email to text passage.
    
    Format: {id} | {timestamp} | {location}
            {short_summary}
            {detail}
    """
    location = email.get("location", "Unknown") or "Unknown"
    lines = [f"{email['id']} | {email.get('timestamp', 'Unknown')} | {location}"]
    
    if email.get("short_summary"):
        lines.append(email["short_summary"])
    
    if include_detail and email.get("detail"):
        lines.append(email["detail"])
    
    return "\n".join(lines)


def format_image_passage(
    image_data: Dict[str, Any],
    image_id: str,
    level: str = "short_caption_only",
) -> str:
    """Convert image batch_results to text passage.
    
    Supports 4-level incremental augmentation (MMRag aligned):
    - short_caption_only: id, timestamp, location, short_caption
    - short_caption_caption: + caption
    - short_caption_caption_tag: + tags
    - short_caption_caption_tag_ocr: + ocr_text
    
    Args:
        image_data: Batch results for this image
        image_id: Image filename/ID
        level: Augmentation level
    """
    timestamp = image_data.get("timestamp", "Unknown") or "Unknown"
    location = image_data.get("location") or image_data.get("location_name") or "Unknown"
    short_caption = image_data.get("short_caption", "")
    
    lines = [f"{image_id} | {timestamp} | {location}"]
    
    if short_caption:
        lines.append(short_caption)
    
    if level in ("short_caption_caption", "short_caption_caption_tag", "short_caption_caption_tag_ocr"):
        caption = image_data.get("caption", "")
        if caption:
            lines.append(f"Caption: {caption}")
    
    if level in ("short_caption_caption_tag", "short_caption_caption_tag_ocr"):
        tags = image_data.get("tags", [])
        if tags:
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
            lines.append(f"Tags: {tags_str}")
    
    if level == "short_caption_caption_tag_ocr":
        ocr_text = image_data.get("ocr_text", "")
        if ocr_text:
            lines.append(f"OCR: {ocr_text}")
    
    return "\n".join(lines)


def format_video_passage(
    video_data: Dict[str, Any],
    video_id: str,
    level: str = "short_caption_only",
) -> str:
    """Convert video batch_results to text passage.
    
    Same 4-level hierarchy as images (without OCR typically).
    """
    timestamp = video_data.get("timestamp", "Unknown") or "Unknown"
    location = video_data.get("location") or video_data.get("location_name") or "Unknown"
    short_caption = video_data.get("short_caption", "")
    
    lines = [f"{video_id} | {timestamp} | {location}"]
    
    if short_caption:
        lines.append(short_caption)
    
    if level in ("short_caption_caption", "short_caption_caption_tag", "short_caption_caption_tag_ocr"):
        caption = video_data.get("caption", "")
        if caption:
            lines.append(f"Caption: {caption}")
    
    if level in ("short_caption_caption_tag", "short_caption_caption_tag_ocr"):
        tags = video_data.get("tags", [])
        if tags:
            tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
            lines.append(f"Tags: {tags_str}")
    
    return "\n".join(lines)


# =============================================================================
# Corpus Builder
# =============================================================================


class CorpusBuilder:
    """Builds HippoRAG 2 corpus from PersonalMemoryQA data sources."""
    
    def __init__(
        self,
        email_file: Optional[str] = None,
        image_batch_results: Optional[str] = None,
        video_batch_results: Optional[str] = None,
        media_source: str = "batch_results",
        image_root: Optional[str] = None,
        video_root: Optional[str] = None,
        augmentation_level: str = "short_caption_only",
    ):
        """Initialize the corpus builder.
        
        Args:
            email_file: Path to merged_emails.json
            image_batch_results: Path to image batch results JSON
            video_batch_results: Path to video batch results JSON
            media_source: "batch_results" or "raw"
            image_root: Root directory for raw images
            video_root: Root directory for raw videos
            augmentation_level: Text augmentation level for images/videos
        """
        self.augmentation_level = augmentation_level
        self.media_source = media_source
        self.image_root = Path(image_root) if image_root else None
        self.video_root = Path(video_root) if video_root else None

        if self.media_source == "raw":
            if not self.image_root:
                logger.warning("Raw mode enabled but image_root is not set.")
            if not self.video_root:
                logger.warning("Raw mode enabled but video_root is not set.")
        
        # Load data sources
        self.emails: Dict[str, Dict] = {}
        self.image_data: Dict[str, Dict] = {}
        self.video_data: Dict[str, Dict] = {}
        
        if email_file:
            self._load_emails(email_file)
        if image_batch_results:
            self._load_image_batch_results(image_batch_results)
        if video_batch_results:
            self._load_video_batch_results(video_batch_results)
    
    def _load_emails(self, path: str) -> None:
        """Load emails from merged_emails.json."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for email in data:
            self.emails[email["id"]] = email
        
        logger.info(f"Loaded {len(self.emails)} emails from {path}")
    
    def _load_image_batch_results(self, path: str) -> None:
        """Load image batch results."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data:
            # Use image_path stem as ID
            image_path = item.get("image_path", "")
            raw_id = image_path if image_path else item.get("id", "")
            image_id = Path(str(raw_id)).stem if raw_id else ""
            if image_id:
                self.image_data[image_id] = item
        
        logger.info(f"Loaded {len(self.image_data)} image records from {path}")
    
    def _load_video_batch_results(self, path: str) -> None:
        """Load video batch results."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data:
            video_path = item.get("video_path", "")
            raw_id = video_path if video_path else item.get("id", "")
            video_id = Path(str(raw_id)).stem if raw_id else ""
            if video_id:
                self.video_data[video_id] = item
        
        logger.info(f"Loaded {len(self.video_data)} video records from {path}")
    
    def build_full_corpus(self) -> HippoRAGCorpus:
        """Build corpus from all loaded data sources.
        
        Returns:
            HippoRAGCorpus with all passages indexed
        """
        corpus = HippoRAGCorpus()
        idx = 0
        
        # Add emails
        for email_id, email in self.emails.items():
            text = format_email_passage(email)
            passage = TextPassage(
                title=f"Email_{email_id}",
                text=text,
                idx=idx,
                evidence_id=email_id,
                modality="email",
                timestamp=email.get("timestamp", ""),
                location="Unknown",
            )
            corpus.add_passage(passage)
            idx += 1
        
        # Add images
        for image_id, image in self.image_data.items():
            text = format_image_passage(
                image, image_id, level=self.augmentation_level
            )
            media_path = ""
            if self.media_source == "raw" and self.image_root:
                from memqa.retrieve.utils import resolve_media_file, IMAGE_EXTENSIONS
                resolved = resolve_media_file(self.image_root, image_id, IMAGE_EXTENSIONS)
                media_path = str(resolved) if resolved else ""
            passage = TextPassage(
                title=f"Image_{image_id}",
                text=text,
                idx=idx,
                evidence_id=image_id,
                modality="image",
                timestamp=image.get("timestamp", ""),
                location=image.get("location") or image.get("location_name") or "Unknown",
                media_path=media_path,
            )
            corpus.add_passage(passage)
            idx += 1
        
        # Add videos
        for video_id, video in self.video_data.items():
            text = format_video_passage(
                video, video_id, level=self.augmentation_level
            )
            media_path = ""
            if self.media_source == "raw" and self.video_root:
                from memqa.retrieve.utils import resolve_media_file, VIDEO_EXTENSIONS
                resolved = resolve_media_file(self.video_root, video_id, VIDEO_EXTENSIONS)
                media_path = str(resolved) if resolved else ""
            passage = TextPassage(
                title=f"Video_{video_id}",
                text=text,
                idx=idx,
                evidence_id=video_id,
                modality="video",
                timestamp=video.get("timestamp", ""),
                location=video.get("location") or video.get("location_name") or "Unknown",
                media_path=media_path,
            )
            corpus.add_passage(passage)
            idx += 1
        
        logger.info(
            f"Built corpus: {len(self.emails)} emails, "
            f"{len(self.image_data)} images, {len(self.video_data)} videos "
            f"= {len(corpus.passages)} total passages"
        )
        
        return corpus
    
    def get_cache_key(self) -> str:
        """Generate a cache key for this corpus configuration.
        
        The key is based on:
        - Data source file paths/mtimes
        - Augmentation level
        """
        key_parts = [
            f"media_source={self.media_source}",
            f"level={self.augmentation_level}",
            f"emails={len(self.emails)}",
            f"images={len(self.image_data)}",
            f"videos={len(self.video_data)}",
        ]
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]


# =============================================================================
# QA Format Converter
# =============================================================================


def convert_qa_to_hipporag_format(
    qa_items: List[Dict[str, Any]],
    corpus: HippoRAGCorpus,
) -> List[Dict[str, Any]]:
    """Convert PersonalMemoryQA format to HippoRAG 2 query format.
    
    PersonalMemoryQA format:
    {
        "id": "...",
        "question": "...",
        "answer": "...",
        "evidence_ids": ["email123", "IMG_456"]
    }
    
    HippoRAG 2 format:
    {
        "id": "...",
        "question": "...",
        "answer": ["..."],
        "paragraphs": [{"title": "...", "text": "...", "is_supporting": true, "idx": 0}]
    }
    """
    converted = []
    
    for qa in qa_items:
        # Get supporting passages from corpus
        paragraphs = []
        evidence_ids = qa.get("evidence_ids", [])
        
        for eid in evidence_ids:
            passage = corpus.get_passage_by_id(eid)
            if passage:
                paragraphs.append({
                    "title": passage.title,
                    "text": passage.text,
                    "is_supporting": True,
                    "idx": passage.idx,
                })
            else:
                logger.warning(f"Evidence ID {eid} not found in corpus for QA {qa['id']}")
        
        # Format answer as list (HippoRAG expectation)
        answer = qa.get("answer", "")
        if isinstance(answer, str):
            answer = [answer]
        
        converted.append({
            "id": qa["id"],
            "question": qa["question"],
            "answer": answer,
            "paragraphs": paragraphs,
        })
    
    return converted


def save_hipporag_qa(
    qa_items: List[Dict[str, Any]],
    path: Path,
) -> None:
    """Save QA items in HippoRAG 2 format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(qa_items, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(qa_items)} QA items to {path}")


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI for corpus conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PersonalMemoryQA to HippoRAG 2 format")
    parser.add_argument("--email-file", type=str, help="Path to merged_emails.json")
    parser.add_argument("--image-batch-results", type=str, help="Path to image batch results")
    parser.add_argument("--video-batch-results", type=str, help="Path to video batch results")
    parser.add_argument(
        "--media-source",
        choices=["batch_results", "raw"],
        default="batch_results",
        help="Source type for media items",
    )
    parser.add_argument("--image-root", type=str, help="Root directory for raw images")
    parser.add_argument("--video-root", type=str, help="Root directory for raw videos")
    parser.add_argument("--qa-file", type=str, help="Path to QA JSON file")
    parser.add_argument(
        "--augmentation-level",
        choices=["short_caption_only", "short_caption_caption", "short_caption_caption_tag", "short_caption_caption_tag_ocr"],
        default="short_caption_only",
        help="Text augmentation level for images/videos",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--extended", action="store_true", help="Save extended format with all metadata")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    output_dir = Path(args.output_dir)
    
    # Build corpus
    builder = CorpusBuilder(
        email_file=args.email_file,
        image_batch_results=args.image_batch_results,
        video_batch_results=args.video_batch_results,
        media_source=args.media_source,
        image_root=args.image_root,
        video_root=args.video_root,
        augmentation_level=args.augmentation_level,
    )
    
    corpus = builder.build_full_corpus()
    
    # Save corpus
    corpus_path = output_dir / "corpus.json"
    corpus.save(corpus_path, extended=args.extended)
    
    # Convert QA if provided
    if args.qa_file:
        with open(args.qa_file, "r", encoding="utf-8") as f:
            qa_items = json.load(f)
        
        converted_qa = convert_qa_to_hipporag_format(qa_items, corpus)
        qa_path = output_dir / "qa.json"
        save_hipporag_qa(converted_qa, qa_path)
    
    print(f"Cache key: {builder.get_cache_key()}")


if __name__ == "__main__":
    main()
