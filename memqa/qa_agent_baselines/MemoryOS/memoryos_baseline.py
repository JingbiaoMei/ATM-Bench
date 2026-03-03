#!/usr/bin/env python3
"""
MemoryOS Baseline for PersonalMemoryQA Benchmark

MemoryOS is a hierarchical memory management system inspired by OS memory management.
It organizes memory into three tiers:
- Short-Term Memory (STM): Recent dialogue pages
- Mid-Term Memory (MTM): Topic-based segments with heat-based eviction
- Long-Term Personal Memory (LPM): User/agent profiles and knowledge base

Paper: Memory OS of AI Agent (EMNLP 2025)
       https://arxiv.org/abs/2506.06326
Code:  https://github.com/BAI-LAB/MemoryOS

This implementation adapts MemoryOS for single-QA benchmarking by:
1. Formatting each memory item (email/image/video) as a dialogue pair
2. Using separate MemoryOS instance per question to prevent contamination
3. Supporting custom LLMs (Qwen3VL-8B) and embeddings (bge-m3)
"""

from __future__ import annotations

import argparse
import json
import inspect
import logging
import os
import re
import shutil
import sys
import time
import uuid
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

from tqdm import tqdm

from memqa.global_config import get_openai_api_key, get_vllm_api_key
from memqa.qa_agent_baselines.MemoryOS.config import (
    MEMORYOS_CONFIG,
    PROMPTS,
    QA_PROMPTS,
)
from memqa.retrieve import (
    EmailTextConfig,
    MediaTextConfig,
    RetrievalItem,
    build_retrieval_items,
    build_cache_key,
    extract_evidence_ids,
    load_json,
    load_qa_list,
    write_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)

RECALL_KS = [1, 5, 10, 25, 50, 100]
_THREAD_SUPPRESS_WARNING_EMITTED = False
_THREAD_SUPPRESS_WARNING_LOCK = threading.Lock()
_EMBEDDING_LOAD_LOCK = threading.Lock()
_EMBEDDING_KWARGS_WARNING_EMITTED = False


def ensure_memoryos_importable() -> bool:
    try:
        import memoryos  # noqa: F401

        return True
    except Exception:
        vendor_root = Path(__file__).resolve().parents[3] / "third_party" / "MemoryOS"
        if (vendor_root / "memoryos").exists():
            sys.path.insert(0, str(vendor_root))
            try:
                import memoryos  # noqa: F401

                return True
            except Exception:
                return False
        return False


@contextmanager
def suppress_memoryos_output(enabled: bool):
    global _THREAD_SUPPRESS_WARNING_EMITTED
    if not enabled:
        yield
        return
    # redirect_stdout/redirect_stderr are process-global and not thread-safe.
    # In threaded QA runs, redirecting in worker threads can leave sys.stdout
    # pointing to a closed file in other workers.
    if threading.current_thread() is not threading.main_thread():
        if not _THREAD_SUPPRESS_WARNING_EMITTED:
            with _THREAD_SUPPRESS_WARNING_LOCK:
                if not _THREAD_SUPPRESS_WARNING_EMITTED:
                    logger.warning(
                        "memoryos_quiet is ignored in worker threads during parallel QA "
                        "to avoid stdout/stderr race conditions."
                    )
                    _THREAD_SUPPRESS_WARNING_EMITTED = True
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MemoryOS QA baseline")
    parser.add_argument("--qa-file", required=True, help="Path to QA annotations JSON")

    # Media source settings
    parser.add_argument(
        "--media-source",
        choices=["batch_results"],
        default=MEMORYOS_CONFIG["media_source"],
    )

    # Data paths
    parser.add_argument(
        "--image-batch-results",
        default=MEMORYOS_CONFIG["image_batch_results"],
        help="Path to image batch_results.json",
    )
    parser.add_argument(
        "--video-batch-results",
        default=MEMORYOS_CONFIG["video_batch_results"],
        help="Path to video batch_results.json",
    )
    parser.add_argument(
        "--image-root",
        default=MEMORYOS_CONFIG["image_root"],
        help="Root directory for raw images",
    )
    parser.add_argument(
        "--video-root",
        default=MEMORYOS_CONFIG["video_root"],
        help="Root directory for raw videos",
    )
    parser.add_argument(
        "--email-file",
        default=MEMORYOS_CONFIG["email_file"],
        help="Path to merged_emails.json",
    )

    # LLM settings
    parser.add_argument(
        "--provider",
        choices=["openai", "vllm", "vllm_local"],
        default=MEMORYOS_CONFIG["provider"],
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Answerer model name (alias for --memoryos-answer-llm-model)",
    )
    parser.add_argument("--api-key", default=None, help="API key (overrides config)")
    parser.add_argument("--vllm-endpoint", default=None, help="VLLM endpoint URL")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout seconds")

    # Evidence and frame settings
    parser.add_argument(
        "--max-evidence-items", type=int, default=MEMORYOS_CONFIG["max_evidence_items"]
    )
    parser.add_argument(
        "--max-workers", type=int, default=MEMORYOS_CONFIG["max_workers"]
    )

    # Output settings
    parser.add_argument(
        "--output-dir-base",
        default=MEMORYOS_CONFIG["output_dir_base"],
        help="Output base",
    )
    parser.add_argument(
        "--method-name", default=MEMORYOS_CONFIG["method_name"], help="Method name"
    )

    # MemoryOS-specific settings
    parser.add_argument(
        "--memoryos-data-storage-path",
        default=MEMORYOS_CONFIG["memoryos_data_storage_path"],
        help="MemoryOS data storage directory",
    )
    parser.add_argument(
        "--memoryos-user-id",
        default=MEMORYOS_CONFIG["memoryos_user_id"],
        help="User ID for MemoryOS storage (single shared user by default)",
    )
    parser.add_argument(
        "--memoryos-checkpoint-dir",
        default=MEMORYOS_CONFIG["memoryos_checkpoint_dir"],
        help="Directory for MemoryOS checkpoints",
    )
    parser.add_argument(
        "--memoryos-reuse-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["memoryos_reuse_checkpoint"],
        help="Reuse existing MemoryOS checkpoint if available",
    )
    parser.add_argument(
        "--memoryos-save-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["memoryos_save_checkpoint"],
        help="Save MemoryOS checkpoint after indexing",
    )
    parser.add_argument(
        "--memoryos-force-rebuild",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["memoryos_force_rebuild"],
        help="Force rebuild MemoryOS checkpoint",
    )
    parser.add_argument(
        "--memoryos-llm-model",
        default=MEMORYOS_CONFIG["memoryos_llm_model"],
        help="Default LLM model for MemoryOS (indexing + answering)",
    )
    parser.add_argument(
        "--memoryos-index-llm-model",
        default=MEMORYOS_CONFIG["memoryos_index_llm_model"],
        help="LLM model used during memory indexing (summaries/meta/keywords)",
    )
    parser.add_argument(
        "--memoryos-answer-llm-model",
        default=MEMORYOS_CONFIG["memoryos_answer_llm_model"],
        help="LLM model used for answering (get_response)",
    )
    parser.add_argument(
        "--memoryos-answer-prompt-style",
        choices=["baseline", "memoryos"],
        default=MEMORYOS_CONFIG["memoryos_answer_prompt_style"],
        help=(
            "QA answer prompt style: baseline aligns with other baselines; "
            "memoryos keeps native MemoryOS prompt."
        ),
    )
    parser.add_argument(
        "--memoryos-embedding-model",
        default=MEMORYOS_CONFIG["memoryos_embedding_model_name"],
        help="Embedding model name",
    )
    parser.add_argument(
        "--memoryos-embedding-kwargs",
        default=None,
        help="JSON string of embedding kwargs (overrides config)",
    )
    parser.add_argument(
        "--memoryos-quiet",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["memoryos_quiet"],
        help="Suppress MemoryOS internal stdout/stderr",
    )
    parser.add_argument(
        "--memoryos-parallel-llm",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["memoryos_parallel_llm"],
        help="Run MemoryOS keyword extraction in parallel",
    )
    parser.add_argument(
        "--memoryos-parallel-llm-workers",
        type=int,
        default=MEMORYOS_CONFIG["memoryos_parallel_llm_workers"],
        help="Parallel workers for MemoryOS LLM calls",
    )
    parser.add_argument(
        "--memoryos-resume-indexing",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["memoryos_resume_indexing"],
        help="Resume checkpoint indexing if partial data exists",
    )

    # Memory capacity settings
    parser.add_argument(
        "--memoryos-short-term-capacity",
        type=int,
        default=MEMORYOS_CONFIG["memoryos_short_term_capacity"],
        help="Short-term memory capacity",
    )
    parser.add_argument(
        "--memoryos-mid-term-capacity",
        type=int,
        default=MEMORYOS_CONFIG["memoryos_mid_term_capacity"],
        help="Mid-term memory capacity",
    )
    parser.add_argument(
        "--memoryos-long-term-knowledge-capacity",
        type=int,
        default=MEMORYOS_CONFIG["memoryos_long_term_knowledge_capacity"],
        help="Long-term knowledge capacity",
    )

    # Retrieval settings
    parser.add_argument(
        "--memoryos-retrieval-queue-capacity",
        type=int,
        default=MEMORYOS_CONFIG["memoryos_retrieval_queue_capacity"],
        help="Retrieval queue capacity (top-k pages)",
    )
    parser.add_argument(
        "--memoryos-top-k-sessions",
        type=int,
        default=MEMORYOS_CONFIG["memoryos_top_k_sessions"],
        help="Top-k mid-term sessions to search",
    )
    parser.add_argument(
        "--memoryos-top-k-knowledge",
        type=int,
        default=MEMORYOS_CONFIG["memoryos_top_k_knowledge"],
        help="Top-k long-term knowledge entries to retrieve",
    )
    parser.add_argument(
        "--memoryos-heat-threshold",
        type=float,
        default=MEMORYOS_CONFIG["memoryos_mid_term_heat_threshold"],
        help="Heat threshold for MTM->LPM updates (set high to disable)",
    )
    parser.add_argument(
        "--memoryos-similarity-threshold",
        type=float,
        default=MEMORYOS_CONFIG["memoryos_mid_term_similarity_threshold"],
        help="Similarity threshold for segment grouping",
    )
    parser.add_argument(
        "--memoryos-segment-similarity-threshold",
        type=float,
        default=MEMORYOS_CONFIG["memoryos_segment_similarity_threshold"],
        help="Segment similarity threshold for MTM retrieval",
    )
    parser.add_argument(
        "--memoryos-page-similarity-threshold",
        type=float,
        default=MEMORYOS_CONFIG["memoryos_page_similarity_threshold"],
        help="Page similarity threshold for MTM retrieval",
    )
    parser.add_argument(
        "--memoryos-knowledge-threshold",
        type=float,
        default=MEMORYOS_CONFIG["memoryos_knowledge_threshold"],
        help="Knowledge similarity threshold for LPM retrieval",
    )

    # Isolation strategy
    parser.add_argument(
        "--memoryos-use-per-qa-instance",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["memoryos_use_per_qa_instance"],
        help="Create separate MemoryOS instance per question",
    )
    parser.add_argument(
        "--memoryos-eval-no-update",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["memoryos_eval_no_update"],
        help=(
            "Disable post-answer memory updates during QA evaluation "
            "(skip add_memory/profile refresh in get_response)"
        ),
    )
    parser.add_argument(
        "--memoryos-full-history-mode",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["memoryos_full_history_mode"],
        help=(
            "Benchmark mode for large corpora: expand MTM capacity/search depth to "
            "retain more historical items (non-paper-default behavior)."
        ),
    )
    parser.add_argument(
        "--memoryos-full-history-mid-term-capacity",
        type=int,
        default=MEMORYOS_CONFIG["memoryos_full_history_mid_term_capacity"],
        help=(
            "Target minimum MTM capacity applied when --memoryos-full-history-mode "
            "is enabled."
        ),
    )
    parser.add_argument(
        "--memoryos-full-history-top-k-sessions",
        type=int,
        default=MEMORYOS_CONFIG["memoryos_full_history_top_k_sessions"],
        help=(
            "Target minimum top-k MTM sessions searched when "
            "--memoryos-full-history-mode is enabled."
        ),
    )
    parser.add_argument(
        "--memoryos-full-history-heat-threshold",
        type=float,
        default=MEMORYOS_CONFIG["memoryos_full_history_heat_threshold"],
        help=(
            "Target minimum MTM heat threshold when --memoryos-full-history-mode "
            "is enabled (high value suppresses profile refresh churn)."
        ),
    )

    # Text augmentation flags
    parser.add_argument(
        "--include-id",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["include_id"],
    )
    parser.add_argument(
        "--include-type",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["include_type"],
    )
    parser.add_argument(
        "--include-timestamp",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["include_timestamp"],
    )
    parser.add_argument(
        "--include-location",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["include_location"],
    )
    parser.add_argument(
        "--include-short-caption",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["include_short_caption"],
    )
    parser.add_argument(
        "--include-caption",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["include_caption"],
    )
    parser.add_argument(
        "--include-ocr-text",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["include_ocr_text"],
    )
    parser.add_argument(
        "--include-tags",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["include_tags"],
    )
    parser.add_argument(
        "--include-email-summary",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["include_email_summary"],
    )
    parser.add_argument(
        "--include-email-detail",
        action=argparse.BooleanOptionalAction,
        default=MEMORYOS_CONFIG["include_email_detail"],
    )

    return parser.parse_args()


def resolve_embedding_kwargs(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if raw is None:
        return MEMORYOS_CONFIG["memoryos_embedding_model_kwargs"]
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON for --memoryos-embedding-kwargs") from exc
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise ValueError("--memoryos-embedding-kwargs must be a JSON object or null")
    return parsed


def resolve_llm_defaults(args: argparse.Namespace) -> None:
    if args.model:
        if (
            args.memoryos_answer_llm_model
            and args.memoryos_answer_llm_model != args.model
        ):
            logger.warning(
                "--model overrides --memoryos-answer-llm-model (%s -> %s)",
                args.memoryos_answer_llm_model,
                args.model,
            )
        args.memoryos_answer_llm_model = args.model

    if args.memoryos_index_llm_model is None:
        args.memoryos_index_llm_model = args.memoryos_llm_model
    if args.memoryos_answer_llm_model is None:
        args.memoryos_answer_llm_model = args.memoryos_llm_model

    if args.provider == "openai":
        defaults = MEMORYOS_CONFIG.get("openai", {})
        if args.api_key is None:
            args.api_key = defaults.get("api_key") or get_openai_api_key()
        if args.timeout is None:
            args.timeout = defaults.get("timeout")
        if args.vllm_endpoint:
            logger.warning("--vllm-endpoint is ignored when --provider=openai.")
        args.vllm_endpoint = None
    else:
        defaults = MEMORYOS_CONFIG.get("vllm_text", {})
        if args.api_key is None:
            args.api_key = defaults.get("api_key") or get_vllm_api_key()
        if args.vllm_endpoint is None:
            args.vllm_endpoint = defaults.get("endpoint")
        if args.timeout is None:
            args.timeout = defaults.get("timeout")


def apply_full_history_mode(args: argparse.Namespace) -> None:
    if not args.memoryos_full_history_mode:
        return

    original = {
        "memoryos_mid_term_capacity": args.memoryos_mid_term_capacity,
        "memoryos_top_k_sessions": args.memoryos_top_k_sessions,
        "memoryos_heat_threshold": args.memoryos_heat_threshold,
    }

    args.memoryos_mid_term_capacity = max(
        args.memoryos_mid_term_capacity,
        args.memoryos_full_history_mid_term_capacity,
    )
    args.memoryos_top_k_sessions = max(
        args.memoryos_top_k_sessions,
        args.memoryos_full_history_top_k_sessions,
    )
    args.memoryos_heat_threshold = max(
        args.memoryos_heat_threshold,
        args.memoryos_full_history_heat_threshold,
    )

    logger.info(
        (
            "Enabled --memoryos-full-history-mode "
            "(non-paper-default benchmark setting). "
            "Applied mid_term_capacity: %s -> %s, "
            "top_k_sessions: %s -> %s, heat_threshold: %s -> %s"
        ),
        original["memoryos_mid_term_capacity"],
        args.memoryos_mid_term_capacity,
        original["memoryos_top_k_sessions"],
        args.memoryos_top_k_sessions,
        original["memoryos_heat_threshold"],
        args.memoryos_heat_threshold,
    )


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        pass
    try:
        from dateutil import parser as date_parser
    except Exception:
        return None
    try:
        parsed = date_parser.parse(text)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except (ValueError, TypeError):
        return None


def normalize_metadata_value(
    metadata: Dict[str, Any], key: str, default: str = "Unknown"
) -> str:
    value = metadata.get(key)
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return str(value)


def sort_retrieval_items(
    items: List[RetrievalItem],
) -> Tuple[List[RetrievalItem], int]:
    indexed: List[Tuple[bool, datetime, int, RetrievalItem]] = []
    missing_count = 0
    for idx, item in enumerate(items):
        timestamp = None
        if item.metadata:
            timestamp = item.metadata.get("timestamp")
        parsed = parse_timestamp(timestamp)
        if parsed is None:
            missing_count += 1
            parsed = datetime.max
        indexed.append((parsed == datetime.max, parsed, idx, item))
    indexed.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    ordered = [entry[3] for entry in indexed]
    return ordered, missing_count


def log_missing_metadata(items: List[RetrievalItem]) -> None:
    counters: Dict[str, Dict[str, int]] = {
        "email": {"timestamp": 0, "location": 0, "sender": 0, "subject": 0},
        "image": {"timestamp": 0, "location": 0},
        "video": {"timestamp": 0, "location": 0},
    }
    for item in items:
        modality = item.modality
        metadata = item.metadata or {}
        if modality not in counters:
            continue
        if not str(metadata.get("timestamp", "") or "").strip():
            counters[modality]["timestamp"] += 1
        if not str(metadata.get("location", "") or "").strip():
            counters[modality]["location"] += 1
        if modality == "email":
            if not str(metadata.get("sender", "") or "").strip():
                counters[modality]["sender"] += 1
            if not str(metadata.get("subject", "") or "").strip():
                counters[modality]["subject"] += 1

    for modality, fields in counters.items():
        for field, count in fields.items():
            if count:
                logger.warning(
                    "Missing %s metadata for %s %s items.",
                    field,
                    count,
                    modality,
                )


def extract_retrieval_ids(
    memoryos_instance: Any, max_items: Optional[int]
) -> Tuple[List[str], List[float]]:
    retrieved_pages = memoryos_instance._last_retrieval.get("retrieved_pages", [])
    retrieval_scores = memoryos_instance._last_retrieval.get("retrieval_scores", [])
    retrieval_ids = [
        extract_item_id_from_text(page.get("user_input", ""))
        for page in retrieved_pages
    ]
    retrieval_ids = [item_id for item_id in retrieval_ids if item_id]
    if max_items is not None:
        retrieval_ids = retrieval_ids[:max_items]
        retrieval_scores = retrieval_scores[:max_items]
    return retrieval_ids, retrieval_scores


def build_recall_detail(
    qa: Dict[str, Any], retrieval_ids: List[str], retrieval_scores: List[float]
) -> Dict[str, Any]:
    gt_ids = extract_evidence_ids(qa)
    recall = compute_recall(gt_ids, retrieval_ids)
    return {
        "id": qa.get("id") or qa.get("qa_id"),
        "question": qa.get("question"),
        "gt_evidence_ids": gt_ids,
        "retrieval_ids": retrieval_ids,
        "retrieval_scores": retrieval_scores,
        "retrieval_recall": recall,
    }


def set_memoryos_answer_llm(memoryos_instance: Any, llm_model: str) -> None:
    if not llm_model:
        return
    memoryos_instance.llm_model = llm_model
    updater = getattr(memoryos_instance, "updater", None)
    if updater is not None and hasattr(updater, "llm_model"):
        updater.llm_model = llm_model

def format_item_as_dialogue(item: RetrievalItem) -> Tuple[str, str]:
    """
    Format a retrieval item as a dialogue pair for MemoryOS.

    Returns:
        (user_input, agent_response) tuple
    """
    if not item.text:
        return ("", "")

    # Extract metadata
    item_id = item.item_id
    modality = item.modality
    metadata = item.metadata or {}

    # Build user input based on modality
    if modality == "email":
        sender = normalize_metadata_value(metadata, "sender")
        timestamp = normalize_metadata_value(metadata, "timestamp")
        location = normalize_metadata_value(metadata, "location")
        subject = normalize_metadata_value(metadata, "subject", default="No subject")
        user_input = PROMPTS["MEMORY_EMAIL_USER"].format(
            item_id=item_id,
            sender=sender,
            timestamp=timestamp,
            location=location,
            subject=subject,
            content=item.text,
        )
        agent_response = PROMPTS["MEMORY_EMAIL_AGENT"]
    elif modality == "image":
        location = normalize_metadata_value(metadata, "location")
        timestamp = normalize_metadata_value(metadata, "timestamp")
        user_input = PROMPTS["MEMORY_IMAGE_USER"].format(
            item_id=item_id,
            location=location,
            timestamp=timestamp,
            caption=item.text,
        )
        agent_response = PROMPTS["MEMORY_IMAGE_AGENT"]
    elif modality == "video":
        location = normalize_metadata_value(metadata, "location")
        timestamp = normalize_metadata_value(metadata, "timestamp")
        user_input = PROMPTS["MEMORY_VIDEO_USER"].format(
            item_id=item_id,
            location=location,
            timestamp=timestamp,
            description=item.text,
        )
        agent_response = PROMPTS["MEMORY_VIDEO_AGENT"]
    else:
        # Generic fallback
        user_input = PROMPTS["MEMORY_GENERIC_USER"].format(
            item_id=item_id,
            content=item.text,
        )
        agent_response = PROMPTS["MEMORY_GENERIC_AGENT"]

    return user_input, agent_response


def load_index_progress(progress_path: Optional[Path]) -> Set[str]:
    if not progress_path or not progress_path.exists():
        return set()
    completed: Set[str] = set()
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "success" and record.get("item_id"):
                completed.add(str(record["item_id"]))
    return completed


def append_index_progress(
    progress_path: Optional[Path],
    item_id: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    if not progress_path:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"item_id": item_id, "status": status}
    if error:
        payload["error"] = error
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def rewrite_index_progress(progress_path: Path, item_ids: Set[str]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for item_id in sorted(item_ids):
            handle.write(
                json.dumps(
                    {"item_id": item_id, "status": "success"},
                    ensure_ascii=True,
                )
                + "\n"
            )
    tmp_path.replace(progress_path)


def load_indexed_item_ids_from_storage(storage_root: Path, user_id: str) -> Set[str]:
    user_dir = storage_root / "users" / user_id
    short_term_path = user_dir / "short_term.json"
    mid_term_path = user_dir / "mid_term.json"
    indexed_ids: Set[str] = set()

    if short_term_path.exists():
        try:
            short_data = load_json(short_term_path)
        except Exception:
            short_data = None

        short_entries: List[Any] = []
        if isinstance(short_data, list):
            short_entries = short_data
        elif isinstance(short_data, dict):
            for key in ("memory", "pages"):
                value = short_data.get(key)
                if isinstance(value, list):
                    short_entries = value
                    break
        for entry in short_entries:
            if not isinstance(entry, dict):
                continue
            item_id = extract_item_id_from_text(str(entry.get("user_input", "") or ""))
            if item_id:
                indexed_ids.add(item_id)

    if mid_term_path.exists():
        try:
            mid_data = load_json(mid_term_path)
        except Exception:
            mid_data = None

        sessions: List[Dict[str, Any]] = []
        if isinstance(mid_data, dict):
            raw_sessions = mid_data.get("sessions")
            if isinstance(raw_sessions, dict):
                sessions = [value for value in raw_sessions.values() if isinstance(value, dict)]
            elif isinstance(raw_sessions, list):
                sessions = [value for value in raw_sessions if isinstance(value, dict)]
        elif isinstance(mid_data, list):
            sessions = [value for value in mid_data if isinstance(value, dict)]

        for session in sessions:
            details = session.get("details")
            if not isinstance(details, list):
                details = session.get("dialogue_pages")
            if not isinstance(details, list):
                continue
            for page in details:
                if not isinstance(page, dict):
                    continue
                item_id = extract_item_id_from_text(
                    str(page.get("user_input", "") or "")
                )
                if item_id:
                    indexed_ids.add(item_id)

    return indexed_ids


def compute_recall(gt_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    total = len(gt_ids)
    gt_set = set(gt_ids)
    recalls: Dict[str, float] = {}
    for k in RECALL_KS:
        top_ids = retrieved_ids[: min(k, len(retrieved_ids))]
        hit = len([item_id for item_id in top_ids if item_id in gt_set])
        recalls[f"R@{k}"] = hit / total if total else 0.0
    return recalls


def summarize_recalls(details: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    totals = {f"R@{k}": 0.0 for k in RECALL_KS}
    count = 0
    for detail in details:
        recall = detail.get(key)
        if not recall:
            continue
        for k in RECALL_KS:
            totals[f"R@{k}"] += recall.get(f"R@{k}", 0.0)
        count += 1
    if count:
        for k in RECALL_KS:
            totals[f"R@{k}"] /= count
    totals["count"] = count
    return totals


def extract_item_id_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\[(.*?)\]", text)
    if match:
        return match.group(1).strip()
    return None


def build_checkpoint_key(
    args: argparse.Namespace,
    media_text_config: MediaTextConfig,
    email_text_config: EmailTextConfig,
) -> str:
    def safe_mtime(path: str) -> Optional[float]:
        return os.path.getmtime(path) if path and os.path.exists(path) else None

    payload = {
        "media_source": args.media_source,
        "image_batch_results": args.image_batch_results,
        "video_batch_results": args.video_batch_results,
        "email_file": args.email_file,
        "image_root": args.image_root,
        "video_root": args.video_root,
        "image_batch_mtime": safe_mtime(args.image_batch_results),
        "video_batch_mtime": safe_mtime(args.video_batch_results),
        "email_mtime": safe_mtime(args.email_file),
        "media_text_config": asdict(media_text_config),
        "email_text_config": asdict(email_text_config),
        "embedding_model": args.memoryos_embedding_model,
        "embedding_kwargs": args.memoryos_embedding_kwargs,
        "index_llm_model": args.memoryos_index_llm_model,
        "short_term_capacity": args.memoryos_short_term_capacity,
        "mid_term_capacity": args.memoryos_mid_term_capacity,
        "long_term_capacity": args.memoryos_long_term_knowledge_capacity,
        "mid_term_heat_threshold": args.memoryos_heat_threshold,
        "mid_term_similarity_threshold": args.memoryos_similarity_threshold,
        "full_history_mode": args.memoryos_full_history_mode,
        "full_history_mid_term_capacity": args.memoryos_full_history_mid_term_capacity,
        "full_history_heat_threshold": args.memoryos_full_history_heat_threshold,
    }
    return build_cache_key(payload)


def has_existing_memory(storage_root: Path, user_id: str) -> bool:
    user_dir = storage_root / "users" / user_id
    return (user_dir / "short_term.json").exists() and (
        user_dir / "mid_term.json"
    ).exists()


def prepare_memory_storage(
    storage_root: Path,
    checkpoint_dir: Path,
    reuse_checkpoint: bool,
    force_rebuild: bool,
) -> bool:
    if not reuse_checkpoint or force_rebuild:
        return False
    if not checkpoint_dir.exists():
        return False
    if storage_root.exists():
        shutil.rmtree(storage_root)
    shutil.copytree(checkpoint_dir, storage_root)
    return True


def normalize_openai_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return base_url
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        trimmed = trimmed[: -len("/chat/completions")]
    parsed = urlparse(trimmed)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or ""
        if path in {"", "/"}:
            path = "/v1"
        return urlunparse(parsed._replace(path=path))
    return trimmed


def patch_memoryos_legacy_defaults(
    args: argparse.Namespace, llm_model_override: Optional[str] = None
) -> None:
    if not ensure_memoryos_importable():
        return
    try:
        import memoryos.utils as memos_utils
        import memoryos.mid_term as memos_mid_term
    except Exception:
        return
    try:
        import memoryos.prompts as memos_prompts
    except Exception:
        memos_prompts = None

    if memos_prompts is not None:
        if not hasattr(memos_prompts, "_pmqa_original_generate_system_prompt"):
            memos_prompts._pmqa_original_generate_system_prompt = (
                memos_prompts.GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT
            )
            memos_prompts._pmqa_original_generate_user_prompt = (
                memos_prompts.GENERATE_SYSTEM_RESPONSE_USER_PROMPT
            )

        prompt_style = (args.memoryos_answer_prompt_style or "memoryos").lower()
        if prompt_style == "baseline":
            memos_prompts.GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT = QA_PROMPTS[
                "BASELINE_SYSTEM"
            ]
            memos_prompts.GENERATE_SYSTEM_RESPONSE_USER_PROMPT = QA_PROMPTS[
                "BASELINE_USER"
            ]
        else:
            memos_prompts.GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT = (
                memos_prompts._pmqa_original_generate_system_prompt
            )
            memos_prompts.GENERATE_SYSTEM_RESPONSE_USER_PROMPT = (
                memos_prompts._pmqa_original_generate_user_prompt
            )

        applied_style = getattr(memos_prompts, "_pmqa_answer_prompt_style", None)
        if applied_style != prompt_style:
            logger.info("MemoryOS QA answer prompt style: %s", prompt_style)
        memos_prompts._pmqa_answer_prompt_style = prompt_style

    if not getattr(memos_utils, "_pmqa_patched", False):
        original_get_embedding = memos_utils.get_embedding
        original_get_embedding_sig = inspect.signature(original_get_embedding)
        embedding_accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in original_get_embedding_sig.parameters.values()
        )

        original_chat_completion = memos_utils.OpenAIClient.chat_completion

        def call_original_embedding(
            text: str,
            model_name: str,
            use_cache: bool,
            kwargs: Dict[str, Any],
        ):
            global _EMBEDDING_KWARGS_WARNING_EMITTED
            if kwargs and embedding_accepts_kwargs:
                return original_get_embedding(
                    text, model_name=model_name, use_cache=use_cache, **kwargs
                )
            if kwargs and not embedding_accepts_kwargs:
                if not _EMBEDDING_KWARGS_WARNING_EMITTED:
                    with _EMBEDDING_LOAD_LOCK:
                        if not _EMBEDDING_KWARGS_WARNING_EMITTED:
                            logger.warning(
                                "Installed MemoryOS get_embedding does not accept kwargs; "
                                "ignoring --memoryos-embedding-kwargs."
                            )
                            _EMBEDDING_KWARGS_WARNING_EMITTED = True
                return original_get_embedding(
                    text, model_name=model_name, use_cache=use_cache
                )
            return original_get_embedding(text, model_name=model_name, use_cache=use_cache)

        def fallback_embedding_cpu(
            text: str,
            model_name: str,
            use_cache: bool,
        ):
            # Fallback path for torch meta-tensor load failures.
            from sentence_transformers import SentenceTransformer

            model_cache = getattr(memos_utils, "_model_cache", None)
            embedding_cache = getattr(memos_utils, "_embedding_cache", None)

            cache_key = None
            if use_cache and isinstance(embedding_cache, dict):
                cache_key = f"{model_name}::{hash(text)}"
                cached = embedding_cache.get(cache_key)
                if cached is not None:
                    return cached

            model = None
            if isinstance(model_cache, dict):
                model = model_cache.get(model_name)
            if model is None:
                logger.warning(
                    "Falling back to CPU for embedding model '%s' after meta-tensor error.",
                    model_name,
                )
                model = SentenceTransformer(model_name, device="cpu")
                if isinstance(model_cache, dict):
                    model_cache[model_name] = model

            embedding = model.encode([text], convert_to_numpy=True)[0]
            if cache_key and isinstance(embedding_cache, dict):
                embedding_cache[cache_key] = embedding
                if len(embedding_cache) > 10000:
                    keys_to_remove = list(embedding_cache.keys())[:1000]
                    for key in keys_to_remove:
                        embedding_cache.pop(key, None)
            return embedding

        def get_embedding(text, model_name="all-MiniLM-L6-v2", use_cache=True, **kwargs):
            if model_name in {None, "all-MiniLM-L6-v2"}:
                model_name = args.memoryos_embedding_model
            if not kwargs and args.memoryos_embedding_kwargs:
                kwargs = dict(args.memoryos_embedding_kwargs)

            model_cache = getattr(memos_utils, "_model_cache", None)
            needs_init = isinstance(model_cache, dict) and model_name not in model_cache
            try:
                if needs_init:
                    # MemoryOS get_embedding initializes SentenceTransformer without a lock.
                    # Parallel QA can race here and trigger torch meta-tensor init failures.
                    with _EMBEDDING_LOAD_LOCK:
                        model_cache = getattr(memos_utils, "_model_cache", None)
                        if isinstance(model_cache, dict) and model_name in model_cache:
                            return call_original_embedding(
                                text, model_name=model_name, use_cache=use_cache, kwargs=kwargs
                            )
                        return call_original_embedding(
                            text, model_name=model_name, use_cache=use_cache, kwargs=kwargs
                        )
                return call_original_embedding(
                    text, model_name=model_name, use_cache=use_cache, kwargs=kwargs
                )
            except Exception as exc:
                error_text = str(exc).lower()
                if "meta tensor" not in error_text:
                    raise
                logger.warning(
                    "Embedding load failed with meta-tensor error for '%s': %s",
                    model_name,
                    exc,
                )
                with _EMBEDDING_LOAD_LOCK:
                    model_cache = getattr(memos_utils, "_model_cache", None)
                    if isinstance(model_cache, dict):
                        model_cache.pop(model_name, None)
                    return fallback_embedding_cpu(
                        text=text, model_name=model_name, use_cache=use_cache
                    )

        def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
            if args.temperature is not None:
                temperature = args.temperature
            if args.max_tokens is not None:
                max_tokens = args.max_tokens
            request_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if args.timeout:
                try:
                    create_sig = inspect.signature(
                        self.client.chat.completions.create
                    )
                except (TypeError, ValueError):
                    create_sig = None
                if create_sig and "timeout" in create_sig.parameters:
                    request_kwargs["timeout"] = args.timeout
            message_count = len(messages) if isinstance(messages, list) else 0
            system_chars = 0
            user_chars = 0
            if isinstance(messages, list):
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    content = str(message.get("content", "") or "")
                    role = str(message.get("role", "") or "")
                    if role == "system":
                        system_chars += len(content)
                    elif role == "user":
                        user_chars += len(content)
            try:
                response = self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                logger.exception(
                    (
                        "MemoryOS LLM call failed: model=%s timeout=%s "
                        "messages=%s system_chars=%s user_chars=%s"
                    ),
                    model,
                    request_kwargs.get("timeout"),
                    message_count,
                    system_chars,
                    user_chars,
                )
                raise RuntimeError(f"MemoryOS LLM call failed: {exc}") from exc
            raw_text = (response.choices[0].message.content or "")
            raw_content = raw_text.strip()
            if not raw_content:
                logger.error(
                    (
                        "MemoryOS LLM returned empty content: model=%s timeout=%s "
                        "messages=%s system_chars=%s user_chars=%s"
                    ),
                    model,
                    request_kwargs.get("timeout"),
                    message_count,
                    system_chars,
                    user_chars,
                )
                raise RuntimeError("MemoryOS LLM returned empty content")
            clean_fn = getattr(memos_utils, "clean_reasoning_model_output", None)
            if callable(clean_fn):
                cleaned = clean_fn(raw_content)
                cleaned = str(cleaned or "").strip()
                if not cleaned:
                    logger.error(
                        (
                            "MemoryOS LLM output became empty after cleanup: model=%s "
                            "messages=%s system_chars=%s user_chars=%s"
                        ),
                        model,
                        message_count,
                        system_chars,
                        user_chars,
                    )
                    raise RuntimeError(
                        "MemoryOS LLM output became empty after cleanup"
                    )
                return cleaned
            return raw_content

        memos_utils.get_embedding = get_embedding
        memos_utils.OpenAIClient.chat_completion = chat_completion
        memos_mid_term.get_embedding = get_embedding
        memos_utils._pmqa_patched = True

    original_llm_extract = getattr(memos_utils, "llm_extract_keywords", None)
    if original_llm_extract and not getattr(memos_utils, "_pmqa_kw_patched", False):

        def llm_extract_keywords(text, client, model="gpt-4o-mini"):
            if model in {None, "gpt-4o-mini"}:
                model = llm_model_override or args.memoryos_llm_model
            response = original_llm_extract(text, client, model=model)
            if isinstance(response, list):
                return response
            if not response:
                return []
            if isinstance(response, str) and response.strip().startswith("Error:"):
                return []
            return response

        memos_utils.llm_extract_keywords = llm_extract_keywords
        memos_mid_term.llm_extract_keywords = llm_extract_keywords
        memos_utils._pmqa_kw_patched = True

    if args.memoryos_parallel_llm and not getattr(
        memos_mid_term, "_pmqa_parallel_patched", False
    ):
        original_add_session = memos_mid_term.MidTermMemory.add_session
        original_insert = memos_mid_term.MidTermMemory.insert_pages_into_session

        def prefill_page_keywords(self, pages: List[Dict[str, Any]]) -> None:
            tasks = []
            for page_data in pages:
                if page_data.get("page_keywords"):
                    continue
                full_text = (
                    f"User: {page_data.get('user_input', '')} Assistant: "
                    f"{page_data.get('agent_response', '')}"
                )
                tasks.append((page_data, full_text))

            if not tasks:
                return

            max_workers = max(1, args.memoryos_parallel_llm_workers)
            max_workers = min(max_workers, len(tasks))
            model_name = getattr(
                self, "llm_model", llm_model_override or args.memoryos_llm_model
            )
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for page_data, full_text in tasks:
                    futures[
                        executor.submit(
                            memos_mid_term.llm_extract_keywords,
                            full_text,
                            client=self.client,
                            model=model_name,
                        )
                    ] = page_data
                for future in as_completed(futures):
                    page_data = futures[future]
                    try:
                        keywords = future.result()
                    except Exception:
                        continue
                    if keywords:
                        page_data["page_keywords"] = list(keywords)

        def add_session(self, summary, details):
            prefill_page_keywords(self, details)
            return original_add_session(self, summary, details)

        def insert_pages_into_session(
            self,
            summary_for_new_pages,
            keywords_for_new_pages,
            pages_to_insert,
            similarity_threshold=0.6,
            keyword_similarity_alpha=1.0,
        ):
            prefill_page_keywords(self, pages_to_insert)
            return original_insert(
                self,
                summary_for_new_pages,
                keywords_for_new_pages,
                pages_to_insert,
                similarity_threshold=similarity_threshold,
                keyword_similarity_alpha=keyword_similarity_alpha,
            )

        memos_mid_term.MidTermMemory.add_session = add_session
        memos_mid_term.MidTermMemory.insert_pages_into_session = (
            insert_pages_into_session
        )
        memos_mid_term._pmqa_parallel_patched = True


def create_memoryos_instance(
    args: argparse.Namespace,
    user_id: str,
    storage_path_override: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    llm_model_override: Optional[str] = None,
) -> Any:
    """
    Create a MemoryOS instance with proper configuration.

    Args:
        args: Parsed arguments
        user_id: Unique user ID for this instance
        openai_base_url: Optional custom OpenAI base URL

    Returns:
        Configured Memoryos instance
    """
    try:
        from memoryos import Memoryos
    except ImportError:
        if ensure_memoryos_importable():
            from memoryos import Memoryos
        else:
            raise RuntimeError(
                "MemoryOS package not found. Install `memoryos-pro` (recommended) or "
                "install from GitHub (`pip install git+https://github.com/BAI-LAB/MemoryOS.git#subdirectory=memoryos-pypi`). "
                "Alternatively, ensure the vendored copy exists at `third_party/MemoryOS/memoryos/`."
            )

    patch_memoryos_legacy_defaults(args, llm_model_override=llm_model_override)

    # Determine API key and base URL
    if args.provider == "openai":
        api_key = args.api_key if args.api_key is not None else get_openai_api_key()
        base_url = openai_base_url
    else:
        api_key = args.api_key if args.api_key is not None else get_vllm_api_key()
        base_url = args.vllm_endpoint or openai_base_url
        if not base_url:
            base_url = MEMORYOS_CONFIG["vllm_text"].get("endpoint")

    base_url = normalize_openai_base_url(base_url) if base_url else None

    storage_path = storage_path_override or args.memoryos_data_storage_path

    # Embedding model kwargs
    embedding_kwargs = args.memoryos_embedding_kwargs

    llm_model_name = llm_model_override or args.memoryos_llm_model
    logger.info("Creating MemoryOS instance for user_id='%s'", user_id)
    logger.info("  LLM: %s", llm_model_name)
    logger.info("  Embedding: %s", args.memoryos_embedding_model)
    logger.info("  Base URL: %s", base_url)
    logger.info("  Storage: %s", storage_path)

    retrieval_queue_capacity = args.memoryos_retrieval_queue_capacity
    if args.max_evidence_items is not None:
        retrieval_queue_capacity = min(
            retrieval_queue_capacity, args.max_evidence_items
        )

    memoryos_kwargs = {
        "user_id": user_id,
        "openai_api_key": api_key,
        "openai_base_url": base_url,
        "data_storage_path": storage_path,
        "llm_model": llm_model_name,
        "assistant_id": MEMORYOS_CONFIG["memoryos_assistant_id"],
        "short_term_capacity": args.memoryos_short_term_capacity,
        "mid_term_capacity": args.memoryos_mid_term_capacity,
        "long_term_knowledge_capacity": args.memoryos_long_term_knowledge_capacity,
        "retrieval_queue_capacity": retrieval_queue_capacity,
        "mid_term_heat_threshold": args.memoryos_heat_threshold,
        "mid_term_similarity_threshold": args.memoryos_similarity_threshold,
        "embedding_model_name": args.memoryos_embedding_model,
        "embedding_model_kwargs": embedding_kwargs,
    }

    alias_map = {
        "openai_api_key": "api_key",
        "openai_base_url": "base_url",
        "llm_model": "model",
        "embedding_model_name": "embedding_model",
        "long_term_knowledge_capacity": "long_term_capacity",
        "mid_term_heat_threshold": "heat_threshold",
        "mid_term_similarity_threshold": "similarity_threshold",
        "retrieval_queue_capacity": "queue_capacity",
    }

    signature = inspect.signature(Memoryos.__init__)
    accepted = set(signature.parameters)
    accepted.discard("self")

    if "embedding_model_name" not in accepted:
        logger.warning(
            "Installed MemoryOS does not accept embedding_model_name; "
            "using runtime patch to set default embedding model."
        )

    filtered_kwargs: Dict[str, Any] = {}
    for key, value in memoryos_kwargs.items():
        if key in accepted:
            filtered_kwargs[key] = value
            continue
        alias = alias_map.get(key)
        if alias and alias in accepted:
            filtered_kwargs[alias] = value

    missing_required = [
        param.name
        for param in signature.parameters.values()
        if param.name != "self"
        and param.default is param.empty
        and param.name not in filtered_kwargs
    ]
    if missing_required:
        raise TypeError(
            "Memoryos.__init__ missing required args: " + ", ".join(missing_required)
        )

    logger.info(
        "MemoryOS init args: %s",
        ", ".join(sorted(filtered_kwargs.keys())),
    )
    with suppress_memoryos_output(args.memoryos_quiet):
        memoryos_instance = Memoryos(**filtered_kwargs)

    original_retrieve_context = memoryos_instance.retriever.retrieve_context

    def retrieve_context_with_metrics(user_query: str, user_id: str) -> Dict[str, Any]:
        matched_sessions = memoryos_instance.mid_term_memory.search_sessions(
            query_text=user_query,
            segment_similarity_threshold=args.memoryos_segment_similarity_threshold,
            page_similarity_threshold=args.memoryos_page_similarity_threshold,
            top_k_sessions=args.memoryos_top_k_sessions,
        )

        top_pages_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        page_counter = 0
        for session_match in matched_sessions:
            for page_match in session_match.get("matched_pages", []):
                page_data = page_match.get("page_data") or {}
                page_score = page_match.get("score") or 0.0
                combined_score = page_score
                if (
                    len(top_pages_heap)
                    < memoryos_instance.retriever.retrieval_queue_capacity
                ):
                    top_pages_heap.append((combined_score, page_counter, page_data))
                    page_counter += 1
                    if len(top_pages_heap) > 1:
                        top_pages_heap.sort(key=lambda x: x[0])
                elif combined_score > top_pages_heap[0][0]:
                    top_pages_heap[0] = (combined_score, page_counter, page_data)
                    page_counter += 1
                    top_pages_heap.sort(key=lambda x: x[0])

        retrieved_pages_scored = sorted(
            top_pages_heap, key=lambda x: x[0], reverse=True
        )
        retrieved_pages = [item[2] for item in retrieved_pages_scored]
        retrieval_scores = [item[0] for item in retrieved_pages_scored]

        retrieved_user_knowledge = memoryos_instance.retriever._retrieve_user_knowledge(
            user_query,
            args.memoryos_knowledge_threshold,
            args.memoryos_top_k_knowledge,
        )
        retrieved_assistant_knowledge = (
            memoryos_instance.retriever._retrieve_assistant_knowledge(
                user_query,
                args.memoryos_knowledge_threshold,
                args.memoryos_top_k_knowledge,
            )
        )

        result = {
            "retrieved_pages": retrieved_pages,
            "retrieved_user_knowledge": retrieved_user_knowledge or [],
            "retrieved_assistant_knowledge": retrieved_assistant_knowledge or [],
            "retrieved_at": time.time(),
        }
        memoryos_instance._last_retrieval = {
            "retrieved_pages": retrieved_pages,
            "retrieval_scores": retrieval_scores,
        }
        return result

    memoryos_instance.retriever.retrieve_context = retrieve_context_with_metrics
    memoryos_instance._original_retrieve_context = original_retrieve_context
    memoryos_instance._last_retrieval = {"retrieved_pages": [], "retrieval_scores": []}

    return memoryos_instance


def index_memories_into_memoryos(
    memoryos_instance: Any,
    items: List[RetrievalItem],
    desc: str = "Indexing memories",
    quiet: bool = False,
    progress_path: Optional[Path] = None,
    resume_ids: Optional[Set[str]] = None,
) -> None:
    """
    Index retrieval items into MemoryOS as dialogue pairs.

    Args:
        memoryos_instance: MemoryOS instance
        items: List of retrieval items
        desc: Progress bar description
    """
    resume_ids = resume_ids or set()
    for item in tqdm(items, desc=desc):
        if item.item_id and item.item_id in resume_ids:
            continue
        user_input, agent_response = format_item_as_dialogue(item)

        if not user_input or not agent_response:
            continue

        # Get timestamp from metadata if available
        timestamp = None
        if item.metadata and "timestamp" in item.metadata:
            timestamp = item.metadata["timestamp"]

        # Add to MemoryOS
        try:
            with suppress_memoryos_output(quiet):
                memoryos_instance.add_memory(
                    user_input=user_input,
                    agent_response=agent_response,
                    timestamp=timestamp,
                )
            if item.item_id:
                append_index_progress(progress_path, item.item_id, "success")
        except Exception as exc:
            logger.exception("MemoryOS add_memory failed for %s", item.item_id)
            if item.item_id:
                append_index_progress(
                    progress_path, item.item_id, "error", error=str(exc)
                )


def answer_question_with_memoryos(
    qa: Dict[str, Any],
    memoryos_instance: Any,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """
    Answer a question using MemoryOS.

    Args:
        qa: QA dict with 'id' and 'question'
        memoryos_instance: Initialized MemoryOS instance with indexed memories
        args: Parsed arguments

    Returns:
        Dict with 'id' and 'answer', or None if error
    """
    qa_id = qa.get("id") or qa.get("qa_id")
    question = qa.get("question")

    if not qa_id or not question:
        logger.warning("QA missing id or question: %s", qa)
        return None

    try:
        disable_update = bool(
            args.memoryos_eval_no_update and hasattr(memoryos_instance, "add_memory")
        )
        saved_add_memory = None
        if disable_update:
            saved_add_memory = memoryos_instance.add_memory

            def _skip_add_memory(*_args, **_kwargs):
                return None

            memoryos_instance.add_memory = _skip_add_memory

        try:
            with suppress_memoryos_output(args.memoryos_quiet):
                answer = memoryos_instance.get_response(query=question)
        finally:
            if disable_update and saved_add_memory is not None:
                memoryos_instance.add_memory = saved_add_memory

        if answer is None or not str(answer).strip():
            logger.error("MemoryOS returned empty answer for question %s", qa_id)
            return {
                "id": qa_id,
                "answer": "Error: MemoryOS returned empty answer",
            }

        return {"id": qa_id, "answer": str(answer)}

    except Exception as e:
        logger.exception("Error answering question %s: %s", qa_id, e)
        import traceback

        traceback.print_exc()
        return {"id": qa_id, "answer": f"Error: {str(e)}"}


def answer_questions_with_shared_instance(
    qa_list: List[Dict[str, Any]],
    memoryos_instance: Any,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Answer all questions using a single shared MemoryOS instance.

    WARNING: This will cause memory contamination as each answered
    question gets added to memory. Only use for testing or if you
    want cumulative memory behavior.
    """
    results = []
    for qa in tqdm(qa_list, desc="Answering questions"):
        result = answer_question_with_memoryos(qa, memoryos_instance, args)
        if result:
            results.append(result)
    return results


def answer_questions_with_per_qa_instances(
    qa_list: List[Dict[str, Any]],
    retrieval_items: List[RetrievalItem],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Answer each question with an isolated MemoryOS instance.

    This prevents memory contamination between questions.
    Instances are created per question; use the main entry point for
    optional parallel execution with checkpoints.
    """

    def process_single_qa(qa: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single QA with isolated instance."""
        qa_id = qa.get("id") or qa.get("qa_id")

        try:
            # Create unique user ID for this question
            user_id = args.memoryos_user_id

            # Create isolated instance (indexing LLM first)
            memoryos_instance = create_memoryos_instance(
                args, user_id, llm_model_override=args.memoryos_index_llm_model
            )

            # Index all memories
            index_memories_into_memoryos(
                memoryos_instance,
                retrieval_items,
                desc=f"Indexing for QA {qa_id}",
            )

            # Switch to answerer LLM before answering
            set_memoryos_answer_llm(
                memoryos_instance, args.memoryos_answer_llm_model
            )
            # Answer question
            result = answer_question_with_memoryos(qa, memoryos_instance, args)

            # Cleanup (optional - MemoryOS saves to disk)
            # Could delete storage directory here if desired

            return result

        except Exception as e:
            logger.exception("Error processing QA %s: %s", qa_id, e)
            import traceback

            traceback.print_exc()
            return {"id": qa_id, "answer": f"Error: {str(e)}"}

    # Per-qa isolation uses shared storage; process sequentially to avoid clobbering.
    results = []
    for qa in tqdm(qa_list, desc="Answering questions (isolated)"):
        result = process_single_qa(qa)
        if result:
            results.append(result)

    # Sort by QA order
    qa_id_to_idx = {
        qa.get("id") or qa.get("qa_id"): idx for idx, qa in enumerate(qa_list)
    }
    results.sort(key=lambda x: qa_id_to_idx.get(x["id"], 999999))

    return results


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.memoryos_embedding_kwargs = resolve_embedding_kwargs(
        args.memoryos_embedding_kwargs
    )
    resolve_llm_defaults(args)
    apply_full_history_mode(args)
    if args.memoryos_eval_no_update:
        logger.info(
            "Enabled --memoryos-eval-no-update: skipping post-answer memory/profile updates."
        )

    # Create output directory
    output_dir = Path(args.output_dir_base) / args.method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "memoryos_answers.jsonl"
    if output_file.exists():
        logger.info("Output exists, skipping inference: %s", output_file)
        return

    # Load QA data
    logger.info("Loading QA data from %s", args.qa_file)
    qa_data = load_json(Path(args.qa_file))
    qa_list = load_qa_list(qa_data)
    logger.info("Loaded %s QA pairs", len(qa_list))

    # Build retrieval items with text augmentation
    logger.info("Building retrieval items...")

    # Configure text augmentation
    media_text_config = MediaTextConfig(
        include_id=args.include_id,
        include_type=args.include_type,
        include_timestamp=args.include_timestamp,
        include_location=args.include_location,
        include_short_caption=args.include_short_caption,
        include_caption=args.include_caption,
        include_ocr_text=args.include_ocr_text,
        include_tags=args.include_tags,
    )

    email_text_config = EmailTextConfig(
        include_id=args.include_id,
        include_timestamp=args.include_timestamp,
        include_summary=args.include_email_summary,
        include_detail=args.include_email_detail,
    )

    image_entries = load_json(Path(args.image_batch_results))
    video_entries = load_json(Path(args.video_batch_results))
    email_entries = load_json(Path(args.email_file))

    if not isinstance(image_entries, list) or not isinstance(video_entries, list):
        raise ValueError("Batch results must be lists")
    if not isinstance(email_entries, list):
        raise ValueError("Email file should be a list")

    retrieval_items = build_retrieval_items(
        email_entries,
        image_entries,
        video_entries,
        media_text_config,
        email_text_config,
        Path(args.image_root),
        Path(args.video_root),
    )

    retrieval_items, missing_timestamps = sort_retrieval_items(retrieval_items)
    if missing_timestamps:
        logger.warning(
            "Missing or invalid timestamps for %s items; appended after dated items.",
            missing_timestamps,
        )
    log_missing_metadata(retrieval_items)

    logger.info("Built %s retrieval items", len(retrieval_items))

    checkpoint_key = build_checkpoint_key(args, media_text_config, email_text_config)
    storage_root = Path(args.memoryos_data_storage_path)
    if args.memoryos_data_storage_path == MEMORYOS_CONFIG["memoryos_data_storage_path"]:
        storage_root = output_dir / "memoryos_data"
    storage_root.mkdir(parents=True, exist_ok=True)
    args.memoryos_data_storage_path = str(storage_root)

    checkpoint_dir = Path(args.memoryos_checkpoint_dir) / checkpoint_key
    progress_path = output_dir / f"index_progress_{checkpoint_key}.jsonl"
    resume_ids: Set[str] = set()
    if args.memoryos_resume_indexing:
        has_memory_snapshot = has_existing_memory(storage_root, args.memoryos_user_id)
        indexed_ids = (
            load_indexed_item_ids_from_storage(storage_root, args.memoryos_user_id)
            if has_memory_snapshot
            else set()
        )

        if progress_path.exists() and has_memory_snapshot:
            progress_ids = load_index_progress(progress_path)
            if indexed_ids:
                resume_ids = indexed_ids
                stale_progress_ids = progress_ids - indexed_ids
                missing_progress_ids = indexed_ids - progress_ids
                if stale_progress_ids or missing_progress_ids:
                    archive_path = progress_path.with_name(
                        f"{progress_path.stem}_stale_{int(time.time())}.jsonl"
                    )
                    shutil.copy2(progress_path, archive_path)
                    rewrite_index_progress(progress_path, resume_ids)
                    logger.warning(
                        (
                            "Resume state mismatch detected. Archived stale progress to %s. "
                            "Progress-only IDs=%s, memory-only IDs=%s. "
                            "Reconciled resume set to %s item IDs from memory snapshot."
                        ),
                        archive_path,
                        len(stale_progress_ids),
                        len(missing_progress_ids),
                        len(resume_ids),
                    )
                logger.info(
                    "Resuming indexing from memory snapshot (%s validated items).",
                    len(resume_ids),
                )
            elif progress_ids:
                logger.warning(
                    (
                        "Index progress exists (%s items) but no parseable item IDs found in "
                        "current memory files at %s. Ignoring resume state and rebuilding."
                    ),
                    len(progress_ids),
                    storage_root,
                )
            else:
                logger.info(
                    "Progress file exists but has no successful items: %s",
                    progress_path,
                )
        elif has_memory_snapshot:
            if indexed_ids:
                resume_ids = indexed_ids
                rewrite_index_progress(progress_path, resume_ids)
                logger.warning(
                    (
                        "Recovered resume state from memory snapshot (%s items) and "
                        "recreated missing progress file: %s"
                    ),
                    len(resume_ids),
                    progress_path,
                )
            elif progress_path.exists():
                logger.warning(
                    (
                        "Progress exists but memory files have no parseable item IDs. "
                        "Ignoring progress and rebuilding from scratch."
                    )
                )
        elif progress_path.exists():
            logger.warning(
                "Index progress exists but no memory storage found. Ignoring progress file."
            )
            progress_path.unlink(missing_ok=True)
    loaded_from_checkpoint = False

    # Answer questions based on isolation strategy
    recall_details: List[Dict[str, Any]] = []
    retrieval_total_ms = 0.0
    index_build_ms_total = 0.0
    model_load_ms_total = 0.0

    if args.memoryos_use_per_qa_instance:
        logger.info("Using PER-QUESTION isolation strategy")
        results = []
        if args.memoryos_reuse_checkpoint:
            if args.memoryos_force_rebuild or not checkpoint_dir.exists():
                logger.info("Building MemoryOS checkpoint: %s", checkpoint_dir)
                if not (args.memoryos_resume_indexing and resume_ids):
                    if storage_root.exists():
                        shutil.rmtree(storage_root)
                    storage_root.mkdir(parents=True, exist_ok=True)
                    resume_ids = set()

                start_load = time.perf_counter()
                memoryos_instance = create_memoryos_instance(
                    args,
                    args.memoryos_user_id,
                    llm_model_override=args.memoryos_index_llm_model,
                )
                model_load_ms_total += (time.perf_counter() - start_load) * 1000.0

                index_start = time.perf_counter()
                index_memories_into_memoryos(
                    memoryos_instance,
                    retrieval_items,
                    desc="Indexing checkpoint",
                    quiet=args.memoryos_quiet,
                    progress_path=progress_path,
                    resume_ids=resume_ids,
                )
                index_build_ms_total += (time.perf_counter() - index_start) * 1000.0

                if args.memoryos_save_checkpoint:
                    if checkpoint_dir.exists():
                        shutil.rmtree(checkpoint_dir)
                    shutil.copytree(storage_root, checkpoint_dir)
        else:
            logger.warning(
                "Per-question isolation without checkpoint reuse will re-index every QA."
            )
        use_parallel = bool(args.max_workers and args.max_workers > 1)
        if use_parallel and not args.memoryos_reuse_checkpoint:
            logger.warning(
                "Parallel per-question runs require checkpoint reuse; "
                "falling back to sequential execution."
            )
            use_parallel = False
        if use_parallel and not checkpoint_dir.exists():
            logger.warning(
                "Checkpoint missing; falling back to sequential per-question runs."
            )
            use_parallel = False
        if use_parallel:
            logger.info(
                "Parallel per-question runs enabled (%s workers).", args.max_workers
            )
        else:
            logger.info("Per-question runs are sequential.")

        qa_storage_root = storage_root
        if use_parallel:
            qa_storage_root = storage_root / "qa_instances"
            qa_storage_root.mkdir(parents=True, exist_ok=True)

        def process_single_qa(index: int, qa: Dict[str, Any]) -> Dict[str, Any]:
            qa_id = qa.get("id") or qa.get("qa_id")
            local_storage = storage_root
            cleanup_storage = False
            try:
                if use_parallel:
                    token = uuid.uuid4().hex[:8]
                    local_storage = qa_storage_root / f"qa_{index}_{token}"
                    shutil.copytree(checkpoint_dir, local_storage)
                    cleanup_storage = True
                else:
                    if args.memoryos_reuse_checkpoint and checkpoint_dir.exists():
                        prepare_memory_storage(
                            storage_root,
                            checkpoint_dir,
                            True,
                            False,
                        )
                    else:
                        if storage_root.exists():
                            shutil.rmtree(storage_root)
                        storage_root.mkdir(parents=True, exist_ok=True)

                llm_model_for_instance = args.memoryos_answer_llm_model
                needs_index = (
                    not args.memoryos_reuse_checkpoint
                    and not has_existing_memory(local_storage, args.memoryos_user_id)
                )
                if needs_index:
                    llm_model_for_instance = args.memoryos_index_llm_model

                start_load = time.perf_counter()
                memoryos_instance = create_memoryos_instance(
                    args,
                    args.memoryos_user_id,
                    storage_path_override=str(local_storage),
                    llm_model_override=llm_model_for_instance,
                )
                model_load_ms = (time.perf_counter() - start_load) * 1000.0

                index_build_ms = 0.0
                if needs_index:
                    index_start = time.perf_counter()
                    index_memories_into_memoryos(
                        memoryos_instance,
                        retrieval_items,
                        desc=f"Indexing for QA {qa_id}",
                        quiet=args.memoryos_quiet,
                        progress_path=progress_path,
                        resume_ids=resume_ids,
                    )
                    index_build_ms = (time.perf_counter() - index_start) * 1000.0
                    if args.memoryos_answer_llm_model != llm_model_for_instance:
                        set_memoryos_answer_llm(
                            memoryos_instance, args.memoryos_answer_llm_model
                        )

                retrieval_start = time.perf_counter()
                result = answer_question_with_memoryos(qa, memoryos_instance, args)
                retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

                retrieval_ids, retrieval_scores = extract_retrieval_ids(
                    memoryos_instance, args.max_evidence_items
                )
                recall_detail = build_recall_detail(
                    qa, retrieval_ids, retrieval_scores
                )
            except Exception as exc:
                logger.exception("Error processing QA %s: %s", qa_id, exc)
                result = {"id": qa_id, "answer": f"Error: {str(exc)}"}
                recall_detail = build_recall_detail(qa, [], [])
                model_load_ms = 0.0
                index_build_ms = 0.0
                retrieval_ms = 0.0
            finally:
                if cleanup_storage:
                    shutil.rmtree(local_storage, ignore_errors=True)

            return {
                "index": index,
                "result": result,
                "recall_detail": recall_detail,
                "model_load_ms": model_load_ms,
                "index_build_ms": index_build_ms,
                "retrieval_ms": retrieval_ms,
            }

        if use_parallel:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_map = {
                    executor.submit(process_single_qa, idx, qa): idx
                    for idx, qa in enumerate(qa_list)
                }
                ordered: List[Dict[str, Any]] = []
                for future in tqdm(
                    as_completed(future_map),
                    total=len(future_map),
                    desc="Answering questions (isolated, parallel)",
                ):
                    payload = future.result()
                    ordered.append(payload)
                ordered.sort(key=lambda item: item["index"])
        else:
            ordered = []
            for idx, qa in enumerate(tqdm(qa_list, desc="Answering questions (isolated)")):
                ordered.append(process_single_qa(idx, qa))

        for payload in ordered:
            result = payload["result"]
            if result:
                results.append(result)
            recall_details.append(payload["recall_detail"])
            model_load_ms_total += payload["model_load_ms"]
            index_build_ms_total += payload["index_build_ms"]
            retrieval_total_ms += payload["retrieval_ms"]
    else:
        logger.info("Using SHARED instance strategy")
        logger.info("Answered questions will be added to memory.")

        loaded_from_checkpoint = prepare_memory_storage(
            storage_root,
            checkpoint_dir,
            args.memoryos_reuse_checkpoint,
            args.memoryos_force_rebuild,
        )

        needs_index = not loaded_from_checkpoint and not has_existing_memory(
            storage_root, args.memoryos_user_id
        )
        llm_model_for_instance = (
            args.memoryos_index_llm_model if needs_index else args.memoryos_answer_llm_model
        )

        start_load = time.perf_counter()
        memoryos_instance = create_memoryos_instance(
            args,
            args.memoryos_user_id,
            llm_model_override=llm_model_for_instance,
        )
        model_load_ms_total += (time.perf_counter() - start_load) * 1000.0

        if needs_index:
            logger.info("Indexing memories into shared instance...")
            index_start = time.perf_counter()
            index_memories_into_memoryos(
                memoryos_instance,
                retrieval_items,
                quiet=args.memoryos_quiet,
                progress_path=progress_path,
                resume_ids=resume_ids,
            )
            index_build_ms_total += (time.perf_counter() - index_start) * 1000.0

            if args.memoryos_save_checkpoint:
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                shutil.copytree(storage_root, checkpoint_dir)
            if args.memoryos_answer_llm_model != llm_model_for_instance:
                set_memoryos_answer_llm(
                    memoryos_instance, args.memoryos_answer_llm_model
                )
        else:
            set_memoryos_answer_llm(memoryos_instance, args.memoryos_answer_llm_model)

        results = []
        for qa in tqdm(qa_list, desc="Answering questions"):
            retrieval_start = time.perf_counter()
            result = answer_question_with_memoryos(qa, memoryos_instance, args)
            retrieval_total_ms += (time.perf_counter() - retrieval_start) * 1000.0

            if result:
                results.append(result)

            retrieval_ids, retrieval_scores = extract_retrieval_ids(
                memoryos_instance, args.max_evidence_items
            )
            recall_details.append(
                build_recall_detail(qa, retrieval_ids, retrieval_scores)
            )

    # Save results
    logger.info("Saving %s results to %s", len(results), output_file)
    write_jsonl(output_file, results)

    # Save configuration
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    logger.info("Configuration saved to %s", config_file)

    summary = {
        "retrieval_top_k": args.memoryos_retrieval_queue_capacity,
        "recall": summarize_recalls(recall_details, "retrieval_recall"),
        "num_questions": len(qa_list),
    }
    write_json(output_dir / "retrieval_recall_summary.json", summary)
    write_json(output_dir / "retrieval_recall_details.json", recall_details)

    retrieval_latency = {
        "model_load_ms": model_load_ms_total,
        "index_build_ms": index_build_ms_total,
        "retrieval_total_ms": retrieval_total_ms,
        "retrieval_per_query_avg_ms": retrieval_total_ms / max(len(qa_list), 1),
        "num_queries": len(qa_list),
    }
    write_json(output_dir / "retrieval_latency.json", retrieval_latency)

    logger.info("Done!")


if __name__ == "__main__":
    main()
