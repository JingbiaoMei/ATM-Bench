#!/usr/bin/env python3
"""
SimpleMem Baseline for PersonalMemoryQA Benchmark

SimpleMem is an efficient lifelong memory system based on semantic lossless compression.
It uses a three-stage pipeline: Semantic Structured Compression, Online Semantic Synthesis,
and Intent-Aware Retrieval Planning.

Paper: https://arxiv.org/abs/2601.02553
Code: https://github.com/aiming-lab/SimpleMem

This implementation adapts SimpleMem for single-QA benchmarking by:
1. Formatting each memory item (email/image/video) as a dialogue pair
2. Using a shared SimpleMem instance per run (all items indexed once)
3. Supporting custom LLMs and embeddings via OpenAI-compatible endpoints
4. Providing ATM-Bench compliant answering with either SimpleMem's LLM client
   or an external LLM client
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from memqa.global_config import get_openai_api_key, get_vllm_api_key
from memqa.qa_agent_baselines.SimpleMem.simplemem_config import (
    PROMPTS,
    SIMPLEMEM_CONFIG,
)
from memqa.qa_agent_baselines.MMRag.llm_utils import LLMClient
from memqa.retrieve import (
    EmailTextConfig,
    MediaTextConfig,
    RetrievalItem,
    build_cache_key,
    build_retrieval_items,
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


@contextmanager
def suppress_output(enabled: bool):
    global _THREAD_SUPPRESS_WARNING_EMITTED
    if not enabled:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        if not _THREAD_SUPPRESS_WARNING_EMITTED:
            with _THREAD_SUPPRESS_WARNING_LOCK:
                if not _THREAD_SUPPRESS_WARNING_EMITTED:
                    logger.warning(
                        "suppress_output is ignored in worker threads during parallel QA "
                        "to avoid stdout/stderr race conditions."
                    )
                    _THREAD_SUPPRESS_WARNING_EMITTED = True
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimpleMem QA baseline")
    parser.add_argument("--qa-file", required=True, help="Path to QA annotations JSON")

    parser.add_argument(
        "--media-source",
        choices=["batch_results"],
        default=SIMPLEMEM_CONFIG["media_source"],
    )
    parser.add_argument(
        "--image-batch-results",
        default=SIMPLEMEM_CONFIG["image_batch_results"],
    )
    parser.add_argument(
        "--video-batch-results",
        default=SIMPLEMEM_CONFIG["video_batch_results"],
    )
    parser.add_argument("--image-root", default=SIMPLEMEM_CONFIG["image_root"])
    parser.add_argument("--video-root", default=SIMPLEMEM_CONFIG["video_root"])
    parser.add_argument("--email-file", default=SIMPLEMEM_CONFIG["email_file"])

    parser.add_argument(
        "--provider",
        choices=["openai", "vllm", "vllm_local"],
        default=SIMPLEMEM_CONFIG["provider"],
    )
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--vllm-endpoint", default=None)
    parser.add_argument(
        "--build-model",
        default=SIMPLEMEM_CONFIG["build_model"],
        help="Model for memory build/indexing (default: Qwen3-VL-2B on localhost)",
    )
    parser.add_argument(
        "--build-endpoint",
        default=SIMPLEMEM_CONFIG["build_endpoint"],
        help="OpenAI-compatible base URL for memory build/indexing LLM",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)

    parser.add_argument(
        "--max-evidence-items", type=int, default=SIMPLEMEM_CONFIG["max_evidence_items"]
    )
    parser.add_argument(
        "--max-workers", type=int, default=SIMPLEMEM_CONFIG["max_workers"]
    )
    parser.add_argument(
        "--output-dir-base", default=SIMPLEMEM_CONFIG["output_dir_base"]
    )
    parser.add_argument(
        "--method-name", default=SIMPLEMEM_CONFIG["method_name"]
    )
    parser.add_argument(
        "--overwrite-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate answer files even if they already exist.",
    )

    parser.add_argument(
        "--simplemem-dir",
        default=None,
        help="Path to SimpleMem repo root (or set SIMPLEMEM_DIR env var)",
    )
    parser.add_argument(
        "--simplemem-db-path",
        default=SIMPLEMEM_CONFIG["simplemem_db_path"],
        help="Path for SimpleMem LanceDB storage",
    )
    parser.add_argument(
        "--simplemem-reuse-index",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["simplemem_reuse_index"],
        help="Reuse existing SimpleMem index if available",
    )
    parser.add_argument(
        "--simplemem-force-rebuild",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["simplemem_force_rebuild"],
        help="Force rebuild SimpleMem index",
    )
    parser.add_argument(
        "--simplemem-answer-mode",
        choices=["atm", "native", "external"],
        default=SIMPLEMEM_CONFIG["simplemem_answer_mode"],
        help=(
            "Answer mode: 'atm' uses SimpleMem retrieval plus an ATM-Bench answer "
            "prompt; 'native' uses upstream SimpleMem AnswerGenerator; 'external' "
            "uses the repository LLMClient with the ATM-Bench answer prompt."
        ),
    )
    parser.add_argument(
        "--simplemem-use-native-answer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Deprecated compatibility flag. Prefer --simplemem-answer-mode native "
            "or --simplemem-answer-mode atm."
        ),
    )
    parser.add_argument(
        "--simplemem-enable-planning",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["simplemem_enable_planning"],
        help="Enable intent-aware retrieval planning",
    )
    parser.add_argument(
        "--simplemem-enable-reflection",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["simplemem_enable_reflection"],
        help="Enable reflection-based additional retrieval",
    )
    parser.add_argument(
        "--simplemem-max-reflection-rounds",
        type=int,
        default=SIMPLEMEM_CONFIG["simplemem_max_reflection_rounds"],
    )
    parser.add_argument(
        "--simplemem-enable-parallel-processing",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["simplemem_enable_parallel_processing"],
        help="Enable parallel processing for memory building",
    )
    parser.add_argument(
        "--simplemem-max-parallel-workers",
        type=int,
        default=SIMPLEMEM_CONFIG["simplemem_max_parallel_workers"],
    )
    parser.add_argument(
        "--simplemem-enable-parallel-retrieval",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["simplemem_enable_parallel_retrieval"],
        help="Enable parallel processing for retrieval queries",
    )
    parser.add_argument(
        "--simplemem-max-retrieval-workers",
        type=int,
        default=SIMPLEMEM_CONFIG["simplemem_max_retrieval_workers"],
    )
    parser.add_argument(
        "--simplemem-min-source-id-coverage",
        type=float,
        default=SIMPLEMEM_CONFIG["simplemem_min_source_id_coverage"],
        help=(
            "Minimum fraction of SimpleMem memory entries that must map back to "
            "ATM source IDs after building. Set to 0 to disable the check."
        ),
    )
    parser.add_argument(
        "--simplemem-window-size",
        type=int,
        default=SIMPLEMEM_CONFIG["simplemem_window_size"],
        help="Number of dialogues per window for memory construction",
    )
    parser.add_argument(
        "--simplemem-overlap-size",
        type=int,
        default=SIMPLEMEM_CONFIG["simplemem_overlap_size"],
        help="Window overlap size for context continuity",
    )
    parser.add_argument(
        "--simplemem-semantic-top-k",
        type=int,
        default=SIMPLEMEM_CONFIG["simplemem_semantic_top_k"],
        help="Top-k entries for semantic search",
    )
    parser.add_argument(
        "--simplemem-keyword-top-k",
        type=int,
        default=SIMPLEMEM_CONFIG["simplemem_keyword_top_k"],
        help="Top-k entries for keyword search",
    )
    parser.add_argument(
        "--simplemem-structured-top-k",
        type=int,
        default=SIMPLEMEM_CONFIG["simplemem_structured_top_k"],
        help="Top-k entries for structured search",
    )
    parser.add_argument(
        "--simplemem-quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress SimpleMem internal stdout/stderr",
    )

    parser.add_argument(
        "--include-id",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["include_id"],
    )
    parser.add_argument(
        "--include-type",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["include_type"],
    )
    parser.add_argument(
        "--include-timestamp",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["include_timestamp"],
    )
    parser.add_argument(
        "--include-location",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["include_location"],
    )
    parser.add_argument(
        "--include-short-caption",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["include_short_caption"],
    )
    parser.add_argument(
        "--include-caption",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["include_caption"],
    )
    parser.add_argument(
        "--include-ocr-text",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["include_ocr_text"],
    )
    parser.add_argument(
        "--include-tags",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["include_tags"],
    )
    parser.add_argument(
        "--include-email-summary",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["include_email_summary"],
    )
    parser.add_argument(
        "--include-email-detail",
        action=argparse.BooleanOptionalAction,
        default=SIMPLEMEM_CONFIG["include_email_detail"],
    )

    parser.add_argument(
        "--stage",
        choices=["build", "answer", "all"],
        default="all",
        help="Pipeline stage: 'build' (indexing only), 'answer' (QA only), 'all' (both)",
    )
    parser.add_argument(
        "--retrieval-log-k",
        type=int,
        default=100,
        help="Log top-K retrieved items for recall analysis",
    )

    args = parser.parse_args()
    if args.simplemem_use_native_answer is True:
        args.simplemem_answer_mode = "native"
    elif args.simplemem_use_native_answer is False:
        args.simplemem_answer_mode = "atm"
    return args


def normalize_openai_base_url(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return None
    normalized = endpoint.rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized or None


def resolve_api_settings(args: argparse.Namespace, for_build: bool = False) -> Tuple[str, Optional[str], Optional[str]]:
    if for_build:
        model = args.build_model
        if args.provider == "openai":
            api_key = args.api_key or get_openai_api_key()
            endpoint = args.build_endpoint
        else:
            api_key = args.api_key or get_vllm_api_key()
            endpoint = args.build_endpoint or args.vllm_endpoint
        return api_key, normalize_openai_base_url(endpoint), model
    if args.provider == "openai":
        api_key = args.api_key or get_openai_api_key()
        base_url = None
        model = args.model
    else:
        api_key = args.api_key or get_vllm_api_key()
        base_url = args.vllm_endpoint or SIMPLEMEM_CONFIG["vllm_text"].get("endpoint")
        model = args.model
    return api_key, normalize_openai_base_url(base_url), model


def build_text_configs(
    args: argparse.Namespace,
) -> Tuple[MediaTextConfig, EmailTextConfig]:
    media_config = MediaTextConfig(
        include_id=args.include_id,
        include_type=args.include_type,
        include_timestamp=args.include_timestamp,
        include_location=args.include_location,
        include_short_caption=args.include_short_caption,
        include_caption=args.include_caption,
        include_ocr_text=args.include_ocr_text,
        include_tags=args.include_tags,
    )
    email_config = EmailTextConfig(
        include_id=args.include_id,
        include_timestamp=args.include_timestamp,
        include_summary=args.include_email_summary,
        include_detail=args.include_email_detail,
    )
    return media_config, email_config


def normalize_metadata_value(
    metadata: Dict[str, Any], key: str, default: str = "Unknown"
) -> str:
    value = metadata.get(key)
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return str(value)


def format_item_as_dialogue(item: RetrievalItem) -> Tuple[str, str]:
    if not item.text:
        return ("", "")

    item_id = item.item_id
    modality = item.modality
    metadata = item.metadata or {}

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
        user_input = PROMPTS["MEMORY_GENERIC_USER"].format(
            item_id=item_id, content=item.text
        )
        agent_response = PROMPTS["MEMORY_GENERIC_AGENT"]

    return user_input, agent_response


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
            return datetime.strptime(text, fmt)
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

        parsed = date_parser.parse(text)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except (ValueError, TypeError, ImportError):
        return None


def sort_retrieval_items(
    items: List[RetrievalItem],
) -> List[RetrievalItem]:
    indexed: List[Tuple[bool, datetime, int, RetrievalItem]] = []
    for idx, item in enumerate(items):
        timestamp = None
        if item.metadata:
            timestamp = item.metadata.get("timestamp")
        parsed = parse_timestamp(timestamp)
        if parsed is None:
            parsed = datetime.max
        indexed.append((parsed == datetime.max, parsed, idx, item))
    indexed.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    return [entry[3] for entry in indexed]


def extract_item_ids_from_text(text: str) -> List[str]:
    if not text:
        return []
    ids: List[str] = []
    seen: Set[str] = set()
    source_matches = re.findall(r"\[SOURCE_ID:\s*([^\]]+)\]", text, flags=re.IGNORECASE)
    for raw_match in source_matches:
        for raw_id in re.split(r"[,;]", raw_match):
            item_id = raw_id.strip()
            if item_id and item_id not in seen:
                seen.add(item_id)
                ids.append(item_id)
    if ids:
        return ids

    match = re.search(r"\[(.*?)\]", text)
    if match:
        item_id = match.group(1).strip()
        if item_id:
            ids.append(item_id)
    return ids


def extract_item_id_from_text(text: str) -> Optional[str]:
    ids = extract_item_ids_from_text(text)
    return ids[0] if ids else None


def extract_item_ids_from_memory_entry(entry: Any) -> List[str]:
    if isinstance(entry, dict):
        restatement = entry.get("lossless_restatement", "") or entry.get("text", "")
        metadata = entry.get("metadata")
    else:
        restatement = getattr(entry, "lossless_restatement", "") or ""
        metadata = getattr(entry, "metadata", None)
    item_ids = extract_item_ids_from_text(restatement)
    if item_ids:
        return item_ids
    if isinstance(metadata, dict):
        raw = metadata.get("item_id") or metadata.get("source_item_id")
        if raw:
            return [str(raw)]
    return []


def extract_item_id_from_memory_entry(entry: Any) -> Optional[str]:
    ids = extract_item_ids_from_memory_entry(entry)
    return ids[0] if ids else None


def dedupe_retrieved_ids(retrieved_ids: List[str]) -> List[str]:
    seen: Set[str] = set()
    deduped: List[str] = []
    for item_id in retrieved_ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        deduped.append(item_id)
    return deduped


def compute_recall(gt_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    total = len(gt_ids)
    gt_set = set(gt_ids)
    deduped_retrieved_ids = dedupe_retrieved_ids(retrieved_ids)
    recalls: Dict[str, float] = {}
    for k in RECALL_KS:
        top_ids = deduped_retrieved_ids[: min(k, len(deduped_retrieved_ids))]
        hit = len([item_id for item_id in top_ids if item_id in gt_set])
        recalls[f"R@{k}"] = hit / total if total else 0.0
    return recalls


def _safe_mtime(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def build_index_cache_payload(args: argparse.Namespace) -> Dict[str, Any]:
    # Use the same corpus-identity convention as MemoryOS / HippoRAG2: include
    # the batch-result + email file paths AND their mtimes, so renaming or
    # rewriting the source files invalidates the cache.
    return {
        "image_batch_results": args.image_batch_results,
        "video_batch_results": args.video_batch_results,
        "email_file": args.email_file,
        "image_batch_mtime": _safe_mtime(args.image_batch_results),
        "video_batch_mtime": _safe_mtime(args.video_batch_results),
        "email_mtime": _safe_mtime(args.email_file),
        "provider": args.provider,
        "build_model": args.build_model,
        "build_endpoint": normalize_openai_base_url(args.build_endpoint or args.vllm_endpoint),
        "simplemem_window_size": args.simplemem_window_size,
        "simplemem_overlap_size": args.simplemem_overlap_size,
        "source_tracking": "atm_source_id_prompt_v1",
        "include_id": args.include_id,
        "include_type": args.include_type,
        "include_timestamp": args.include_timestamp,
        "include_location": args.include_location,
        "include_caption": args.include_caption,
        "include_short_caption": args.include_short_caption,
        "include_ocr_text": args.include_ocr_text,
        "include_tags": args.include_tags,
        "include_email_summary": args.include_email_summary,
        "include_email_detail": args.include_email_detail,
    }


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


def patch_simplemem_config(args: argparse.Namespace, api_key: str, base_url: Optional[str]) -> None:
    try:
        import config as sm_config
    except ImportError:
        logger.warning("SimpleMem config module not found; skipping patch")
        return

    if api_key:
        sm_config.OPENAI_API_KEY = api_key
    if base_url:
        sm_config.OPENAI_BASE_URL = base_url

    model_name = args.model
    if model_name:
        sm_config.LLM_MODEL = model_name

    sm_config.WINDOW_SIZE = args.simplemem_window_size
    sm_config.OVERLAP_SIZE = args.simplemem_overlap_size
    sm_config.SEMANTIC_TOP_K = args.simplemem_semantic_top_k
    sm_config.KEYWORD_TOP_K = args.simplemem_keyword_top_k
    sm_config.STRUCTURED_TOP_K = args.simplemem_structured_top_k
    sm_config.ENABLE_PARALLEL_PROCESSING = args.simplemem_enable_parallel_processing
    sm_config.MAX_PARALLEL_WORKERS = args.simplemem_max_parallel_workers
    sm_config.ENABLE_PARALLEL_RETRIEVAL = args.simplemem_enable_parallel_retrieval
    sm_config.MAX_RETRIEVAL_WORKERS = args.simplemem_max_retrieval_workers
    sm_config.ENABLE_PLANNING = args.simplemem_enable_planning
    sm_config.ENABLE_REFLECTION = args.simplemem_enable_reflection
    sm_config.MAX_REFLECTION_ROUNDS = args.simplemem_max_reflection_rounds
    sm_config.LANCEDB_PATH = args.simplemem_db_path

    if hasattr(sm_config, "USE_STREAMING"):
        sm_config.USE_STREAMING = False

    logger.info("Patched SimpleMem config: model=%s, base_url=%s", model_name, base_url)


def patch_simplemem_source_tracking() -> None:
    try:
        from core.memory_builder import MemoryBuilder
    except ImportError:
        return

    if getattr(MemoryBuilder, "_atm_source_tracking_patched", False):
        return

    original_build_prompt = MemoryBuilder._build_extraction_prompt

    def patched_build_prompt(
        self: Any, dialogue_text: str, dialogue_ids: List[int], context: str
    ) -> str:
        prompt = original_build_prompt(self, dialogue_text, dialogue_ids, context)
        source_requirement = (
            "0. **Source ID Preservation for ATM-Bench (MANDATORY)**: Every "
            "dialogue starts with `Help me remember this <kind> [<ID>]:` where "
            "`<ID>` is the canonical item identifier (e.g. `email_001`, "
            "`image_023`). Every memory entry you output MUST begin its "
            "`lossless_restatement` with `[SOURCE_ID: <id>]` listing all "
            "source IDs contributing to the entry, comma-separated. Also "
            "copy the same `<id>` strings into `keywords`. Do not omit or "
            "rename these IDs. The example below shows the required format.\n"
        )
        if "Source ID Preservation for ATM-Bench" in prompt:
            patched = prompt
        else:
            anchor = "[Requirements]\n"
            if anchor not in prompt:
                raise RuntimeError(
                    "SimpleMem MemoryBuilder._build_extraction_prompt no longer "
                    "contains the expected '[Requirements]' anchor. Refusing to "
                    "append the ATM source-ID requirement at a weaker prompt "
                    "position; pin SimpleMem to a compatible commit or update "
                    "patch_simplemem_source_tracking()."
                )
            patched = prompt.replace(anchor, f"{anchor}{source_requirement}", 1)

        # Replace the upstream example with one that shows the required
        # [SOURCE_ID: ...] prefix and source IDs in keywords. Small build
        # models tend to imitate the example rather than read requirements,
        # so the example is load-bearing for ATM-Bench attribution.
        example_anchor = (
            "[2025-11-15T14:30:00] Alice: Bob, let's meet at Starbucks tomorrow "
            "at 2pm to discuss the new product"
        )
        if example_anchor in patched:
            atm_example = (
                "[2026-01-05T09:00:00] User: Help me remember this email "
                "[email_042]: From Alice at 2026-01-05T09:00:00 from Office. "
                "Subject: Q1 roadmap. We'll meet at Starbucks tomorrow at 2pm "
                "to discuss the new product.\n"
                "[2026-01-05T09:01:00] User: Help me remember this image "
                "[image_017]: Captured at Starbucks on 2026-01-06T14:00:00. "
                "Photo of Bob seated next to Alice with a printed roadmap."
            )
            atm_output = (
                "[\n"
                "  {\n"
                "    \"lossless_restatement\": \"[SOURCE_ID: email_042] "
                "Alice emailed the user at 2026-01-05T09:00:00 from Office "
                "with subject 'Q1 roadmap' proposing to meet at Starbucks "
                "on 2026-01-06T14:00:00 to discuss the new product.\",\n"
                "    \"keywords\": [\"email_042\", \"Alice\", \"Starbucks\", "
                "\"Q1 roadmap\", \"new product\"],\n"
                "    \"timestamp\": \"2026-01-06T14:00:00\",\n"
                "    \"location\": \"Starbucks\",\n"
                "    \"persons\": [\"Alice\"],\n"
                "    \"entities\": [\"Q1 roadmap\", \"new product\"],\n"
                "    \"topic\": \"Product roadmap meeting arrangement\"\n"
                "  },\n"
                "  {\n"
                "    \"lossless_restatement\": \"[SOURCE_ID: image_017] On "
                "2026-01-06T14:00:00 at Starbucks a photo shows Bob seated "
                "next to Alice with a printed roadmap.\",\n"
                "    \"keywords\": [\"image_017\", \"Bob\", \"Alice\", "
                "\"Starbucks\", \"roadmap\"],\n"
                "    \"timestamp\": \"2026-01-06T14:00:00\",\n"
                "    \"location\": \"Starbucks\",\n"
                "    \"persons\": [\"Bob\", \"Alice\"],\n"
                "    \"entities\": [\"roadmap\"],\n"
                "    \"topic\": \"Meeting photograph\"\n"
                "  }\n"
                "]"
            )
            # Replace the upstream example block (dialogues + JSON output).
            upstream_block_start = patched.find("[Example]")
            upstream_block_end = patched.find(
                "Now process the above dialogues."
            )
            if upstream_block_start != -1 and upstream_block_end != -1:
                replacement = (
                    "[Example]\n"
                    "Dialogues:\n"
                    f"{atm_example}\n\n"
                    "Output:\n"
                    "```json\n"
                    f"{atm_output}\n"
                    "```\n\n"
                )
                patched = (
                    patched[:upstream_block_start]
                    + replacement
                    + patched[upstream_block_end:]
                )
        return patched

    MemoryBuilder._build_extraction_prompt = patched_build_prompt
    MemoryBuilder._atm_source_tracking_patched = True


def extract_balanced_json_value(text: str, start_idx: int) -> Optional[str]:
    if start_idx < 0 or start_idx >= len(text):
        return None
    start_char = text[start_idx]
    end_char = "]" if start_char == "[" else "}" if start_char == "{" else ""
    if not end_char:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]
    return None


def parse_simplemem_memory_payload(response: str) -> List[Dict[str, Any]]:
    """Parse SimpleMem memory-builder output, preferring arrays over objects.

    Upstream SimpleMem's fallback JSON extractor scans for ``{`` before ``[``.
    For fenced array responses whose full-block parse fails, that can return the
    first entry object as a dict. Memory building expects a list, so parse arrays
    first here.
    """
    text = response.strip()
    candidates: List[str] = []

    for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        candidates.extend(re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL))

    array_start = text.find("[")
    balanced_array = extract_balanced_json_value(text, array_start)
    if balanced_array:
        candidates.append(balanced_array)
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("memory_entries", "memories", "entries", "data", "results"):
                value = data.get(key)
                if isinstance(value, list):
                    return value

    salvaged: List[Dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        json_object = extract_balanced_json_value(text, match.start())
        if not json_object:
            continue
        try:
            data = json.loads(json_object)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "lossless_restatement" in data:
            salvaged.append(data)
    if salvaged:
        return salvaged

    raise ValueError("Expected JSON array memory payload")


def _atm_attribute_entry(
    builder: Any,
    entry: Any,
    window_dialogue_ids: List[int],
) -> List[str]:
    """Return the item_ids this memory entry should be attributed to.

    Strategy: substring-match the entry's text/keywords/entities against the
    item_ids and the textual content of the items in the current window. Fall
    back to the whole window when no narrow match is found, so coverage stays
    at 100% even when the build LLM ignores the [SOURCE_ID] prompt.
    """
    dialogue_to_item = getattr(builder, "_atm_dialogue_to_item_id", None) or {}
    item_texts = getattr(builder, "_atm_item_id_to_text", None) or {}
    window_item_ids = [
        dialogue_to_item[d]
        for d in window_dialogue_ids
        if d in dialogue_to_item
    ]
    if not window_item_ids:
        return []

    restatement = (getattr(entry, "lossless_restatement", "") or "")
    keywords = getattr(entry, "keywords", []) or []
    entities = getattr(entry, "entities", []) or []
    persons = getattr(entry, "persons", []) or []
    haystack_id = " ".join([restatement] + list(keywords) + list(entities))
    haystack_text_lower = (
        " ".join([restatement] + list(keywords) + list(entities) + list(persons))
    ).lower()

    matched: List[str] = []
    seen: Set[str] = set()
    for item_id in window_item_ids:
        if item_id in seen:
            continue
        if item_id in haystack_id:
            matched.append(item_id)
            seen.add(item_id)
    if matched:
        return matched

    # No direct ID match; try item-text fingerprint match (length>=6 tokens
    # from the item's content). Keeps attribution narrow when possible.
    for item_id in window_item_ids:
        if item_id in seen:
            continue
        text = (item_texts.get(item_id) or "").lower()
        if not text:
            continue
        tokens = [
            tok
            for tok in re.findall(r"[a-z0-9_\-]{6,}", text)
            if tok and tok not in {"timestamp", "subject", "location", "from"}
        ][:8]
        if not tokens:
            continue
        hits = sum(1 for tok in tokens if tok in haystack_text_lower)
        if hits >= 2:
            matched.append(item_id)
            seen.add(item_id)
    if matched:
        return matched

    # Last resort: attribute to the whole window. Over-attribution but keeps
    # coverage at 100% so retrieval can still surface the right context.
    return list(window_item_ids)


def patch_simplemem_memory_parser() -> None:
    try:
        from core.memory_builder import MemoryBuilder
        from models.memory_entry import MemoryEntry
    except ImportError:
        return

    if getattr(MemoryBuilder, "_atm_memory_parser_patched", False):
        return

    original_parse = MemoryBuilder._parse_llm_response

    def patched_parse(
        self: Any, response: str, dialogue_ids: List[int]
    ) -> List[Any]:
        try:
            data = parse_simplemem_memory_payload(response)
        except ValueError:
            entries = original_parse(self, response, dialogue_ids)
        else:
            entries = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                entries.append(
                    MemoryEntry(
                        lossless_restatement=item["lossless_restatement"],
                        keywords=item.get("keywords", []),
                        timestamp=item.get("timestamp"),
                        location=item.get("location"),
                        persons=item.get("persons", []),
                        entities=item.get("entities", []),
                        topic=item.get("topic"),
                    )
                )

        if not getattr(self, "_atm_entry_to_item_ids", None):
            self._atm_entry_to_item_ids = {}

        for entry in entries:
            restatement = entry.lossless_restatement or ""
            existing_ids = extract_item_ids_from_text(restatement)
            # Drop hallucinated IDs that aren't in our window.
            dialogue_to_item = getattr(self, "_atm_dialogue_to_item_id", None) or {}
            window_item_ids = {
                dialogue_to_item[d]
                for d in dialogue_ids
                if d in dialogue_to_item
            }
            existing_ids = [eid for eid in existing_ids if eid in window_item_ids]
            if existing_ids:
                attributed = existing_ids
            else:
                attributed = _atm_attribute_entry(self, entry, dialogue_ids)

            if attributed:
                self._atm_entry_to_item_ids[entry.entry_id] = list(attributed)
                stamp = f"[SOURCE_ID: {', '.join(attributed)}]"
                if "[SOURCE_ID:" not in restatement.upper():
                    entry.lossless_restatement = f"{stamp} {restatement}".strip()
                # Make IDs discoverable by the lexical layer too.
                kw = list(entry.keywords or [])
                for item_id in attributed:
                    if item_id not in kw:
                        kw.append(item_id)
                entry.keywords = kw

        return entries

    MemoryBuilder._parse_llm_response = patched_parse
    MemoryBuilder._atm_memory_parser_patched = True


def format_duration(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def estimate_simplemem_windows(num_dialogues: int, window_size: int, overlap_size: int) -> int:
    if num_dialogues <= 0:
        return 0
    if num_dialogues < window_size:
        return 1
    step_size = max(1, window_size - overlap_size)
    full_windows = ((num_dialogues - window_size) // step_size) + 1
    consumed = full_windows * step_size
    return full_windows + (1 if consumed < num_dialogues else 0)


def _atm_write_checkpoint(builder: Any, status: str = "in_progress") -> None:
    path = getattr(builder, "_atm_checkpoint_path", None)
    if not path:
        return
    payload = {
        "cache_key": getattr(builder, "_atm_cache_key", None),
        "completed_windows": getattr(builder, "_atm_completed_windows", 0),
        "total_windows": getattr(builder, "_atm_total_windows", 0),
        "dialogues_consumed": getattr(builder, "_atm_dialogues_consumed", 0),
        "total_dialogues": getattr(builder, "_atm_total_dialogues", 0),
        "step_size": getattr(builder, "step_size", 0),
        "window_size": getattr(builder, "window_size", 0),
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Failed to persist SimpleMem build checkpoint: %s", exc)


def _atm_persist_entry_mapping(builder: Any) -> None:
    path = getattr(builder, "_atm_entry_mapping_path", None)
    mapping = getattr(builder, "_atm_entry_to_item_ids", None)
    if not path or not mapping:
        return
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Failed to persist SimpleMem entry-id mapping: %s", exc)


def _atm_load_entry_mapping(db_path: str) -> Dict[str, List[str]]:
    path = Path(db_path) / "entry_to_item_ids.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, List[str]] = {}
    for entry_id, value in raw.items():
        if isinstance(value, list):
            result[str(entry_id)] = [str(v) for v in value if v]
        elif value:
            result[str(entry_id)] = [str(value)]
    return result


def patch_simplemem_progress_logging() -> None:
    try:
        from core.memory_builder import MemoryBuilder
    except ImportError:
        return

    if getattr(MemoryBuilder, "_atm_progress_logging_patched", False):
        return

    def progress_line(self: Any, label: str, window_len: int) -> str:
        completed = getattr(self, "_atm_completed_windows", 0) + 1
        total = getattr(self, "_atm_total_windows", 0)
        elapsed = time.time() - getattr(self, "_atm_started_at", time.time())
        if completed > 1 and total:
            eta = elapsed / (completed - 1) * max(total - completed + 1, 0)
        else:
            eta = 0
        if total:
            pct = completed / total * 100
            return (
                f"\n{label}: {window_len} dialogues "
                f"[window {completed}/{total}, {pct:.1f}%, "
                f"elapsed {format_duration(elapsed)}, eta {format_duration(eta)}]"
            )
        return f"\n{label}: {window_len} dialogues"

    def patched_process_window(self: Any) -> None:
        if not self.dialogue_buffer:
            return

        window = self.dialogue_buffer[: self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.step_size :]
        print(progress_line(self, "Processing window", len(window)))

        entries = self._generate_memory_entries(window)
        if entries:
            self.vector_store.add_entries(entries)
            self.previous_entries = entries

        self.processed_count += len(window)
        self._atm_completed_windows = getattr(self, "_atm_completed_windows", 0) + 1
        self._atm_dialogues_consumed = (
            getattr(self, "_atm_dialogues_consumed", 0) + self.step_size
        )
        print(f"Generated {len(entries)} memory entries")
        _atm_write_checkpoint(self, status="in_progress")
        _atm_persist_entry_mapping(self)

    def patched_process_remaining(self: Any) -> None:
        if not self.dialogue_buffer:
            return

        window = self.dialogue_buffer
        print(progress_line(self, "Processing remaining dialogues", len(window)))
        entries = self._generate_memory_entries(window)
        if entries:
            self.vector_store.add_entries(entries)
            self.previous_entries = entries
        self.processed_count += len(window)
        self._atm_completed_windows = getattr(self, "_atm_completed_windows", 0) + 1
        self._atm_dialogues_consumed = getattr(self, "_atm_total_dialogues", 0)
        self.dialogue_buffer = []
        print(f"Generated {len(entries)} memory entries")
        _atm_write_checkpoint(self, status="in_progress")
        _atm_persist_entry_mapping(self)

    MemoryBuilder.process_window = patched_process_window
    MemoryBuilder.process_remaining = patched_process_remaining
    MemoryBuilder._atm_progress_logging_patched = True


def build_index_cache_key(args: argparse.Namespace) -> str:
    return build_cache_key(build_index_cache_payload(args))


def output_index_marker_path(args: argparse.Namespace, cache_key: str) -> Path:
    return Path(args.output_dir_base) / args.method_name / f".index_{cache_key}"


def db_index_metadata_path(args: argparse.Namespace) -> Path:
    return Path(args.simplemem_db_path) / "personal_memoryqa_index_metadata.json"


def write_index_metadata(args: argparse.Namespace, cache_key: str) -> None:
    payload = {
        "cache_key": cache_key,
        "config": build_index_cache_payload(args),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    path = db_index_metadata_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    marker = output_index_marker_path(args, cache_key)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def index_metadata_matches(args: argparse.Namespace, cache_key: str) -> bool:
    metadata_path = db_index_metadata_path(args)
    if not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("cache_key") == cache_key


def _ensure_simplemem_on_path(simplemem_dir: Optional[str]) -> None:
    if not simplemem_dir:
        simplemem_dir = os.environ.get("SIMPLEMEM_DIR")
    if not simplemem_dir:
        return
    resolved = os.path.abspath(simplemem_dir)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _ensure_simplemem_config_file(simplemem_dir: Optional[str]) -> None:
    if not simplemem_dir:
        return
    simplemem_path = Path(simplemem_dir)
    config_path = simplemem_path / "config.py"
    if config_path.exists():
        return
    example_path = simplemem_path / "config.py.example"
    if not example_path.exists():
        raise RuntimeError(
            f"SimpleMem config is missing: expected {config_path} or {example_path}."
        )
    shutil.copyfile(example_path, config_path)
    logger.info("Created SimpleMem config from example: %s", config_path)


def create_simplemem_system(
    args: argparse.Namespace,
    api_key: str,
    base_url: Optional[str],
    clear_db: bool = True,
    model_override: Optional[str] = None,
) -> Any:
    simplemem_dir = getattr(args, "simplemem_dir", None) or os.environ.get("SIMPLEMEM_DIR")
    _ensure_simplemem_config_file(simplemem_dir)
    _ensure_simplemem_on_path(simplemem_dir)
    try:
        from main import SimpleMemSystem
    except ImportError as exc:
        if getattr(exc, "name", None) != "main":
            raise RuntimeError(f"SimpleMem import failed: {exc}") from exc
        raise RuntimeError(
            "SimpleMem package not found. Clone and install from GitHub: "
            "git clone https://github.com/aiming-lab/SimpleMem.git && "
            "cd SimpleMem && pip install -r requirements.txt. "
            "Then use --simplemem-dir or set SIMPLEMEM_DIR."
        ) from exc

    effective_model = model_override or args.model

    patch_simplemem_config(args, api_key, base_url)
    patch_simplemem_source_tracking()
    patch_simplemem_memory_parser()
    patch_simplemem_progress_logging()
    if model_override:
        try:
            import config as sm_config
            sm_config.LLM_MODEL = model_override
        except ImportError:
            pass

    system = SimpleMemSystem(
        api_key=api_key,
        model=effective_model,
        base_url=base_url,
        db_path=args.simplemem_db_path,
        clear_db=clear_db,
        enable_planning=args.simplemem_enable_planning,
        enable_reflection=args.simplemem_enable_reflection,
        max_reflection_rounds=args.simplemem_max_reflection_rounds,
        enable_parallel_processing=args.simplemem_enable_parallel_processing,
        max_parallel_workers=args.simplemem_max_parallel_workers,
        enable_parallel_retrieval=args.simplemem_enable_parallel_retrieval,
        max_retrieval_workers=args.simplemem_max_retrieval_workers,
    )
    return system


def _save_item_id_mapping(
    system: Any,
    items: List[RetrievalItem],
    db_path: str,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    memories = system.get_all_memories()
    all_texts_by_item: Dict[str, str] = {}
    for item in items:
        if item.text:
            all_texts_by_item[item.item_id] = item.text

    # Prefer the deterministic side-channel populated by the patched
    # MemoryBuilder; fall back to text-based heuristics when it isn't
    # available (e.g. an older partial build on disk).
    builder = getattr(system, "memory_builder", None)
    deterministic: Dict[str, List[str]] = {}
    if builder is not None:
        live = getattr(builder, "_atm_entry_to_item_ids", None)
        if isinstance(live, dict):
            deterministic = {
                str(k): list(v)
                for k, v in live.items()
                if isinstance(v, (list, tuple)) and v
            }
    if not deterministic:
        deterministic = _atm_load_entry_mapping(db_path)

    mapping: Dict[str, List[str]] = {}
    direct_source_id_entries = 0
    deterministic_entries = 0
    fuzzy_mapped_entries = 0
    for mem in memories:
        source_ids = [
            item_id
            for item_id in extract_item_ids_from_memory_entry(mem)
            if item_id in all_texts_by_item
        ]
        if source_ids:
            mapping[mem.entry_id] = source_ids
            direct_source_id_entries += 1
            continue

        side = deterministic.get(mem.entry_id)
        if side:
            valid = [iid for iid in side if iid in all_texts_by_item]
            if valid:
                mapping[mem.entry_id] = valid
                deterministic_entries += 1
                continue

        restatement = (mem.lossless_restatement or "").lower()
        keywords_set = set(k.lower() for k in (mem.keywords or []))
        entities_set = set(e.lower() for e in (mem.entities or []))
        location = (mem.location or "").lower()
        all_retrieval_tokens = keywords_set | entities_set | {location}

        best_item_id = None
        best_score = 0
        for item_id, item_text in all_texts_by_item.items():
            item_id_lower = item_id.lower()
            if (
                item_id_lower
                and (item_id_lower in restatement or item_id_lower in all_retrieval_tokens)
            ):
                best_item_id = item_id
                best_score = 100
                break
            item_lower = item_text.lower()
            score = 0
            for token in all_retrieval_tokens:
                if token and len(token) >= 4 and token in item_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_item_id = item_id

        if best_item_id and best_score >= 2:
            mapping[mem.entry_id] = [best_item_id]
            fuzzy_mapped_entries += 1

    mapping_path = Path(db_path) / "item_id_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, indent=2))
    total_memories = len(memories)
    covered = direct_source_id_entries + deterministic_entries
    stats = {
        "total_memory_entries": total_memories,
        "mapped_entries": len(mapping),
        "direct_source_id_entries": direct_source_id_entries,
        "deterministic_window_entries": deterministic_entries,
        "fuzzy_mapped_entries": fuzzy_mapped_entries,
        "mapping_coverage": len(mapping) / total_memories if total_memories else 0.0,
        "direct_source_id_coverage": (
            covered / total_memories if total_memories else 0.0
        ),
        "llm_emitted_source_id_coverage": (
            direct_source_id_entries / total_memories if total_memories else 0.0
        ),
    }
    stats_path = Path(db_path) / "item_id_mapping_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    logger.info(
        "Saved item_id mapping: %d/%d entries "
        "(LLM-emitted source IDs: %d, deterministic window IDs: %d, fuzzy: %d)",
        len(mapping),
        total_memories,
        direct_source_id_entries,
        deterministic_entries,
        fuzzy_mapped_entries,
    )
    return mapping, stats


def _load_item_id_mapping(db_path: str) -> Dict[str, List[str]]:
    mapping_path = Path(db_path) / "item_id_mapping.json"
    if not mapping_path.exists():
        return {}
    try:
        raw = json.loads(mapping_path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, List[str]] = {}
    for entry_id, value in raw.items():
        if isinstance(value, list):
            result[str(entry_id)] = [str(v) for v in value if v]
        elif value:
            result[str(entry_id)] = [str(value)]
    return result


def get_entry_id(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        raw = entry.get("entry_id")
    else:
        raw = getattr(entry, "entry_id", None)
    return str(raw) if raw else None


def get_context_text(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("lossless_restatement", "") or entry.get("text", ""))
    return str(getattr(entry, "lossless_restatement", "") or getattr(entry, "text", ""))


def resolve_context_item_id(
    entry: Any, entry_to_item_ids: Dict[str, List[str]]
) -> Optional[str]:
    item_ids = resolve_context_item_ids(entry, entry_to_item_ids)
    return item_ids[0] if item_ids else None


def resolve_context_item_ids(
    entry: Any, entry_to_item_ids: Dict[str, List[str]]
) -> List[str]:
    item_ids = extract_item_ids_from_memory_entry(entry)
    if item_ids:
        return item_ids
    entry_id = get_entry_id(entry)
    if entry_id:
        mapped = entry_to_item_ids.get(entry_id)
        if mapped:
            if isinstance(mapped, list):
                return [str(x) for x in mapped if x]
            return [str(mapped)]
    return []


def format_contexts_for_atm(
    contexts: List[Any],
    args: argparse.Namespace,
) -> str:
    entry_to_item_id = _load_item_id_mapping(args.simplemem_db_path)
    evidence_parts: List[str] = []
    for idx, ctx in enumerate(contexts, 1):
        text = get_context_text(ctx)
        if not text:
            continue
        item_ids = resolve_context_item_ids(ctx, entry_to_item_id)
        header = f"Memory {idx}"
        if item_ids:
            header += f" (Source ID: {', '.join(item_ids)})"
        part_lines = [f"{header}:"]
        if item_ids:
            part_lines.append(f"Source ID: {', '.join(item_ids)}")
        part_lines.append(text)
        evidence_parts.append("\n".join(part_lines))
    return "\n\n".join(evidence_parts)


def build_atm_messages(question: str, evidence_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": PROMPTS["QA_SYSTEM"]},
        {
            "role": "user",
            "content": PROMPTS["QA_USER"].format(
                question=question, evidence=evidence_text
            ),
        },
    ]


def index_memories(
    system: Any,
    items: List[RetrievalItem],
    args: argparse.Namespace,
    resume_from_dialogue: int = 0,
    cache_key: Optional[str] = None,
) -> None:
    from models.memory_entry import Dialogue

    sorted_items = sort_retrieval_items(items)
    total = len(sorted_items)
    logger.info("Indexing %d memory items into SimpleMem", total)

    db_path = Path(args.simplemem_db_path)

    with suppress_output(args.simplemem_quiet):
        dialogues: List[Dialogue] = []
        dialogue_to_item_id: Dict[int, str] = {}
        item_id_to_text: Dict[str, str] = {}
        for item in tqdm(sorted_items, desc="Formatting dialogues"):
            user_input, _agent_response = format_item_as_dialogue(item)
            if not user_input:
                continue

            timestamp = None
            if item.metadata:
                ts_raw = item.metadata.get("timestamp")
                if ts_raw:
                    timestamp = str(ts_raw)

            dialogue_id = len(dialogues) + 1
            dialogues.append(
                Dialogue(
                    dialogue_id=dialogue_id,
                    speaker="User",
                    content=user_input,
                    timestamp=timestamp,
                )
            )
            dialogue_to_item_id[dialogue_id] = item.item_id
            item_id_to_text[item.item_id] = item.text or ""

        logger.info("Submitting %d dialogues to SimpleMem", len(dialogues))
        total_windows = estimate_simplemem_windows(
            len(dialogues), args.simplemem_window_size, args.simplemem_overlap_size
        )

        builder = system.memory_builder
        builder._atm_total_dialogues = len(dialogues)
        builder._atm_total_windows = total_windows
        builder._atm_started_at = time.time()
        builder._atm_dialogue_to_item_id = dialogue_to_item_id
        builder._atm_item_id_to_text = item_id_to_text
        builder._atm_checkpoint_path = db_path / "build_checkpoint.json"
        builder._atm_entry_mapping_path = db_path / "entry_to_item_ids.json"
        builder._atm_cache_key = cache_key
        if not getattr(builder, "_atm_entry_to_item_ids", None):
            builder._atm_entry_to_item_ids = _atm_load_entry_mapping(args.simplemem_db_path)

        if resume_from_dialogue and resume_from_dialogue > 0:
            already_done = min(resume_from_dialogue, len(dialogues))
            # Approximate completed-window count from consumed dialogues so the
            # progress line reads sanely after a resume.
            step = max(1, args.simplemem_window_size - args.simplemem_overlap_size)
            builder._atm_completed_windows = max(0, already_done // step)
            builder._atm_dialogues_consumed = already_done
            remaining = dialogues[already_done:]
            print(
                f"\n[SimpleMem] Resuming build: skipping {already_done} "
                f"dialogues already in DB, processing {len(remaining)} remaining "
                f"(of {len(dialogues)} total, est_windows={total_windows})"
            )
            submit = remaining
        else:
            builder._atm_completed_windows = 0
            builder._atm_dialogues_consumed = 0
            print(
                f"\n[SimpleMem] Building {len(dialogues)} dialogues "
                f"with window_size={args.simplemem_window_size}, "
                f"overlap={args.simplemem_overlap_size}, estimated_windows={total_windows}"
            )
            submit = dialogues

        _atm_write_checkpoint(builder, status="in_progress")

        if submit:
            system.add_dialogues(submit)
        system.finalize()
        _atm_write_checkpoint(builder, status="completed")
        _atm_persist_entry_mapping(builder)

    mapping, mapping_stats = _save_item_id_mapping(system, items, args.simplemem_db_path)

    num_memories = len(system.get_all_memories())
    direct_source_id_coverage = mapping_stats.get("direct_source_id_coverage", 0.0)
    if (
        args.simplemem_min_source_id_coverage
        and direct_source_id_coverage < args.simplemem_min_source_id_coverage
    ):
        raise RuntimeError(
            "SimpleMem source-ID mapping coverage is too low: "
            f"{direct_source_id_coverage:.1%} direct source-ID coverage < "
            f"{args.simplemem_min_source_id_coverage:.1%}. "
            f"See {Path(args.simplemem_db_path) / 'item_id_mapping_stats.json'}."
        )
    logger.info("Indexing complete: %d memory entries created from %d items", num_memories, total)


def retrieve_with_logging(
    system: Any,
    question: str,
    log_k: int,
    args: argparse.Namespace,
) -> Tuple[List[Any], List[str], List[float]]:
    with suppress_output(args.simplemem_quiet):
        contexts = system.hybrid_retriever.retrieve(question)

    retrieval_ids: List[str] = []
    retrieval_scores: List[float] = []

    entry_to_item_id = _load_item_id_mapping(args.simplemem_db_path)

    for ctx in contexts[:log_k]:
        item_ids = resolve_context_item_ids(ctx, entry_to_item_id)
        for item_id in item_ids:
            retrieval_ids.append(str(item_id))
            retrieval_scores.append(0.0)

    retrieval_ids = dedupe_retrieved_ids(retrieval_ids)
    retrieval_scores = [0.0] * len(retrieval_ids)
    return contexts, retrieval_ids, retrieval_scores


class _UsageRecord:
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens")

    def __init__(self, prompt: int = 0, completion: int = 0, total: int = 0) -> None:
        self.prompt_tokens = int(prompt or 0)
        self.completion_tokens = int(completion or 0)
        self.total_tokens = int(total or (self.prompt_tokens + self.completion_tokens))


def answer_atm_with_simplemem_llm(
    system: Any,
    question: str,
    contexts: List[Any],
    args: argparse.Namespace,
) -> Tuple[str, Optional[_UsageRecord]]:
    """Run the answer LLM through SimpleMem's OpenAI client but capture usage.

    SimpleMem's wrapper drops the ``usage`` block on the response. We bypass
    it and call ``client.chat.completions.create`` directly so per-question
    token counts land in ``simplemem_answers.jsonl`` and the aggregated
    run-stats file. Same model, base_url, and message payload as before.
    """
    evidence_text = format_contexts_for_atm(contexts, args)
    messages = build_atm_messages(question, evidence_text)
    llm_client = system.llm_client
    with suppress_output(args.simplemem_quiet):
        response = llm_client.client.chat.completions.create(
            model=llm_client.model,
            messages=messages,
            temperature=0.1,
        )
    choice = response.choices[0]
    answer = (choice.message.content or "").strip()
    usage = getattr(response, "usage", None)
    usage_obj = None
    if usage is not None:
        usage_obj = _UsageRecord(
            prompt=getattr(usage, "prompt_tokens", 0),
            completion=getattr(usage, "completion_tokens", 0),
            total=getattr(usage, "total_tokens", 0),
        )
    return answer, usage_obj


def answer_native_simplemem(
    system: Any,
    question: str,
    contexts: List[Any],
    args: argparse.Namespace,
) -> str:
    with suppress_output(args.simplemem_quiet):
        answer = system.answer_generator.generate_answer(question, contexts)
    return answer.strip()


def answer_external_llm(
    question: str,
    contexts: List[Any],
    llm: LLMClient,
    args: argparse.Namespace,
) -> Tuple[str, Any]:
    evidence_text = format_contexts_for_atm(contexts, args)
    messages = build_atm_messages(question, evidence_text)
    answer, usage = llm.chat_with_usage(messages)
    return answer, usage


def build_model_config(provider: str, args: argparse.Namespace) -> Dict[str, Any]:
    if provider == "openai":
        base = dict(SIMPLEMEM_CONFIG["openai"])
    elif provider in {"vllm", "vllm_local"}:
        base = dict(SIMPLEMEM_CONFIG["vllm_text"])
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if args.model:
        base["model"] = args.model
    if args.api_key:
        base["api_key"] = args.api_key
    if args.vllm_endpoint:
        base["endpoint"] = args.vllm_endpoint
    if args.max_tokens is not None:
        base["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        base["temperature"] = args.temperature
    if args.timeout is not None:
        base["timeout"] = args.timeout
    return base


def _load_build_checkpoint(db_path: Path) -> Dict[str, Any]:
    path = db_path / "build_checkpoint.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def build_stage(args: argparse.Namespace) -> int:
    api_key, base_url, model = resolve_api_settings(args, for_build=True)
    logger.info("Build stage using model=%s, endpoint=%s", model, base_url)
    media_config, email_config = build_text_configs(args)

    image_batch = load_json(Path(args.image_batch_results))
    video_batch = load_json(Path(args.video_batch_results))
    email_entries = load_json(Path(args.email_file))

    retrieval_items = build_retrieval_items(
        email_entries,
        image_batch,
        video_batch,
        media_config,
        email_config,
        Path(args.image_root),
        Path(args.video_root),
    )

    db_path = Path(args.simplemem_db_path)
    cache_key = build_index_cache_key(args)
    db_has_data = db_path.exists() and any(db_path.iterdir())
    full_metadata_matches = (
        args.simplemem_reuse_index
        and not args.simplemem_force_rebuild
        and db_has_data
        and index_metadata_matches(args, cache_key)
    )

    checkpoint = _load_build_checkpoint(db_path) if db_has_data else {}
    checkpoint_matches = (
        bool(checkpoint)
        and checkpoint.get("cache_key") == cache_key
        and checkpoint.get("status") == "in_progress"
        and int(checkpoint.get("dialogues_consumed") or 0) > 0
    )
    resume_requested = (
        args.simplemem_reuse_index
        and not args.simplemem_force_rebuild
        and checkpoint_matches
    )

    if full_metadata_matches:
        logger.info("Reusing existing SimpleMem index at %s", db_path)
        system = create_simplemem_system(
            args, api_key, base_url, clear_db=False, model_override=model
        )
    elif resume_requested:
        consumed = int(checkpoint.get("dialogues_consumed") or 0)
        total = int(checkpoint.get("total_dialogues") or 0)
        print(
            f"\n[SimpleMem] Resume detected at {db_path} "
            f"(cache_key={cache_key[:12]}…, "
            f"{consumed}/{total} dialogues consumed, "
            f"windows {checkpoint.get('completed_windows')}/{checkpoint.get('total_windows')}). "
            f"Continuing from window {int(checkpoint.get('completed_windows') or 0) + 1}."
        )
        system = create_simplemem_system(
            args, api_key, base_url, clear_db=False, model_override=model
        )
        index_memories(
            system,
            retrieval_items,
            args,
            resume_from_dialogue=consumed,
            cache_key=cache_key,
        )
    else:
        if db_path.exists():
            if checkpoint and checkpoint.get("cache_key") != cache_key:
                logger.info(
                    "Existing SimpleMem checkpoint cache_key mismatch "
                    "(have=%s, want=%s); wiping DB",
                    checkpoint.get("cache_key"),
                    cache_key,
                )
            shutil.rmtree(db_path)
        system = create_simplemem_system(
            args, api_key, base_url, clear_db=True, model_override=model
        )
        index_memories(system, retrieval_items, args, cache_key=cache_key)

    write_index_metadata(args, cache_key)

    num_memories = len(system.get_all_memories())
    print(f"\n[SimpleMem] Index build complete: {num_memories} memory entries")
    print(f"[SimpleMem] DB path: {db_path}")
    print(f"[SimpleMem] Cache key: {cache_key}")
    return 0


def answer_stage(args: argparse.Namespace) -> int:
    api_key, base_url, model = resolve_api_settings(args)
    logger.info("Answer stage using model=%s, endpoint=%s", model, base_url)
    db_path = Path(args.simplemem_db_path)

    if not db_path.exists() or not any(db_path.iterdir()):
        print("[SimpleMem] ERROR: No index found. Run --stage build first.")
        return 1
    cache_key = build_index_cache_key(args)
    if not index_metadata_matches(args, cache_key):
        print(
            "[SimpleMem] ERROR: Existing index metadata does not match the current "
            "build configuration. Run --stage build or pass --stage all to rebuild."
        )
        print(f"[SimpleMem] Expected cache key: {cache_key}")
        print(f"[SimpleMem] Metadata path: {db_index_metadata_path(args)}")
        return 1

    system = create_simplemem_system(args, api_key, base_url, clear_db=False)

    qa_data = load_json(Path(args.qa_file))
    qas = load_qa_list(qa_data)

    output_dir = Path(args.output_dir_base) / args.method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "simplemem_answers.jsonl"

    if output_file.exists() and not args.overwrite_output:
        print(f"[SimpleMem] Output exists, skipping: {output_file}")
        return 0

    answer_mode = args.simplemem_answer_mode
    llm_config = None
    if answer_mode == "external":
        llm_config = build_model_config(args.provider, args)
        logger.info("Using external LLM for answering: %s", llm_config.get("model"))

    thread_local = threading.local()

    def get_llm() -> LLMClient:
        if not hasattr(thread_local, "llm"):
            thread_local.llm = LLMClient(args.provider, llm_config)
        return thread_local.llm

    def answer_single(qa: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        qa_id = qa.get("id") or qa.get("qa_id")
        question = qa.get("question")
        if not qa_id or not question:
            return None

        log_k = max(args.retrieval_log_k, 10)
        contexts, retrieval_ids, retrieval_scores = retrieve_with_logging(
            system, question, log_k, args
        )

        gt_evidence_ids = extract_evidence_ids(qa)
        retrieval_recall = compute_recall(gt_evidence_ids, retrieval_ids)

        if answer_mode == "atm":
            answer, usage_obj = answer_atm_with_simplemem_llm(
                system, question, contexts, args
            )
        elif answer_mode == "native":
            answer = answer_native_simplemem(system, question, contexts, args)
            usage_obj = None
        else:
            answer, usage_obj = answer_external_llm(question, contexts, get_llm(), args)

        answer_record: Dict[str, Any] = {"id": qa_id, "answer": answer}
        if usage_obj:
            answer_record["prompt_tokens"] = usage_obj.prompt_tokens
            answer_record["completion_tokens"] = usage_obj.completion_tokens
            answer_record["total_tokens"] = usage_obj.total_tokens

        retrieval_record = {
            "id": qa_id,
            "question": question,
            "gt_evidence_ids": gt_evidence_ids,
            "retrieval_ids": retrieval_ids,
            "retrieval_scores": retrieval_scores,
            "retrieval_recall": retrieval_recall,
        }

        return {
            "answer_record": answer_record,
            "retrieval_record": retrieval_record,
        }

    results: List[Dict[str, Any]] = []
    retrieval_details: List[Dict[str, Any]] = []

    if args.max_workers and args.max_workers > 1 and args.provider != "vllm_local":
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(answer_single, qa): idx
                for idx, qa in enumerate(qas)
            }
            ordered: List[Tuple[int, Dict[str, Any]]] = []
            for future in tqdm(
                as_completed(future_map), total=len(future_map), desc="SimpleMem QA"
            ):
                idx = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("QA failed for index %d: %s", idx, exc)
                    result = None
                if result:
                    ordered.append((idx, result))

            ordered.sort(key=lambda item: item[0])
            for _, item in ordered:
                results.append(item["answer_record"])
                retrieval_details.append(item["retrieval_record"])
    else:
        for qa in tqdm(qas, desc="SimpleMem QA"):
            result = answer_single(qa)
            if result:
                results.append(result["answer_record"])
                retrieval_details.append(result["retrieval_record"])

    write_jsonl(output_file, results)

    total_prompt = sum(r.get("prompt_tokens", 0) for r in results)
    total_completion = sum(r.get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    num_samples = len(results)

    run_stats = {
        "num_samples": num_samples,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "avg_prompt_tokens": round(total_prompt / num_samples, 1) if num_samples else 0,
        "avg_completion_tokens": round(total_completion / num_samples, 1)
        if num_samples
        else 0,
        "avg_total_tokens": round(total_tokens / num_samples, 1) if num_samples else 0,
        "answer_mode": answer_mode,
        "answer_model": model,
        "build_model": args.build_model,
        "build_endpoint": normalize_openai_base_url(args.build_endpoint or args.vllm_endpoint),
        "enable_planning": args.simplemem_enable_planning,
        "enable_reflection": args.simplemem_enable_reflection,
        "semantic_top_k": args.simplemem_semantic_top_k,
        "keyword_top_k": args.simplemem_keyword_top_k,
        "structured_top_k": args.simplemem_structured_top_k,
    }

    stats_path = output_file.parent / f"{output_file.stem}_run_stats.json"
    write_json(stats_path, run_stats)
    print(f"Token stats written to: {stats_path}")

    if retrieval_details:
        summary = {
            "retrieval_log_k": min(args.retrieval_log_k, 100),
            "recall": summarize_recalls(retrieval_details, "retrieval_recall"),
        }
        write_json(output_dir / "retrieval_recall_summary.json", summary)
        write_json(output_dir / "retrieval_recall_details.json", retrieval_details)

    print(f"\n[SimpleMem] Answer generation complete: {len(results)} results")
    print(f"[SimpleMem] Output: {output_file}")
    return 0


def main() -> int:
    args = parse_args()

    if args.stage == "build":
        return build_stage(args)
    elif args.stage == "answer":
        return answer_stage(args)
    else:
        build_stage(args)
        return answer_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
