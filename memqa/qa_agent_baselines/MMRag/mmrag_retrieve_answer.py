#!/usr/bin/env python3
"""MMRag retrieve-answer baseline."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from memqa.qa_agent_baselines.MMRag.config import MMRAG_CONFIG, RECALL_KS
from memqa.qa_agent_baselines.MMRag.llm_utils import LLMClient
from memqa.qa_agent_baselines.MMRag.mmrag_utils import (
    append_jsonl,
    build_agentic_critic_messages,
    build_multimodal_messages,
    build_text_evidence,
    build_text_messages,
    is_failed_record,
    load_resume_map,
    merge_token_usage,
    parse_agentic_critic_response,
)
from memqa.retrieve import (
    EmailTextConfig,
    MediaTextConfig,
    RetrievalItem,
    build_cache_key,
    extract_evidence_ids,
    load_qa_list,
    load_json,
    minimal_email_text_config,
    minimal_media_text_config,
    build_retrieval_items,
    write_json,
    write_jsonl,
)
from memqa.retrieve.retrievers import (
    ClipRetriever,
    Qwen3VLRetriever,
    SentenceTransformerRetriever,
    TextRetriever,
    VistaRetriever,
)

VL_RETRIEVERS = {"qwen3_vl_embedding", "vista", "clip"}


def is_vl_retriever(retriever_name: str) -> bool:
    return retriever_name in VL_RETRIEVERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMRag retrieve-answer baseline")
    parser.add_argument("--qa-file", required=True, help="Path to QA annotations JSON")
    parser.add_argument(
        "--media-source",
        choices=["batch_results", "raw"],
        default=MMRAG_CONFIG["media_source"],
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        default=MMRAG_CONFIG["no_evidence"],
        help="Do not provide evidence to the model",
    )
    parser.add_argument(
        "--image-batch-results",
        default=MMRAG_CONFIG["image_batch_results"],
        help="Path to image batch_results.json",
    )
    parser.add_argument(
        "--video-batch-results",
        default=MMRAG_CONFIG["video_batch_results"],
        help="Path to video batch_results.json",
    )
    parser.add_argument(
        "--image-root",
        default=MMRAG_CONFIG["image_root"],
        help="Root directory for raw images",
    )
    parser.add_argument(
        "--video-root",
        default=MMRAG_CONFIG["video_root"],
        help="Root directory for raw videos",
    )
    parser.add_argument(
        "--email-file",
        default=MMRAG_CONFIG["email_file"],
        help="Path to merged_emails.json",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "vllm", "vllm_local"],
        default=MMRAG_CONFIG["provider"],
    )
    parser.add_argument("--model", default=None, help="Model name (overrides config)")
    parser.add_argument("--api-key", default=None, help="API key (overrides config)")
    parser.add_argument("--vllm-endpoint", default=None, help="VLLM endpoint URL")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout seconds")
    parser.add_argument(
        "--max-evidence-items", type=int, default=MMRAG_CONFIG["max_evidence_items"]
    )
    parser.add_argument("--num-frames", type=int, default=MMRAG_CONFIG["num_frames"])
    parser.add_argument(
        "--max-total-frames",
        type=int,
        default=MMRAG_CONFIG["max_total_frames"],
        help="Cap total frames extracted across all videos (default: num-frames)",
    )
    parser.add_argument("--frame-strategy", default=MMRAG_CONFIG["frame_strategy"])
    parser.add_argument("--max-workers", type=int, default=MMRAG_CONFIG["max_workers"])
    parser.add_argument("--output-dir-base", default=MMRAG_CONFIG["output_dir_base"])
    parser.add_argument(
        "--method-name", default="debug_method", help="Method identifier for output"
    )

    parser.add_argument(
        "--retriever",
        choices=[
            "qwen3_vl_embedding",
            "vista",
            "clip",
            "text",
            "sentence_transformer",
        ],
        default=MMRAG_CONFIG["retriever"],
    )
    parser.add_argument(
        "--retriever-batch-size", type=int, default=MMRAG_CONFIG["retriever_batch_size"]
    )
    parser.add_argument(
        "--retrieval-top-k", type=int, default=MMRAG_CONFIG["retrieval_top_k"]
    )
    parser.add_argument(
        "--retrieval-max-k", type=int, default=MMRAG_CONFIG["retrieval_max_k"]
    )
    parser.add_argument("--index-cache", default=MMRAG_CONFIG["index_cache_dir"])
    parser.add_argument("--force-rebuild", action="store_true", default=False)
    parser.add_argument(
        "--reuse-retrieval-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached retrieval results when available",
    )

    parser.add_argument(
        "--text-embedding-model", default=MMRAG_CONFIG["text_embedding_model"]
    )
    parser.add_argument(
        "--vl-embedding-model", default=MMRAG_CONFIG["vl_embedding_model"]
    )
    parser.add_argument("--clip-model", default=MMRAG_CONFIG["clip_model"])
    parser.add_argument("--vista-model-name", default=MMRAG_CONFIG["vista_model_name"])
    parser.add_argument("--vista-weights", default=MMRAG_CONFIG["vista_weights"])

    parser.add_argument(
        "--insert-raw-images",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["insert_raw_images"],
        help="Insert raw images into LLM prompt even for text retrieval",
    )
    parser.add_argument(
        "--vl-text-augment",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["vl_text_augment"],
    )
    parser.add_argument(
        "--include-id",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["include_id"],
    )
    parser.add_argument(
        "--include-type",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["include_type"],
    )
    parser.add_argument(
        "--include-timestamp",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["include_timestamp"],
    )
    parser.add_argument(
        "--include-location",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["include_location"],
    )
    parser.add_argument(
        "--include-short-caption",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["include_short_caption"],
    )
    parser.add_argument(
        "--include-caption",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["include_caption"],
    )
    parser.add_argument(
        "--include-ocr-text",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["include_ocr_text"],
    )
    parser.add_argument(
        "--include-tags",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["include_tags"],
    )
    parser.add_argument(
        "--include-email-summary",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["include_email_summary"],
    )
    parser.add_argument(
        "--include-email-detail",
        action=argparse.BooleanOptionalAction,
        default=MMRAG_CONFIG["include_email_detail"],
    )
    parser.add_argument(
        "--critic-answerer",
        "--agentic-answser",
        "--agentic-answer",
        dest="critic_answerer",
        action="store_true",
        default=MMRAG_CONFIG["critic_answerer"],
        help="Enable critic answerer (draft + critic verification)",
    )

    return parser.parse_args()


def build_model_config(
    provider: str, has_multimodal: bool, args: argparse.Namespace
) -> Dict[str, Any]:
    if provider == "openai":
        base = dict(MMRAG_CONFIG["openai"])
    elif provider in {"vllm", "vllm_local"}:
        base = dict(
            MMRAG_CONFIG["vllm_vl"] if has_multimodal else MMRAG_CONFIG["vllm_text"]
        )
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


def build_cache_config(
    args: argparse.Namespace,
    retriever_name: str,
    media_config: MediaTextConfig,
    email_config: EmailTextConfig,
) -> Dict[str, Any]:
    def safe_mtime(path: str) -> Optional[float]:
        return os.path.getmtime(path) if path and os.path.exists(path) else None

    return {
        "retriever": retriever_name,
        "media_source": args.media_source,
        "text_embedding_model": args.text_embedding_model,
        "vl_embedding_model": args.vl_embedding_model,
        "clip_model": args.clip_model,
        "vista_model_name": args.vista_model_name,
        "vista_weights": args.vista_weights,
        "media_text_config": asdict(media_config),
        "email_text_config": asdict(email_config),
        "image_batch_results": args.image_batch_results,
        "video_batch_results": args.video_batch_results,
        "email_file": args.email_file,
        "image_root": args.image_root,
        "video_root": args.video_root,
        "image_batch_mtime": safe_mtime(args.image_batch_results),
        "video_batch_mtime": safe_mtime(args.video_batch_results),
        "email_mtime": safe_mtime(args.email_file),
    }


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


def select_evidence_items(
    items: List[RetrievalItem], max_items: Optional[int]
) -> List[RetrievalItem]:
    if max_items is None:
        return items
    return items[:max_items]


def build_llm_messages(
    question: str, items: List[RetrievalItem], args: argparse.Namespace
) -> List[Dict[str, Any]]:
    has_media = any(item.modality in {"image", "video"} for item in items)

    # If explicitly forcing raw images or if we have raw media candidates
    force_multimodal = args.insert_raw_images and args.media_source == "batch_results"

    if args.media_source == "batch_results" and not force_multimodal:
        evidence_chunks = build_text_evidence(items)
        return build_text_messages(question, evidence_chunks)

    if not has_media and not force_multimodal:
        evidence_chunks = build_text_evidence(items)
        return build_text_messages(question, evidence_chunks)

    text_chunks = build_text_evidence(items)
    max_total_frames = (
        args.max_total_frames if args.max_total_frames is not None else args.num_frames
    )
    return build_multimodal_messages(
        question,
        items,
        text_chunks,
        args.num_frames,
        max_total_frames,
        args.frame_strategy,
    )


def main() -> int:
    args = parse_args()

    qa_data = load_json(Path(args.qa_file))
    qas = load_qa_list(qa_data)

    media_config, email_config = build_text_configs(args)
    vl_media_config = (
        media_config if args.vl_text_augment else minimal_media_text_config()
    )
    vl_email_config = (
        email_config if args.vl_text_augment else minimal_email_text_config()
    )

    vl_retriever = is_vl_retriever(args.retriever)
    retrieval_media_config = vl_media_config if vl_retriever else media_config
    retrieval_email_config = vl_email_config if vl_retriever else email_config

    cache_config = build_cache_config(
        args, args.retriever, retrieval_media_config, retrieval_email_config
    )

    def safe_mtime(path: str) -> Optional[float]:
        return os.path.getmtime(path) if path and os.path.exists(path) else None

    retrieval_cache_config = dict(cache_config)
    retrieval_cache_config["retrieval_max_k"] = args.retrieval_max_k
    retrieval_cache_config["qa_file"] = args.qa_file
    retrieval_cache_config["qa_mtime"] = safe_mtime(args.qa_file)
    retrieval_cache_key = build_cache_key(retrieval_cache_config)
    cache_dir = Path(args.index_cache)
    retrieval_cache_dir = cache_dir / "retrieval_results"
    retrieval_items_cache_dir = cache_dir / "retrieval_items"
    retrieval_cache_path = retrieval_cache_dir / f"{retrieval_cache_key}.json"
    retrieval_items_cache_path = (
        retrieval_items_cache_dir / f"{retrieval_cache_key}.json"
    )

    retrieval_results_cached = False
    cached_results: List[Dict[str, Any]] = []
    cached_items: Dict[str, RetrievalItem] = {}
    retrieval_items: List[RetrievalItem] = []

    print(
        "[RetrievalResultsCache] Lookup: "
        f"results={retrieval_cache_path} (exists={retrieval_cache_path.exists()}), "
        f"items={retrieval_items_cache_path} (exists={retrieval_items_cache_path.exists()})"
    )

    if (
        args.reuse_retrieval_results
        and not args.force_rebuild
        and retrieval_cache_path.exists()
        and retrieval_items_cache_path.exists()
    ):
        retrieval_cache_data = load_json(retrieval_cache_path)
        cached_results = retrieval_cache_data.get("results", [])
        cached_items_list = load_json(retrieval_items_cache_path)
        cached_items = {
            item_data["item_id"]: RetrievalItem.from_dict(item_data)
            for item_data in cached_items_list
        }
        cached_ids = {str(item.get("qa_id")) for item in cached_results}
        expected_ids = {str(qa.get("id") or qa.get("qa_id")) for qa in qas}
        if cached_ids == expected_ids:
            retrieval_results_cached = True
            print(f"[RetrievalResultsCache] Cache hit: {retrieval_cache_path}")
        else:
            print(
                "[RetrievalResultsCache] Cache miss: cached results do not match QA set"
            )

    if not retrieval_results_cached:
        print(f"[RetrievalResultsCache] Cache miss: {retrieval_cache_path}")

        image_batch_data = load_json(Path(args.image_batch_results))
        video_batch_data = load_json(Path(args.video_batch_results))
        email_entries = load_json(Path(args.email_file))
        if not isinstance(image_batch_data, list) or not isinstance(
            video_batch_data, list
        ):
            raise ValueError("Batch results must be lists")
        if not isinstance(email_entries, list):
            raise ValueError("Email file should be a list")

        retrieval_items = build_retrieval_items(
            email_entries,
            image_batch_data,
            video_batch_data,
            retrieval_media_config,
            retrieval_email_config,
            Path(args.image_root),
            Path(args.video_root),
        )

    if args.retriever in {"clip", "vista"} and args.media_source == "batch_results":
        raise ValueError(
            "clip/vista retrievers require raw media (use --media-source raw)"
        )

    prepared: List[Dict[str, Any]] = []
    recall_details: List[Dict[str, Any]] = []
    retrieval_total_ms = 0.0
    model_load_ms = 0.0
    index_build_ms = 0.0

    if retrieval_results_cached:
        retrieval_items = list(cached_items.values())
        for qa in tqdm(qas, desc="Retrieval"):
            qa_id = qa.get("id") or qa.get("qa_id")
            question = qa.get("question")
            if not qa_id or not question:
                continue
            gt_ids = extract_evidence_ids(qa)

            cached_entry = next(
                (
                    item
                    for item in cached_results
                    if str(item.get("qa_id")) == str(qa_id)
                ),
                None,
            )
            if not cached_entry:
                continue
            retrieved = cached_entry.get("retrieved", [])
            retrieval_ids = [item["item_id"] for item in retrieved]
            retrieval_scores = [item["score"] for item in retrieved]
            recall = compute_recall(gt_ids, retrieval_ids)
            recall_details.append(
                {
                    "id": qa_id,
                    "question": question,
                    "gt_evidence_ids": gt_ids,
                    "retrieval_ids": retrieval_ids,
                    "retrieval_scores": retrieval_scores,
                    "retrieval_recall": recall,
                }
            )

            if args.no_evidence:
                evidence_items = []
            else:
                evidence_items = [
                    cached_items[item_id]
                    for item_id in retrieval_ids[: args.retrieval_top_k]
                    if item_id in cached_items
                ]
                evidence_items = select_evidence_items(
                    evidence_items, args.max_evidence_items
                )

            prepared.append(
                {
                    "id": qa_id,
                    "question": question,
                    "evidence_items": evidence_items,
                }
            )
    else:
        start_model_load = time.perf_counter()

        if args.retriever == "qwen3_vl_embedding":
            retriever = Qwen3VLRetriever(
                model_name=args.vl_embedding_model,
                cache_dir=cache_dir,
                batch_size=args.retriever_batch_size,
                num_frames=args.num_frames,
                max_frames=args.num_frames,
            )
        elif args.retriever == "vista":
            if not args.vista_weights:
                raise ValueError("--vista-weights is required for VISTA retriever")
            retriever = VistaRetriever(
                model_name=args.vista_model_name,
                weights_path=args.vista_weights,
                cache_dir=cache_dir,
                batch_size=args.retriever_batch_size,
            )
        elif args.retriever == "clip":
            retriever = ClipRetriever(
                model_name=args.clip_model,
                cache_dir=cache_dir,
                batch_size=args.retriever_batch_size,
            )
        elif args.retriever == "sentence_transformer":
            retriever = SentenceTransformerRetriever(
                model_name=args.text_embedding_model,
                cache_dir=cache_dir,
                batch_size=args.retriever_batch_size,
            )
        else:
            retriever = TextRetriever(
                model_name=args.text_embedding_model,
                cache_dir=cache_dir,
                batch_size=args.retriever_batch_size,
            )

        model_load_ms = (time.perf_counter() - start_model_load) * 1000

        start_index_build = time.perf_counter()

        retriever.build_index(
            retrieval_items, cache_config, force_rebuild=args.force_rebuild
        )

        index_build_ms = (time.perf_counter() - start_index_build) * 1000

        retrieval_results_list: List[Dict[str, Any]] = []
        all_retrieved_items: Dict[str, RetrievalItem] = {}

        for qa in tqdm(qas, desc="Retrieval"):
            qa_id = qa.get("id") or qa.get("qa_id")
            question = qa.get("question")
            if not qa_id or not question:
                continue
            gt_ids = extract_evidence_ids(qa)

            start_retrieval = time.perf_counter()
            retrieval_results = retriever.retrieve(question, args.retrieval_max_k)
            retrieval_total_ms += (time.perf_counter() - start_retrieval) * 1000

            retrieval_ids = [result.item.item_id for result in retrieval_results]
            retrieval_scores = [result.score for result in retrieval_results]
            recall = compute_recall(gt_ids, retrieval_ids)
            recall_details.append(
                {
                    "id": qa_id,
                    "question": question,
                    "gt_evidence_ids": gt_ids,
                    "retrieval_ids": retrieval_ids,
                    "retrieval_scores": retrieval_scores,
                    "retrieval_recall": recall,
                }
            )

            retrieval_results_list.append(
                {
                    "qa_id": qa_id,
                    "question": question,
                    "gt_evidence_ids": gt_ids,
                    "retrieved": [
                        {"item_id": item_id, "score": float(score)}
                        for item_id, score in zip(retrieval_ids, retrieval_scores)
                    ],
                }
            )

            for result in retrieval_results:
                if result.item.item_id not in all_retrieved_items:
                    all_retrieved_items[result.item.item_id] = result.item

            if args.no_evidence:
                evidence_items = []
            else:
                evidence_items = [
                    result.item for result in retrieval_results[: args.retrieval_top_k]
                ]
                evidence_items = select_evidence_items(
                    evidence_items, args.max_evidence_items
                )

            prepared.append(
                {
                    "id": qa_id,
                    "question": question,
                    "evidence_items": evidence_items,
                }
            )

        retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
        retrieval_items_cache_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            retrieval_cache_path,
            {
                "metadata": {
                    "retriever": args.retriever,
                    "retrieval_max_k": args.retrieval_max_k,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                "results": retrieval_results_list,
            },
        )
        write_json(
            retrieval_items_cache_path,
            [item.to_dict() for item in all_retrieved_items.values()],
        )

    output_base = Path(args.output_dir_base)
    method_dir = output_base / args.method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    output_file = method_dir / "mmrag_answers.jsonl"
    checkpoint_file = method_dir / "mmrag_answers_checkpoint.jsonl"

    summary = {
        "retriever": args.retriever,
        "retrieval_top_k": args.retrieval_top_k,
        "retrieval_max_k": args.retrieval_max_k,
        "recall": summarize_recalls(recall_details, "retrieval_recall"),
    }
    write_json(method_dir / "retrieval_recall_summary.json", summary)
    write_json(method_dir / "retrieval_recall_details.json", recall_details)

    retrieval_latency = {
        "model_load_ms": model_load_ms,
        "index_build_ms": index_build_ms,
        "retrieval_total_ms": retrieval_total_ms,
        "retrieval_per_query_avg_ms": retrieval_total_ms / len(qas) if qas else 0.0,
        "num_queries": len(qas),
        "cache_hit": retrieval_results_cached,
    }
    write_json(method_dir / "retrieval_latency.json", retrieval_latency)

    resume_map = load_resume_map(output_file, checkpoint_file)
    pending = [
        entry
        for entry in prepared
        if is_failed_record(resume_map.get(str(entry["id"])))
    ]
    if resume_map:
        completed = len(prepared) - len(pending)
        print(f"Resume: {completed}/{len(prepared)} complete, {len(pending)} pending.")

    if pending:
        if args.provider == "vllm_local" and args.max_workers > 1:
            raise ValueError("vllm_local does not support concurrent execution.")

        force_multimodal = (
            args.insert_raw_images and args.media_source == "batch_results"
        )
        has_multimodal = force_multimodal or (
            args.media_source == "raw"
            and any(
                any(item.modality in {"image", "video"} for item in entry["evidence_items"])
                for entry in pending
            )
        )
        if args.provider == "vllm_local" and has_multimodal:
            raise ValueError(
                "vllm_local does not support multimodal inputs (raw media or insert_raw_images)."
            )

        llm_config = build_model_config(args.provider, has_multimodal, args)
        thread_local = threading.local()

        def get_llm() -> LLMClient:
            if not hasattr(thread_local, "llm"):
                thread_local.llm = LLMClient(args.provider, llm_config)
            return thread_local.llm

        def answer_single(entry: Dict[str, Any]) -> Dict[str, Any]:
            try:
                messages = build_llm_messages(
                    entry["question"], entry["evidence_items"], args
                )
                llm = get_llm()
                answer, usage = llm.chat_with_usage(messages)
                critic_meta: Dict[str, Any] = {}
                if args.critic_answerer:
                    has_multimodal = (
                        (args.media_source == "raw" or args.insert_raw_images)
                        and any(
                            item.modality in {"image", "video"}
                            for item in entry["evidence_items"]
                        )
                    )
                    critic_meta["critic_used"] = False
                    critic_meta["critic_rounds"] = 1
                    if entry["evidence_items"] and not has_multimodal:
                        try:
                            critic_messages = build_agentic_critic_messages(
                                entry["question"], entry["evidence_items"], answer
                            )
                            critique, critic_usage = llm.chat_with_usage(
                                critic_messages
                            )
                            if critic_usage:
                                critic_meta["critic_prompt_tokens"] = (
                                    critic_usage.prompt_tokens
                                )
                                critic_meta["critic_completion_tokens"] = (
                                    critic_usage.completion_tokens
                                )
                                critic_meta["critic_total_tokens"] = (
                                    critic_usage.total_tokens
                                )
                            usage = merge_token_usage(usage, critic_usage)
                            critic = parse_agentic_critic_response(critique)
                            final_answer = critic.get("final_answer")
                            support = critic.get("support")
                            critic_meta["critic_used"] = True
                            critic_meta["critic_rounds"] = 2
                            if support:
                                critic_meta["critic_support"] = support
                            if critic.get("utility") is not None:
                                critic_meta["critic_utility"] = critic.get("utility")
                            if support == "none":
                                answer = "Unknown"
                            elif final_answer:
                                answer = final_answer
                        except Exception as exc:
                            print(
                                f"[MMRag QA] Critic failed id={entry['id']}: {exc}",
                                file=sys.stderr,
                            )
                result: Dict[str, Any] = {
                    "id": entry["id"],
                    "answer": answer,
                    "status": "ok",
                }
                if critic_meta:
                    result.update(critic_meta)
                if usage:
                    result["prompt_tokens"] = usage.prompt_tokens
                    result["completion_tokens"] = usage.completion_tokens
                    result["total_tokens"] = usage.total_tokens
                return result
            except Exception as exc:
                print(
                    f"[MMRag QA] Failed id={entry['id']}: {exc}",
                    file=sys.stderr,
                )
                return {
                    "id": entry["id"],
                    "answer": "",
                    "status": "failed",
                    "error": str(exc),
                }

        if args.max_workers and args.max_workers > 1:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_map = {
                    executor.submit(answer_single, entry): entry for entry in pending
                }
                for future in tqdm(
                    as_completed(future_map), total=len(future_map), desc="MMRag QA"
                ):
                    entry = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(
                            f"[MMRag QA] Failed id={entry['id']}: {exc}",
                            file=sys.stderr,
                        )
                        result = {
                            "id": entry["id"],
                            "answer": "",
                            "status": "failed",
                            "error": str(exc),
                        }
                    if result:
                        append_jsonl(checkpoint_file, result)
                        resume_map[str(result["id"])] = result
        else:
            for entry in tqdm(pending, desc="MMRag QA"):
                result = answer_single(entry)
                if result:
                    append_jsonl(checkpoint_file, result)
                    resume_map[str(result["id"])] = result
    else:
        print("All answers already present; skipping inference.")

    results: List[Dict[str, Any]] = []
    for entry in prepared:
        record = resume_map.get(str(entry["id"]))
        if record and not is_failed_record(record):
            output_record = {
                key: value
                for key, value in record.items()
                if key not in {"status", "error"}
            }
            results.append(output_record)

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
    }
    if results and any("critic_rounds" in r or "agentic_rounds" in r for r in results):
        run_stats["critic_total_rounds"] = sum(
            int(r.get("critic_rounds", r.get("agentic_rounds", 0))) for r in results
        )
        run_stats["critic_avg_rounds"] = round(
            run_stats["critic_total_rounds"] / num_samples, 2
        )

    stats_path = output_file.parent / f"{output_file.stem}_run_stats.json"
    if pending or not stats_path.exists():
        write_json(stats_path, run_stats)
        print(f"Token stats written to: {stats_path}")
    else:
        print(f"Token stats already present; leaving unchanged: {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
