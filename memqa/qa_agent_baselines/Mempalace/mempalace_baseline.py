#!/usr/bin/env python3
"""MemPalace baseline for ATM-Bench.

Two-stage pipeline:
  Stage 1 (build): Ingest RetrievalItems into a MemPalace palace (ChromaDB).
  Stage 2 (answer): Use MemPalace hybrid BM25+vector search + LLM for QA.

Alignment with upstream MemPalace (>= 3.3.5):
  - Embedding model: all-MiniLM-L6-v2 (MemPalace default ONNX model)
  - Chunking: CHUNK_SIZE=800, CHUNK_OVERLAP=100, MIN_CHUNK_SIZE=50 (via chunk_text)
  - Retrieval: MemPalace search_memories() hybrid BM25+vector reranking
  - Closets: upstream regex closets are built for each virtual RetrievalItem
  - Closet boost + drawer-grep enrichment: enabled through search_memories()
  - candidate_strategy: "vector" (default, aligned with upstream)

Necessary adaptations:
  - Data is injected programmatically via batched upsert rather than file-system
    scanning. This avoids temporary files while preserving chunking and closets.
  - Wing assignment is by modality (email/image/video) rather than directory
    structure. This preserves modality-based filtering without filesystem semantics.
  - n_results=100 is the default to over-fetch for retrieval-recall reporting,
    then top --retrieve-k evidence items are passed to the answerer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from memqa.qa_agent_baselines.Mempalace.config import MEMPALACE_CONFIG, PROMPTS
from memqa.qa_agent_baselines.Mempalace.data_adapter import (
    build_palace_from_items,
    search_results_to_item_ids,
    search_results_to_scores,
    source_file_to_item_id,
)
from memqa.qa_agent_baselines.MMRag.llm_utils import LLMClient, TokenUsage
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


RECALL_KS = [1, 5, 10, 25, 50, 100]
INDEX_ADAPTER_VERSION = "mempalace_retrieval_items_drawers_closets_v2"
MEMPALACE_VECTOR_WEIGHT = MEMPALACE_CONFIG["vector_weight"]
MEMPALACE_BM25_WEIGHT = MEMPALACE_CONFIG["bm25_weight"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MemPalace baseline for ATM-Bench")

    parser.add_argument("--qa-file", required=True)
    parser.add_argument(
        "--image-batch-results", default=MEMPALACE_CONFIG["image_batch_results"]
    )
    parser.add_argument(
        "--video-batch-results", default=MEMPALACE_CONFIG["video_batch_results"]
    )
    parser.add_argument("--image-root", default=MEMPALACE_CONFIG["image_root"])
    parser.add_argument("--video-root", default=MEMPALACE_CONFIG["video_root"])
    parser.add_argument("--email-file", default=MEMPALACE_CONFIG["email_file"])
    parser.add_argument(
        "--provider",
        choices=["openai", "vllm", "vllm_local"],
        default=MEMPALACE_CONFIG["provider"],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--vllm-endpoint", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument(
        "--max-evidence-items",
        type=int,
        default=MEMPALACE_CONFIG["max_evidence_items"],
    )
    parser.add_argument(
        "--no-evidence", action="store_true", default=MEMPALACE_CONFIG["no_evidence"]
    )
    parser.add_argument(
        "--num-frames", type=int, default=MEMPALACE_CONFIG["num_frames"]
    )
    parser.add_argument("--frame-strategy", default=MEMPALACE_CONFIG["frame_strategy"])
    parser.add_argument(
        "--max-workers", type=int, default=MEMPALACE_CONFIG["max_workers"]
    )
    parser.add_argument(
        "--output-dir-base", default=MEMPALACE_CONFIG["output_dir_base"]
    )
    parser.add_argument("--method-name", default="mempalace_default")

    parser.add_argument(
        "--retrieve-k", type=int, default=MEMPALACE_CONFIG["retrieve_k"]
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=MEMPALACE_CONFIG["n_results"],
        help="MemPalace search n_results (over-fetch for reranking)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=MEMPALACE_CONFIG["max_distance"],
        help="Max cosine distance filter (0.0 = disabled)",
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=["vector", "union"],
        default=MEMPALACE_CONFIG["candidate_strategy"],
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="ChromaDB collection name (default: mempalace default)",
    )

    parser.add_argument("--index-cache", default=MEMPALACE_CONFIG["index_cache_dir"])
    parser.add_argument("--force-rebuild", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing answer files for this method.",
    )
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--include-id",
        action=argparse.BooleanOptionalAction,
        default=MEMPALACE_CONFIG["include_id"],
    )
    parser.add_argument(
        "--include-type",
        action=argparse.BooleanOptionalAction,
        default=MEMPALACE_CONFIG["include_type"],
    )
    parser.add_argument(
        "--include-timestamp",
        action=argparse.BooleanOptionalAction,
        default=MEMPALACE_CONFIG["include_timestamp"],
    )
    parser.add_argument(
        "--include-location",
        action=argparse.BooleanOptionalAction,
        default=MEMPALACE_CONFIG["include_location"],
    )
    parser.add_argument(
        "--include-short-caption",
        action=argparse.BooleanOptionalAction,
        default=MEMPALACE_CONFIG["include_short_caption"],
    )
    parser.add_argument(
        "--include-caption",
        action=argparse.BooleanOptionalAction,
        default=MEMPALACE_CONFIG["include_caption"],
    )
    parser.add_argument(
        "--include-ocr-text",
        action=argparse.BooleanOptionalAction,
        default=MEMPALACE_CONFIG["include_ocr_text"],
    )
    parser.add_argument(
        "--include-tags",
        action=argparse.BooleanOptionalAction,
        default=MEMPALACE_CONFIG["include_tags"],
    )
    parser.add_argument(
        "--include-email-summary",
        action=argparse.BooleanOptionalAction,
        default=MEMPALACE_CONFIG["include_email_summary"],
    )
    parser.add_argument(
        "--include-email-detail",
        action=argparse.BooleanOptionalAction,
        default=MEMPALACE_CONFIG["include_email_detail"],
    )

    parser.add_argument(
        "--stage",
        choices=["build", "answer", "all"],
        default="all",
        help="Pipeline stage",
    )
    parser.add_argument("--print-cache-key", action="store_true")

    return parser.parse_args()


def build_model_config(provider: str, args: argparse.Namespace) -> Dict[str, Any]:
    if provider == "openai":
        base = dict(MEMPALACE_CONFIG["openai"])
    elif provider in {"vllm", "vllm_local"}:
        base = dict(MEMPALACE_CONFIG["vllm_text"])
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


def build_cache_key_from_args(args: argparse.Namespace) -> str:
    def safe_mtime(path: str) -> Optional[float]:
        return os.path.getmtime(path) if path and os.path.exists(path) else None

    return build_cache_key(
        {
            "embedding_model": MEMPALACE_CONFIG["embedding_model"],
            "adapter_version": INDEX_ADAPTER_VERSION,
            "collection_name": args.collection_name,
            "image_batch_results": args.image_batch_results,
            "video_batch_results": args.video_batch_results,
            "email_file": args.email_file,
            "include_id": args.include_id,
            "include_type": args.include_type,
            "include_timestamp": args.include_timestamp,
            "include_location": args.include_location,
            "include_short_caption": args.include_short_caption,
            "include_caption": args.include_caption,
            "include_ocr_text": args.include_ocr_text,
            "include_tags": args.include_tags,
            "include_email_summary": args.include_email_summary,
            "include_email_detail": args.include_email_detail,
            "index_limit": args.limit,
            "image_batch_mtime": safe_mtime(args.image_batch_results),
            "video_batch_mtime": safe_mtime(args.video_batch_results),
            "email_mtime": safe_mtime(args.email_file),
        }
    )


def answer_question(
    llm: LLMClient,
    question: str,
    context: str,
) -> Tuple[str, Optional[Any]]:
    evidence_block = context if context else ""
    messages = [
        {"role": "system", "content": PROMPTS["SYSTEM"]},
        {
            "role": "user",
            "content": PROMPTS["USER"].format(
                question=question, evidence=evidence_block
            ),
        },
    ]
    return llm.chat_with_usage(messages)


def build_stage(args: argparse.Namespace) -> int:
    print("=" * 80)
    print("STAGE 1: PALACE CONSTRUCTION")
    print("=" * 80)

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

    if args.limit:
        retrieval_items = retrieval_items[: args.limit]

    cache_key = build_cache_key_from_args(args)
    cache_dir = Path(args.index_cache)
    palace_path = cache_dir / f"palace_{cache_key}"
    marker_file = cache_dir / f"palace_{cache_key}_built.json"

    if marker_file.exists() and not args.force_rebuild:
        print(f"[Mempalace] Palace cache hit: {palace_path}")
        with marker_file.open("r") as f:
            stats = json.load(f)
        print(
            f"  Items: {stats.get('total_items')}, Drawers: {stats.get('total_drawers')}"
        )
        print(f"  Palace path: {palace_path}")
        return 0

    if args.force_rebuild and palace_path.exists():
        print(f"[Mempalace] Force rebuild: removing {palace_path}")
        shutil.rmtree(palace_path, ignore_errors=True)
    elif palace_path.exists() and not marker_file.exists():
        print(f"[Mempalace] Incomplete palace cache found; removing {palace_path}")
        shutil.rmtree(palace_path, ignore_errors=True)

    start_time = time.perf_counter()
    stats = build_palace_from_items(
        retrieval_items,
        palace_path=str(palace_path),
        collection_name=args.collection_name,
        build_closets=True,
    )
    build_ms = (time.perf_counter() - start_time) * 1000

    stats["build_ms"] = build_ms
    stats["palace_path"] = str(palace_path)
    stats["cache_key"] = cache_key

    cache_dir.mkdir(parents=True, exist_ok=True)
    with marker_file.open("w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print("PALACE CONSTRUCTION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Cache key:      {cache_key}")
    print(f"Palace path:    {palace_path}")
    print(f"Total items:    {stats['total_items']}")
    print(f"Total drawers:  {stats['total_drawers']}")
    print(f"Total closets:  {stats['total_closets']}")
    print(f"Skipped (empty): {stats['skipped_empty']}")
    print(f"Build time:     {build_ms:.0f} ms")
    print(f"{'=' * 80}")

    return 0


def answer_stage(args: argparse.Namespace) -> int:
    print("=" * 80)
    print("STAGE 2: ANSWER GENERATION")
    print("=" * 80)

    qa_data = load_json(Path(args.qa_file))
    qas = load_qa_list(qa_data)
    if args.limit:
        qas = qas[: args.limit]

    cache_key = build_cache_key_from_args(args)
    cache_dir = Path(args.index_cache)
    palace_path = cache_dir / f"palace_{cache_key}"
    marker_file = cache_dir / f"palace_{cache_key}_built.json"

    if not marker_file.exists():
        print("ERROR: Palace not found. Run Stage 1 first.")
        print(f"  Expected: {marker_file}")
        return 1

    from mempalace.searcher import search_memories

    print("[Mempalace] Warming up search (handles HNSW quarantine on NTFS)...")
    for _ in range(3):
        try:
            search_memories(
                query="warmup",
                palace_path=str(palace_path),
                n_results=1,
                max_distance=args.max_distance,
                candidate_strategy=args.candidate_strategy,
                collection_name=args.collection_name,
            )
        except Exception:
            pass
    print("[Mempalace] Search warmup complete.")

    model_config = build_model_config(args.provider, args)

    output_dir = Path(args.output_dir_base) / args.method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mempalace_answers.jsonl"

    if args.overwrite and output_path.exists():
        output_path.unlink()

    if output_path.exists():
        print(f"Output exists, skipping inference: {output_path}")
        return 0

    media_config, email_config = build_text_configs(args)
    image_batch = load_json(Path(args.image_batch_results))
    video_batch = load_json(Path(args.video_batch_results))
    email_entries = load_json(Path(args.email_file))

    all_items = build_retrieval_items(
        email_entries,
        image_batch,
        video_batch,
        media_config,
        email_config,
        Path(args.image_root),
        Path(args.video_root),
    )
    items_by_id: Dict[str, RetrievalItem] = {item.item_id: item for item in all_items}

    n_results = max(args.n_results, args.retrieve_k)

    thread_local = threading.local()

    def get_llm() -> LLMClient:
        if not hasattr(thread_local, "llm"):
            thread_local.llm = LLMClient(args.provider, model_config)
        return thread_local.llm

    def answer_single(qa: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        qa_id = qa.get("id") or qa.get("qa_id")
        question = qa.get("question")
        if not qa_id or not question:
            return None

        gt_ids = extract_evidence_ids(qa)

        if args.no_evidence:
            context_text = ""
            retrieval_ids: List[str] = []
            retrieval_scores: List[float] = []
        else:
            search_result: Dict[str, Any] = {"results": []}
            for attempt in range(3):
                try:
                    search_result = search_memories(
                        query=question,
                        palace_path=str(palace_path),
                        n_results=n_results,
                        max_distance=args.max_distance,
                        candidate_strategy=args.candidate_strategy,
                        collection_name=args.collection_name,
                    )
                    break
                except Exception as exc:
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    print(
                        f"[Mempalace] Search failed id={qa_id}: {exc}", file=sys.stderr
                    )

            hits = search_result.get("results", [])
            if "error" in search_result:
                print(
                    f"[Mempalace] Search error id={qa_id}: {search_result['error']}",
                    file=sys.stderr,
                )
                hits = []

            retrieval_ids = search_results_to_item_ids(hits)
            retrieval_scores = search_results_to_scores(hits)

            top_item_ids = retrieval_ids[: args.retrieve_k]
            evidence_texts: List[str] = []
            for item_id in top_item_ids:
                item = items_by_id.get(item_id)
                if item:
                    evidence_texts.append(item.text)
                else:
                    for hit in hits:
                        if (
                            source_file_to_item_id(hit.get("source_file", ""))
                            == item_id
                        ):
                            evidence_texts.append(hit.get("text", ""))
                            break

            if args.max_evidence_items is not None:
                evidence_texts = evidence_texts[: args.max_evidence_items]

            context_text = "\n\n---\n\n".join(evidence_texts)

        retrieval_recall = compute_recall(gt_ids, retrieval_ids)

        llm_client = get_llm()
        answer, usage = answer_question(llm_client, question, context_text)

        retrieval_record = {
            "id": qa_id,
            "question": question,
            "gt_evidence_ids": gt_ids,
            "retrieval_ids": retrieval_ids,
            "retrieval_scores": retrieval_scores,
            "retrieval_recall": retrieval_recall,
        }

        answer_record: Dict[str, Any] = {"id": qa_id, "answer": answer}
        if usage:
            answer_record["prompt_tokens"] = usage.prompt_tokens
            answer_record["completion_tokens"] = usage.completion_tokens
            answer_record["total_tokens"] = usage.total_tokens

        return {
            "answer_record": answer_record,
            "retrieval_record": retrieval_record,
        }

    results: List[Dict[str, Any]] = []
    retrieval_details: List[Dict[str, Any]] = []

    if args.max_workers and args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(answer_single, qa): idx for idx, qa in enumerate(qas)
            }
            ordered: List[Tuple[int, Dict[str, Any]]] = []
            for future in tqdm(
                as_completed(future_map), total=len(future_map), desc="Mempalace QA"
            ):
                idx = future_map[future]
                result = future.result()
                if result:
                    ordered.append((idx, result))
            ordered.sort(key=lambda item: item[0])
            for _, item in ordered:
                results.append(item["answer_record"])
                retrieval_details.append(item["retrieval_record"])
    else:
        for qa in tqdm(qas, desc="Mempalace QA"):
            result = answer_single(qa)
            if result:
                results.append(result["answer_record"])
                retrieval_details.append(result["retrieval_record"])

    write_jsonl(output_path, results)

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

    stats_path = output_path.parent / f"{output_path.stem}_run_stats.json"
    write_json(stats_path, run_stats)
    print(f"Token stats written to: {stats_path}")

    summary = {
        "retrieval_top_k": args.retrieve_k,
        "n_results": args.n_results,
        "candidate_strategy": args.candidate_strategy,
        "vector_weight": MEMPALACE_VECTOR_WEIGHT,
        "bm25_weight": MEMPALACE_BM25_WEIGHT,
        "max_distance": args.max_distance,
        "recall": summarize_recalls(retrieval_details, "retrieval_recall"),
    }
    write_json(output_dir / "retrieval_recall_summary.json", summary)
    write_json(output_dir / "retrieval_recall_details.json", retrieval_details)

    print(f"\n{'=' * 80}")
    print("ANSWER GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved: {output_path}")
    print(f"Total QAs:     {len(results)}")
    print(f"{'=' * 80}")

    return 0


def main() -> int:
    args = parse_args()

    if args.print_cache_key:
        cache_key = build_cache_key_from_args(args)
        print(f"Cache key: {cache_key}")
        print(f"Cache dir: {args.index_cache}")
        return 0

    if args.stage == "build":
        return build_stage(args)
    elif args.stage == "answer":
        return answer_stage(args)
    else:
        build_stage(args)
        return answer_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
