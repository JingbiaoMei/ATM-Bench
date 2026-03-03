#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CURRENT_DIR / "config.py"
MEMORY_LAYER_PATH = CURRENT_DIR / "memory_layer.py"


def load_amem_config_module():
    spec = importlib.util.spec_from_file_location("amem_config", CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load A-Mem config module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_memory_layer_module():
    spec = importlib.util.spec_from_file_location(
        "amem_memory_layer", MEMORY_LAYER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load A-Mem memory layer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


amem_config = load_amem_config_module()
memory_layer = load_memory_layer_module()

A_MEM_CONFIG = amem_config.A_MEM_CONFIG
PROMPTS = amem_config.PROMPTS
AgenticMemorySystem = memory_layer.AgenticMemorySystem
LLMController = memory_layer.LLMController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A-Mem baseline")
    parser.add_argument("--qa-file", required=True)
    parser.add_argument(
        "--image-batch-results", default=A_MEM_CONFIG["image_batch_results"]
    )
    parser.add_argument(
        "--video-batch-results", default=A_MEM_CONFIG["video_batch_results"]
    )
    parser.add_argument("--image-root", default=A_MEM_CONFIG["image_root"])
    parser.add_argument("--video-root", default=A_MEM_CONFIG["video_root"])
    parser.add_argument("--email-file", default=A_MEM_CONFIG["email_file"])
    parser.add_argument(
        "--provider",
        choices=["openai", "vllm", "vllm_local"],
        default=A_MEM_CONFIG["provider"],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--vllm-endpoint", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument(
        "--max-evidence-items", type=int, default=A_MEM_CONFIG["max_evidence_items"]
    )
    parser.add_argument(
        "--no-evidence", action="store_true", default=A_MEM_CONFIG["no_evidence"]
    )
    parser.add_argument("--num-frames", type=int, default=A_MEM_CONFIG["num_frames"])
    parser.add_argument("--frame-strategy", default=A_MEM_CONFIG["frame_strategy"])
    parser.add_argument("--max-workers", type=int, default=A_MEM_CONFIG["max_workers"])
    parser.add_argument(
        "--memory-workers",
        type=int,
        default=A_MEM_CONFIG["memory_workers"],
        help="Parallel workers for memory construction (Phase 1). Default 1 (sequential). Set higher for faster construction with openai/vllm providers.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=A_MEM_CONFIG["checkpoint_interval"],
        help="Save memory cache every N ingests (0 disables).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=A_MEM_CONFIG["resume"],
        help="Resume from existing partial memory cache if present.",
    )
    parser.add_argument(
        "--memory-retry-count",
        type=int,
        default=A_MEM_CONFIG["memory_retry_count"],
        help="Retry count for memory note construction failures.",
    )
    parser.add_argument(
        "--memory-retry-backoff",
        type=float,
        default=A_MEM_CONFIG["memory_retry_backoff"],
        help="Seconds to sleep between memory note retries.",
    )
    parser.add_argument(
        "--allow-incomplete-cache",
        action="store_true",
        default=A_MEM_CONFIG["allow_incomplete_cache"],
        help="Allow saving cache even if some notes failed to construct.",
    )
    parser.add_argument("--output-dir-base", default=A_MEM_CONFIG["output_dir_base"])
    parser.add_argument("--method-name", default="amem_default")
    parser.add_argument("--embedding-model", default=A_MEM_CONFIG["embedding_model"])
    parser.add_argument("--retrieve-k", type=int, default=A_MEM_CONFIG["retrieve_k"])
    parser.add_argument(
        "--retrieval-log-k",
        type=int,
        default=100,
        help="Log top-K retrieved items for analysis (default 100).",
    )
    parser.add_argument(
        "--evo-threshold", type=int, default=A_MEM_CONFIG["evo_threshold"]
    )
    parser.add_argument(
        "--disable-evolution",
        action="store_true",
        default=A_MEM_CONFIG["disable_evolution"],
    )
    parser.add_argument("--index-cache", default=A_MEM_CONFIG["index_cache_dir"])
    parser.add_argument("--force-rebuild", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--stage",
        choices=["build", "answer", "all"],
        default="all",
        help="Pipeline stage: 'build' (memory construction only), 'answer' (QA only), 'all' (both)",
    )
    parser.add_argument(
        "--print-cache-key",
        action="store_true",
        help="Print cache key and exit (useful for Stage 2 validation)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=A_MEM_CONFIG["alpha"],
        help="Hybrid retrieval weight: alpha*BM25 + (1-alpha)*embedding. Default 0.5",
    )
    parser.add_argument(
        "--use-hybrid",
        action=argparse.BooleanOptionalAction,
        default=A_MEM_CONFIG["use_hybrid"],
        help="Use hybrid BM25+embedding retrieval (aligned with original)",
    )
    parser.add_argument(
        "--follow-links",
        action=argparse.BooleanOptionalAction,
        default=A_MEM_CONFIG["follow_links"],
        help="Traverse memory links during retrieval (aligned with original)",
    )
    parser.add_argument(
        "--caption-only",
        action="store_true",
        default=False,
        help="Use minimal text (caption + ID/Time/Location only, aligned with original paper)",
    )
    parser.add_argument(
        "--use-short-caption",
        action="store_true",
        default=False,
        help="Use short_caption instead of full caption (requires --caption-only)",
    )
    parser.add_argument(
        "--memory-provider",
        choices=["openai", "vllm", "vllm_local"],
        default=None,
        help="Provider for memory operations (defaults to --provider)",
    )
    parser.add_argument(
        "--memory-model",
        default=None,
        help="Model for memory operations (defaults to --model). Original paper uses gpt-4o-mini.",
    )
    parser.add_argument(
        "--memory-api-key", default=None, help="API key for memory operations"
    )
    parser.add_argument(
        "--memory-vllm-endpoint", default=None, help="Endpoint for memory operations"
    )
    # New arguments for raw media construction
    parser.add_argument(
        "--construct-from-raw",
        action="store_true",
        default=False,
        help="Use raw image/video for memory construction instead of batch captions",
    )
    parser.add_argument(
        "--answer-from-raw",
        action="store_true",
        default=False,
        help="Provide raw media paths to the answer generation LLM (if supported)",
    )
    parser.add_argument(
        "--include-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include item ID in text context",
    )
    parser.add_argument(
        "--include-type",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include item type (image/video/email) in text context",
    )
    parser.add_argument(
        "--include-timestamp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include timestamp in text context",
    )
    parser.add_argument(
        "--include-location",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include location in text context",
    )
    parser.add_argument(
        "--include-caption",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include full caption in text context",
    )
    parser.add_argument(
        "--include-short-caption",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include short caption in text context",
    )
    parser.add_argument(
        "--include-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include tags in text context",
    )
    parser.add_argument(
        "--include-ocr-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include OCR text in text context",
    )
    return parser.parse_args()


def build_model_config(provider: str, args: argparse.Namespace) -> Dict[str, Any]:
    if provider == "openai":
        base = dict(A_MEM_CONFIG["openai"])
    elif provider in {"vllm", "vllm_local"}:
        base = dict(A_MEM_CONFIG["vllm_text"])
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
    if args.caption_only:
        use_short = args.use_short_caption
        use_full = not args.use_short_caption

        media_config = MediaTextConfig(
            include_id=True,
            include_type=False,
            include_timestamp=True,
            include_location=True,
            include_short_caption=use_short,
            include_caption=use_full,
            include_ocr_text=False,
            include_tags=False,
        )
        email_config = EmailTextConfig(
            include_id=True,
            include_timestamp=True,
            include_summary=True,
            include_detail=False,
        )
    else:
        media_config = MediaTextConfig(
            include_id=True,
            include_type=args.include_type,
            include_timestamp=True,
            include_location=True,
            include_short_caption=args.include_short_caption,
            include_caption=args.include_caption,
            include_ocr_text=args.include_ocr_text,
            include_tags=args.include_tags,
        )
        email_config = EmailTextConfig(
            include_id=True,
            include_timestamp=True,
            include_summary=True,
            include_detail=True,
        )
    return media_config, email_config


def memory_from_items(
    items: List[RetrievalItem],
    construct_from_raw: bool = False,
    image_root: Optional[Path] = None,
) -> List[Tuple[str, str, str, Optional[str], str]]:
    """
    Returns list of (id, content_text, timestamp, optional_image_path, modality)
    """
    memory_rows: List[Tuple[str, str, str, Optional[str], str]] = []
    for item in items:
        timestamp = ""
        if item.metadata:
            timestamp = str(item.metadata.get("timestamp", ""))

        image_path = None
        if construct_from_raw and item.modality == "image":
            if item.image_path:
                image_path = str(item.image_path)
            elif image_root:
                potential_path = image_root / item.item_id
                if potential_path.exists():
                    image_path = str(potential_path)

            if not image_path:
                print(
                    "[A-Mem][warn] Raw image missing for item "
                    f"{item.item_id}; falling back to text-only."
                )
            elif not Path(image_path).exists():
                print(
                    "[A-Mem][warn] Raw image path not found for item "
                    f"{item.item_id}: {image_path}"
                )
                image_path = None

        memory_rows.append(
            (item.item_id, item.text, timestamp, image_path, item.modality)
        )
    return memory_rows


def build_memory_system(
    args: argparse.Namespace,
    model_config: Dict[str, Any],
) -> AgenticMemorySystem:
    mem_provider = args.memory_provider or args.provider

    if args.memory_provider and args.memory_provider != args.provider:
        if args.memory_provider == "openai":
            mem_config = dict(A_MEM_CONFIG["openai"])
        elif args.memory_provider in {"vllm", "vllm_local"}:
            mem_config = dict(A_MEM_CONFIG["vllm_text"])
        else:
            mem_config = {}
    else:
        mem_config = dict(model_config)

    if args.memory_model:
        mem_config["model"] = args.memory_model
    if args.memory_api_key:
        mem_config["api_key"] = args.memory_api_key
    if args.memory_vllm_endpoint:
        mem_config["endpoint"] = args.memory_vllm_endpoint

    api_base = None
    if mem_provider == "vllm":
        api_base = mem_config.get("endpoint")

    timeout = mem_config.get("timeout", 120)

    return AgenticMemorySystem(
        model_name=args.embedding_model,
        llm_backend=mem_provider,
        llm_model=mem_config.get("model"),
        evo_threshold=args.evo_threshold,
        api_key=mem_config.get("api_key"),
        api_base=api_base,
        alpha=args.alpha,
        use_hybrid=args.use_hybrid,
        follow_links=args.follow_links,
        timeout=timeout,
    )


def _save_construction_cache(
    constructed_notes: List[Optional[Any]],
    construction_cache: Path,
) -> None:
    import pickle

    with construction_cache.open("wb") as handle:
        pickle.dump(constructed_notes, handle)


def _save_memory_cache(
    memory_system: AgenticMemorySystem,
    memory_cache: Path,
    retriever_cache: Path,
    embeddings_cache: Path,
) -> None:
    import pickle

    with memory_cache.open("wb") as handle:
        pickle.dump(memory_system.memories, handle)
    memory_system.retriever.save(str(retriever_cache), str(embeddings_cache))


def load_or_build_memory(
    memory_system: AgenticMemorySystem,
    cache_dir: Path,
    cache_key: str,
    memory_rows: List[Tuple[str, str, str, Optional[str], str]],
    disable_evolution: bool,
    force_rebuild: bool,
    memory_workers: int,
    memory_provider: str,
    checkpoint_interval: int,
    resume: bool,
    memory_retry_count: int,
    memory_retry_backoff: float,
    allow_incomplete_cache: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    memory_cache = cache_dir / f"{cache_key}_memories.pkl"
    retriever_cache = cache_dir / f"{cache_key}_retriever.pkl"
    embeddings_cache = cache_dir / f"{cache_key}_retriever.npy"
    construction_cache = cache_dir / f"{cache_key}_constructed.pkl"

    total_items = len(memory_rows)
    constructed_notes: Optional[List[Optional[Any]]] = None

    if resume and construction_cache.exists() and not force_rebuild:
        import pickle

        try:
            with construction_cache.open("rb") as handle:
                constructed_notes = pickle.load(handle)
            if not isinstance(constructed_notes, list) or (
                total_items and len(constructed_notes) != total_items
            ):
                constructed_notes = None
        except Exception:
            constructed_notes = None

    cache_available = memory_cache.exists() and retriever_cache.exists()
    if cache_available and not force_rebuild:
        import pickle

        print(f"[A-Mem] Cache hit: {memory_cache.name}")
        with memory_cache.open("rb") as handle:
            memory_system.memories = pickle.load(handle)
        memory_system.memory_ids = list(memory_system.memories.keys())
        memory_system.retriever.load(str(retriever_cache), str(embeddings_cache))
        if not resume:
            return
        if len(memory_system.memories) >= total_items:
            print("[A-Mem] Cache complete; skipping build")
            return
        print("[A-Mem] Cache incomplete; resuming build")

    start_index = len(memory_system.memory_ids)
    if start_index:
        print(f"[A-Mem] Resume enabled: {start_index} memories already loaded")
    if start_index >= total_items:
        return

    if constructed_notes is None:
        constructed_notes = [None for _ in range(total_items)]

    # Two-phase memory construction:
    # Phase 1: Concurrent construction (LLM analysis to create MemoryNotes)
    # Phase 2: Sequential ingestion (evolution + indexing)

    if disable_evolution:
        print("[A-Mem] Evolution disabled: ingest will skip evolution step")

    use_concurrent = memory_workers > 1 and memory_provider != "vllm_local"

    if use_concurrent:
        print(f"[A-Mem] Parallel construction enabled: workers={memory_workers}")

        # Phase 1: Parallel construction
        def construct_note(idx: int) -> Tuple[int, Optional[Any]]:
            item_id, content, timestamp, image_path, modality = memory_rows[idx]
            attempt = 0
            max_attempts = max(1, memory_retry_count)
            while attempt < max_attempts:
                try:
                    note = memory_system.analyze_and_create_note(
                        content,
                        timestamp=timestamp or None,
                        note_id=item_id,
                        image_path=image_path,
                        modality=modality,
                    )
                    return idx, note
                except Exception as exc:
                    attempt += 1
                    if attempt >= max_attempts:
                        print(f"Error constructing note {item_id}: {exc}")
                        return idx, None
                    print(
                        f"Retrying note {item_id} ({attempt}/{max_attempts}) after error: {exc}"
                    )
                    if memory_retry_backoff:
                        import time

                        time.sleep(memory_retry_backoff)
            return idx, None

        from concurrent.futures import ThreadPoolExecutor, as_completed

        failed_indices = []
        completed = 0
        missing_indices = [
            idx
            for idx in range(start_index, total_items)
            if constructed_notes[idx] is None
        ]
        if not missing_indices:
            print("[A-Mem] All notes already constructed; skipping construction")
        else:
            print(f"[A-Mem] Constructing {len(missing_indices)} notes")
        with ThreadPoolExecutor(max_workers=memory_workers) as executor:
            futures = {
                executor.submit(construct_note, idx): idx for idx in missing_indices
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="A-Mem construct (parallel)",
            ):
                try:
                    idx, note = future.result()
                    if note is None:
                        failed_indices.append(idx)
                    else:
                        constructed_notes[idx] = note
                except Exception as e:
                    idx = futures[future]
                    failed_indices.append(idx)
                    print(f"Error constructing note {idx}: {e}")
                completed += 1
                if checkpoint_interval and completed % checkpoint_interval == 0:
                    _save_construction_cache(constructed_notes, construction_cache)

        if checkpoint_interval:
            _save_construction_cache(constructed_notes, construction_cache)

        if failed_indices and not allow_incomplete_cache:
            failed_preview = ", ".join(str(idx) for idx in failed_indices[:10])
            print(
                "[A-Mem] Aborting memory build due to failed constructions. "
                f"Failed count={len(failed_indices)}. "
                f"First failures: {failed_preview}"
            )
            return

        # Phase 2: Sequential ingestion
        for idx in tqdm(
            range(start_index, total_items), desc="A-Mem ingest (sequential)"
        ):
            note = constructed_notes[idx]
            if note is None:
                if allow_incomplete_cache:
                    continue
                print(f"[A-Mem] Missing constructed note at index {idx}")
                return
            memory_system.ingest_note(note, disable_evolution=disable_evolution)
            total_count = idx + 1
            if checkpoint_interval and total_count % checkpoint_interval == 0:
                _save_memory_cache(
                    memory_system,
                    memory_cache,
                    retriever_cache,
                    embeddings_cache,
                )
    else:
        print("[A-Mem] Sequential construction (single worker)")
        # Sequential construction + ingestion (original behavior)
        failed_indices = []
        max_attempts = max(1, memory_retry_count)
        for idx in tqdm(
            range(start_index, total_items), desc="A-Mem build (sequential)"
        ):
            if constructed_notes[idx] is not None:
                continue
            item_id, content, timestamp, image_path, modality = memory_rows[idx]
            attempt = 0
            note_created = False
            while attempt < max_attempts and not note_created:
                try:
                    note = memory_system.analyze_and_create_note(
                        content,
                        timestamp=timestamp or None,
                        note_id=item_id,
                        image_path=image_path,
                        modality=modality,
                    )
                    constructed_notes[idx] = note
                    note_created = True
                except Exception as exc:
                    attempt += 1
                    if attempt >= max_attempts:
                        failed_indices.append(idx)
                        print(f"Error constructing note {item_id}: {exc}")
                        break
                    print(
                        f"Retrying note {item_id} ({attempt}/{max_attempts}) after error: {exc}"
                    )
                    if memory_retry_backoff:
                        import time

                        time.sleep(memory_retry_backoff)

            total_count = idx + 1
            if checkpoint_interval and total_count % checkpoint_interval == 0:
                _save_construction_cache(constructed_notes, construction_cache)

        if failed_indices and not allow_incomplete_cache:
            failed_preview = ", ".join(str(idx) for idx in failed_indices[:10])
            print(
                "[A-Mem] Aborting memory build due to failed constructions. "
                f"Failed count={len(failed_indices)}. "
                f"First failures: {failed_preview}"
            )
            return

        if checkpoint_interval:
            _save_construction_cache(constructed_notes, construction_cache)

        for idx in tqdm(
            range(start_index, total_items), desc="A-Mem ingest (sequential)"
        ):
            note = constructed_notes[idx]
            if note is None:
                if allow_incomplete_cache:
                    continue
                print(f"[A-Mem] Missing constructed note at index {idx}")
                return
            memory_system.ingest_note(note, disable_evolution=disable_evolution)
            total_count = idx + 1
            if checkpoint_interval and total_count % checkpoint_interval == 0:
                _save_memory_cache(
                    memory_system,
                    memory_cache,
                    retriever_cache,
                    embeddings_cache,
                )

    _save_memory_cache(
        memory_system,
        memory_cache,
        retriever_cache,
        embeddings_cache,
    )


def build_evidence_block(context: str) -> str:
    return context if context else ""


def answer_question(
    llm: LLMClient,
    question: str,
    context: str,
) -> Tuple[str, Optional[TokenUsage]]:
    evidence_block = build_evidence_block(context)
    messages = [
        {"role": "system", "content": PROMPTS["SYSTEM"]},
        {
            "role": "user",
            "content": PROMPTS["USER"].format(
                question=question,
                evidence=evidence_block,
            ),
        },
    ]
    return llm.chat_with_usage(messages)


def compute_recall(gt_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    total = len(gt_ids)
    gt_set = set(gt_ids)
    recalls: Dict[str, float] = {}
    recall_ks = [1, 5, 10, 25, 50, 100]
    for k in recall_ks:
        top_ids = retrieved_ids[: min(k, len(retrieved_ids))]
        hit = len([item_id for item_id in top_ids if item_id in gt_set])
        recalls[f"R@{k}"] = hit / total if total else 0.0
    return recalls


def summarize_recalls(details: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    recall_ks = [1, 5, 10, 25, 50, 100]
    totals = {f"R@{k}": 0.0 for k in recall_ks}
    count = 0
    for detail in details:
        recall = detail.get(key)
        if not recall:
            continue
        for k in recall_ks:
            totals[f"R@{k}"] += recall.get(f"R@{k}", 0.0)
        count += 1
    if count:
        for k in recall_ks:
            totals[f"R@{k}"] /= count
    totals["count"] = count
    return totals


def build_cache_key_from_args(
    args: argparse.Namespace, model_config: Dict[str, Any]
) -> str:
    mem_model_name = args.memory_model or model_config.get("model")
    if args.memory_provider and args.memory_provider != args.provider:
        if not args.memory_model:
            if args.memory_provider == "openai":
                mem_model_name = A_MEM_CONFIG["openai"].get("model")
            elif args.memory_provider in {"vllm", "vllm_local"}:
                mem_model_name = A_MEM_CONFIG["vllm_text"].get("model")

    return build_cache_key(
        {
            "embedding_model": args.embedding_model,
            "evo_threshold": args.evo_threshold,
            "disable_evolution": args.disable_evolution,
            "image_batch_results": args.image_batch_results,
            "video_batch_results": args.video_batch_results,
            "email_file": args.email_file,
            "alpha": args.alpha,
            "use_hybrid": args.use_hybrid,
            "caption_only": args.caption_only,
            "memory_provider": args.memory_provider or args.provider,
            "memory_model": mem_model_name,
            "construct_from_raw": args.construct_from_raw,
            "use_short_caption": args.use_short_caption if args.caption_only else False,
            "include_type": args.include_type,
            "include_caption": args.include_caption,
            "include_short_caption": args.include_short_caption,
            "include_ocr_text": args.include_ocr_text,
            "include_tags": args.include_tags,
        }
    )


def build_memory_stage(args: argparse.Namespace) -> int:
    print("=" * 80)
    print("STAGE 1: MEMORY CONSTRUCTION")
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

    model_config = build_model_config(args.provider, args)
    memory_system = build_memory_system(args, model_config)

    cache_key = build_cache_key_from_args(args, model_config)

    img_root_path = Path(args.image_root) if args.construct_from_raw else None
    memory_rows = memory_from_items(
        retrieval_items,
        construct_from_raw=args.construct_from_raw,
        image_root=img_root_path,
    )

    load_or_build_memory(
        memory_system,
        Path(args.index_cache),
        cache_key,
        memory_rows,
        args.disable_evolution,
        args.force_rebuild,
        args.memory_workers,
        args.memory_provider or args.provider,
        args.checkpoint_interval,
        args.resume,
        args.memory_retry_count,
        args.memory_retry_backoff,
        args.allow_incomplete_cache,
    )

    cache_dir = Path(args.index_cache)
    memory_cache = cache_dir / f"{cache_key}_memories.pkl"
    retriever_cache = cache_dir / f"{cache_key}_retriever.pkl"
    embeddings_cache = cache_dir / f"{cache_key}_retriever.npy"

    print("\n" + "=" * 80)
    print("✓ MEMORY CONSTRUCTION COMPLETE")
    print("=" * 80)
    print(f"Cache key:      {cache_key}")
    print(f"Cache location: {cache_dir}")
    print("Files saved:")
    print(f"  - {memory_cache.name}")
    print(f"  - {retriever_cache.name}")
    print(f"  - {embeddings_cache.name}")
    print(f"\nTotal memories: {len(memory_system.memories)}")
    print("\nTo run answer generation (Stage 2):")
    print("  python amem_baseline.py --stage answer \\")
    print("    [same cache-affecting args] \\")
    print("    --provider <answer-provider> \\")
    print("    --model <answer-model> \\")
    print("    --output-dir-base <output-dir> \\")
    print("    --method-name <method>")
    print("=" * 80)

    return 0


def answer_stage(args: argparse.Namespace) -> int:
    print("=" * 80)
    print("STAGE 2: ANSWER GENERATION")
    print("=" * 80)

    qa_data = load_json(Path(args.qa_file))
    qas = load_qa_list(qa_data)
    if args.limit:
        qas = qas[: args.limit]

    model_config = build_model_config(args.provider, args)

    cache_key = build_cache_key_from_args(args, model_config)
    cache_dir = Path(args.index_cache)
    memory_cache = cache_dir / f"{cache_key}_memories.pkl"
    retriever_cache = cache_dir / f"{cache_key}_retriever.pkl"
    embeddings_cache = cache_dir / f"{cache_key}_retriever.npy"

    if not memory_cache.exists() or not retriever_cache.exists():
        print("\n" + "=" * 80)
        print("ERROR: Memory cache not found!")
        print("=" * 80)
        print(f"Expected cache key:  {cache_key}")
        print(f"Expected location:   {cache_dir}")
        print("Required files:")
        print(
            f"  - {memory_cache.name} {'✗ MISSING' if not memory_cache.exists() else '✓'}"
        )
        print(
            f"  - {retriever_cache.name} {'✗ MISSING' if not retriever_cache.exists() else '✓'}"
        )
        print(
            f"  - {embeddings_cache.name} {'✗ MISSING' if not embeddings_cache.exists() else '✓'}"
        )
        print("\nYou must run Stage 1 (memory construction) first:")
        print("  python amem_baseline.py --stage build \\")
        print("    [all cache-affecting args]")
        print("\nCache-affecting arguments:")
        print("  --image-batch-results, --video-batch-results, --email-file")
        print("  --embedding-model, --memory-provider, --memory-model")
        print("  --disable-evolution, --evo-threshold, --use-hybrid")
        print("  --caption-only, --use-short-caption, --construct-from-raw")
        print("  --include-type, --include-caption, --include-short-caption")
        print("  --include-ocr-text, --include-tags")
        print("=" * 80)
        return 1

    memory_system = build_memory_system(args, model_config)

    import pickle

    with memory_cache.open("rb") as handle:
        memory_system.memories = pickle.load(handle)
    memory_system.memory_ids = list(memory_system.memories.keys())
    memory_system.retriever.load(str(retriever_cache), str(embeddings_cache))

    if hasattr(memory_system.retriever, "alpha"):
        memory_system.retriever.alpha = args.alpha
        memory_system.alpha = args.alpha

    print(f"✓ Loaded {len(memory_system.memories)} memories from cache")
    print(f"  Cache key: {cache_key}")
    print(f"  Location:  {cache_dir}")

    output_dir = Path(args.output_dir_base) / args.method_name
    output_path = output_dir / "amem_answers.jsonl"
    if output_path.exists():
        print(f"Output exists, skipping inference: {output_path}")
        return 0

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

        retrieval_indices: List[int] = []
        retrieval_ids: List[str] = []
        retrieval_scores: List[float] = []
        top_indices: List[int] = []

        if args.no_evidence:
            context_text = ""
        else:
            log_k = max(args.retrieval_log_k, args.retrieve_k)
            if hasattr(memory_system.retriever, "search_with_scores"):
                retrieval_indices, retrieval_scores = (
                    memory_system.retriever.search_with_scores(question, log_k)
                )
            else:
                retrieval_indices = memory_system.retriever.search(question, log_k)
                retrieval_scores = [0.0 for _ in retrieval_indices]

            retrieval_ids = [
                memory_system.memory_ids[idx]
                for idx in retrieval_indices
                if idx < len(memory_system.memory_ids)
            ]

            top_indices = retrieval_indices[: args.retrieve_k]
            context_text, _ = memory_system.find_related_memories(
                question, args.retrieve_k, follow_links=args.follow_links
            )

            if args.answer_from_raw:
                pass

        if args.answer_from_raw and not args.no_evidence:
            images = []
            valid_indices = [
                idx for idx in top_indices if idx < len(memory_system.memory_ids)
            ]
            for idx in valid_indices:
                mem_id = memory_system.memory_ids[idx]
                note = memory_system.memories.get(mem_id)
                if not note:
                    continue
                modality = getattr(note, "modality", None)
                if modality is not None and modality != "image":
                    continue
                if modality is None and not note.image_path:
                    continue
                if not note.image_path:
                    print(
                        "[A-Mem][warn] Missing raw image for memory "
                        f"{mem_id}; skipping."
                    )
                    continue
                if not Path(note.image_path).exists():
                    print(
                        "[A-Mem][warn] Raw image path not found for memory "
                        f"{mem_id}: {note.image_path}"
                    )
                    continue
                images.append(note.image_path)

            llm_client = get_llm()

            messages = [{"role": "system", "content": PROMPTS["SYSTEM"]}]

            user_content = []
            evidence_block = build_evidence_block(context_text)
            text_prompt = PROMPTS["USER"].format(
                question=question, evidence=evidence_block
            )
            user_content.append({"type": "text", "text": text_prompt})

            import base64
            import mimetypes

            for img_path in images:
                if img_path.startswith("http"):
                    user_content.append(
                        {"type": "image_url", "image_url": {"url": img_path}}
                    )
                elif Path(img_path).exists():
                    mime_type, _ = mimetypes.guess_type(img_path)
                    if not mime_type:
                        mime_type = "image/jpeg"
                    try:
                        with open(img_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                            }
                        )
                    except Exception:
                        pass

            messages.append({"role": "user", "content": user_content})
            answer, usage = llm_client.chat_with_usage(messages)

        else:
            if args.no_evidence:
                context = ""
            else:
                context = context_text

            llm_client = get_llm()
            answer, usage = answer_question(llm_client, question, context)

        retrieval_recall = compute_recall(extract_evidence_ids(qa), retrieval_ids)
        retrieval_record = {
            "id": qa_id,
            "question": question,
            "gt_evidence_ids": extract_evidence_ids(qa),
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
                as_completed(future_map), total=len(future_map), desc="A-Mem QA"
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
        for qa in tqdm(qas, desc="A-Mem QA"):
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

    method_dir = output_dir
    summary = {
        "retrieval_top_k": args.retrieve_k,
        "retrieval_log_k": args.retrieval_log_k,
        "recall": summarize_recalls(retrieval_details, "retrieval_recall"),
    }
    write_json(method_dir / "retrieval_recall_summary.json", summary)
    write_json(method_dir / "retrieval_recall_details.json", retrieval_details)

    print("\n" + "=" * 80)
    print("✓ ANSWER GENERATION COMPLETE")
    print("=" * 80)
    print(f"Results saved: {output_path}")
    print(f"Total QAs:     {len(results)}")
    print("=" * 80)

    return 0


def main() -> int:
    args = parse_args()

    if args.print_cache_key:
        model_config = build_model_config(args.provider, args)
        cache_key = build_cache_key_from_args(args, model_config)
        print(f"Cache key: {cache_key}")
        print(f"Cache dir: {args.index_cache}")
        return 0

    if args.stage == "build":
        return build_memory_stage(args)
    elif args.stage == "answer":
        return answer_stage(args)
    else:
        build_memory_stage(args)
        return answer_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
