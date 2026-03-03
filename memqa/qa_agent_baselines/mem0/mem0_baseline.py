#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from memqa.global_config import get_vllm_api_key
from memqa.qa_agent_baselines.mem0.config import (
    CUSTOM_FACT_EXTRACTION_PROMPT,
    MEM0_CONFIG,
    MEM0_DEFAULT_CONFIG,
    PROMPTS,
)
from memqa.qa_agent_baselines.MMRag.llm_utils import LLMClient
from memqa.retrieve import (
    ClipRetriever,
    EmailTextConfig,
    MediaTextConfig,
    Qwen3VLRetriever,
    RetrievalItem,
    SentenceTransformerRetriever,
    TextRetriever,
    VistaRetriever,
    build_retrieval_items,
    extract_evidence_ids,
    load_json,
    load_qa_list,
    minimal_email_text_config,
    minimal_media_text_config,
    write_json,
    write_jsonl,
)

RECALL_KS = [1, 5, 10, 25, 50, 100]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_resume_map(output_file: Path, checkpoint_file: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(output_file):
        qa_id = str(row.get("id")) if row.get("id") is not None else ""
        if qa_id:
            results[qa_id] = row
    for row in load_jsonl(checkpoint_file):
        qa_id = str(row.get("id")) if row.get("id") is not None else ""
        if qa_id:
            results[qa_id] = row
    return results


def is_failed_record(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return True
    status = str(record.get("status", "")).lower()
    if status in {"failed", "fail", "error"}:
        return True
    if record.get("error"):
        return True
    answer = record.get("answer")
    if answer is None:
        return True
    if isinstance(answer, str) and not answer.strip():
        return True
    return False


def get_qa_id(qa: Dict[str, Any]) -> Optional[str]:
    qa_id = qa.get("id") or qa.get("qa_id")
    if qa_id is None:
        return None
    return str(qa_id)


def build_checkpoint_record(
    result: Optional[Dict[str, Any]],
    qa_id: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"id": qa_id}
    if result:
        record.update(result.get("answer_record") or {})
        record["retrieval_record"] = result.get("retrieval_record")
    if error:
        record["status"] = "failed"
        record["error"] = error
        record.setdefault("answer", "")
        return record
    if is_failed_record(record):
        record["status"] = "failed"
        record.setdefault("error", "empty answer")
    else:
        record["status"] = "ok"
    return record


def build_local_embedder(args: argparse.Namespace):
    try:
        from langchain.embeddings.base import Embeddings
    except ImportError as exc:
        raise RuntimeError(
            "langchain is required for local embedders. Install with `pip install langchain`."
        ) from exc

    class LocalRetrieverEmbeddings(Embeddings):
        def __init__(self, retriever):
            self.retriever = retriever

        def embed_query(self, text: str) -> List[float]:
            embedding = self.retriever.encode_query(text)
            return embedding.squeeze(0).detach().cpu().tolist()

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            embeddings = self.retriever.encode_items(
                [
                    RetrievalItem(item_id=str(idx), modality="text", text=doc)
                    for idx, doc in enumerate(texts)
                ]
            )
            return embeddings.detach().cpu().tolist()

        def __call__(self, text: str) -> List[float]:
            return self.embed_query(text)

    retriever = build_local_retriever(args)
    return LocalRetrieverEmbeddings(retriever)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mem0 QA baseline")
    parser.add_argument("--qa-file", required=True, help="Path to QA annotations JSON")
    parser.add_argument(
        "--media-source",
        choices=["batch_results", "raw"],
        default=MEM0_CONFIG["media_source"],
    )
    parser.add_argument(
        "--image-batch-results",
        default=MEM0_CONFIG["image_batch_results"],
        help="Path to image batch_results.json",
    )
    parser.add_argument(
        "--video-batch-results",
        default=MEM0_CONFIG["video_batch_results"],
        help="Path to video batch_results.json",
    )
    parser.add_argument(
        "--image-root",
        default=MEM0_CONFIG["image_root"],
        help="Root directory for raw images",
    )
    parser.add_argument(
        "--video-root",
        default=MEM0_CONFIG["video_root"],
        help="Root directory for raw videos",
    )
    parser.add_argument(
        "--email-file",
        default=MEM0_CONFIG["email_file"],
        help="Path to merged_emails.json",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "vllm", "vllm_local"],
        default=MEM0_CONFIG["provider"],
    )
    parser.add_argument("--model", default=None, help="Model name (overrides config)")
    parser.add_argument("--api-key", default=None, help="API key (overrides config)")
    parser.add_argument("--vllm-endpoint", default=None, help="VLLM endpoint URL")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout seconds")
    parser.add_argument(
        "--max-evidence-items", type=int, default=MEM0_CONFIG["max_evidence_items"]
    )
    parser.add_argument("--num-frames", type=int, default=MEM0_CONFIG["num_frames"])
    parser.add_argument("--frame-strategy", default=MEM0_CONFIG["frame_strategy"])
    parser.add_argument("--max-workers", type=int, default=MEM0_CONFIG["max_workers"])
    parser.add_argument(
        "--output-dir-base", default=MEM0_CONFIG["output_dir_base"], help="Output base"
    )
    parser.add_argument(
        "--method-name", default=MEM0_CONFIG["method_name"], help="Method name"
    )

    parser.add_argument(
        "--mem0-config",
        default=MEM0_CONFIG["mem0_config"],
        help="Optional mem0 config (JSON file path)",
    )
    parser.add_argument(
        "--mem0-user-id",
        default="dataset",
        help="mem0 user_id namespace for this run",
    )
    parser.add_argument(
        "--mem0-collection-name",
        default=MEM0_CONFIG["mem0_collection_name"],
    )
    parser.add_argument(
        "--mem0-vector-path",
        default=MEM0_CONFIG["mem0_vector_path"],
    )
    parser.add_argument(
        "--mem0-history-db-path",
        default=MEM0_CONFIG["mem0_history_db_path"],
    )
    parser.add_argument(
        "--mem0-llm-provider",
        default=MEM0_CONFIG["mem0_llm_provider"],
    )
    parser.add_argument(
        "--mem0-llm-model",
        default=MEM0_CONFIG["mem0_llm_model"],
    )
    parser.add_argument(
        "--mem0-llm-base-url",
        default=MEM0_CONFIG["mem0_llm_base_url"],
    )
    parser.add_argument(
        "--mem0-llm-temperature",
        type=float,
        default=MEM0_CONFIG["mem0_llm_temperature"],
    )
    parser.add_argument(
        "--mem0-llm-max-tokens",
        type=int,
        default=MEM0_CONFIG["mem0_llm_max_tokens"],
    )
    parser.add_argument(
        "--mem0-embedder-provider",
        default=MEM0_CONFIG["mem0_embedder_provider"],
        choices=["local", "openai", "huggingface", "ollama", "langchain", "fastembed"],
    )
    parser.add_argument(
        "--mem0-embedder-model",
        default=MEM0_CONFIG["mem0_embedder_model"],
    )
    parser.add_argument(
        "--mem0-embedder-base-url",
        default=MEM0_CONFIG["mem0_embedder_base_url"],
    )
    parser.add_argument(
        "--mem0-embedder-api-key",
        default=MEM0_CONFIG["mem0_embedder_api_key"],
    )
    parser.add_argument(
        "--mem0-embedder-dims",
        type=int,
        default=MEM0_CONFIG["mem0_embedder_dims"],
    )
    parser.add_argument(
        "--mem0-local-retriever",
        choices=["text", "sentence_transformer", "clip", "qwen3_vl", "vista"],
        default=MEM0_CONFIG["mem0_local_retriever"],
    )
    parser.add_argument(
        "--mem0-local-cache-dir",
        default=MEM0_CONFIG["mem0_local_cache_dir"],
    )
    parser.add_argument(
        "--mem0-local-device",
        default=MEM0_CONFIG["mem0_local_device"],
    )
    parser.add_argument(
        "--retriever-batch-size",
        type=int,
        default=MEM0_CONFIG["retriever_batch_size"],
    )
    parser.add_argument(
        "--text-embedding-model",
        default=MEM0_CONFIG["text_embedding_model"],
    )
    parser.add_argument(
        "--vl-embedding-model",
        default=MEM0_CONFIG["vl_embedding_model"],
    )
    parser.add_argument(
        "--clip-model",
        default=MEM0_CONFIG["clip_model"],
    )
    parser.add_argument(
        "--vista-model-name",
        default=MEM0_CONFIG["vista_model_name"],
    )
    parser.add_argument(
        "--vista-weights",
        default=MEM0_CONFIG["vista_weights"],
    )
    parser.add_argument(
        "--mem0-top-k",
        type=int,
        default=MEM0_CONFIG["mem0_top_k"],
        help="Top-k memories to retrieve",
    )
    parser.add_argument(
        "--mem0-infer",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["mem0_infer"],
        help="Enable Mem0 extraction/update during indexing",
    )
    parser.add_argument(
        "--mem0-resume",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["mem0_resume"],
        help="Skip items already indexed in the current cache",
    )
    parser.add_argument(
        "--mem0-progress-path",
        default=MEM0_CONFIG["mem0_progress_path"],
        help="Optional progress file for mem0 indexing (JSONL)",
    )
    parser.add_argument(
        "--mem0-log-k",
        type=int,
        default=MEM0_CONFIG["mem0_log_k"],
        help="Max memories to log for recall (capped at 100)",
    )
    parser.add_argument(
        "--mem0-debug-mode",
        action="store_true",
        default=False,
        help="Enable debug mode to log raw LLM requests/responses",
    )
    parser.add_argument(
        "--mem0-chat-api",
        action="store_true",
        default=False,
        help="Use mem0 Chat API for agentic answering (reuses --vllm-endpoint and --model)",
    )

    parser.add_argument(
        "--vl-text-augment",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["vl_text_augment"],
    )
    parser.add_argument(
        "--include-id",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["include_id"],
    )
    parser.add_argument(
        "--include-type",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["include_type"],
    )
    parser.add_argument(
        "--include-timestamp",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["include_timestamp"],
    )
    parser.add_argument(
        "--include-location",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["include_location"],
    )
    parser.add_argument(
        "--include-short-caption",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["include_short_caption"],
    )
    parser.add_argument(
        "--include-caption",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["include_caption"],
    )
    parser.add_argument(
        "--include-ocr-text",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["include_ocr_text"],
    )
    parser.add_argument(
        "--include-tags",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["include_tags"],
    )
    parser.add_argument(
        "--include-email-summary",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["include_email_summary"],
    )
    parser.add_argument(
        "--include-email-detail",
        action=argparse.BooleanOptionalAction,
        default=MEM0_CONFIG["include_email_detail"],
    )
    parser.add_argument(
        "--indexing-limit",
        type=int,
        default=None,
        help="Limit number of items to index (for testing)",
    )
    parser.add_argument(
        "--use-conversational-format",
        action="store_true",
        default=False,
        help="Use conversational_text field from converted batch files",
    )
    parser.add_argument(
        "--use-custom-extraction-prompt",
        action="store_true",
        default=False,
        help="Use custom fact extraction prompt for personal memory retrieval",
    )

    return parser.parse_args()


def build_model_config(provider: str, args: argparse.Namespace) -> Dict[str, Any]:
    if provider == "openai":
        base = dict(MEM0_CONFIG["openai"])
    elif provider in {"vllm", "vllm_local"}:
        base = dict(MEM0_CONFIG["vllm_text"])
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


def load_mem0_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return dict(MEM0_DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"mem0 config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_index_progress(progress_path: Path, mem0_config: Dict[str, Any]) -> set[str]:
    completed: set[str] = set()
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = {"item_id": line}
                item_id = payload.get("item_id") if isinstance(payload, dict) else None
                if item_id:
                    completed.add(str(item_id))

    vector_store = (
        mem0_config.get("vector_store") if isinstance(mem0_config, dict) else None
    )
    if not isinstance(vector_store, dict):
        return completed
    if vector_store.get("provider") != "chroma":
        return completed
    config = vector_store.get("config")
    if not isinstance(config, dict):
        return completed
    chroma_path = config.get("path")
    if not chroma_path:
        return completed

    sqlite_path = Path(chroma_path) / "chroma.sqlite3"
    if not sqlite_path.exists():
        return completed

    try:
        import sqlite3
    except ImportError:
        return completed

    try:
        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT em1.string_value
            FROM embedding_metadata em1
            JOIN embedding_metadata em2 ON em1.id = em2.id
            WHERE em1.key='item_id'
              AND em1.string_value IS NOT NULL
              AND em1.string_value != ''
              AND em2.key='data'
              AND em2.string_value IS NOT NULL
              AND em2.string_value != ''
            """
        )
        rows = cur.fetchall()
        completed.update({row[0] for row in rows if row and row[0]})
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return completed


def append_index_progress(handle, item_id: str, payload: Dict[str, Any]) -> None:
    record = {
        "item_id": item_id,
        "modality": payload.get("modality"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def build_local_retriever(args: argparse.Namespace):
    cache_dir = Path(args.mem0_local_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    retriever_name = args.mem0_local_retriever
    if retriever_name == "text":
        return TextRetriever(
            model_name=args.text_embedding_model,
            cache_dir=cache_dir,
            batch_size=args.retriever_batch_size,
            device=args.mem0_local_device,
        )
    if retriever_name == "sentence_transformer":
        return SentenceTransformerRetriever(
            model_name=args.text_embedding_model,
            cache_dir=cache_dir,
            batch_size=args.retriever_batch_size,
            device=args.mem0_local_device,
        )
    if retriever_name == "clip":
        return ClipRetriever(
            model_name=args.clip_model,
            cache_dir=cache_dir,
            batch_size=args.retriever_batch_size,
            device=args.mem0_local_device,
        )
    if retriever_name == "qwen3_vl":
        return Qwen3VLRetriever(
            model_name=args.vl_embedding_model,
            cache_dir=cache_dir,
            batch_size=args.retriever_batch_size,
            device=args.mem0_local_device,
            num_frames=args.num_frames,
            max_frames=args.num_frames,
        )
    if retriever_name == "vista":
        if not args.vista_weights:
            raise ValueError("vista_weights is required for vista retriever")
        return VistaRetriever(
            model_name=args.vista_model_name,
            weights_path=args.vista_weights,
            cache_dir=cache_dir,
            batch_size=args.retriever_batch_size,
            device=args.mem0_local_device,
        )
    raise ValueError(f"Unsupported local retriever: {retriever_name}")


def create_mem0_instance(mem0_config: Dict[str, Any]):
    import importlib

    try:
        mem0_module = importlib.import_module("mem0")
    except ImportError:
        vendored_root = Path(__file__).resolve().parents[3] / "third_party" / "mem0"
        if (vendored_root / "mem0").exists():
            sys.path.insert(0, str(vendored_root))
            mem0_module = importlib.import_module("mem0")
        else:
            raise RuntimeError(
                "mem0 is required. Install with `pip install mem0ai` (or equivalent), "
                "or ensure the vendored copy exists at `third_party/mem0/mem0/`."
            )

    mem0_memory = getattr(mem0_module, "Memory")
    return mem0_memory.from_config(mem0_config)


def build_mem0_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(MEM0_DEFAULT_CONFIG)
    config["vector_store"] = {
        "provider": "chroma",
        "config": {
            "collection_name": args.mem0_collection_name,
            "path": args.mem0_vector_path,
        },
    }
    config["history_db_path"] = args.mem0_history_db_path

    if args.mem0_llm_provider:
        config["llm"] = {
            "provider": args.mem0_llm_provider,
            "config": {
                "model": args.mem0_llm_model,
                "temperature": args.mem0_llm_temperature,
                "max_tokens": args.mem0_llm_max_tokens,
                "vllm_base_url": args.mem0_llm_base_url,
                "api_key": get_vllm_api_key(),
            },
        }

    if args.mem0_embedder_provider != "local":
        config["embedder"] = {
            "provider": args.mem0_embedder_provider,
            "config": {
                "model": args.mem0_embedder_model,
                "openai_base_url": args.mem0_embedder_base_url,
                "api_key": args.mem0_embedder_api_key,
                "embedding_dims": args.mem0_embedder_dims,
            },
        }
    else:
        config["embedder"] = {
            "provider": "langchain",
            "config": {
                "model": build_local_embedder(args),
                "embedding_dims": args.mem0_embedder_dims,
            },
        }

    if args.use_custom_extraction_prompt:
        config["custom_fact_extraction_prompt"] = CUSTOM_FACT_EXTRACTION_PROMPT

    return config


def build_memory_payloads(items: List[RetrievalItem]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for item in items:
        if not item.text:
            continue
        payloads.append(
            {
                "item_id": item.item_id,
                "modality": item.modality,
                "text": item.text,
                "metadata": {
                    "item_id": item.item_id,
                    "modality": item.modality,
                    **(item.metadata or {}),
                },
            }
        )
    return payloads


def build_conversational_payloads(
    image_entries: List[Dict[str, Any]],
    video_entries: List[Dict[str, Any]],
    email_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []

    for entry in email_entries:
        text = entry.get("conversational_text")
        if not text:
            continue
        item_id = entry.get("id", "")
        payloads.append(
            {
                "item_id": item_id,
                "modality": "email",
                "text": text,
                "metadata": {
                    "item_id": item_id,
                    "modality": "email",
                    "source": "email",
                    "timestamp": entry.get("timestamp", ""),
                },
            }
        )

    for entry in image_entries:
        text = entry.get("conversational_text")
        if not text:
            continue
        raw_path = entry.get("image_path", "")
        item_id = Path(raw_path).stem if raw_path else ""
        payloads.append(
            {
                "item_id": item_id,
                "modality": "image",
                "text": text,
                "metadata": {
                    "item_id": item_id,
                    "modality": "image",
                    "source": "image",
                    "timestamp": entry.get("timestamp", ""),
                    "location": entry.get("location_name", "")
                    or entry.get("location", ""),
                },
            }
        )

    for entry in video_entries:
        text = entry.get("conversational_text")
        if not text:
            continue
        raw_path = entry.get("video_path", "")
        item_id = Path(raw_path).stem if raw_path else ""
        payloads.append(
            {
                "item_id": item_id,
                "modality": "video",
                "text": text,
                "metadata": {
                    "item_id": item_id,
                    "modality": "video",
                    "source": "video",
                    "timestamp": entry.get("timestamp", ""),
                    "location": entry.get("location_name", "")
                    or entry.get("location", ""),
                },
            }
        )

    payloads.sort(key=lambda x: x.get("metadata", {}).get("timestamp", ""))
    return payloads


def index_memories(
    mem0_instance,
    payloads: List[Dict[str, Any]],
    user_id: str,
    infer: bool,
    resume: bool,
    progress_path: Path,
    mem0_config: Dict[str, Any],
    debug_mode: bool = False,
) -> None:
    completed: set[str] = set()
    if resume:
        completed = load_index_progress(progress_path, mem0_config)
        if completed:
            print(f"[mem0] resume enabled: {len(completed)} items already indexed")

    progress_handle = None
    if resume:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_handle = progress_path.open("a", encoding="utf-8")

    debug_dir = None
    if debug_mode:
        debug_dir = progress_path.parent / "debug_output"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[mem0] debug mode enabled: logging to {debug_dir}")

    for payload in tqdm(payloads, desc="mem0 indexing"):
        item_id = payload.get("item_id") or payload.get("metadata", {}).get("item_id")
        if resume and item_id and str(item_id) in completed:
            continue
        messages = [
            {"role": "user", "content": payload["text"]},
        ]

        # Debug: log input text
        if debug_mode and debug_dir and item_id:
            debug_file = debug_dir / f"{item_id}.json"
            debug_data = {
                "item_id": item_id,
                "modality": payload.get("modality"),
                "metadata": payload.get("metadata"),
                "input_text": payload["text"],
                "messages": messages,
            }

        try:
            result = mem0_instance.add(
                messages,
                user_id=user_id,
                metadata=payload.get("metadata"),
                infer=infer,
            )
        except Exception as exc:
            item_label = item_id if item_id else "unknown"
            print(f"[mem0] index failed for {item_label}: {exc}")

            # Debug: log exception
            if debug_mode and debug_dir and item_id:
                debug_data["error"] = str(exc)
                debug_data["error_type"] = type(exc).__name__
                debug_data["status"] = "failed"
                try:
                    with open(debug_file, "w", encoding="utf-8") as f:
                        json.dump(debug_data, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            continue

        results = result.get("results") if isinstance(result, dict) else None

        # Debug: log response
        if debug_mode and debug_dir and item_id:
            debug_data["raw_response"] = result
            debug_data["extracted_results"] = results
            debug_data["results_count"] = len(results) if results else 0
            debug_data["status"] = (
                "empty_results" if (results is not None and not results) else "success"
            )
            try:
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump(debug_data, f, indent=2, ensure_ascii=False)
            except Exception as write_exc:
                print(f"[mem0] debug write failed for {item_id}: {write_exc}")

        if results is not None and not results:
            item_label = item_id if item_id else "unknown"
            print(f"[mem0] no memories extracted for {item_label}; will retry")
            continue
        if progress_handle and item_id:
            append_index_progress(progress_handle, str(item_id), payload)
            completed.add(str(item_id))

    if progress_handle:
        progress_handle.close()


def extract_item_id_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("ID:"):
            return line.replace("ID:", "", 1).strip() or None
    return None


def retrieve_memories(
    mem0_instance, query: str, user_id: str, top_k: int
) -> List[Dict[str, Any]]:
    results = mem0_instance.search(query=query, user_id=user_id, limit=top_k)
    memories: List[Dict[str, Any]] = []
    if isinstance(results, dict):
        data = results.get("results") or results.get("data")
        results = data if isinstance(data, list) else []
    if not isinstance(results, list):
        return memories
    for result in results:
        metadata: Dict[str, Any] = {}
        score = 0.0
        if isinstance(result, dict):
            memory_text = result.get("memory") or result.get("text")
            score = float(result.get("score") or 0.0)
            metadata = result.get("metadata") or {}
            item_id = (
                metadata.get("item_id")
                or result.get("item_id")
                or extract_item_id_from_text(memory_text)
            )
        else:
            memory_text = getattr(result, "memory", None) or getattr(
                result, "text", None
            )
            score = float(getattr(result, "score", 0.0) or 0.0)
            metadata = getattr(result, "metadata", None) or {}
            item_id = metadata.get("item_id") or extract_item_id_from_text(memory_text)
        if not memory_text:
            continue
        memories.append(
            {
                "text": str(memory_text),
                "item_id": str(item_id) if item_id else None,
                "score": score,
            }
        )
    return memories


def build_messages(question: str, memories: List[str]) -> List[Dict[str, Any]]:
    evidence_text = "\n".join(
        [f"Memory {idx + 1}:\n{chunk}" for idx, chunk in enumerate(memories)]
    )
    return [
        {"role": "system", "content": PROMPTS["SYSTEM"]},
        {
            "role": "user",
            "content": PROMPTS["USER_TEXT"].format(
                question=question, evidence=evidence_text
            ),
        },
    ]


def answer_question_chat_api(
    qa: Dict[str, Any],
    args: argparse.Namespace,
    mem0_instance,
) -> Optional[Dict[str, Any]]:
    """
    Answer question using mem0-style agentic answering with automatic memory retrieval.
    Implements the same logic as mem0's Chat API but without the function calling requirement.
    Reuses --vllm-endpoint and --model for the answering LLM.
    """
    qa_id = qa.get("id") or qa.get("qa_id")
    question = qa.get("question")
    if not qa_id or not question:
        return None

    # First, retrieve memories for logging/evaluation purposes
    search_k = max(args.mem0_top_k, args.mem0_log_k)
    retrieved = retrieve_memories(
        mem0_instance, question, user_id=args.mem0_user_id, top_k=search_k
    )

    log_k = min(args.mem0_log_k, 100, len(retrieved))
    retrieval_ids: List[str] = []
    retrieval_scores: List[float] = []
    for item in retrieved[:log_k]:
        if not item.get("item_id"):
            continue
        retrieval_ids.append(item["item_id"])
        retrieval_scores.append(float(item.get("score") or 0.0))

    gt_evidence_ids = extract_evidence_ids(qa)
    retrieval_recall = compute_recall(gt_evidence_ids, retrieval_ids)

    # Implement agentic answering manually (same as Chat API but without function calling requirement)
    # Step 1: Retrieve relevant memories (already done above)
    # Step 2: Build augmented prompt with memories
    memories = [item["text"] for item in retrieved[: args.mem0_top_k]]
    memory_context = "\n".join(
        [f"Memory {idx + 1}: {mem}" for idx, mem in enumerate(memories)]
    )

    # Use mem0's MEMORY_ANSWER_PROMPT format
    from mem0.configs.prompts import MEMORY_ANSWER_PROMPT

    augmented_prompt = f"""Relevant Memories/Facts:
{memory_context}

User Question: {question}"""

    messages = [
        {"role": "system", "content": MEMORY_ANSWER_PROMPT},
        {"role": "user", "content": augmented_prompt},
    ]

    # Step 3: Call LLM directly with augmented prompt
    from memqa.qa_agent_baselines.MMRag.llm_utils import LLMClient

    llm_config = build_model_config(args.provider, args)
    llm = LLMClient(args.provider, llm_config)

    try:
        answer, usage = llm.chat_with_usage(messages)
    except Exception as exc:
        print(f"[mem0 Agentic] Error for QA {qa_id}: {exc}")
        answer = ""
        usage = None

    answer_record: Dict[str, Any] = {"id": qa_id, "answer": answer}
    if usage:
        answer_record["prompt_tokens"] = usage.prompt_tokens
        answer_record["completion_tokens"] = usage.completion_tokens
        answer_record["total_tokens"] = usage.total_tokens

    return {
        "answer_record": answer_record,
        "retrieval_record": {
            "id": qa_id,
            "question": question,
            "gt_evidence_ids": gt_evidence_ids,
            "retrieval_ids": retrieval_ids,
            "retrieval_scores": retrieval_scores,
            "retrieval_recall": retrieval_recall,
        },
    }


def answer_question(
    qa: Dict[str, Any],
    args: argparse.Namespace,
    mem0_instance,
    llm_getter,
) -> Optional[Dict[str, Any]]:
    qa_id = qa.get("id") or qa.get("qa_id")
    question = qa.get("question")
    if not qa_id or not question:
        return None

    search_k = max(args.mem0_top_k, args.mem0_log_k)
    retrieved = retrieve_memories(
        mem0_instance, question, user_id=args.mem0_user_id, top_k=search_k
    )
    memories = [item["text"] for item in retrieved[: args.mem0_top_k]]

    log_k = min(args.mem0_log_k, 100, len(retrieved))
    retrieval_ids: List[str] = []
    retrieval_scores: List[float] = []
    for item in retrieved[:log_k]:
        if not item.get("item_id"):
            continue
        retrieval_ids.append(item["item_id"])
        retrieval_scores.append(float(item.get("score") or 0.0))

    gt_evidence_ids = extract_evidence_ids(qa)
    retrieval_recall = compute_recall(gt_evidence_ids, retrieval_ids)

    messages = build_messages(question, memories)
    llm = llm_getter()
    answer, usage = llm.chat_with_usage(messages)
    answer_record: Dict[str, Any] = {"id": qa_id, "answer": answer}
    if usage:
        answer_record["prompt_tokens"] = usage.prompt_tokens
        answer_record["completion_tokens"] = usage.completion_tokens
        answer_record["total_tokens"] = usage.total_tokens
    return {
        "answer_record": answer_record,
        "retrieval_record": {
            "id": qa_id,
            "question": question,
            "gt_evidence_ids": gt_evidence_ids,
            "retrieval_ids": retrieval_ids,
            "retrieval_scores": retrieval_scores,
            "retrieval_recall": retrieval_recall,
        },
    }


def main() -> int:
    args = parse_args()

    output_base = Path(args.output_dir_base)
    method_dir = output_base / args.method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    output_file = method_dir / "mem0_answers.jsonl"
    checkpoint_file = method_dir / "mem0_answers_checkpoint.jsonl"

    qa_data = load_json(Path(args.qa_file))
    qas = load_qa_list(qa_data)

    media_config, email_config = build_text_configs(args)
    vl_media_config = (
        media_config if args.vl_text_augment else minimal_media_text_config()
    )
    vl_email_config = (
        email_config if args.vl_text_augment else minimal_email_text_config()
    )

    image_batch_data = load_json(Path(args.image_batch_results))
    video_batch_data = load_json(Path(args.video_batch_results))
    email_entries = load_json(Path(args.email_file))

    if not isinstance(image_batch_data, list) or not isinstance(video_batch_data, list):
        raise ValueError("Batch results must be lists")
    if not isinstance(email_entries, list):
        raise ValueError("Email file should be a list")

    retrieval_items = build_retrieval_items(
        email_entries,
        image_batch_data,
        video_batch_data,
        vl_media_config,
        vl_email_config,
        Path(args.image_root),
        Path(args.video_root),
    )

    mem0_config = load_mem0_config(args.mem0_config)
    mem0_config = build_mem0_config(args) if args.mem0_config is None else mem0_config
    mem0_instance = create_mem0_instance(mem0_config)

    if args.use_conversational_format:
        print("[mem0] Using conversational format from pre-converted data")
        payloads = build_conversational_payloads(
            image_batch_data,
            video_batch_data,
            email_entries,
        )
        if not payloads:
            raise ValueError(
                "No conversational_text fields found. Run convert_to_conversational.py first."
            )
        print(f"[mem0] Built {len(payloads)} conversational payloads")
    else:
        payloads = build_memory_payloads(retrieval_items)
    if args.indexing_limit:
        payloads = payloads[: args.indexing_limit]
    progress_path = (
        Path(args.mem0_progress_path)
        if args.mem0_progress_path
        else Path(args.output_dir_base) / args.method_name / "mem0_index_progress.jsonl"
    )
    index_memories(
        mem0_instance,
        payloads,
        user_id=args.mem0_user_id,
        infer=args.mem0_infer,
        resume=args.mem0_resume,
        progress_path=progress_path,
        mem0_config=mem0_config,
        debug_mode=args.mem0_debug_mode,
    )

    qas_with_ids = [qa for qa in qas if get_qa_id(qa)]
    resume_map = load_resume_map(output_file, checkpoint_file)
    pending_qas = [
        qa
        for qa in qas_with_ids
        if is_failed_record(resume_map.get(get_qa_id(qa)))
    ]
    if resume_map:
        completed = len(qas_with_ids) - len(pending_qas)
        print(
            f"[mem0] Resume: {completed}/{len(qas_with_ids)} complete, "
            f"{len(pending_qas)} pending."
        )

    if pending_qas:
        if args.provider == "vllm_local" and args.max_workers > 1:
            raise ValueError("vllm_local does not support concurrent execution.")

        if args.mem0_chat_api:
            # Use mem0-style agentic answering with automatic memory retrieval
            print(
                "[mem0] Using agentic answering mode (auto memory retrieval + augmented prompts)"
            )
            print(f"[mem0] Model: {args.model}")
            print(f"[mem0] Endpoint: {args.vllm_endpoint}")
            print(f"[mem0] Top-k memories: {args.mem0_top_k}")

            for qa in tqdm(pending_qas, desc="mem0 Agentic QA"):
                qa_id = get_qa_id(qa)
                if not qa_id:
                    continue
                try:
                    result = answer_question_chat_api(qa, args, mem0_instance)
                    if not result:
                        record = build_checkpoint_record(
                            None, qa_id, error="missing question or id"
                        )
                    else:
                        record = build_checkpoint_record(result, qa_id)
                except Exception as exc:
                    print(
                        f"[mem0 Agentic] Failed id={qa_id}: {exc}",
                        file=sys.stderr,
                    )
                    record = build_checkpoint_record(None, qa_id, error=str(exc))
                append_jsonl(checkpoint_file, record)
                resume_map[qa_id] = record
        else:
            # Standard mode - use manual retrieval + LLM answering
            llm_config = build_model_config(args.provider, args)
            thread_local = threading.local()

            def get_llm() -> LLMClient:
                if not hasattr(thread_local, "llm"):
                    thread_local.llm = LLMClient(args.provider, llm_config)
                return thread_local.llm

            def answer_single(qa: Dict[str, Any]) -> Dict[str, Any]:
                qa_id = get_qa_id(qa) or "unknown"
                try:
                    result = answer_question(qa, args, mem0_instance, get_llm)
                    if not result:
                        return build_checkpoint_record(
                            None, qa_id, error="missing question or id"
                        )
                    return build_checkpoint_record(result, qa_id)
                except Exception as exc:
                    print(f"[mem0 QA] Failed id={qa_id}: {exc}", file=sys.stderr)
                    return build_checkpoint_record(None, qa_id, error=str(exc))

            if args.max_workers and args.max_workers > 1:
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    future_map = {
                        executor.submit(answer_single, qa): qa for qa in pending_qas
                    }
                    for future in tqdm(
                        as_completed(future_map),
                        total=len(future_map),
                        desc="mem0 QA",
                    ):
                        qa = future_map[future]
                        qa_id = get_qa_id(qa) or "unknown"
                        try:
                            record = future.result()
                        except Exception as exc:
                            print(
                                f"[mem0 QA] Failed id={qa_id}: {exc}",
                                file=sys.stderr,
                            )
                            record = build_checkpoint_record(
                                None, qa_id, error=str(exc)
                            )
                        append_jsonl(checkpoint_file, record)
                        resume_map[qa_id] = record
            else:
                for qa in tqdm(pending_qas, desc="mem0 QA"):
                    record = answer_single(qa)
                    append_jsonl(checkpoint_file, record)
                    resume_map[record["id"]] = record
    else:
        print("[mem0] All answers already present; skipping inference.")

    results: List[Dict[str, Any]] = []
    retrieval_details: List[Dict[str, Any]] = []
    for qa in qas_with_ids:
        qa_id = get_qa_id(qa)
        if not qa_id:
            continue
        record = resume_map.get(qa_id)
        if not record:
            continue
        retrieval_record = record.get("retrieval_record")
        if retrieval_record:
            retrieval_details.append(retrieval_record)
        if not is_failed_record(record):
            output_record = {
                key: value
                for key, value in record.items()
                if key not in {"status", "error", "retrieval_record"}
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

    stats_path = output_file.parent / f"{output_file.stem}_run_stats.json"
    write_json(stats_path, run_stats)
    print(f"Token stats written to: {stats_path}")

    if retrieval_details:
        summary = {
            "retrieval_top_k": args.mem0_top_k,
            "retrieval_log_k": min(args.mem0_log_k, 100),
            "recall": summarize_recalls(retrieval_details, "retrieval_recall"),
        }
        write_json(method_dir / "retrieval_recall_summary.json", summary)
        write_json(method_dir / "retrieval_recall_details.json", retrieval_details)
    else:
        print("[mem0] No retrieval details available; skipping recall summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
