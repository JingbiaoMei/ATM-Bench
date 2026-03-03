#!/usr/bin/env python3
"""
HippoRAG 2 Baseline for PersonalMemoryQA (strict reproduction).

This implementation delegates indexing/retrieval to the original HippoRAG 2
codebase, while adapting PersonalMemoryQA data formats and output conventions.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from memqa.qa_agent_baselines.HippoRag2.config import (
    HIPPORAG2_CONFIG,
    HIPPORAG2_PROMPTS,
)
from memqa.qa_agent_baselines.HippoRag2.data_adapter import CorpusBuilder
from memqa.retrieve import (
    extract_evidence_ids,
    load_json,
    load_qa_list,
    write_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HippoRAG 2 Integration Helpers
# =============================================================================


def import_hipporag(hipporag_repo: Optional[str]) -> Tuple[Any, Any]:
    if hipporag_repo:
        repo_path = Path(hipporag_repo).expanduser().resolve()
        if repo_path.exists():
            src_path = repo_path / "src"
            sys.path.insert(0, str(src_path if src_path.exists() else repo_path))

    try:
        from hipporag import HippoRAG  # type: ignore
        from hipporag.utils.config_utils import BaseConfig  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "igraph":
            raise RuntimeError(
                "Missing dependency: python-igraph. Install it (e.g., "
                "`pip install python-igraph`) and retry."
            ) from exc
        raise RuntimeError(
            "HippoRAG source not found. Set --hipporag-repo to the cloned repo "
            "or install the HippoRAG package."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "HippoRAG import failed. Verify the repo path and dependencies."
        ) from exc

    return HippoRAG, BaseConfig


def normalize_embedding_model(
    embedding_model: str,
    embedding_endpoint: Optional[str],
    embedding_mode: str = "auto",
) -> Tuple[str, Optional[str], str]:
    """
    Normalize embedding model name and determine the embedding mode.

    Args:
        embedding_model: Model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
        embedding_endpoint: Optional API endpoint for VLLM embeddings
        embedding_mode: "auto", "local", or "api"
            - "auto": Automatically determine based on model name and endpoint
            - "local": Force local HuggingFace inference (no endpoint required)
            - "api": Force API-based inference (endpoint required for Qwen)

    Returns:
        Tuple of (normalized_model_name, endpoint, effective_mode)
    """
    # Explicit local mode: use HuggingFace directly
    if embedding_mode == "local":
        return f"HuggingFace/{embedding_model}", None, "local"

    # Already prefixed models
    if embedding_model.startswith("Transformers/"):
        return embedding_model, embedding_endpoint, "transformers"
    if embedding_model.startswith("VLLM/"):
        if not embedding_endpoint:
            raise ValueError("--embedding-endpoint is required for VLLM embeddings.")
        return embedding_model, embedding_endpoint, "api"
    if embedding_model.startswith("HuggingFace/"):
        return embedding_model, None, "local"

    # NV-Embed-v2
    if "NV-Embed-v2" in embedding_model:
        return embedding_model, embedding_endpoint, "nvembed"

    # all-MiniLM (sentence-transformers)
    if embedding_model.lower().startswith("all-minilm"):
        return f"Transformers/{embedding_model}", embedding_endpoint, "transformers"

    # Qwen embeddings
    if "Qwen" in embedding_model or "qwen" in embedding_model:
        if embedding_mode == "api" or embedding_endpoint:
            if not embedding_endpoint:
                raise ValueError(
                    "--embedding-endpoint is required for Qwen embeddings in API mode. "
                    "Use --embedding-mode local to run locally with HuggingFace."
                )
            return f"VLLM/{embedding_model}", embedding_endpoint, "api"
        else:
            # Default to local for Qwen when no endpoint provided
            logger.info(
                f"No embedding endpoint provided for {embedding_model}; using local HuggingFace mode."
            )
            return f"HuggingFace/{embedding_model}", None, "local"

    return embedding_model, embedding_endpoint, "auto"


def normalize_llm_base_url(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return None
    if endpoint.endswith("/chat/completions"):
        return endpoint.rsplit("/chat/completions", 1)[0]
    return endpoint


def build_cache_key(values: Dict[str, Any]) -> str:
    payload = hashlib.md5(str(sorted(values.items())).encode("utf-8")).hexdigest()
    return payload[:12]


def normalize_evidence_id(value: str) -> str:
    value = value.strip()
    for prefix in ("Image_", "Video_", "Email_"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def build_text_to_id_map(passages: Iterable[Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for passage in passages:
        keys: List[str] = []
        if getattr(passage, "text", ""):
            keys.append(passage.text)
            first_line = passage.text.splitlines()[0].strip()
            if first_line:
                keys.append(first_line)
                keys.append(normalize_evidence_id(first_line))
        if getattr(passage, "title", ""):
            keys.append(passage.title)
            keys.append(normalize_evidence_id(passage.title))
        for key in keys:
            if not key:
                continue
            if key in mapping and mapping[key] != passage.evidence_id:
                logger.warning(
                    "Duplicate passage key with conflicting evidence IDs: %s", key
                )
                continue
            mapping[key] = passage.evidence_id
    return mapping


def parse_evidence_id(text: str, fallback_map: Dict[str, str]) -> str:
    if text in fallback_map:
        return fallback_map[text]

    first_line = text.splitlines()[0].strip()
    if " | " in first_line:
        return normalize_evidence_id(first_line.split(" | ", 1)[0].strip())
    if first_line.lower().startswith("wikipedia title:"):
        return normalize_evidence_id(first_line.split(":", 1)[1].strip())
    if first_line.lower().startswith("id:"):
        return normalize_evidence_id(first_line.split(":", 1)[1].strip())
    return normalize_evidence_id(first_line)


def extract_token_usage(metadata: Any) -> Dict[str, int]:
    usage: Dict[str, Any] = {}
    if isinstance(metadata, dict):
        if isinstance(metadata.get("usage"), dict):
            usage = metadata["usage"]
        else:
            usage = metadata
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    try:
        prompt_tokens = int(prompt)
    except (TypeError, ValueError):
        prompt_tokens = 0
    try:
        completion_tokens = int(completion)
    except (TypeError, ValueError):
        completion_tokens = 0
    total_tokens = usage.get("total_tokens")
    try:
        total_tokens_int = int(total_tokens)
    except (TypeError, ValueError):
        total_tokens_int = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens_int,
    }


def summarize_token_usage(
    usages: List[Dict[str, int]], num_samples: int
) -> Dict[str, Any]:
    total_prompt = sum(item.get("prompt_tokens", 0) for item in usages)
    total_completion = sum(item.get("completion_tokens", 0) for item in usages)
    total_tokens = sum(item.get("total_tokens", 0) for item in usages)
    avg_prompt = total_prompt / num_samples if num_samples else 0
    avg_completion = total_completion / num_samples if num_samples else 0
    avg_total = total_tokens / num_samples if num_samples else 0
    summary: Dict[str, Any] = {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "avg_prompt_tokens": avg_prompt,
        "avg_completion_tokens": avg_completion,
        "avg_total_tokens": avg_total,
    }
    if total_tokens == 0:
        summary["note"] = (
            "Token usage not reported in HippoRAG LLM metadata; totals are zero."
        )
    return summary


# =============================================================================
# Retrieval Metrics
# =============================================================================


def compute_recall(gt_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    total = len(gt_ids)
    gt_set = set(gt_ids)
    recall_ks = [1, 5, 10, 25, 50, 100]
    recalls: Dict[str, float] = {}
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


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HippoRAG 2 baseline for PersonalMemoryQA"
    )

    # Input/Output
    parser.add_argument("--qa-file", required=True, help="Path to QA JSON file")
    parser.add_argument(
        "--output-dir-base", default=HIPPORAG2_CONFIG["output_dir_base"]
    )
    parser.add_argument("--method-name", default="hipporag2_default")
    parser.add_argument("--index-cache", default=HIPPORAG2_CONFIG["index_cache_dir"])

    # HippoRAG repo
    parser.add_argument(
        "--hipporag-repo",
        default=HIPPORAG2_CONFIG.get("hipporag_repo") or None,
        help="Path to cloned HippoRAG repository",
    )

    # Data sources
    parser.add_argument(
        "--media-source",
        choices=["batch_results", "raw"],
        default=HIPPORAG2_CONFIG["media_source"],
    )
    parser.add_argument(
        "--image-batch-results", default=HIPPORAG2_CONFIG["image_batch_results"]
    )
    parser.add_argument(
        "--video-batch-results", default=HIPPORAG2_CONFIG["video_batch_results"]
    )
    parser.add_argument("--email-file", default=HIPPORAG2_CONFIG["email_file"])
    parser.add_argument("--image-root", default=HIPPORAG2_CONFIG["image_root"])
    parser.add_argument("--video-root", default=HIPPORAG2_CONFIG["video_root"])

    # Augmentation level (MMRag aligned)
    parser.add_argument(
        "--augmentation-level",
        choices=[
            "short_caption_only",
            "short_caption_caption",
            "short_caption_caption_tag",
            "short_caption_caption_tag_ocr",
        ],
        default="short_caption_only",
        help="Text augmentation level for images/videos",
    )

    # Embedding model
    parser.add_argument(
        "--embedding-model", default=HIPPORAG2_CONFIG["embedding_model"]
    )
    parser.add_argument(
        "--embedding-endpoint", default=HIPPORAG2_CONFIG.get("embedding_base_url")
    )
    parser.add_argument(
        "--embedding-mode",
        choices=["auto", "local", "api"],
        default="auto",
        help=(
            "Embedding mode: 'auto' (determine from model/endpoint), "
            "'local' (HuggingFace on device), 'api' (VLLM endpoint required)"
        ),
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Device for local embeddings (e.g., 'cuda', 'cpu'). Auto-detect if not set.",
    )

    # LLM provider
    parser.add_argument(
        "--provider",
        choices=["openai", "vllm", "vllm_local"],
        default=HIPPORAG2_CONFIG["provider"],
    )
    parser.add_argument("--model", default=None, help="Model name (overrides config)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--vllm-endpoint", default=None)
    parser.add_argument(
        "--answerer-model",
        default=None,
        help="Override model for QA answerer + rerank (keeps index cache from --model).",
    )
    parser.add_argument(
        "--answerer-endpoint",
        default=None,
        help="Override endpoint for QA answerer + rerank.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=HIPPORAG2_CONFIG["timeout"])

    # HippoRAG 2 hyperparameters (paper defaults)
    parser.add_argument(
        "--retrieval-top-k", type=int, default=HIPPORAG2_CONFIG["retrieval_top_k"]
    )
    parser.add_argument(
        "--linking-top-k", type=int, default=HIPPORAG2_CONFIG["linking_top_k"]
    )
    parser.add_argument("--qa-top-k", type=int, default=HIPPORAG2_CONFIG["qa_top_k"])
    parser.add_argument(
        "--max-qa-steps", type=int, default=HIPPORAG2_CONFIG["max_qa_steps"]
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=HIPPORAG2_CONFIG["embedding_batch_size"],
    )
    parser.add_argument("--graph-type", default=HIPPORAG2_CONFIG["graph_type"])
    parser.add_argument(
        "--directed-graph",
        action=argparse.BooleanOptionalAction,
        default=HIPPORAG2_CONFIG["is_directed_graph"],
        help="Whether to use a directed graph for PPR",
    )
    parser.add_argument(
        "--damping", type=float, default=HIPPORAG2_CONFIG.get("damping", 0.5)
    )

    # Pipeline control
    parser.add_argument(
        "--stage",
        choices=["build", "answer", "all"],
        default="all",
        help="Pipeline stage: 'build' (index only), 'answer' (QA only), 'all' (both)",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true", help="Rebuild index from scratch"
    )
    parser.add_argument(
        "--force-openie", action="store_true", help="Re-run OpenIE from scratch"
    )
    parser.add_argument(
        "--max-workers", type=int, default=HIPPORAG2_CONFIG["max_workers"]
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=HIPPORAG2_CONFIG["checkpoint_interval"],
    )
    parser.add_argument(
        "--force-answer",
        action="store_true",
        help="Re-run answer stage even if answers already exist.",
    )

    # QA settings
    parser.add_argument(
        "--no-evidence", action="store_true", default=HIPPORAG2_CONFIG["no_evidence"]
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit QA items for debugging"
    )
    parser.add_argument("--max-evidence-items", type=int, default=None)

    # Retrieval logging
    parser.add_argument(
        "--retrieval-log-k", type=int, default=100, help="Log top-K retrieved items"
    )

    # OpenIE mode
    parser.add_argument(
        "--openie-mode",
        choices=["online", "offline", "Transformers-offline"],
        default=HIPPORAG2_CONFIG.get("openie_mode", "online"),
    )
    parser.add_argument(
        "--openie-workers",
        type=int,
        default=None,
        help="Max workers for OpenIE thread pool (online mode).",
    )

    return parser.parse_args()


# =============================================================================
# Pipeline
# =============================================================================


def resolve_llm_settings(
    args: argparse.Namespace,
) -> Tuple[str, Optional[str], Optional[str]]:
    if args.provider == "openai":
        model_name = args.model or HIPPORAG2_CONFIG["openai"]["model"]
        api_key = args.api_key or HIPPORAG2_CONFIG["openai"].get("api_key")
        llm_base_url = None
    else:
        model_name = args.model or HIPPORAG2_CONFIG["vllm_text"]["model"]
        api_key = args.api_key or HIPPORAG2_CONFIG["vllm_text"].get("api_key")
        llm_base_url = normalize_llm_base_url(
            args.vllm_endpoint or HIPPORAG2_CONFIG["vllm_text"]["endpoint"]
        )

    return model_name, api_key, llm_base_url


def build_hipporag(
    args: argparse.Namespace,
    index_dir: Path,
    corpus_size: int,
    use_answerer: bool = False,
) -> Any:
    HippoRAG, BaseConfig = import_hipporag(args.hipporag_repo)

    embedding_model, embedding_base_url, embedding_mode = normalize_embedding_model(
        args.embedding_model,
        args.embedding_endpoint,
        getattr(args, "embedding_mode", "auto"),
    )

    model_name, api_key, llm_base_url = resolve_llm_settings(args)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "sk-"

    # For local mode, we still need to pass a valid embedding config to HippoRAG
    # We'll inject our local embedder after initialization
    hipporag_embedding_model = embedding_model
    hipporag_embedding_base_url = embedding_base_url
    if embedding_mode == "local":
        # Use the actual model name with Transformers/ prefix for directory naming.
        # HippoRAG will try to load this via SentenceTransformer, but we immediately
        # replace the embedder with our local implementation in inject_local_embedder().
        # Strip HuggingFace/ prefix if present and use Transformers/ for config.
        model_name_clean = args.embedding_model
        if model_name_clean.startswith("HuggingFace/"):
            model_name_clean = model_name_clean[len("HuggingFace/") :]
        hipporag_embedding_model = f"Transformers/{model_name_clean}"
        hipporag_embedding_base_url = None
        logger.info(f"Using local HuggingFace embeddings: {args.embedding_model}")

    config = BaseConfig(
        save_dir=str(index_dir),
        llm_base_url=llm_base_url,
        llm_name=model_name,
        embedding_model_name=hipporag_embedding_model,
        embedding_base_url=hipporag_embedding_base_url,
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        max_qa_steps=args.max_qa_steps,
        qa_top_k=args.qa_top_k,
        graph_type=args.graph_type,
        embedding_batch_size=args.embedding_batch_size,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        force_index_from_scratch=args.force_rebuild,
        force_openie_from_scratch=args.force_openie,
        openie_mode=args.openie_mode,
        openie_workers=args.openie_workers,
        corpus_len=corpus_size,
        is_directed_graph=args.directed_graph,
        damping=args.damping,
        save_openie=HIPPORAG2_CONFIG.get("save_openie", True),
        rerank_dspy_file_path=HIPPORAG2_CONFIG.get("rerank_dspy_file_path"),
    )

    hipporag = HippoRAG(global_config=config)

    # Inject local embedder if using local mode
    if embedding_mode == "local":
        hipporag = inject_local_embedder(hipporag, args)

    # Fix upstream VLLM embedding attribute mismatch if needed.
    if hasattr(hipporag, "embedding_model") and hasattr(
        hipporag.embedding_model, "url"
    ):
        if not hasattr(hipporag.embedding_model, "base_url"):
            setattr(hipporag.embedding_model, "base_url", hipporag.embedding_model.url)

    if use_answerer:
        configure_answerer_llm(args, hipporag)

    return hipporag


def inject_local_embedder(hipporag: Any, args: argparse.Namespace) -> Any:
    """
    Inject a local HuggingFace embedder into the HippoRAG instance.

    This replaces HippoRAG's default embedding model with our local implementation,
    allowing Qwen embeddings to run on-device without an API endpoint.

    Args:
        hipporag: HippoRAG instance
        args: CLI arguments containing embedding_model and embedding_device

    Returns:
        Modified HippoRAG instance with local embedder
    """
    import gc

    import torch

    from memqa.qa_agent_baselines.HippoRag2.local_embeddings import (
        create_local_embedder,
    )

    model_name = args.embedding_model
    if model_name.startswith("HuggingFace/"):
        model_name = model_name[len("HuggingFace/") :]

    old_embedder = hipporag.embedding_model

    local_embedder = create_local_embedder(
        model_name=model_name,
        batch_size=args.embedding_batch_size,
        device=getattr(args, "embedding_device", None),
    )

    hipporag.embedding_model = local_embedder

    for store in [
        hipporag.chunk_embedding_store,
        hipporag.entity_embedding_store,
        hipporag.fact_embedding_store,
    ]:
        if hasattr(store, "embedding_model"):
            store.embedding_model = local_embedder

    if hasattr(hipporag, "global_config"):
        hipporag.global_config.embedding_model_name = f"HuggingFace/{model_name}"

    if old_embedder is not None:
        del old_embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Injected local embedder: {model_name}")
    logger.info(f"Embedding dimension: {local_embedder.embedding_dim}")

    return hipporag


def configure_answerer_llm(args: argparse.Namespace, hipporag: Any) -> None:
    if not args.answerer_model and not args.answerer_endpoint:
        return

    try:
        from hipporag.llm.openai_gpt import CacheOpenAI  # type: ignore
    except Exception as exc:
        logger.warning("Failed to import HippoRAG CacheOpenAI: %s", exc)
        return

    answerer_config = deepcopy(hipporag.global_config)
    if args.answerer_model:
        answerer_config.llm_name = args.answerer_model
    if args.answerer_endpoint:
        answerer_config.llm_base_url = normalize_llm_base_url(args.answerer_endpoint)

    hipporag.llm_model = CacheOpenAI.from_experiment_config(answerer_config)
    if hasattr(hipporag, "rerank_filter"):
        hipporag.rerank_filter.llm_infer_fn = hipporag.llm_model.infer
        hipporag.rerank_filter.model_name = answerer_config.llm_name


def build_index(
    args: argparse.Namespace,
    hipporag: Any,
    corpus: Any,
) -> None:
    logger.info("=" * 60)
    logger.info("Stage 1: Building HippoRAG 2 Index")
    logger.info("=" * 60)
    docs = [p.text for p in corpus.passages]
    hipporag.index(docs)


def build_qa_messages(
    hipporag: Any,
    query_solution: Any,
    qa_top_k: int,
) -> List[Dict[str, str]]:
    evidence_blocks = [
        str(passage).strip() for passage in query_solution.docs[:qa_top_k]
    ]
    evidence = "\n\n".join(block for block in evidence_blocks if block)
    system_prompt = HIPPORAG2_PROMPTS["QA_SYSTEM"]
    user_prompt = HIPPORAG2_PROMPTS["QA_USER"].format(
        question=query_solution.question,
        evidence=evidence,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def run_qa(
    hipporag: Any,
    query_solutions: List[Any],
    qa_top_k: int,
    max_workers: int,
) -> Tuple[List[str], List[Dict[str, int]]]:
    messages_list = [
        build_qa_messages(hipporag, qs, qa_top_k) for qs in query_solutions
    ]

    results: List[Tuple[int, str, Dict[str, int]]] = []

    def submit_one(
        idx: int, messages: List[Dict[str, str]]
    ) -> Tuple[int, str, Dict[str, int]]:
        response_message, metadata, _cache_hit = hipporag.llm_model.infer(
            messages=messages
        )
        return idx, response_message, extract_token_usage(metadata)

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(submit_one, idx, messages)
                for idx, messages in enumerate(messages_list)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="QA"):
                idx, response, usage = future.result()
                results.append((idx, response, usage))
    else:
        for idx, messages in enumerate(tqdm(messages_list, desc="QA")):
            _, response, usage = submit_one(idx, messages)
            results.append((idx, response, usage))

    results.sort(key=lambda item: item[0])
    ordered_responses = [response for _, response, _ in results]
    ordered_usages = [usage for _, _, usage in results]

    answers = []
    for response in ordered_responses:
        try:
            answer = response.split("Answer:")[1].strip()
        except Exception:
            answer = response.strip()
        answers.append(answer)

    return answers, ordered_usages


def answer_no_evidence(
    hipporag: Any,
    questions: List[str],
    max_workers: int,
) -> Tuple[List[str], List[Dict[str, int]]]:
    messages_list = [
        [
            {
                "role": "system",
                "content": (
                    "Answer the question based on your knowledge. "
                    "If unsure, answer 'Unknown'."
                ),
            },
            {"role": "user", "content": question},
        ]
        for question in questions
    ]

    results: List[Tuple[int, str, Dict[str, int]]] = []

    def submit_one(
        idx: int, messages: List[Dict[str, str]]
    ) -> Tuple[int, str, Dict[str, int]]:
        response_message, metadata, _cache_hit = hipporag.llm_model.infer(
            messages=messages
        )
        return idx, response_message, extract_token_usage(metadata)

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(submit_one, idx, messages)
                for idx, messages in enumerate(messages_list)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="QA"):
                idx, response, usage = future.result()
                results.append((idx, response, usage))
    else:
        for idx, messages in enumerate(tqdm(messages_list, desc="QA")):
            _, response, usage = submit_one(idx, messages)
            results.append((idx, response, usage))

    results.sort(key=lambda item: item[0])
    ordered_responses = [response for _, response, _ in results]
    ordered_usages = [usage for _, _, usage in results]
    return [response.strip() for response in ordered_responses], ordered_usages


def answer_questions(
    args: argparse.Namespace,
    hipporag: Any,
    qa_items: List[Dict[str, Any]],
    corpus: Any,
    output_dir: Path,
) -> None:
    logger.info("=" * 60)
    logger.info("Stage 2: Answering Questions")
    logger.info("=" * 60)

    text_to_id = build_text_to_id_map(corpus.passages)
    questions = [qa["question"] for qa in qa_items]

    retrieval_details: List[Dict[str, Any]] = []
    answers: List[Dict[str, Any]] = []

    if args.no_evidence:
        raw_answers, usage_list = answer_no_evidence(
            hipporag, questions, args.max_workers
        )
        for qa, answer in zip(qa_items, raw_answers):
            answers.append({"id": qa["id"], "answer": answer})
            retrieval_details.append(
                {
                    "id": qa["id"],
                    "question": qa["question"],
                    "gt_evidence_ids": extract_evidence_ids(qa),
                    "retrieval_ids": [],
                    "retrieval_scores": [],
                    "retrieval_recall": compute_recall(extract_evidence_ids(qa), []),
                }
            )
    else:
        query_solutions = hipporag.retrieve(
            queries=questions,
            num_to_retrieve=args.retrieval_top_k,
        )

        qa_top_k = args.qa_top_k
        if args.max_evidence_items:
            qa_top_k = min(qa_top_k, args.max_evidence_items)

        raw_answers, usage_list = run_qa(
            hipporag, query_solutions, qa_top_k, args.max_workers
        )

        for qa, query_solution, answer in zip(qa_items, query_solutions, raw_answers):
            retrieved_texts = query_solution.docs
            retrieved_ids = [
                parse_evidence_id(text, text_to_id) for text in retrieved_texts
            ]
            retrieved_scores = (
                [float(score) for score in query_solution.doc_scores.tolist()]
                if query_solution.doc_scores is not None
                else []
            )

            gt_ids = extract_evidence_ids(qa)
            retrieval_record = {
                "id": qa["id"],
                "question": qa["question"],
                "gt_evidence_ids": gt_ids,
                "retrieval_ids": retrieved_ids,
                "retrieval_scores": retrieved_scores,
                "retrieval_recall": compute_recall(gt_ids, retrieved_ids),
            }
            retrieval_details.append(retrieval_record)
            answers.append({"id": qa["id"], "answer": answer})

    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / "hipporag2_answers.jsonl"
    write_jsonl(answers_path, answers)

    details_path = output_dir / "retrieval_recall_details.json"
    write_json(details_path, retrieval_details)

    summary = {
        "retrieval_top_k": args.retrieval_top_k,
        "retrieval_log_k": args.retrieval_log_k,
        "recall": summarize_recalls(retrieval_details, "retrieval_recall"),
    }
    summary_path = output_dir / "retrieval_recall_summary.json"
    write_json(summary_path, summary)

    run_stats = {"num_samples": len(answers)}
    run_stats.update(summarize_token_usage(usage_list, len(answers)))
    stats_path = output_dir / "hipporag2_answers_run_stats.json"
    write_json(stats_path, run_stats)

    logger.info("Saved answers to %s", answers_path)
    logger.info("Saved retrieval details to %s", details_path)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir_base) / args.method_name
    index_cache = Path(args.index_cache)

    corpus_builder = CorpusBuilder(
        email_file=args.email_file,
        image_batch_results=args.image_batch_results,
        video_batch_results=args.video_batch_results,
        media_source=args.media_source,
        image_root=args.image_root,
        video_root=args.video_root,
        augmentation_level=args.augmentation_level,
    )
    corpus = corpus_builder.build_full_corpus()

    normalized_embedding, normalized_endpoint, embedding_mode = (
        normalize_embedding_model(
            args.embedding_model, args.embedding_endpoint, args.embedding_mode
        )
    )
    if args.provider == "openai":
        llm_model_key = args.model or HIPPORAG2_CONFIG["openai"]["model"]
    else:
        llm_model_key = args.model or HIPPORAG2_CONFIG["vllm_text"]["model"]

    cache_key = build_cache_key(
        {
            "embedding_model": normalized_embedding,
            "embedding_mode": embedding_mode,
            "embedding_endpoint": normalized_endpoint,
            "llm_model": llm_model_key,
            "media_source": args.media_source,
            "augmentation": args.augmentation_level,
            "corpus": corpus_builder.get_cache_key(),
        }
    )
    index_path = index_cache / cache_key

    if args.stage in ("build", "all"):
        hipporag = build_hipporag(args, index_path, len(corpus.passages))
        build_index(args, hipporag, corpus)

    if args.stage in ("answer", "all"):
        if args.stage == "answer":
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Index cache not found at {index_path}. Run with --stage build first."
                )
        answers_path = output_dir / "hipporag2_answers.jsonl"
        if answers_path.exists() and not args.force_answer:
            logger.info(
                "Answers already exist at %s; skipping answer stage.", answers_path
            )
        else:
            hipporag = build_hipporag(
                args, index_path, len(corpus.passages), use_answerer=True
            )

            qa_data = load_json(Path(args.qa_file))
            qa_items = load_qa_list(qa_data)
            if args.limit:
                qa_items = qa_items[: args.limit]
            answer_questions(args, hipporag, qa_items, corpus, output_dir)

    logger.info("HippoRAG 2 baseline complete!")


if __name__ == "__main__":
    main()
