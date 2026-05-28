#!/usr/bin/env python3
"""
Configuration for SimpleMem baseline.

SimpleMem is an efficient lifelong memory system based on semantic lossless compression.
Three-stage pipeline:
1. Semantic Structured Compression: distills unstructured interactions into compact
   memory units with multi-view indexing (semantic, lexical, symbolic).
2. Online Semantic Synthesis: on-the-fly consolidation during the write phase to
   eliminate redundancy and maintain a compact memory topology.
3. Intent-Aware Retrieval Planning: infers search intent to dynamically determine
   retrieval scope and construct precise context.

Paper: SimpleMem: Efficient Lifelong Memory for LLM Agents (arXiv 2601.02553)
Code: https://github.com/aiming-lab/SimpleMem

This implementation adapts SimpleMem for single-QA benchmarking by:
1. Formatting each memory item (email/image/video) as a dialogue pair
2. Using a shared SimpleMem instance per run (all items indexed once)
3. Supporting custom LLMs and embeddings via OpenAI-compatible endpoints
"""

from pathlib import Path

from memqa.global_config import (
    DEFAULT_PATHS,
    IMAGE_PROCESSING_CONFIG,
    OPENAI_CONFIG as GLOBAL_OPENAI_CONFIG,
    VIDEO_PROCESSING_CONFIG,
    VLLM_TEXT_CONFIG,
)

PROJECT_ROOT = Path(DEFAULT_PATHS["project_root"])

SIMPLEMEM_CONFIG = {
    "provider": "vllm",
    "media_source": "batch_results",
    "max_workers": 128,
    "max_evidence_items": None,
    # Data paths
    "image_root": str(PROJECT_ROOT / "data/raw_memory/image"),
    "video_root": str(PROJECT_ROOT / "data/raw_memory/video"),
    "email_file": str(PROJECT_ROOT / "data/raw_memory/email/emails.json"),
    "image_batch_results": str(
        PROJECT_ROOT / "output/image/qwen3vl2b/batch_results.json"
    ),
    "video_batch_results": str(
        PROJECT_ROOT / "output/video/qwen3vl2b/batch_results.json"
    ),
    "output_dir_base": str(PROJECT_ROOT / "output/QA_Agent/SimpleMem"),
    "method_name": "simplemem_base",
    # Text augmentation flags (what to include in memory items)
    "include_id": True,
    "include_type": True,
    "include_timestamp": True,
    "include_location": True,
    "include_short_caption": True,
    "include_caption": True,
    "include_ocr_text": True,
    "include_tags": True,
    "include_email_summary": True,
    "include_email_detail": True,
    # SimpleMem storage paths
    "simplemem_db_path": str(
        PROJECT_ROOT / "output/QA_Agent/SimpleMem/simplemem_lancedb"
    ),
    # Answer mode:
    # - "atm":      SimpleMem retrieval + ATM-Bench QA prompt (aligned with
    #               HippoRAG2/A-Mem/MemoryOS baselines).
    # - "native":   upstream SimpleMem answer_generator.generate_answer.
    # - "external": run via memqa.LLMClient (matches MMRag-style answer path).
    "simplemem_answer_mode": "atm",
    # Source-ID attribution: minimum direct LLM-emitted coverage before the
    # build raises. Deterministic window-fallback covers any remainder.
    "simplemem_min_source_id_coverage": 0.9,
    # Planning/reflection (SimpleMem paper defaults)
    "simplemem_enable_planning": True,
    "simplemem_enable_reflection": True,
    "simplemem_max_reflection_rounds": 2,
    # Parallelism for build/retrieval
    "simplemem_enable_parallel_processing": True,
    "simplemem_max_parallel_workers": 8,
    "simplemem_enable_parallel_retrieval": True,
    "simplemem_max_retrieval_workers": 4,
    # Sliding window over the dialogue corpus
    "simplemem_window_size": 40,
    "simplemem_overlap_size": 2,
    # Multi-view retrieval top-k (semantic + lexical + symbolic)
    "simplemem_semantic_top_k": 25,
    "simplemem_keyword_top_k": 5,
    "simplemem_structured_top_k": 5,
    # Cache behavior
    "simplemem_reuse_index": True,
    "simplemem_force_rebuild": False,
    # Indexing model (smaller/cheaper, paired with the larger answer model)
    "build_model": "Qwen/Qwen3-VL-2B-Instruct",
    "build_endpoint": None,
    "openai": {
        **GLOBAL_OPENAI_CONFIG,
    },
    "vllm_text": {
        **VLLM_TEXT_CONFIG,
    },
    "image_extensions": IMAGE_PROCESSING_CONFIG.get("supported_extensions", []),
    "video_extensions": VIDEO_PROCESSING_CONFIG.get("supported_extensions", []),
}

PROMPTS = {
    "MEMORY_EMAIL_USER": (
        "Help me remember this email [{item_id}]: "
        "From {sender} at {timestamp} from {location}. "
        "Subject: {subject}. "
        "{content}"
    ),
    "MEMORY_EMAIL_AGENT": "I've stored this email in my memory.",
    "MEMORY_IMAGE_USER": (
        "Help me remember this image [{item_id}]: "
        "Captured at {location} on {timestamp}. "
        "{caption}"
    ),
    "MEMORY_IMAGE_AGENT": "I've stored this image memory.",
    "MEMORY_VIDEO_USER": (
        "Help me remember this video [{item_id}]: "
        "Recorded at {location} on {timestamp}. "
        "{description}"
    ),
    "MEMORY_VIDEO_AGENT": "I've stored this video memory.",
    "MEMORY_GENERIC_USER": "Help me remember this information [{item_id}]: {content}",
    "MEMORY_GENERIC_AGENT": "I've stored this information in my memory.",
    "QA_SYSTEM": (
        "You are a memory QA assistant. Use ONLY the provided evidence to answer. "
        "If the evidence is insufficient, answer 'Unknown'. Respond with only the answer. "
        "If the question asks to recall or list items (photos/emails/videos), respond "
        "with the corresponding evidence IDs only, comma-separated, with no extra text."
    ),
    "QA_USER": (
        "Question: {question}\n\n"
        "Evidence:\n"
        "{evidence}\n\n"
        "Provide the answer based solely on the evidence."
    ),
}
