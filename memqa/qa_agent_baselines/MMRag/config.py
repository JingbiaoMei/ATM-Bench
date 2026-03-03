#!/usr/bin/env python3
"""Configuration for MMRag baselines."""

from pathlib import Path

from memqa.global_config import (
    DEFAULT_PATHS,
    IMAGE_PROCESSING_CONFIG,
    VIDEO_PROCESSING_CONFIG,
    OPENAI_CONFIG as GLOBAL_OPENAI_CONFIG,
    VLLM_VL_CONFIG,
    VLLM_TEXT_CONFIG,
)

PROJECT_ROOT = Path(DEFAULT_PATHS["project_root"])

MMRAG_CONFIG = {
    "provider": "vllm",
    "media_source": "batch_results",
    "no_evidence": False,
    "max_workers": 8,
    "num_frames": 8,
    "max_total_frames": 32,
    "frame_strategy": "uniform",
    "max_evidence_items": None,
    # Public repo defaults (user-provided, not committed):
    #   data/raw_memory/{image,video,email}/...
    "image_root": str(PROJECT_ROOT / "data/raw_memory/image"),
    "video_root": str(PROJECT_ROOT / "data/raw_memory/video"),
    "email_file": str(PROJECT_ROOT / "data/raw_memory/email/merged_emails.json"),
    "image_batch_results": str(PROJECT_ROOT / "data/raw_memory/image/batch_results.json"),
    "video_batch_results": str(PROJECT_ROOT / "data/raw_memory/video/batch_results.json"),
    "output_dir_base": str(PROJECT_ROOT / "output/QA_Agent/MMRag"),
    "index_cache_dir": str(PROJECT_ROOT / "output/retrieval/index_cache"),
    "retriever": "qwen3_vl_embedding",
    "reranker": "qwen3_vl_reranker",
    "reuse_rerank_results": True,
    "retrieval_top_k": 5,
    "retrieval_max_k": 100,
    "rerank_input_k": 20,
    "rerank_top_k": 5,
    "retriever_batch_size": 8,
    "reranker_batch_size": 4,
    "text_embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "vl_embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
    "clip_model": "openai/clip-vit-large-patch14",
    "vista_model_name": "BAAI/bge-base-en-v1.5",
    "vista_weights": None,
    "text_reranker_model": "Qwen/Qwen3-Reranker-2B",
    "vl_reranker_model": "Qwen/Qwen3-VL-Reranker-2B",
    "vl_text_augment": True,
    "insert_raw_images": False,
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
    "critic_answerer": False,
    "openai": {
        **GLOBAL_OPENAI_CONFIG,
    },
    "vllm_vl": {
        **VLLM_VL_CONFIG,
    },
    "vllm_text": {
        **VLLM_TEXT_CONFIG,
    },
    "image_extensions": IMAGE_PROCESSING_CONFIG.get("supported_extensions", []),
    "video_extensions": VIDEO_PROCESSING_CONFIG.get("supported_extensions", []),
}

RECALL_KS = [1, 5, 10, 25, 50, 100, 200]

PROMPTS = {
    "SYSTEM": (
        "You are a memory QA assistant. Use ONLY the provided evidence to answer. "
        "If the evidence is insufficient, answer 'Unknown'. Respond with only the answer. "
        "If the question asks to recall or list items (photos/emails/videos), respond with the corresponding evidence IDs only, comma-separated, with no extra text."
    ),
    "USER_TEXT": (
        "Question: {question}\n\n"
        "Evidence:\n"
        "{evidence}\n\n"
        "Provide the answer based solely on the evidence."
    ),
    "USER_MULTIMODAL": (
        "Question: {question}\n"
        "You will be given evidence items as images or video frames. "
        "Use only the evidence to answer, and reply with only the answer."
    ),
    "AGENTIC_CRITIC_SYSTEM": (
        "You are a strict memory QA critic. Use ONLY the provided evidence to verify "
        "the draft answer. If the evidence is insufficient, the final answer must be "
        "'Unknown'. If the question asks to recall or list items (photos/emails/videos), "
        "the final answer must be the corresponding evidence IDs only, comma-separated, "
        "with no extra text. Return ONLY a JSON object with keys: final_answer, support, "
        "utility. support must be one of: fully, partially, none. utility is an integer "
        "1-5 (1=not useful, 5=very useful)."
    ),
    "AGENTIC_CRITIC_USER": (
        "Question: {question}\n\n"
        "Evidence:\n"
        "{evidence}\n\n"
        "Draft answer: {draft}\n\n"
        "JSON:"
    ),
}
