#!/usr/bin/env python3
"""
Configuration for oracle QA baseline.
"""

from pathlib import Path

from memqa.global_config import (
    OPENAI_CONFIG as GLOBAL_OPENAI_CONFIG,
    VLLM_VL_CONFIG,
    VLLM_TEXT_CONFIG,
    IMAGE_PROCESSING_CONFIG,
    VIDEO_PROCESSING_CONFIG,
    DEFAULT_PATHS,
)

PROJECT_ROOT = Path(DEFAULT_PATHS["project_root"])

ORACLE_CONFIG = {
    "provider": "vllm",
    "media_source": "batch_results",
    "max_evidence_items": None,
    "batch_fields": [
        "type",
        "timestamp",
        "location",
        "short_caption",
        "caption",
        "ocr",
        "tags",
    ],
    "num_frames": 8,
    "frame_strategy": "uniform",
    "max_workers": 1,
    "max_retries": 3,
    "request_delay": 1.0,
    "no_evidence": False,
    # Public repo defaults (user-provided, not committed):
    #   data/raw_memory/{image,video,email}/...
    "image_root": str(PROJECT_ROOT / "data/raw_memory/image"),
    "video_root": str(PROJECT_ROOT / "data/raw_memory/video"),
    "email_file": str(PROJECT_ROOT / "data/raw_memory/email/emails.json"),
    "image_batch_results": str(PROJECT_ROOT / "output/image/qwen3vl2b/batch_results.json"),
    "video_batch_results": str(PROJECT_ROOT / "output/video/qwen3vl2b/batch_results.json"),
    "output_file": str(PROJECT_ROOT / "output/QA_Agent/Oracle/oracle_answers.jsonl"),
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

PROMPTS = {
    "ORACLE_SYSTEM": (
        "You are a QA assistant. Use ONLY the provided evidence to answer. "
        "If the evidence is insufficient, answer 'Unknown'. Respond with only the answer. "
        "If the question asks to recall or list items (photos/emails/videos), respond with the corresponding evidence IDs only, comma-separated, with no extra text."
    ),
    "ORACLE_USER_TEXT": (
        "Question: {question}\n\n"
        "Evidence:\n"
        "{evidence}\n\n"
        "Provide the answer based solely on the evidence."
    ),
    "ORACLE_USER_MULTIMODAL": (
        "Question: {question}\n"
        "You will be given evidence items as images or video frames. "
        "Use only the evidence to answer, and reply with only the answer."
    ),
}
