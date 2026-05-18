from pathlib import Path

from memqa.global_config import (
    DEFAULT_PATHS,
    OPENAI_CONFIG as GLOBAL_OPENAI_CONFIG,
    VLLM_TEXT_CONFIG,
    VLLM_VL_CONFIG,
)

PROJECT_ROOT = Path(DEFAULT_PATHS["project_root"])

MEMPALACE_CONFIG = {
    "provider": "vllm",
    "no_evidence": False,
    "max_workers": 8,
    "max_evidence_items": None,
    "num_frames": 8,
    "frame_strategy": "uniform",
    "image_root": str(PROJECT_ROOT / "data/raw_memory/image"),
    "video_root": str(PROJECT_ROOT / "data/raw_memory/video"),
    "email_file": str(PROJECT_ROOT / "data/raw_memory/email/emails.json"),
    "image_batch_results": str(
        PROJECT_ROOT / "output/image/qwen3vl2b/batch_results.json"
    ),
    "video_batch_results": str(
        PROJECT_ROOT / "output/video/qwen3vl2b/batch_results.json"
    ),
    "output_dir_base": str(PROJECT_ROOT / "output/QA_Agent/Mempalace"),
    "index_cache_dir": str(PROJECT_ROOT / "output/QA_Agent/Mempalace/index_cache"),
    "embedding_model": "all-MiniLM-L6-v2",
    "retrieve_k": 10,
    "n_results": 100,
    "max_distance": 0.0,
    "candidate_strategy": "vector",
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
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
    "openai": {
        **GLOBAL_OPENAI_CONFIG,
    },
    "vllm_vl": {
        **VLLM_VL_CONFIG,
    },
    "vllm_text": {
        **VLLM_TEXT_CONFIG,
    },
}

PROMPTS = {
    "SYSTEM": (
        "You are a memory QA assistant. Use ONLY the provided evidence to answer. "
        "If the evidence is insufficient, answer 'Unknown'. Respond with only the answer. "
        "If the question asks to recall or list items (photos/emails/videos), respond with the corresponding evidence IDs only, comma-separated, with no extra text."
    ),
    "USER": (
        "Question: {question}\n\n"
        "Evidence:\n"
        "{evidence}\n\n"
        "Provide the answer based solely on the evidence."
    ),
}
