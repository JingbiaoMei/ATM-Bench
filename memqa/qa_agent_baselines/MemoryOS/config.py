#!/usr/bin/env python3
"""
Configuration for MemoryOS baseline.

MemoryOS is a hierarchical memory management system with:
- Short-term memory (STM): Recent dialogue pages
- Mid-term memory (MTM): Topic-based segments
- Long-term personal memory (LPM): User/agent profiles and knowledge

Paper: https://arxiv.org/abs/2506.06326
Code: https://github.com/BAI-LAB/MemoryOS
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

MEMORYOS_CONFIG = {
    # Standard baseline settings
    "provider": "vllm",
    "media_source": "batch_results",
    "max_workers": 128,
    "max_evidence_items": None,
    # Data paths
    "image_root": str(PROJECT_ROOT / "data/raw_memory/image"),
    "video_root": str(PROJECT_ROOT / "data/raw_memory/video"),
    "email_file": str(PROJECT_ROOT / "data/raw_memory/email/emails.json"),
    "image_batch_results": str(PROJECT_ROOT / "output/image/qwen3vl2b/batch_results.json"),
    "video_batch_results": str(PROJECT_ROOT / "output/video/qwen3vl2b/batch_results.json"),
    "output_dir_base": str(PROJECT_ROOT / "output/QA_Agent/MemoryOS"),
    "method_name": "memoryos_base",
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
    # MemoryOS-specific settings
    # Storage paths
    "memoryos_data_storage_path": str(
        PROJECT_ROOT / "output/QA_Agent/MemoryOS/memoryos_data"
    ),
    "memoryos_user_id": "benchmark_user",
    "memoryos_checkpoint_dir": str(
        PROJECT_ROOT / "output/QA_Agent/MemoryOS/checkpoints"
    ),
    "memoryos_reuse_checkpoint": True,
    "memoryos_save_checkpoint": True,
    "memoryos_force_rebuild": False,
    # Memory hierarchy capacities
    # Paper defaults: short=10, mid=2000, retrieval_queue=7
    "memoryos_short_term_capacity": 10,
    "memoryos_mid_term_capacity": 2000,
    "memoryos_long_term_knowledge_capacity": 100,
    # Retrieval settings
    # Use top-50 retrieval for benchmark comparability with other baselines.
    "memoryos_retrieval_queue_capacity": 50,  # Top-k pages to retrieve
    "memoryos_top_k_sessions": 50,  # Top-k segments to search
    "memoryos_top_k_knowledge": 20,  # Top-k LPM entries
    # Heat and similarity thresholds
    # Paper default: heat_threshold=5.0, similarity=0.6
    "memoryos_mid_term_heat_threshold": 5.0,
    "memoryos_mid_term_similarity_threshold": 0.6,
    # Retrieval thresholds (paper defaults from retriever.py)
    "memoryos_segment_similarity_threshold": 0.1,  # For segment matching
    "memoryos_page_similarity_threshold": 0.1,  # For page matching
    "memoryos_knowledge_threshold": 0.01,  # For LPM knowledge retrieval
    # Model settings
    # memoryos_llm_model is the shared default for both indexing + answering.
    "memoryos_llm_model": "Qwen/Qwen3-VL-8B-Instruct",
    "memoryos_index_llm_model": None,
    "memoryos_answer_llm_model": None,
    "memoryos_assistant_id": "benchmark_assistant",  # Assistant persona ID
    # QA answer prompt style:
    # - "baseline": align with A-Mem/HippoRAG/Oracle evidence-only QA prompt
    # - "memoryos": use MemoryOS package native conversational response prompt
    "memoryos_answer_prompt_style": "baseline",
    # Embedding settings
    "memoryos_embedding_model_name": "all-MiniLM-L6-v2",
    # None uses MemoryOS defaults (e.g., bge-m3 -> use_fp16).
    "memoryos_embedding_model_kwargs": None,
    "memoryos_quiet": True,
    "memoryos_parallel_llm": True,
    "memoryos_parallel_llm_workers": 4,
    "memoryos_resume_indexing": True,
    # Isolation strategy
    "memoryos_use_per_qa_instance": True,  # Create separate instance per question
    "memoryos_eval_no_update": False,  # Skip post-answer add_memory/profile updates
    # Benchmark-oriented full-history mode:
    # keep substantially more MTM sessions and search a wider MTM frontier.
    # This is NOT paper-default behavior and should be reported explicitly.
    "memoryos_full_history_mode": False,
    "memoryos_full_history_mid_term_capacity": 20000,
    "memoryos_full_history_top_k_sessions": 200,
    "memoryos_full_history_heat_threshold": 1000000000.0,
    # OpenAI-compatible LLM settings
    "openai": {
        **GLOBAL_OPENAI_CONFIG,
    },
    "vllm_text": {
        **VLLM_TEXT_CONFIG,
    },
    # Media extensions
    "image_extensions": IMAGE_PROCESSING_CONFIG.get("supported_extensions", []),
    "video_extensions": VIDEO_PROCESSING_CONFIG.get("supported_extensions", []),
}

# Prompts for formatting memory items as dialogue
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
    "MEMORY_GENERIC_USER": ("Help me remember this information [{item_id}]: {content}"),
    "MEMORY_GENERIC_AGENT": "I've stored this information in my memory.",
}

# QA answer prompts used to patch MemoryOS runtime response templates.
QA_PROMPTS = {
    "BASELINE_SYSTEM": (
        "You are a memory QA assistant. Use ONLY the provided evidence to answer. "
        "If the evidence is insufficient, answer 'Unknown'. Respond with only the answer. "
        "If the question asks to recall or list items (photos/emails/videos), respond "
        "with the corresponding evidence IDs only, comma-separated, with no extra text."
    ),
    "BASELINE_USER": (
        "Question: {query}\n\n"
        "Evidence:\n"
        "{retrieval_text}\n\n"
        "Provide the answer based solely on the evidence."
    ),
}

# Additional configuration notes:
#
# MemoryOS Architecture:
# - STM: FIFO queue, stores recent dialogue pages
# - MTM: Segmented paging, groups pages by topic, uses heat-based eviction
# - LPM: User profile (90 dimensions), User KB (FIFO 100), Agent traits
#
# Heat Score Formula (Eq. 4 in paper):
#   Heat = α·N_visit + β·L_interaction + γ·R_recency
#   - N_visit: retrieval count
#   - L_interaction: dialogue page count
#   - R_recency: time decay (exp(-Δt/τ))
#
# Update Flow:
#   STM → MTM (FIFO when STM full)
#   MTM → LPM (when segment heat > threshold)
#
# For benchmark evaluation, we:
# 1. Match paper defaults unless explicitly overridden
# 2. Use separate instance per question to prevent contamination
# 3. Match embedding models with other baselines for fair comparison
