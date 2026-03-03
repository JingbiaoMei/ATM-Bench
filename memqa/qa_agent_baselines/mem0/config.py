#!/usr/bin/env python3

from pathlib import Path

from memqa.global_config import (
    DEFAULT_PATHS,
    IMAGE_PROCESSING_CONFIG,
    OPENAI_CONFIG as GLOBAL_OPENAI_CONFIG,
    VIDEO_PROCESSING_CONFIG,
    VLLM_TEXT_CONFIG,
)

PROJECT_ROOT = Path(DEFAULT_PATHS["project_root"])

MEM0_CONFIG = {
    "provider": "vllm",
    "media_source": "batch_results",
    "max_workers": 128,
    "num_frames": 8,
    "frame_strategy": "uniform",
    "max_evidence_items": None,
    "image_root": str(PROJECT_ROOT / "data/raw_memory/image"),
    "video_root": str(PROJECT_ROOT / "data/raw_memory/video"),
    "email_file": str(PROJECT_ROOT / "data/raw_memory/email/merged_emails.json"),
    "image_batch_results": str(PROJECT_ROOT / "data/raw_memory/image/batch_results.json"),
    "video_batch_results": str(PROJECT_ROOT / "data/raw_memory/video/batch_results.json"),
    "output_dir_base": str(PROJECT_ROOT / "output/QA_Agent/Mem0"),
    "method_name": "mem0_base",
    "vl_text_augment": True,
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
    "mem0_config": None,
    "mem0_top_k": 10,
    "mem0_log_k": 100,
    "mem0_infer": True,
    "mem0_resume": True,
    "mem0_progress_path": None,
    "mem0_collection_name": "mem0",
    "mem0_vector_path": str(PROJECT_ROOT / "output/QA_Agent/Mem0/mem0_chroma"),
    "mem0_history_db_path": str(PROJECT_ROOT / "output/QA_Agent/Mem0/mem0_history.db"),
    "mem0_llm_provider": "vllm",
    "mem0_llm_model": "Qwen/Qwen3-VL-8B-Instruct",
    "mem0_llm_base_url": None,
    "mem0_llm_temperature": 0.1,
    "mem0_llm_max_tokens": 1000,
    "mem0_embedder_provider": "local",
    "mem0_embedder_model": "text-embedding-3-small",
    "mem0_embedder_base_url": None,
    "mem0_embedder_api_key": None,
    "mem0_embedder_dims": None,
    "mem0_local_retriever": "text",
    "mem0_local_cache_dir": str(PROJECT_ROOT / "output/QA_Agent/Mem0/mem0_local_cache"),
    "mem0_local_device": None,
    "retriever_batch_size": 16,
    "text_embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "vl_embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
    "clip_model": "openai/clip-vit-large-patch14",
    "vista_model_name": "BAAI/bge-m3",
    "vista_weights": None,
    "openai": {
        **GLOBAL_OPENAI_CONFIG,
    },
    "vllm_text": {
        **VLLM_TEXT_CONFIG,
    },
    "image_extensions": IMAGE_PROCESSING_CONFIG.get("supported_extensions", []),
    "video_extensions": VIDEO_PROCESSING_CONFIG.get("supported_extensions", []),
}

MEM0_DEFAULT_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": MEM0_CONFIG["mem0_collection_name"],
            "path": MEM0_CONFIG["mem0_vector_path"],
        },
    },
    "history_db_path": MEM0_CONFIG["mem0_history_db_path"],
}

PROMPTS = {
    "SYSTEM": (
        "You are a memory QA assistant. Use ONLY the provided memories to answer. "
        "If the memories are insufficient, answer 'Unknown'. Respond with only the answer. "
        "If the question asks to recall or list items (photos/emails/videos), respond with the corresponding evidence IDs only, comma-separated, with no extra text."
    ),
    "USER_TEXT": (
        "Question: {question}\n\n"
        "Memories:\n"
        "{evidence}\n\n"
        "Provide the answer based solely on the memories."
    ),
}

CONVERSION_PROMPTS = {
    "IMAGE_SYSTEM": (
        "You are helping me create a personal memory diary. "
        "Write natural first-person entries that capture experiences vividly. "
        "Always include the Memory ID exactly as provided."
    ),
    "IMAGE_USER": (
        "Create a personal memory diary entry for this photo.\n\n"
        "Memory ID: {id}\n"
        "Date/Time: {timestamp}\n"
        "Location: {location}\n\n"
        "Write a natural first-person diary entry (2-3 sentences) describing what I "
        "experienced or saw. Start with 'Memory {id}: ' and include the date, location, "
        "and what's visible in the photo. Write as natural prose, not structured data."
    ),
    "VIDEO_SYSTEM": (
        "You are helping me create a personal memory diary. "
        "Write natural first-person entries that capture experiences vividly. "
        "Always include the Memory ID exactly as provided."
    ),
    "VIDEO_USER": (
        "Create a personal memory diary entry for this video clip.\n\n"
        "Memory ID: {id}\n"
        "Date/Time: {timestamp}\n"
        "Location: {location}\n\n"
        "Write a natural first-person diary entry (2-3 sentences) describing what "
        "happened in this moment. Start with 'Memory {id}: ' and include the date, "
        "location, and what's happening in the video. Write as natural prose."
    ),
    "EMAIL_TEMPLATE": (
        "Memory {id}: On {date}, I received an email about '{subject}'"
        "{sender_part}. {summary_sentence}"
    ),
}

CUSTOM_FACT_EXTRACTION_PROMPT = """You are a Personal Memory Organizer specialized in extracting facts from someone's personal life archive (emails, photos, videos). Your role is to extract ALL relevant information that could help answer future questions about the person's life.

Types of Information to Extract:

1. Events and Activities: What happened, when, where, with whom
2. People and Relationships: Names, roles, relationships mentioned
3. Places and Locations: Where events occurred, travel destinations
4. Dates and Times: When things happened (specific dates, times, durations)
5. Objects and Items: Things purchased, received, owned, or seen
6. Communications: Email subjects, senders, key content, notifications received
7. Visual Content: What was seen in photos/videos, scenes, subjects
8. Plans and Schedules: Upcoming events, deadlines, appointments
9. Numbers and Quantities: Prices, quantities, measurements, IDs

Here are some few shot examples:

Input: Memory email202201040001: On 2022-01-04, I received an email about 'Engineering Department IT Maintenance Notice'. Maintenance will be performed on the SSH servers between 18:00 and 20:00 on January 4th.
Output: {"facts": ["Received IT maintenance notice on 2022-01-04", "SSH servers maintenance scheduled 18:00-20:00 on January 4th", "Engineering Department sent maintenance notification"]}

Input: Memory email202201050001: On 2022-01-05, I received an email about 'Order Confirmation and Delivery Details'. The email provides details about a grocery order delivered on January 5, 2022. Key items include onions, carrots, mushrooms, with total cost £42.49.
Output: {"facts": ["Grocery order delivered on January 5, 2022", "Order included onions, carrots, mushrooms", "Order total was £42.49"]}

Input: Memory 20220507_150929: On a sunny afternoon on May 7th, 2022, I was walking through OMFC Patch in Wolvercote, Oxford when I spotted a wild rabbit darting through the tall grass.
Output: {"facts": ["Saw a wild rabbit on May 7, 2022", "Was at OMFC Patch in Wolvercote, Oxford", "Rabbit was in tall grass", "It was a sunny afternoon"]}

Input: Memory VID_20220815: On August 15th, 2022, I was at the beach in Brighton watching the sunset. The waves were calm and children were playing nearby.
Output: {"facts": ["Was at Brighton beach on August 15, 2022", "Watched the sunset", "Waves were calm", "Children were playing nearby"]}

Input: Memory email202201060001: On 2022-01-06, I received an email about 'College Bill Payment'. The bill is due by January 31, 2022. The Accounts Office will be closed December 23 - January 4.
Output: {"facts": ["Received college bill notice on 2022-01-06", "Bill due by January 31, 2022", "Accounts Office closed December 23 to January 4"]}

Input: Hi, how are you?
Output: {"facts": []}

Return the facts in JSON format as shown above.

Remember:
- Extract ALL factual information that could be useful for answering questions later
- Include dates, times, locations, people, and specific details
- Keep facts atomic and self-contained
- Return an empty list only if there is truly no factual content
- Make sure to return JSON with a "facts" key containing a list of strings
"""
