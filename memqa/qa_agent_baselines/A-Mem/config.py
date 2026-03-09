from pathlib import Path

from memqa.global_config import (
    DEFAULT_PATHS,
    OPENAI_CONFIG as GLOBAL_OPENAI_CONFIG,
    VLLM_TEXT_CONFIG,
    VLLM_VL_CONFIG,
)

PROJECT_ROOT = Path(DEFAULT_PATHS["project_root"])

A_MEM_CONFIG = {
    "provider": "vllm",
    "no_evidence": False,
    "max_workers": 8,
    "memory_workers": 1,
    "max_evidence_items": None,
    "checkpoint_interval": 100,
    "resume": True,
    "memory_retry_count": 3,
    "memory_retry_backoff": 5.0,
    "allow_incomplete_cache": False,
    "num_frames": 8,
    "frame_strategy": "uniform",
    "image_root": str(PROJECT_ROOT / "data/raw_memory/image"),
    "video_root": str(PROJECT_ROOT / "data/raw_memory/video"),
    "email_file": str(PROJECT_ROOT / "data/raw_memory/email/emails.json"),
    "image_batch_results": str(PROJECT_ROOT / "output/image/qwen3vl2b/batch_results.json"),
    "video_batch_results": str(PROJECT_ROOT / "output/video/qwen3vl2b/batch_results.json"),
    "output_dir_base": str(PROJECT_ROOT / "output/QA_Agent/A-Mem"),
    "index_cache_dir": str(PROJECT_ROOT / "output/QA_Agent/A-Mem/index_cache"),
    "embedding_model": "all-MiniLM-L6-v2",
    "retrieve_k": 10,
    "evo_threshold": 100,
    "disable_evolution": False,
    "alpha": 0.5,
    "use_hybrid": False,
    "follow_links": True,
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

# A-Mem Paper-Aligned Prompts (from Appendix B)
# Reference: https://arxiv.org/pdf/2502.12110

AMEM_PROMPTS = {
    "NOTE_CONSTRUCTION": """Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object:
{{
    "keywords": [
        // several specific, distinct keywords that capture key concepts and terminology
        // Order from most to least important
        // Don't include keywords that are the name of the speaker or time
        // At least three keywords, but don't be too redundant.
    ],
    "context": 
        // one sentence summarizing:
        // - Main topic/domain
        // - Key arguments/points
        // - Intended audience/purpose
    ,
    "tags": [
        // several broad categories/themes for classification
        // Include domain, format, and type tags
        // At least three tags, but don't be too redundant.
    ]
}}

Content for analysis:
{content}""",
    "NOTE_CONSTRUCTION_MULTIMODAL": """Analyze the provided image and its caption to generate a structured memory note.
1. Identifying the most salient keywords (focus on nouns, verbs, and key visual concepts)
2. Extracting core themes and contextual elements from both visual and textual clues
3. Creating relevant categorical tags

Format the response as a JSON object:
{{
    "keywords": [
        // several specific, distinct keywords that capture key concepts and terminology
        // Order from most to least important
        // Don't include keywords that are the name of the speaker or time
        // At least three keywords, but don't be too redundant.
    ],
    "context": 
        // one sentence summarizing:
        // - Main topic/domain
        // - Key arguments/points
        // - Intended audience/purpose
    ,
    "tags": [
        // several broad categories/themes for classification
        // Include domain, format, and type tags
        // At least three tags, but don't be too redundant.
    ]
}}

Caption/Text: {content}""",
    "MEMORY_EVOLUTION": """You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.
Return your decision in JSON format with the following structure:
{{
    "should_evolve": True or False,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["neighbor_memory_ids"],
    "tags_to_update": ["tag_1",..."tag_n"], 
    "new_context_neighborhood": ["new context",...,"new context"],
    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
}}""",
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
