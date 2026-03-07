"""
HippoRAG 2 Configuration

Paper defaults from:
- "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models"
- arXiv: https://arxiv.org/abs/2502.14802
- Table 5: Hyperparameter Settings
"""

from pathlib import Path
from memqa.global_config import (
    DEFAULT_PATHS,
    OPENAI_CONFIG as GLOBAL_OPENAI_CONFIG,
    VLLM_TEXT_CONFIG,
    VLLM_VL_CONFIG,
)

PROJECT_ROOT = Path(DEFAULT_PATHS["project_root"])
DEFAULT_HIPPORAG_REPO = PROJECT_ROOT / "third_party" / "HippoRAG"

# =============================================================================
# HippoRAG 2 Paper Hyperparameters (Table 5)
# =============================================================================
#
# These values match the paper's reported settings for fair comparison.
#
# | Parameter           | Paper Value      | Description                          |
# |---------------------|------------------|--------------------------------------|
# | retrieval_top_k     | 200              | Initial dense retrieval candidates   |
# | linking_top_k       | 5                | Entity linking candidates per query  |
# | max_qa_steps        | 3                | Multi-step QA reasoning depth        |
# | qa_top_k            | 5                | Passages for QA context              |
# | graph_type          | facts_and_sim... | Graph construction strategy          |
# | embedding_batch     | 8                | Batch size for embeddings            |
# =============================================================================

HIPPORAG2_CONFIG = {
    # -------------------------------------------------------------------------
    # Provider & Model Settings
    # -------------------------------------------------------------------------
    "provider": "vllm",
    "media_source": "batch_results",  # batch_results (text) or raw (multimodal)
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    "image_root": str(PROJECT_ROOT / "data/raw_memory/image"),
    "video_root": str(PROJECT_ROOT / "data/raw_memory/video"),
    "email_file": str(PROJECT_ROOT / "data/raw_memory/email/merged_emails.json"),
    "image_batch_results": str(PROJECT_ROOT / "output/image/qwen3vl2b/batch_results.json"),
    "video_batch_results": str(PROJECT_ROOT / "output/video/qwen3vl2b/batch_results.json"),
    "output_dir_base": str(PROJECT_ROOT / "output/QA_Agent/HippoRag2"),
    "index_cache_dir": str(PROJECT_ROOT / "output/QA_Agent/HippoRag2/index_cache"),
    "hipporag_repo": str(DEFAULT_HIPPORAG_REPO)
    if DEFAULT_HIPPORAG_REPO.exists()
    else "",
    "embedding_base_url": None,
    # -------------------------------------------------------------------------
    # HippoRAG 2 Paper Hyperparameters (Table 5)
    # -------------------------------------------------------------------------
    "retrieval_top_k": 200,  # Top-K passages for dense retrieval
    "linking_top_k": 5,  # Entity linking top-K
    "max_qa_steps": 3,  # Multi-step QA reasoning depth
    "qa_top_k": 5,  # Final passages for QA context
    "embedding_batch_size": 8,  # Embedding batch size
    "graph_type": "facts_and_sim_passage_node_unidirectional",
    "is_directed_graph": True,
    "damping": 0.5,
    # -------------------------------------------------------------------------
    # Embedding Model (Paper: NV-Embed-v2)
    # We support multiple options for ablation:
    # - nvidia/NV-Embed-v2 (paper default, very large)
    # - all-MiniLM-L6-v2 (lightweight, for comparison with A-Mem)
    # - Qwen/Qwen3-Embedding-0.6B (our codebase standard)
    #
    # Embedding Mode:
    # - "auto": Automatically determine based on model name and endpoint presence
    # - "local": Run embeddings locally using HuggingFace (no endpoint required)
    # - "api": Use VLLM/OpenAI-compatible endpoint (requires embedding_base_url)
    # -------------------------------------------------------------------------
    "embedding_model": "all-MiniLM-L6-v2",  # Default: A-Mem aligned
    "embedding_mode": "auto",  # auto, local, or api
    "supported_embeddings": [
        "nvidia/NV-Embed-v2",  # Paper default
        "all-MiniLM-L6-v2",  # A-Mem aligned (local sentence-transformers)
        "Qwen/Qwen3-Embedding-0.6B",  # Can use local HuggingFace or VLLM endpoint
        "Qwen/Qwen3-Embedding-4B",  # Larger Qwen embedding
    ],
    # -------------------------------------------------------------------------
    # LLM for OpenIE (NER + Triple Extraction)
    # Paper: gpt-4o-mini or Llama-3.3-70B-Instruct
    # We support OpenAI-compatible endpoints (Qwen3VL-2B/8B, etc.)
    # -------------------------------------------------------------------------
    "openie_mode": "online",  # online (API) or offline (vLLM batch)
    "force_index_from_scratch": False,
    "force_openie_from_scratch": False,
    "save_openie": True,
    # -------------------------------------------------------------------------
    # Reranking (DSPy-based triple filtering)
    # Paper uses LLM-based "recognition memory" for triple filtering
    # -------------------------------------------------------------------------
    "enable_triple_filtering": True,  # Enable LLM-based triple filtering
    "rerank_dspy_file_path": None,  # Path to DSPy filter weights (optional)
    # -------------------------------------------------------------------------
    # Concurrency & Caching
    # -------------------------------------------------------------------------
    "max_workers": 8,
    "memory_workers": 1,  # Parallel workers for OpenIE
    "checkpoint_interval": 100,
    "resume": True,
    "timeout": 120,
    # -------------------------------------------------------------------------
    # QA Settings
    # -------------------------------------------------------------------------
    "no_evidence": False,
    "max_evidence_items": None,
    "num_frames": 8,
    "frame_strategy": "uniform",
    # -------------------------------------------------------------------------
    # Provider-specific configs (inherited from global)
    # -------------------------------------------------------------------------
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


# =============================================================================
# HippoRAG 2 Prompts (from original codebase)
# =============================================================================
#
# These are the EXACT prompts from the HippoRAG 2 paper/code for:
# 1. Named Entity Recognition (NER)
# 2. Triple Extraction (OpenIE)
# 3. QA Generation
#
# Reference: https://github.com/OSU-NLP-Group/HippoRAG/tree/main/src/hipporag/prompts
# =============================================================================

HIPPORAG2_PROMPTS = {
    # -------------------------------------------------------------------------
    # NER Prompt (from ner.py)
    # Extracts named entities from passages for graph construction
    # -------------------------------------------------------------------------
    "NER_SYSTEM": """Extract all named entities from the given passage.
Return a JSON object with a single key "named_entities" containing a list of named entities.
Named entities include: person names, organizations, locations, dates, events, products, 
scientific terms, and other proper nouns.
Be thorough and extract ALL named entities, not just the most important ones.""",
    "NER_ONE_SHOT_INPUT": """Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English songs. It also plays and songs in Regional languages. Radio City recently forayed into New Media in 2008 with the launch of its music portal - Loss.com. It offers listeners an interactive experience with music downloads, news, ed0s, ed7s, ed8s and all available information about Radio City and its jockeys.""",
    "NER_ONE_SHOT_OUTPUT": """{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "New Media", "2008", "PlanetRadiocity.com"]}""",
    # -------------------------------------------------------------------------
    # Triple Extraction Prompt (from triple_extraction.py)
    # Converts passages + NER results into RDF triples
    # -------------------------------------------------------------------------
    "TRIPLE_EXTRACTION_SYSTEM": """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.""",
    "TRIPLE_EXTRACTION_ONE_SHOT_INPUT": """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English songs. Radio City recently forayed into New Media in May 2008 with the launch of its music portal - PlanetRadiocity.com. It offers news, videos, ed0s songs.
```

{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "New Media", "May 2008", "PlanetRadiocity.com"]}""",
    "TRIPLE_EXTRACTION_ONE_SHOT_OUTPUT": """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}""",
    "TRIPLE_EXTRACTION_USER": """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}""",
    # -------------------------------------------------------------------------
    # QA Prompt (standard RAG-style QA)
    # -------------------------------------------------------------------------
    "QA_SYSTEM": """You are a memory QA assistant. Use ONLY the provided evidence to answer.
If the evidence is insufficient, answer 'Unknown'. Respond with only the answer.
If the question asks to recall or list items (photos/emails/videos), respond with the corresponding evidence IDs only, comma-separated, with no extra text.""",
    "QA_USER": """Question: {question}

Evidence:
{evidence}

Provide the answer based solely on the evidence.""",
}


# =============================================================================
# Text Conversion Formats (for multimodal → text)
# =============================================================================
#
# These formats define how multimodal evidence (images, videos, emails) are
# converted to text passages for HippoRAG 2's text-only graph construction.
#
# Aligns with MMRag text-only retrieval setup (qwen3-embedding, all-MiniLM).
# =============================================================================

TEXT_PASSAGE_FORMATS = {
    "EMAIL": """{id} | {timestamp} | {location}
{short_summary}
{detail}""",
    "IMAGE": """{id} | {timestamp} | {location}
{short_caption}""",
    "VIDEO": """{id} | {timestamp} | {location}
{short_caption}""",
    # Extended format with all available fields (for ablation)
    "IMAGE_EXTENDED": """{id} | {timestamp} | {location}
Caption: {short_caption}
Detailed: {caption}
Tags: {tags}
OCR: {ocr_text}""",
    "VIDEO_EXTENDED": """{id} | {timestamp} | {location}
Caption: {short_caption}
Detailed: {caption}
Tags: {tags}""",
}
