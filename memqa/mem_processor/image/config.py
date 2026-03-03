#!/usr/bin/env python3
"""
Configuration file for image processing setup.
Imports common configurations from global_config and defines image-specific settings.
"""

from pathlib import Path

# Import common configurations from global config
from memqa.global_config import (
    OPENAI_CONFIG,
    VLLM_VL_CONFIG as VLLM_CONFIG,
    OCR_CONFIG,
    GEOCODING_CONFIG,
    PROCESSING_CONFIG,
    IMAGE_PROCESSING_CONFIG,
    OUTPUT_CONFIG,
    ENTITY_CATEGORIES,
    DEFAULT_PATHS,
    LOGGING_CONFIG,
)

# Update processing config with image-specific settings
PROCESSING_CONFIG = {
    **PROCESSING_CONFIG,
    **IMAGE_PROCESSING_CONFIG,
    "vllm_timeout": 60,  # Image-specific timeout
}

# ============================================================================
# IMAGE-SPECIFIC PROMPTS
# ============================================================================

PROMPTS = {
    "short_caption": """Generate a single, highly condensed sentence that serves as a core memory snapshot for this image. This snapshot will be used to quickly inform an LLM-based QA agent about the most essential elements.

Context: I took this image on {time_stamp} at {location_name}.


Your sentence must include:
- The main subject(s) or activity.
- The location: Use place name (if exist), city and country. 
- The date (e.g., "on 10 June 2025").

You should strictly follow the format: [Date phrase], [location], [main action or subject description].

Use natural, vivid language and factual tone. Focus only on the most relevant, recall-triggering details.
""",

    
    
    "caption": """Generate descriptive captions for personal memory retrieval system. Your goal is to provide details that would help me vividly recall this specific moment.

The image is taken on {time_stamp} at {location_name}. 

Include details such as:
- primary activity, event, or scene 
- key subjects visible: people, animals, and prominent objects
- specific visual context of the location if applicable
- time of day and lighting conditions, correlating with {time_stamp}
- prominent text, brands, or unique markings visible
- overall mood, emotional context, or atmosphere 

Aim for a comprehensive, flowing paragraph (approximately 2-4 sentences) that effectively captures the essence of the moment for future recall.""",

    "tags": """List 8-12 relevant tags or keywords for this image that would be useful for personal memory search and retrieval.
The image is taken on {time_stamp} at {location_name}. 

Include:
- Objects and items visible
- Activities and actions
- People categories (if visible)
- Setting/location types
- Time of day and season
- Colors and visual elements
- Mood/atmosphere

Return only the tags as a comma-separated list, no explanations.""",

    "entities": """Identify named entities visible in this image. 
Look for:
- Person names (if readable on signs, badges, name tags, etc.)
- Brand names and logos
- Store/restaurant/business names
- Location names (cities, landmarks, venues, street signs)
- Event names (on banners, signs, etc.)
- Product names
- Organization names
- Dates/times (if visible as text)
- Any other proper nouns

For each entity found, specify the type: person, location, organization, brand, event, date, product, or other.

Return as JSON format: [{{"entity": "name", "type": "category"}}]

If no clear named entities are visible, return an empty list.""",

    "ocr": """Extract all text visible in this image, including handwritten and printed text.
Preserve the original line breaks and formatting as much as possible.
If no text is visible, return an empty string.""",

    "safety_content": """Analyze this image for any sensitive content that should be flagged:
- Personal information (IDs, documents, credit cards, etc.)
- Private/intimate moments
- Children in potentially sensitive contexts
- Any content that should be treated with extra privacy

Return a single word: 'safe', 'sensitive', or 'private'""",

    "city_extraction": """Extract the city and country from this address in the format "City, Country" (e.g., "Cambridge, United Kingdom", "Shanghai, China", "New York City, United States").

Address: {location_name}

Return only the city and country in the specified format. If the address is unclear or incomplete, return your best estimate based on available information."""
}
