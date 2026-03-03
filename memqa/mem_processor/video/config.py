#!/usr/bin/env python3
"""
Configuration file for video processing setup.
Imports common configurations from global_config and defines video-specific settings.
"""

from pathlib import Path

# Import common configurations from global config
from memqa.global_config import (
    OPENAI_CONFIG,
    VLLM_VL_CONFIG as VLLM_CONFIG,
    OCR_CONFIG,
    GEOCODING_CONFIG,
    PROCESSING_CONFIG,
    OUTPUT_CONFIG,
    ENTITY_CATEGORIES,
    DEFAULT_PATHS,
    LOGGING_CONFIG,
)

# Video processing specific configuration
VIDEO_PROCESSING_CONFIG = {
    "supported_extensions": ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'],
    "max_video_size": 500 * 1024 * 1024,  # 500MB
    "frame_extraction": {
        "num_frames": 8,  # Number of frames to extract for analysis
        "strategy": "uniform",  # uniform, keyframes, start_middle_end
        "max_frames": 16,
        "min_frames": 4,
    },
    "thumbnail_size": (512, 512),
}

# Update processing config with video-specific settings
PROCESSING_CONFIG = {
    **PROCESSING_CONFIG,
    **VIDEO_PROCESSING_CONFIG,
    "vllm_timeout": 120,  # Video-specific timeout (longer due to multiple frames)
}

# ============================================================================
# VIDEO-SPECIFIC PROMPTS
# ============================================================================

PROMPTS = {
    "short_caption": """Generate a single, highly condensed sentence that serves as a core memory snapshot for this video. This snapshot will be used to quickly inform an LLM-based QA agent about the most essential elements.

Context: I recorded this video on {time_stamp} at {location_name}.


Your sentence must include:
- The main subject(s) or activity/action in the video.
- The location: Use place name (if exist), city and country.
- The date (e.g., "on 10 June 2025").

You should strictly follow the format: [Date phrase], [location], [main action or subject description].

Use natural, vivid language and factual tone. Focus only on the most relevant, recall-triggering details.
""",

    "caption": """Generate descriptive captions for personal memory retrieval system based on these video frames. Your goal is to provide details that would help me vividly recall this specific moment.

The video was recorded on {time_stamp} at {location_name}. 

Include details such as:
- Primary activity, event, or scene captured in the video
- Key subjects visible: people, animals, and prominent objects
- Motion and actions happening in the video
- Specific visual context of the location if applicable
- Time of day and lighting conditions, correlating with {time_stamp}
- Prominent text, brands, or unique markings visible
- Overall mood, emotional context, or atmosphere

Aim for a comprehensive, flowing paragraph (approximately 2-4 sentences) that effectively captures the essence of the moment for future recall.""",

    "tags": """List 8-12 relevant tags or keywords for this video that would be useful for personal memory search and retrieval.
The video was recorded on {time_stamp} at {location_name}. 

Include:
- Objects and items visible
- Activities and actions
- People categories (if visible)
- Setting/location types
- Time of day and season
- Motion types (walking, driving, etc.)
- Colors and visual elements
- Mood/atmosphere
- Audio context if apparent (silent, talking, music, nature sounds)

Return only the tags as a comma-separated list, no explanations.""",

    "entities": """Identify named entities visible in these video frames. 
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

    "ocr": """Extract all text visible in these video frames, including handwritten and printed text.
Preserve the original line breaks and formatting as much as possible.
If no text is visible, return an empty string.""",

    "safety_content": """Analyze these video frames for any sensitive content that should be flagged:
- Personal information (IDs, documents, credit cards, etc.)
- Private/intimate moments
- Children in potentially sensitive contexts
- Any content that should be treated with extra privacy

Return a single word: 'safe', 'sensitive', or 'private'""",

    "city_extraction": """Extract the city and country from this address in the format "City, Country" (e.g., "Cambridge, United Kingdom", "Shanghai, China", "New York City, United States").

Address: {location_name}

Return only the city and country in the specified format. If the address is unclear or incomplete, return your best estimate based on available information.""",

    "video_summary": """Analyze these sequential frames from a video recording and provide a comprehensive description of what is happening.

The video was recorded on {time_stamp} at {location_name}.

Describe:
1. The main action or activity across the frames
2. Any changes or movement you observe between frames
3. The setting and environment
4. Notable objects, people, or elements
5. The apparent purpose or context of the recording

Provide a natural, flowing description (2-4 sentences) that captures the essence of this video moment.""",
}
