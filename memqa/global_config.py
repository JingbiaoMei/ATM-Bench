#!/usr/bin/env python3
"""
Global configuration file for the ATMBench system.
This centralizes all common configurations including API keys, model settings, 
paths, and shared parameters used across different modules.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# ============================================================================
# PROJECT ROOT AND PATHS
# ============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

PROJECT_ROOT = get_project_root()
API_KEYS_DIR = PROJECT_ROOT / "api_keys"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data layout (public repo defaults)
# - data/atm-bench: benchmark QA files + NIAH pools
# - data/raw_memory: user-provided raw personal artifacts (images/videos/emails)
# - data/processed_memory: outputs of preprocessing pipelines
ATM_BENCH_DIR = DATA_DIR / "atm-bench"
RAW_MEMORY_DIR = DATA_DIR / "raw_memory"
PROCESSED_MEMORY_DIR = DATA_DIR / "processed_memory"

# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

# If set, missing API keys raise instead of printing a warning.
STRICT_API_KEYS = os.getenv('ATMBENCH_STRICT_API_KEYS', '').strip().lower() in {'1', 'true', 'yes'}


def get_api_key(service: str, env_var: str, file_name: str) -> str:
    """
    Generic function to get API key from environment variables or local file.
    
    Priority:
    1. Environment variable
    2. Local file in api_keys directory
    
    Args:
        service: Name of the service (for error messages)
        env_var: Environment variable name
        file_name: Filename in api_keys directory
    
    Returns:
        API key string (empty if not found)
    """
    # Try environment variable first
    api_key = os.getenv(env_var)
    if api_key:
        return api_key
    
    # Try local file
    api_key_file = API_KEYS_DIR / file_name
    if api_key_file.exists():
        with open(api_key_file, "r") as f:
            return f.read().strip()
    
    msg = (
        f"Warning: {service} API key not found in environment variable '{env_var}' "
        f"or file '{api_key_file}'."
    )
    if STRICT_API_KEYS:
        raise RuntimeError(msg)
    print(msg)
    return ""

def get_openai_api_key() -> str:
    """Get OpenAI API key."""
    return get_api_key("OpenAI", "OPENAI_API_KEY", ".openai_key")

def get_vllm_api_key() -> str:
    """Get VLLM API key."""
    return get_api_key("VLLM", "VLLM_API_KEY", ".vllm_key")

def get_anthropic_api_key() -> str:
    """Get Anthropic API key."""
    return get_api_key("Anthropic", "ANTHROPIC_API_KEY", ".anthropic_key")

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# OpenAI Configuration
OPENAI_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": get_openai_api_key(),
    "max_tokens": 1000,
    "temperature": 0.2,
    "timeout": 30
}

# VLLM Configuration (for vision-language models)
VLLM_VL_CONFIG = {
    "provider": "vllm",
    "endpoint": "http://localhost:8000/v1/chat/completions",
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "api_key": get_vllm_api_key(),
    "max_tokens": 1000,
    "temperature": 0.2,
    "timeout": 60
}

# VLLM Configuration (for text-only models)
VLLM_TEXT_CONFIG = {
    "provider": "vllm",
    "endpoint": "http://localhost:8000/v1/chat/completions",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "api_key": get_vllm_api_key(),
    "max_tokens": 1000,
    "temperature": 0.2,
    "timeout": 60
}

# Embedding Model Configuration
EMBEDDING_CONFIG = {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "batch_size": 64,
    "device": "cuda",  # or "cpu"
}

# ============================================================================
# DEFAULT PATHS
# ============================================================================

DEFAULT_PATHS = {
    # Root directories
    "project_root": str(PROJECT_ROOT),
    "data_dir": str(DATA_DIR),
    "atm_bench_dir": str(ATM_BENCH_DIR),
    "raw_memory_dir": str(RAW_MEMORY_DIR),
    "processed_memory_dir": str(PROCESSED_MEMORY_DIR),
    "output_dir": str(OUTPUT_DIR),
    
    # Output subdirectories
    "cache_dir": str(OUTPUT_DIR / "cache"),
    "logs_dir": str(OUTPUT_DIR / "logs"),
    "temp_dir": str(OUTPUT_DIR / "temp"),
    
    # Data subdirectories
    "image_data_dir": str(RAW_MEMORY_DIR / "image"),
    "video_data_dir": str(RAW_MEMORY_DIR / "video"),
    "email_data_dir": str(RAW_MEMORY_DIR / "email"),
    
    # API keys directory
    "api_keys_dir": str(API_KEYS_DIR),
}

# ============================================================================
# PROCESSING CONFIGURATIONS
# ============================================================================

# General processing settings
PROCESSING_CONFIG = {
    "max_workers": 12,
    "batch_size": 1000,
    "cache_enabled": True,
    "max_concurrent_requests": 1000,
    "chunk_size": 5000,
    "max_retries": 3,
    "retry_delay": 5.0,  # seconds
}

# Image processing specific
IMAGE_PROCESSING_CONFIG = {
    "supported_extensions": ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp', '.dng'],
    "max_image_size": 50 * 1024 * 1024,  # 50MB
    "max_total_pixels": 1_000_000,  # For downsampling
}

# Video processing specific
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

# Email processing specific
EMAIL_PROCESSING_CONFIG = {
    "supported_extensions": ['.html', '.eml', '.msg'],
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "ignore_embedded_images": True,
    "ignore_urls": True,
    "max_text_length": 50000,
}

# ============================================================================
# OCR AND GEOCODING
# ============================================================================

OCR_CONFIG = {
    "enabled": True,
    "provider": "vision_model",  # or "paddleocr", "tesseract"
}

GEOCODING_CONFIG = {
    "enabled": True,
    "service": "nominatim",  # nominatim, google (future)
    "rate_limit": 1.0,  # seconds between requests
    "timeout": 10,
    "language": "en",
    "cache_enabled": True,
}

# ============================================================================
# OUTPUT CONFIGURATIONS
# ============================================================================

OUTPUT_CONFIG = {
    "format": "json",
    "indent": 2,
    "ensure_ascii": False,
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file_rotation": True,
    "max_file_size": "10MB",
    "backup_count": 5
}

# ============================================================================
# ENTITY AND TAG CATEGORIES
# ============================================================================

ENTITY_CATEGORIES = {
    "person": ["person", "people", "individual", "celebrity", "friend", "family"],
    "location": ["city", "country", "landmark", "venue", "place", "address", "gps"],
    "organization": ["company", "business", "store", "restaurant", "institution"],
    "brand": ["brand", "logo", "product", "manufacturer"],
    "event": ["event", "occasion", "celebration", "meeting", "conference"],
    "date": ["date", "time", "timestamp", "when"],
    "object": ["item", "thing", "object", "equipment", "tool"],
    "activity": ["action", "activity", "sport", "hobby", "work"],
    "emotion": ["mood", "feeling", "emotion", "atmosphere"]
}

EMAIL_CATEGORIES = {
    "personal": ["family", "friends", "personal", "private"],
    "work": ["business", "professional", "work", "office", "meeting"],
    "finance": ["bank", "payment", "invoice", "receipt", "transaction", "money"],
    "travel": ["booking", "hotel", "flight", "travel", "reservation", "trip"],
    "shopping": ["order", "purchase", "delivery", "shopping", "product", "cart"],
    "notifications": ["alert", "notification", "reminder", "update", "news"],
    "social": ["social media", "facebook", "twitter", "linkedin", "instagram"],
    "education": ["university", "course", "academic", "research", "study"],
    "health": ["medical", "health", "doctor", "appointment", "prescription"],
    "utility": ["bill", "utility", "service", "subscription", "renewal"]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config(provider: str = "vllm", model_type: str = "vision") -> Dict[str, Any]:
    """
    Get model configuration based on provider and type.
    
    Args:
        provider: "openai" or "vllm"
        model_type: "vision" or "text" (only for vllm)
    
    Returns:
        Model configuration dictionary
    """
    if provider.lower() == "openai":
        return OPENAI_CONFIG.copy()
    elif provider.lower() == "vllm":
        if model_type.lower() == "vision":
            return VLLM_VL_CONFIG.copy()
        else:
            return VLLM_TEXT_CONFIG.copy()
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_output_path(module: str, filename: str = None) -> Path:
    """
    Get output path for a specific module.
    
    Args:
        module: Module name (e.g., "image", "email", "event")
        filename: Optional filename to append
    
    Returns:
        Path object
    """
    base_path = OUTPUT_DIR / module
    if filename:
        return base_path / filename
    return base_path

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    dirs_to_create = [
        OUTPUT_DIR,
        OUTPUT_DIR / "image",
        OUTPUT_DIR / "video",
        API_KEYS_DIR,
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

# Create directories on import
ensure_directories()

# ============================================================================
# EXPORTS (for convenient importing)
# ============================================================================

__all__ = [
    # Path functions and constants
    'PROJECT_ROOT',
    'API_KEYS_DIR', 
    'DATA_DIR',
    'OUTPUT_DIR',
    'DEFAULT_PATHS',
    'get_project_root',
    'get_output_path',
    'ensure_directories',
    
    # API key functions
    'get_openai_api_key',
    'get_vllm_api_key',
    'get_anthropic_api_key',
    'get_api_key',
    
    # Model configurations
    'OPENAI_CONFIG',
    'VLLM_VL_CONFIG',
    'VLLM_TEXT_CONFIG',
    'EMBEDDING_CONFIG',
    'get_model_config',
    
    # Processing configurations
    'PROCESSING_CONFIG',
    'IMAGE_PROCESSING_CONFIG',
    'VIDEO_PROCESSING_CONFIG',
    'EMAIL_PROCESSING_CONFIG',
    'OCR_CONFIG',
    'GEOCODING_CONFIG',
    
    # Output and logging
    'OUTPUT_CONFIG',
    'LOGGING_CONFIG',
    
    # Categories
    'ENTITY_CATEGORIES',
    'EMAIL_CATEGORIES',
]
