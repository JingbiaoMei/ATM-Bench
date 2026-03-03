#!/usr/bin/env python3
"""
Video Preprocessing for Personal Memory Retrieval

This module processes video files to extract comprehensive metadata including
visual content analysis from extracted frames.
"""

import json
import argparse
import os
import logging
import base64
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

import requests
from PIL import Image
from geopy.geocoders import Nominatim
from tqdm import tqdm

from memqa.mem_processor.video.config import (
    OPENAI_CONFIG, VLLM_CONFIG, OCR_CONFIG, GEOCODING_CONFIG, 
    PROCESSING_CONFIG, OUTPUT_CONFIG, DEFAULT_PATHS, PROMPTS,
    VIDEO_PROCESSING_CONFIG
)
from memqa.mem_processor.video.utils import (
    extract_video_metadata, extract_frames, check_gps_in_video,
    parse_gps_location, get_video_info, format_duration
)
from openai import OpenAI


class VideoProcessor:
    """Main class for processing videos and extracting comprehensive metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VideoProcessor with a configuration dictionary.
        
        Args:
            config: A dictionary containing all configuration settings.
        """
        self.config = config
        self.provider = config.get("provider", "none")
        
        # Validate provider and check dependencies
        if self.provider not in ['vllm', 'openai', 'none']:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'vllm', 'openai', or 'none'")
        
        # Initialize OpenAI client if using OpenAI
        self.openai_client = None
        if self.provider == 'openai':
            self.openai_client = OpenAI(api_key=self.config.get("api_key"))
        
        # OCR settings
        self.use_ocr = self.config.get("ocr_config", {}).get("enabled", False)
        
        # Geocoding settings
        geocoding_config = self.config.get("geocoding_config", {})
        self.use_geocoding = geocoding_config.get("enabled", False)
        self.geolocator = None
        if self.use_geocoding:
            try:
                self.geolocator = Nominatim(
                    user_agent="Howard's mem assistant",
                    timeout=10,
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize geocoder: {e}")
                self.use_geocoding = False
        
        # Frame extraction settings
        frame_config = config.get("frame_extraction", VIDEO_PROCESSING_CONFIG.get("frame_extraction", {}))
        self.num_frames = frame_config.get("num_frames", 8)
        self.frame_strategy = frame_config.get("strategy", "uniform")
        
        # Create directories
        self.output_dir = Path(self.config.get("output_dir", DEFAULT_PATHS["output_dir"]))
        self.cache_dir = Path(self.config.get("cache_dir", self.output_dir / "cache"))
        self.logs_dir = Path(self.config.get("logs_dir", self.output_dir / "logs"))
        self.frames_dir = Path(self.config.get("frames_dir", self.output_dir / "frames"))
        
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / "video_processor.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate MD5 hash for a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_cache_path(self, file_path: Path, cache_type: str) -> Path:
        """Get cache file path for a given video and cache type."""
        file_hash = self._get_file_hash(file_path)
        return self.cache_dir / f"{file_hash}_{cache_type}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load data from cache if it exists."""
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save data to cache."""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_path}: {e}")
    
    def extract_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract metadata from video file."""
        try:
            metadata = extract_video_metadata(video_path)
            
            return {
                "timestamp": metadata.get("timestamp"),
                "location": metadata.get("location"),
                "device": metadata.get("device", ""),
                "duration": metadata.get("duration", 0),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "rotation": metadata.get("rotation", 0),
                "codec": metadata.get("codec"),
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {video_path}: {e}")
            return {"timestamp": None, "location": None, "device": "", "duration": 0}

    def reverse_geocode(self, location: List[float], video_path: Path) -> Optional[str]:
        """Convert GPS coordinates to human-readable address."""
        if not self.geolocator or not location:
            return None
        
        cache_path = self._get_cache_path(video_path, "location_name")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("location_name", "")

        try:
            self.logger.info(f"Geocoding location for {video_path.name}")
            result = self.geolocator.reverse(f"{location[0]}, {location[1]}", language='en')
            # Add a small delay to respect rate limits
            time.sleep(3)
            self._save_to_cache(cache_path, {"location_name": result.address if result else None})
            return result.address if result else None
        except Exception as e:
            self.logger.warning(f"Geocoding failed for {location}: {e}")
            raise e
            return None
    
    def _extract_and_encode_frames(self, video_path: Path) -> Tuple[List[str], Path]:
        """
        Extract frames from video and encode them as base64.
        
        Returns:
            Tuple of (list of base64 encoded frames, path to frames directory)
        """
        # Create a unique directory for this video's frames
        video_hash = self._get_file_hash(video_path)
        frames_output_dir = self.frames_dir / video_hash
        
        # Check if frames already exist in cache
        if frames_output_dir.exists() and list(frames_output_dir.glob("*.jpg")):
            frame_paths = sorted(frames_output_dir.glob("*.jpg"))
        else:
            # Extract frames
            frame_paths = extract_frames(
                video_path, 
                num_frames=self.num_frames,
                output_dir=frames_output_dir,
                strategy=self.frame_strategy
            )
        
        # Encode frames as base64
        encoded_frames = []
        for frame_path in frame_paths:
            try:
                with open(frame_path, "rb") as f:
                    encoded_frames.append(base64.b64encode(f.read()).decode('utf-8'))
            except Exception as e:
                self.logger.warning(f"Failed to encode frame {frame_path}: {e}")
        
        return encoded_frames, frames_output_dir
    
    def extract_ocr_text(self, video_path: Path) -> str:
        """Extract text from video frames using the vision model."""
        if not self.use_ocr or self.provider == "none":
            return ""

        cache_path = self._get_cache_path(video_path, "ocr_text")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("ocr_text", "")

        prompt = PROMPTS["ocr"]
        
        # Extract and encode frames
        encoded_frames, _ = self._extract_and_encode_frames(video_path)
        
        if not encoded_frames:
            return ""

        # Use a subset of frames for OCR (first, middle, last)
        ocr_frames = [encoded_frames[0]]
        if len(encoded_frames) > 2:
            ocr_frames.append(encoded_frames[len(encoded_frames) // 2])
            ocr_frames.append(encoded_frames[-1])

        ocr_text = self.query_vision_model_multi_frame(ocr_frames, prompt)
        self._save_to_cache(cache_path, {"ocr_text": ocr_text})
        return ocr_text
    
    def query_vision_model(self, image_path: Path, prompt: str) -> str:
        """Query vision model with a single image and prompt."""
        if self.provider == "none":
            return ""
        
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        return self._query_vision_with_images([image_data], prompt)
    
    def query_vision_model_multi_frame(self, encoded_frames: List[str], prompt: str) -> str:
        """Query vision model with multiple frames and prompt."""
        if self.provider == "none":
            return ""
        
        return self._query_vision_with_images(encoded_frames, prompt)
    
    def _query_vision_with_images(self, encoded_images: List[str], prompt: str) -> str:
        """Query vision model with multiple base64-encoded images."""
        if self.provider == "none":
            return ""
        elif self.provider == "openai":
            return self._query_openai_multi(encoded_images, prompt)
        elif self.provider == "vllm":
            return self._query_vllm_multi(encoded_images, prompt)
        else:
            self.logger.warning(f"Unknown provider: {self.provider}")
            return ""
    
    def query_text_model(self, prompt: str) -> str:
        """Query text model with text-only prompt."""
        if self.provider == "none":
            return ""
        elif self.provider == "openai":
            return self._query_openai_text(prompt)
        elif self.provider == "vllm":
            return self._query_vllm_text(prompt)
        else:
            self.logger.warning(f"Unknown provider: {self.provider}")
            return ""
    
    def _query_openai_multi(self, encoded_images: List[str], prompt: str) -> str:
        """Query OpenAI vision model with multiple images."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        model = self.config.get("model")
        max_tokens_value = self.config.get("max_tokens")
        
        # Build content with text and multiple images
        content = [{"type": "text", "text": prompt}]
        for img_data in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
            })
        
        # Check if using newer reasoning models
        is_newer_model = any(x in model.lower() for x in ["gpt-5", "o1", "o3"])
        
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": content}]
        }
        
        if is_newer_model:
            kwargs["max_completion_tokens"] = max_tokens_value * 3 if max_tokens_value else 3000
        else:
            kwargs["max_tokens"] = max_tokens_value
            kwargs["temperature"] = self.config.get("temperature")
        
        response = self.openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    
    def _query_vllm_multi(self, encoded_images: List[str], prompt: str) -> str:
        """Query VLLM model with multiple images."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.get('api_key')}"
        }
        
        # Build content with text and multiple images
        content = [{"type": "text", "text": prompt}]
        for img_data in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
            })
        
        data = {
            "model": self.config.get("model"),
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config.get("max_tokens"),
            "temperature": self.config.get("temperature")
        }
        
        response = requests.post(
            self.config.get("endpoint"), 
            headers=headers, 
            json=data, 
            timeout=self.config.get("timeout", 120)
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    
    def _query_openai_text(self, prompt: str) -> str:
        """Query OpenAI text model with text-only prompt."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        model = self.config.get("model")
        max_tokens_value = self.config.get("max_tokens")
        
        is_newer_model = any(x in model.lower() for x in ["gpt-5", "o1", "o3"])
        
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if is_newer_model:
            kwargs["max_completion_tokens"] = max_tokens_value * 3 if max_tokens_value else 3000
        else:
            kwargs["max_tokens"] = max_tokens_value
            kwargs["temperature"] = self.config.get("temperature")
        
        response = self.openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    
    def _query_vllm_text(self, prompt: str) -> str:
        """Query VLLM model with text-only prompt."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.get('api_key')}"
        }
        
        data = {
            "model": self.config.get("model"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.get("max_tokens"),
            "temperature": self.config.get("temperature")
        }
        
        response = requests.post(
            self.config.get("endpoint"), 
            headers=headers, 
            json=data, 
            timeout=self.config.get("timeout", 60)
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()

    def generate_caption(self, video_path: Path, time_stamp: str, location_name: str) -> str:
        """Generate detailed caption for the video."""
        cache_path = self._get_cache_path(video_path, "caption")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("caption", "")

        prompt = PROMPTS["caption"].format(time_stamp=time_stamp, location_name=location_name)
        
        # Extract and encode frames
        encoded_frames, _ = self._extract_and_encode_frames(video_path)
        
        if not encoded_frames:
            return ""

        caption = self.query_vision_model_multi_frame(encoded_frames, prompt)
        
        self._save_to_cache(cache_path, {"caption": caption})
        return caption

    def generate_short_caption(self, video_path: Path, time_stamp: str, location_name: str) -> str:
        """Generate a short, condensed caption for the video."""
        cache_path = self._get_cache_path(video_path, "short_caption")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("short_caption", "")

        prompt = PROMPTS["short_caption"].format(time_stamp=time_stamp, location_name=location_name)
        
        # Extract and encode frames
        encoded_frames, _ = self._extract_and_encode_frames(video_path)
        
        if not encoded_frames:
            return ""
        
        # Use fewer frames for short caption
        short_frames = encoded_frames[:3] if len(encoded_frames) > 3 else encoded_frames

        short_caption = self.query_vision_model_multi_frame(short_frames, prompt)
        
        self._save_to_cache(cache_path, {"short_caption": short_caption})
        return short_caption

    def extract_city_from_address(self, location_name: str, video_path: Path) -> str:
        """Extract city and country from the geocoded address using LLM."""
        if not location_name or self.provider == "none":
            return ""
        
        cache_path = self._get_cache_path(video_path, "city")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("city", "")

        prompt = PROMPTS["city_extraction"].format(location_name=location_name)

        city = self.query_text_model(prompt)
        
        # Clean up the response
        city = city.strip().replace('"', '').replace("'", "")
        
        self._save_to_cache(cache_path, {"city": city})
        return city

    def extract_tags(self, video_path: Path, time_stamp: str, location_name: str) -> List[str]:
        """Extract relevant tags and keywords from the video."""
        cache_path = self._get_cache_path(video_path, "tags")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("tags", [])
        
        prompt = PROMPTS["tags"].format(time_stamp=time_stamp, location_name=location_name)
        
        # Extract and encode frames
        encoded_frames, _ = self._extract_and_encode_frames(video_path)
        
        if not encoded_frames:
            return []

        response = self.query_vision_model_multi_frame(encoded_frames, prompt)
        
        # Parse tags from response
        tags = []
        if response:
            tags = [tag.strip().lower() for tag in response.split(',') if tag.strip()]
            tags = tags[:12]  # Limit to 12 tags
        
        self._save_to_cache(cache_path, {"tags": tags})
        return tags

    def extract_entities(self, video_path: Path, ocr_text: str = "", time_stamp: str = "", location_name: str = "") -> List[Dict[str, str]]:
        """Extract named entities from the video."""
        cache_path = self._get_cache_path(video_path, "entities")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("entities", [])
        
        prompt = PROMPTS["entities"].format(time_stamp=time_stamp, location_name=location_name)
        
        # Extract and encode frames
        encoded_frames, _ = self._extract_and_encode_frames(video_path)
        
        if not encoded_frames:
            return []

        response = self.query_vision_model_multi_frame(encoded_frames, prompt)
        
        # Parse entities from response
        entities = []
        if response:
            try:
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    entities = json.loads(json_match.group())
            except:
                # Fallback: parse simple format
                lines = response.split('\n')
                for line in lines:
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            entity_name = parts[0].strip()
                            entity_type = parts[1].strip().lower()
                            entities.append({"entity": entity_name, "type": entity_type})
        
        self._save_to_cache(cache_path, {"entities": entities})
        return entities
    
    def extract_safety_content(self, video_path: Path) -> str:
        """Analyze video for sensitive content."""
        cache_path = self._get_cache_path(video_path, "safety_content")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("safety_content", "safe")

        prompt = PROMPTS["safety_content"]
        
        # Extract and encode frames
        encoded_frames, _ = self._extract_and_encode_frames(video_path)
        
        if not encoded_frames:
            return "safe"

        response = self.query_vision_model_multi_frame(encoded_frames[:3], prompt)
        
        safety_content = response.strip() if response else "safe"
        
        self._save_to_cache(cache_path, {"safety_content": safety_content})
        return safety_content
    
    def check_gps_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Check if video has GPS metadata."""
        return check_gps_in_video(video_path)
    
    def process_single_video(self, video_path: Path) -> Dict[str, Any]:
        """Process a single video and extract all metadata."""
        self.logger.info(f"Processing: {video_path}")
        
        # Extract video metadata
        metadata = self.extract_metadata(video_path)
        time_stamp = metadata.get("timestamp")
        
        # Check GPS metadata
        gps_info = self.check_gps_metadata(video_path)
        has_gps = gps_info.get("has_gps", False)
        
        # Reverse geocoding
        location_name = None
        city = None
        if metadata.get("location") and self.use_geocoding:
            location_name = self.reverse_geocode(metadata["location"], video_path)
            if location_name:
                city = self.extract_city_from_address(location_name, video_path)
        
        # OCR text extraction
        ocr_text = self.extract_ocr_text(video_path)
        
        # Generate caption
        caption = self.generate_caption(video_path, time_stamp, location_name)
        
        # Generate short caption
        short_caption = self.generate_short_caption(video_path, time_stamp, location_name)
        
        # Extract tags
        tags = self.extract_tags(video_path, time_stamp, location_name)
        
        # Extract entities
        entities = self.extract_entities(video_path, ocr_text, time_stamp, location_name)
        
        # Safety content analysis
        safety_content = self.extract_safety_content(video_path)

        # Get file info
        file_stat = video_path.stat()
        
        # Compile all metadata
        result = {
            "video_path": str(video_path),
            "file_size": file_stat.st_size,
            "file_modified": datetime.fromtimestamp(file_stat.st_mtime),
            "processed_at": datetime.now(),
            
            # Video metadata
            "timestamp": metadata["timestamp"],
            "location": metadata["location"],
            "location_name": location_name,
            "city": city,
            "has_gps": has_gps,
            "duration": metadata.get("duration", 0),
            "duration_formatted": format_duration(metadata.get("duration", 0)),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "rotation": metadata.get("rotation", 0),
            "codec": metadata.get("codec"),
            "device": metadata.get("device", ""),
            
            # AI-generated content
            "caption": caption,
            "short_caption": short_caption,
            "tags": tags,
            "entities": entities,
            "ocr_text": ocr_text,
            "safety_content": safety_content,
            
            # Additional metadata
            "processing_version": "1.0",
            "model_used": f"{self.provider}:{self.config.get('model')}",
            "num_frames_analyzed": self.num_frames,
        }
        
        return result
    
    def process_directory(self, 
                         input_dir: str, 
                         recursive: bool = True,
                         extensions: List[str] = None,
                         output_file: str = None) -> List[Dict[str, Any]]:
        """Process all videos in a directory."""
        if extensions is None:
            extensions = VIDEO_PROCESSING_CONFIG.get("supported_extensions", ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'])
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all video files
        video_files = []
        if recursive:
            for ext in extensions:
                video_files.extend(input_path.rglob(f"*{ext}"))
                video_files.extend(input_path.rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                video_files.extend(input_path.glob(f"*{ext}"))
                video_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(video_files)} videos to process")
        
        # Process videos
        results = []
        with ThreadPoolExecutor(max_workers=PROCESSING_CONFIG.get("max_workers", 4)) as executor:
            future_to_path = {
                executor.submit(self.process_single_video, path): path 
                for path in video_files
            }
            
            for future in tqdm(as_completed(future_to_path), total=len(video_files), desc="Processing videos"):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process {path}: {e}")
                    results.append({
                        "video_path": str(path),
                        "error": str(e),
                        "processed_at": datetime.now()
                    })
        
        # Save results
        if output_file is None:
            output_file = self.output_dir / f"processed_videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Results saved to: {output_file}")
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Process videos for personal memory retrieval")
    
    parser.add_argument("input_dir", help="Input directory containing videos")
    parser.add_argument("--provider", choices=["vllm", "openai", "none"], default="openai",
                       help="Vision model provider")
    parser.add_argument("--output-dir", default="output/video",
                       help="Output directory")
    parser.add_argument("--num-frames", type=int, default=8,
                       help="Number of frames to extract from each video")
    parser.add_argument("--recursive", action="store_true", default=True,
                       help="Process directories recursively")
    
    args = parser.parse_args()
    
    # Setup config
    config = {
        "ocr_config": OCR_CONFIG,
        "geocoding_config": GEOCODING_CONFIG,
        "output_dir": args.output_dir,
        "provider": args.provider,
        "frame_extraction": {"num_frames": args.num_frames, "strategy": "uniform"},
    }
    
    if args.provider == "vllm":
        config.update(VLLM_CONFIG)
    elif args.provider == "openai":
        config.update(OPENAI_CONFIG)
    
    # Initialize processor
    processor = VideoProcessor(config)
    
    # Process videos
    results = processor.process_directory(
        input_dir=args.input_dir,
        recursive=args.recursive
    )
    
    print(f"\nProcessed {len(results)} videos")
    
    return 0


if __name__ == "__main__":
    exit(main())
