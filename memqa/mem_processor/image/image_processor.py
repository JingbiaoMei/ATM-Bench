#!/usr/bin/env python3
"""
Image Preprocessing for Personal Memory Retrieval

This module processes gallery images to extract comprehensive metadata.
"""

import json
import argparse
import os
import logging
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
import piexif
from geopy.geocoders import Nominatim
from tqdm import tqdm

from memqa.mem_processor.image.config import (
    OPENAI_CONFIG, VLLM_CONFIG, OCR_CONFIG, GEOCODING_CONFIG, 
    PROCESSING_CONFIG, OUTPUT_CONFIG, DEFAULT_PATHS, PROMPTS
)
from openai import OpenAI

class ImageProcessor:
    """Main class for processing images and extracting comprehensive metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ImageProcessor with a configuration dictionary.
        
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
            

        # Create directories
        self.output_dir = Path(self.config.get("output_dir", DEFAULT_PATHS["output_dir"]))
        self.cache_dir = Path(self.config.get("cache_dir", self.output_dir / "cache"))
        self.logs_dir = Path(self.config.get("logs_dir", self.output_dir / "logs"))
        
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        

            
        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / "image_processor.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _get_cache_key(self, file_path: Path) -> str:
        """Get a stable cache key from the filename (stem without extension)."""
        return file_path.stem

    def _get_cache_path(self, file_path: Path, cache_type: str) -> Path:
        """Get cache file path for a given image and cache type."""
        cache_key = self._get_cache_key(file_path)
        return self.cache_dir / f"{cache_key}_{cache_type}.json"
    
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
    
    def extract_exif_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF metadata from image."""
        try:
            img = Image.open(image_path)
            if 'exif' not in img.info:
                return {"timestamp": None, "location": None, "device": "", "camera_settings": {}}
            
            exif_dict = piexif.load(img.info['exif'])
            
            # Extract timestamp
            timestamp = None
            datetime_original = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal)
            if datetime_original:
                try:
                    timestamp = datetime.strptime(datetime_original.decode(), '%Y:%m:%d %H:%M:%S')
                except:
                    pass
            
            # Extract GPS data
            location = None
            gps_info = exif_dict.get("GPS", {})
            if gps_info and 2 in gps_info and 4 in gps_info:
                try:
                    lat = self._get_decimal_from_dms(gps_info[2], gps_info[1].decode())
                    lon = self._get_decimal_from_dms(gps_info[4], gps_info[3].decode())
                    location = [lat, lon]
                except:
                    pass
            
            # Extract device info
            make = exif_dict['0th'].get(piexif.ImageIFD.Make, b'').decode('utf-8', errors='ignore')
            model = exif_dict['0th'].get(piexif.ImageIFD.Model, b'').decode('utf-8', errors='ignore')
            device = f"{make} {model}".strip()
            
            # Extract camera settings
            camera_settings = {}
            if 'Exif' in exif_dict:
                exif_data = exif_dict['Exif']
                settings_map = {
                    piexif.ExifIFD.FNumber: 'f_number',
                    piexif.ExifIFD.ExposureTime: 'exposure_time',
                    piexif.ExifIFD.ISOSpeedRatings: 'iso',
                    piexif.ExifIFD.FocalLength: 'focal_length',
                    piexif.ExifIFD.Flash: 'flash',
                    piexif.ExifIFD.WhiteBalance: 'white_balance'
                }
                
                for exif_key, setting_name in settings_map.items():
                    if exif_key in exif_data:
                        try:
                            value = exif_data[exif_key]
                            if isinstance(value, tuple) and len(value) == 2:
                                camera_settings[setting_name] = value[0] / value[1] if value[1] != 0 else 0
                            else:
                                camera_settings[setting_name] = value
                        except:
                            pass
            
            return {
                "timestamp": timestamp,
                "location": location,
                "device": device,
                "camera_settings": camera_settings
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting EXIF from {image_path}: {e}")
            return {"timestamp": None, "location": None, "device": "", "camera_settings": {}}
    
    def _get_decimal_from_dms(self, dms: Tuple[Tuple[int, int], ...], ref: str) -> float:
        """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees."""
        degrees, minutes, seconds = dms
        result = degrees[0] / degrees[1] + \
                 minutes[0] / minutes[1] / 60 + \
                 seconds[0] / seconds[1] / 3600
        if ref in ['S', 'W']:
            result = -result
        return result

    def reverse_geocode(self, location: List[float], image_path: Path) -> Optional[str]:
        """Convert GPS coordinates to human-readable address."""
        if not self.geolocator or not location:
            return None
        cache_path = self._get_cache_path(image_path, "location_name")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("location_name", "")

        try:
            print(f"no cache found for {cache_path}")
            result = self.geolocator.reverse(f"{location[0]}, {location[1]}", language='en')
            # Add a small delay to respect rate limits
            time.sleep(3)
            self._save_to_cache(cache_path, {"location_name": result.address if result else None})
            print("Geo Success")
            return result.address if result else None
        except Exception as e:
            self.logger.warning(f"Geocoding failed for {location}: {e}")
            # Raise the exception to shutdown the program
            raise e
            return None
    
    def extract_ocr_text(self, image_path: Path) -> str:
        """Extract text from image using the vision model."""
        if not self.use_ocr or self.provider == "none":
            return ""

        cache_path = self._get_cache_path(image_path, "ocr_text")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("ocr_text", "")

        prompt = PROMPTS["ocr"]

        # Let the exception propagate to shutdown the program
        ocr_text = self.query_vision_model(image_path, prompt)
        self._save_to_cache(cache_path, {"ocr_text": ocr_text})
        return ocr_text
    
    def query_vision_model(self, image_path: Path, prompt: str) -> str:
        """Query vision model (VLLM, OpenAI, or none) with image and prompt."""
        if self.provider == "none":
            return ""  # No AI analysis
        elif self.provider == "openai":
            return self._query_openai(image_path, prompt)
        elif self.provider == "vllm":
            return self._query_vllm(image_path, prompt)
        else:
            self.logger.warning(f"Unknown provider: {self.provider}")
            return ""
    
    def query_text_model(self, prompt: str) -> str:
        """Query text model (VLLM, OpenAI, or none) with text-only prompt."""
        if self.provider == "none":
            return ""  # No AI analysis
        elif self.provider == "openai":
            return self._query_openai_text(prompt)
        elif self.provider == "vllm":
            return self._query_vllm_text(prompt)
        else:
            self.logger.warning(f"Unknown provider: {self.provider}")
            return ""
    
    def _query_openai(self, image_path: Path, prompt: str) -> str:
        """Query OpenAI vision model with image and prompt."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        model = self.config.get("model")
        max_tokens_value = self.config.get("max_tokens")
        
        # Check if using newer reasoning models (GPT-5, o1, o3)
        is_newer_model = any(x in model.lower() for x in ["gpt-5", "o1", "o3"])
        
        kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ]
                }
            ]
        }
        
        if is_newer_model:
            # Newer models use max_completion_tokens and don't support custom temperature
            # Reasoning models need 3x tokens (reasoning + output)
            kwargs["max_completion_tokens"] = max_tokens_value * 3 if max_tokens_value else 3000
        else:
            # Older models support temperature and use max_tokens
            kwargs["max_tokens"] = max_tokens_value
            kwargs["temperature"] = self.config.get("temperature")
        
        response = self.openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    
    def _query_vllm(self, image_path: Path, prompt: str) -> str:
        """Query VLLM model with image and prompt."""
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.get('api_key')}"
        }
        
        data = {
            "model": self.config.get("model"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ]
                }
            ],
            "max_tokens": self.config.get("max_tokens"),
            "temperature": self.config.get("temperature")
        }
        
        response = requests.post(
            self.config.get("endpoint"), 
            headers=headers, 
            json=data, 
            timeout=self.config.get("timeout")
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
        
        # Check if using newer reasoning models (GPT-5, o1, o3)
        is_newer_model = any(x in model.lower() for x in ["gpt-5", "o1", "o3"])
        
        kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        if is_newer_model:
            # Newer models use max_completion_tokens and don't support custom temperature
            # Reasoning models need 3x tokens (reasoning + output)
            kwargs["max_completion_tokens"] = max_tokens_value * 3 if max_tokens_value else 3000
        else:
            # Older models support temperature and use max_tokens
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
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.get("max_tokens"),
            "temperature": self.config.get("temperature")
        }
        
        response = requests.post(
            self.config.get("endpoint"), 
            headers=headers, 
            json=data, 
            timeout=self.config.get("timeout")
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()

    def generate_caption(self, image_path: Path, time_stamp: str, location_name: str) -> str:
        """Generate detailed caption for the image."""
        cache_path = self._get_cache_path(image_path, "caption")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("caption", "")

        prompt = PROMPTS["caption"].format(time_stamp=time_stamp, location_name=location_name)

        # Let the exception propagate to shutdown the program
        caption = self.query_vision_model(image_path, prompt)
        
        # Cache result
        self._save_to_cache(cache_path, {"caption": caption})
        return caption

    def generate_short_caption(self, image_path: Path, time_stamp: str, location_name: str) -> str:
        """Generate a short, condensed caption for the image."""
        cache_path = self._get_cache_path(image_path, "short_caption")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("short_caption", "")

        prompt = PROMPTS["short_caption"].format(time_stamp=time_stamp, location_name=location_name)

        # Let the exception propagate to shutdown the program
        short_caption = self.query_vision_model(image_path, prompt)
        
        # Cache result
        self._save_to_cache(cache_path, {"short_caption": short_caption})
        return short_caption

    def extract_city_from_address(self, location_name: str, image_path: Path) -> str:
        """Extract city and country from the geocoded address using LLM."""
        if not location_name or self.provider == "none":
            return ""
        
        cache_path = self._get_cache_path(image_path, "city")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("city", "")

        prompt = PROMPTS["city_extraction"].format(location_name=location_name)

        # Let the exception propagate to shutdown the program
        city = self.query_text_model(prompt)
        
        # Clean up the response
        city = city.strip().replace('"', '').replace("'", "")
        
        # Cache result
        self._save_to_cache(cache_path, {"city": city})
        return city

    def extract_tags(self, image_path: Path, time_stamp: str, location_name: str) -> List[str]:
        """Extract relevant tags and keywords from the image."""
        cache_path = self._get_cache_path(image_path, "tags")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("tags", [])
        
        prompt = PROMPTS["tags"].format(time_stamp=time_stamp, location_name=location_name)
        
        # Let the exception propagate to shutdown the program
        response = self.query_vision_model(image_path, prompt)
        
        # Parse tags from response
        tags = []
        if response:
            tags = [tag.strip().lower() for tag in response.split(',') if tag.strip()]
            tags = tags[:12]  # Limit to 12 tags
        
        # Cache result
        self._save_to_cache(cache_path, {"tags": tags})
        return tags

    def extract_entities(self, image_path: Path, ocr_text: str = "", time_stamp: str = "", location_name: str = "") -> List[Dict[str, str]]:
        """Extract named entities from the image."""
        cache_path = self._get_cache_path(image_path, "entities")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("entities", [])
        prompt = PROMPTS["entities"].format(time_stamp=time_stamp, location_name=location_name)

        # Let the exception propagate to shutdown the program
        response = self.query_vision_model(image_path, prompt)
        
        # Parse entities from response
        entities = []
        if response:
            try:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\\[.*\]', response, re.DOTALL)
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
        
        # Cache result
        self._save_to_cache(cache_path, {"entities": entities})
        return entities
    
    def extract_safety_content(self, image_path: Path) -> str:
        """Analyze image for sensitive content."""
        cache_path = self._get_cache_path(image_path, "safety_content")
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            return cached_result.get("safety_content", [])

        prompt = PROMPTS["safety_content"]

        # Let the exception propagate to shutdown the program
        response = self.query_vision_model(image_path, prompt)
        
        # Default to 'safe' if no response
        safety_content = response.strip() if response else "safe"
        
        # Cache result
        self._save_to_cache(cache_path, {"safety_content": safety_content})
        return safety_content
    
    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image and extract all metadata."""
        self.logger.info(f"Processing: {image_path}")
        
        # Extract EXIF metadata
        exif_data = self.extract_exif_metadata(image_path)
        time_stamp = exif_data.get("timestamp")
        # Reverse geocoding
        location_name = None
        city = None
        if exif_data.get("location") and self.use_geocoding:
            location_name = self.reverse_geocode(exif_data["location"], image_path)
            if location_name:
                city = self.extract_city_from_address(location_name, image_path)
            
        # OCR text extraction
        ocr_text = self.extract_ocr_text(image_path)
        # Generate caption
        caption = self.generate_caption(image_path, time_stamp, location_name)
        # Generate short caption
        short_caption = self.generate_short_caption(image_path, time_stamp, location_name)
        tags = self.extract_tags(image_path, time_stamp, location_name)
        entities = self.extract_entities(image_path, ocr_text, time_stamp, location_name)
        safety_content = self.extract_safety_content(image_path)

        # Get file info
        file_stat = image_path.stat()
        
        # Compile all metadata
        result = {
            "image_path": str(image_path),
            "file_size": file_stat.st_size,
            "file_modified": datetime.fromtimestamp(file_stat.st_mtime),
            "processed_at": datetime.now(),
            
            # EXIF metadata
            "timestamp": exif_data["timestamp"],
            "location": exif_data["location"],
            "location_name": location_name,
            "city": city,
            "camera_settings": exif_data["camera_settings"],
            
            # AI-generated content
            "caption": caption,
            "short_caption": short_caption,
            "tags": tags,
            "entities": entities,
            "ocr_text": ocr_text,
            "safety_content": safety_content,
            
            # Additional metadata
            "processing_version": "1.1",
            "model_used": f"{self.provider}:{self.config.get('model')}"
        }
        
        return result
    
    def process_directory(self, 
                         input_dir: str, 
                         recursive: bool = True,
                         extensions: List[str] = None,
                         output_file: str = None) -> List[Dict[str, Any]]:
        """Process all images in a directory."""
        if extensions is None:
            extensions = PROCESSING_CONFIG.get("supported_extensions")
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all image files
        image_files = []
        if recursive:
            for ext in extensions:
                image_files.extend(input_path.rglob(f"*{ext}"))
                image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process images
        results = []
        with ThreadPoolExecutor(max_workers=PROCESSING_CONFIG.get("max_workers")) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_image, path): path 
                for path in image_files
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_path), total=len(image_files), desc="Processing images"):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process {path}: {e}")
                    results.append({
                        "image_path": str(path),
                        "error": str(e),
                        "processed_at": datetime.now()
                    })
        
        # Save results
        if output_file is None:
            output_file = self.output_dir / f"processed_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Results saved to: {output_file}")
        return results

