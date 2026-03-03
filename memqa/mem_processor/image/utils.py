#!/usr/bin/env python3
"""
Utility functions for image processing and analysis.
"""

import json
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

import numpy as np
from PIL import Image, ExifTags


def setup_directories(base_path: str, dirs: List[str]) -> Dict[str, Path]:
    """Create necessary directories and return paths."""
    base = Path(base_path)
    paths = {}
    
    for dir_name in dirs:
        dir_path = base / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        paths[dir_name] = dir_path
    
    return paths


def get_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """Generate hash for a file."""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def is_image_file(file_path: Path, supported_extensions: List[str] = None) -> bool:
    """Check if file is a supported image format."""
    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp', '.dng']
    
    return file_path.suffix.lower() in supported_extensions


def get_image_info(image_path: Path) -> Dict[str, Any]:
    """Get basic image information."""
    try:
        with Image.open(image_path) as img:
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
    except Exception as e:
        return {"error": str(e)}


def validate_image(image_path: Path, max_size: int = 50 * 1024 * 1024) -> Tuple[bool, str]:
    """Validate image file."""
    if not image_path.exists():
        return False, "File does not exist"
    
    if image_path.stat().st_size > max_size:
        return False, f"File too large (>{max_size} bytes)"
    
    if not is_image_file(image_path):
        return False, "Not a supported image format"
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, "Valid"
    except Exception as e:
        return False, f"Corrupted image: {str(e)}"


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace('\x00', '').replace('\r', ' ').replace('\n', ' ')
    
    return text.strip()


def parse_tags(tag_string: str) -> List[str]:
    """Parse comma-separated tags and clean them."""
    if not tag_string:
        return []
    
    tags = []
    for tag in tag_string.split(','):
        cleaned_tag = clean_text(tag).lower()
        if cleaned_tag and len(cleaned_tag) > 1:
            tags.append(cleaned_tag)
    
    return list(set(tags))  # Remove duplicates


def merge_entities(entities_list: List[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Merge multiple entity lists, removing duplicates."""
    seen = set()
    merged = []
    
    for entities in entities_list:
        for entity in entities:
            key = (entity.get('entity', '').lower(), entity.get('type', ''))
            if key not in seen and entity.get('entity'):
                seen.add(key)
                merged.append(entity)
    
    return merged


def format_timestamp(timestamp: Optional[datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
    """Format timestamp for JSON serialization."""
    if timestamp is None:
        return None
    return timestamp.strftime(format_str)


def create_thumbnail(image_path: Path, size: Tuple[int, int] = (256, 256), quality: int = 85) -> Optional[bytes]:
    """Create thumbnail image."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Create thumbnail
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            return buffer.getvalue()
    except Exception as e:
        logging.warning(f"Failed to create thumbnail for {image_path}: {e}")
        return None


def estimate_processing_time(num_images: int, avg_time_per_image: float = 30.0) -> str:
    """Estimate total processing time."""
    total_seconds = num_images * avg_time_per_image
    
    if total_seconds < 60:
        return f"{total_seconds:.0f} seconds"
    elif total_seconds < 3600:
        return f"{total_seconds/60:.1f} minutes"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def load_json_file(file_path: Path) -> Optional[Dict]:
    """Safely load JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file {file_path}: {e}")
        return None


def save_json_file(data: Any, file_path: Path, indent: int = 2) -> bool:
    """Safely save data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON file {file_path}: {e}")
        return False


def find_images_in_directory(directory: Path, recursive: bool = True, extensions: List[str] = None) -> List[Path]:
    """Find all image files in directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp', '.dng']
    
    images = []
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        images.extend(directory.glob(f"{pattern}{ext}"))
        images.extend(directory.glob(f"{pattern}{ext.upper()}"))
    
    return sorted(set(images))  # Remove duplicates and sort


def group_images_by_date(images_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group images by date."""
    grouped = {}
    
    for image_data in images_data:
        timestamp = image_data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    continue
            
            date_key = timestamp.strftime('%Y-%m-%d')
        else:
            date_key = 'unknown'
        
        if date_key not in grouped:
            grouped[date_key] = []
        grouped[date_key].append(image_data)
    
    return grouped


def create_summary_stats(images_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics for processed images."""
    total_images = len(images_data)
    
    # Count successful vs failed processing
    successful = sum(1 for img in images_data if 'error' not in img)
    failed = total_images - successful
    
    # Count images with different metadata types
    with_gps = sum(1 for img in images_data if img.get('location'))
    with_ocr = sum(1 for img in images_data if img.get('ocr_text', '').strip())
    with_entities = sum(1 for img in images_data if img.get('entities'))
    
    # Date range
    timestamps = [img.get('timestamp') for img in images_data if img.get('timestamp')]
    date_range = None
    if timestamps:
        try:
            dates = []
            for ts in timestamps:
                if isinstance(ts, str):
                    dates.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                elif isinstance(ts, datetime):
                    dates.append(ts)
            
            if dates:
                date_range = {
                    'earliest': min(dates).isoformat(),
                    'latest': max(dates).isoformat()
                }
        except:
            pass
    
    # Most common tags
    all_tags = []
    for img in images_data:
        all_tags.extend(img.get('tags', []))
    
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'total_images': total_images,
        'successful_processing': successful,
        'failed_processing': failed,
        'success_rate': successful / total_images if total_images > 0 else 0,
        'images_with_gps': with_gps,
        'images_with_ocr': with_ocr,
        'images_with_entities': with_entities,
        'date_range': date_range,
        'top_tags': top_tags,
        'processing_timestamp': datetime.now().isoformat()
    }


class ProgressTracker:
    """Track processing progress and estimate completion time."""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.completed_items = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.completed_items += increment
        self.last_update = datetime.now()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        elapsed = (self.last_update - self.start_time).total_seconds()
        
        if self.completed_items > 0:
            avg_time = elapsed / self.completed_items
            remaining_items = self.total_items - self.completed_items
            estimated_remaining = remaining_items * avg_time
        else:
            avg_time = 0
            estimated_remaining = 0
        
        return {
            'total': self.total_items,
            'completed': self.completed_items,
            'remaining': self.total_items - self.completed_items,
            'percentage': (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0,
            'elapsed_seconds': elapsed,
            'estimated_remaining_seconds': estimated_remaining,
            'avg_time_per_item': avg_time
        }
