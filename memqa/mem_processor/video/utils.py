#!/usr/bin/env python3
"""
Utility functions for video processing and analysis.
"""

import json
import hashlib
import subprocess
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import tempfile
import re

from PIL import Image


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


def is_video_file(file_path: Path, supported_extensions: List[str] = None) -> bool:
    """Check if file is a supported video format."""
    if supported_extensions is None:
        supported_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']
    
    return file_path.suffix.lower() in supported_extensions


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get basic video information using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
             '-show_format', '-show_streams', str(video_path)],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return {"error": f"ffprobe failed: {result.stderr}"}
        
        data = json.loads(result.stdout)
        
        # Extract relevant info
        format_info = data.get('format', {})
        video_stream = None
        audio_stream = None
        
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video' and video_stream is None:
                video_stream = stream
            elif stream.get('codec_type') == 'audio' and audio_stream is None:
                audio_stream = stream
        
        info = {
            "format": format_info.get('format_name'),
            "duration": float(format_info.get('duration', 0)),
            "size": int(format_info.get('size', 0)),
            "bit_rate": int(format_info.get('bit_rate', 0)),
        }
        
        if video_stream:
            info.update({
                "width": video_stream.get('width'),
                "height": video_stream.get('height'),
                "codec": video_stream.get('codec_name'),
                "fps": video_stream.get('r_frame_rate'),
                "rotation": get_rotation_from_stream(video_stream),
            })
        
        if audio_stream:
            info["has_audio"] = True
            info["audio_codec"] = audio_stream.get('codec_name')
        else:
            info["has_audio"] = False
        
        return info
        
    except Exception as e:
        return {"error": str(e)}


def get_rotation_from_stream(video_stream: Dict[str, Any]) -> int:
    """Extract rotation angle from video stream metadata."""
    # Check side_data_list for display matrix
    side_data_list = video_stream.get('side_data_list', [])
    for side_data in side_data_list:
        if side_data.get('side_data_type') == 'Display Matrix':
            rotation = side_data.get('rotation', 0)
            return abs(int(rotation))
    
    # Check tags for rotation
    tags = video_stream.get('tags', {})
    if 'rotate' in tags:
        return int(tags['rotate'])
    
    return 0


def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Extract comprehensive metadata from video file using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
             '-show_format', '-show_streams', str(video_path)],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return {}
        
        data = json.loads(result.stdout)
        format_info = data.get('format', {})
        tags = format_info.get('tags', {})
        
        # Extract creation time
        creation_time = None
        for key in ['creation_time', 'date', 'CREATION_TIME']:
            if key in tags:
                try:
                    creation_time = datetime.fromisoformat(tags[key].replace('Z', '+00:00').replace('.000000', ''))
                    break
                except:
                    pass
        
        # Extract GPS location
        location = None
        for key in ['location', 'location-eng', 'LOCATION']:
            if key in tags:
                location = parse_gps_location(tags[key])
                if location:
                    break
        
        # Get video stream info
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        # Extract device info from tags
        device = ""
        make = tags.get('make', tags.get('com.apple.quicktime.make', ''))
        model = tags.get('model', tags.get('com.apple.quicktime.model', ''))
        if make or model:
            device = f"{make} {model}".strip()
        
        metadata = {
            "timestamp": creation_time,
            "location": location,
            "device": device,
            "duration": float(format_info.get('duration', 0)),
            "tags": tags,
        }
        
        if video_stream:
            metadata["width"] = video_stream.get('width')
            metadata["height"] = video_stream.get('height')
            metadata["rotation"] = get_rotation_from_stream(video_stream)
            metadata["codec"] = video_stream.get('codec_name')
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error extracting video metadata: {e}")
        return {}


def parse_gps_location(location_str: str) -> Optional[List[float]]:
    """
    Parse GPS location from video metadata.
    Format is typically: +52.2006+000.1184/ or +52.2006-000.1184/
    """
    if not location_str:
        return None
    
    try:
        # Remove trailing slash
        location_str = location_str.rstrip('/')
        
        # Pattern: +/-DD.DDDD+/-DDD.DDDD
        # Match patterns like +52.2006+000.1184 or +52.2006-000.1184
        pattern = r'([+-]?\d+\.?\d*)\s*([+-]\d+\.?\d*)'
        match = re.search(pattern, location_str)
        
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            
            # Validate coordinates
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return [lat, lon]
        
        return None
        
    except Exception as e:
        logging.warning(f"Failed to parse GPS location '{location_str}': {e}")
        return None


def extract_frames(video_path: Path, 
                   num_frames: int = 8, 
                   output_dir: Path = None,
                   strategy: str = "uniform") -> List[Path]:
    """
    Extract frames from video for analysis.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        output_dir: Directory to save frames (uses temp if None)
        strategy: Frame extraction strategy (uniform, keyframes, start_middle_end)
    
    Returns:
        List of paths to extracted frame images
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video duration
    video_info = get_video_info(video_path)
    duration = video_info.get('duration', 0)
    
    if duration <= 0:
        logging.warning(f"Could not determine duration for {video_path}")
        # Try to extract at least one frame
        duration = 1.0
        num_frames = 1
    
    frame_paths = []
    
    if strategy == "uniform":
        # Extract frames at uniform intervals
        timestamps = [i * duration / (num_frames + 1) for i in range(1, num_frames + 1)]
    elif strategy == "start_middle_end":
        # Extract frames at start, middle, and end
        timestamps = [0.1, duration / 2, duration - 0.1]
        num_frames = 3
    elif strategy == "keyframes":
        # Extract keyframes (I-frames)
        return extract_keyframes(video_path, output_dir, max_frames=num_frames)
    else:
        # Default to uniform
        timestamps = [i * duration / (num_frames + 1) for i in range(1, num_frames + 1)]
    
    for i, ts in enumerate(timestamps):
        frame_path = output_dir / f"frame_{i:03d}.jpg"
        
        try:
            result = subprocess.run([
                'ffmpeg', '-y', '-ss', str(ts), '-i', str(video_path),
                '-vframes', '1', '-q:v', '2', str(frame_path)
            ], capture_output=True, text=True)
            
            if frame_path.exists():
                frame_paths.append(frame_path)
            else:
                logging.warning(f"Failed to extract frame at {ts}s from {video_path}")
                
        except Exception as e:
            logging.error(f"Error extracting frame: {e}")
    
    return frame_paths


def extract_keyframes(video_path: Path, 
                      output_dir: Path,
                      max_frames: int = 8) -> List[Path]:
    """Extract keyframes (I-frames) from video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "keyframe_%03d.jpg")
    
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', str(video_path),
            '-vf', f'select=eq(pict_type\\,I)',
            '-vsync', 'vfr', '-q:v', '2',
            output_pattern
        ], capture_output=True, text=True)
        
        # Get extracted frame paths
        frame_paths = sorted(output_dir.glob("keyframe_*.jpg"))
        
        # Limit to max_frames
        if len(frame_paths) > max_frames:
            # Select uniformly from keyframes
            step = len(frame_paths) // max_frames
            selected = frame_paths[::step][:max_frames]
            
            # Remove unselected frames
            for fp in frame_paths:
                if fp not in selected:
                    fp.unlink()
            
            return selected
        
        return frame_paths
        
    except Exception as e:
        logging.error(f"Error extracting keyframes: {e}")
        return []


def create_video_thumbnail(video_path: Path, 
                          size: Tuple[int, int] = (256, 256),
                          timestamp: float = None) -> Optional[bytes]:
    """Create thumbnail image from video."""
    try:
        # Get video duration if timestamp not specified
        if timestamp is None:
            video_info = get_video_info(video_path)
            duration = video_info.get('duration', 1)
            timestamp = duration / 2  # Middle of video
        
        # Create temporary file for frame
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        # Extract frame
        subprocess.run([
            'ffmpeg', '-y', '-ss', str(timestamp), '-i', str(video_path),
            '-vframes', '1', '-q:v', '2', str(tmp_path)
        ], capture_output=True)
        
        if not tmp_path.exists():
            return None
        
        # Resize to thumbnail
        with Image.open(tmp_path) as img:
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            
            # Clean up
            tmp_path.unlink()
            
            return buffer.getvalue()
            
    except Exception as e:
        logging.warning(f"Failed to create thumbnail for {video_path}: {e}")
        return None


def validate_video(video_path: Path, max_size: int = 500 * 1024 * 1024) -> Tuple[bool, str]:
    """Validate video file."""
    if not video_path.exists():
        return False, "File does not exist"
    
    if video_path.stat().st_size > max_size:
        return False, f"File too large (>{max_size} bytes)"
    
    if not is_video_file(video_path):
        return False, "Not a supported video format"
    
    # Try to get video info
    video_info = get_video_info(video_path)
    if "error" in video_info:
        return False, f"Invalid video: {video_info['error']}"
    
    return True, "Valid"


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


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def estimate_processing_time(num_videos: int, avg_time_per_video: float = 60.0) -> str:
    """Estimate total processing time."""
    total_seconds = num_videos * avg_time_per_video
    
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


def find_videos_in_directory(directory: Path, recursive: bool = True, extensions: List[str] = None) -> List[Path]:
    """Find all video files in directory."""
    if extensions is None:
        extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']
    
    videos = []
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        videos.extend(directory.glob(f"{pattern}{ext}"))
        videos.extend(directory.glob(f"{pattern}{ext.upper()}"))
    
    return sorted(set(videos))  # Remove duplicates and sort


def group_videos_by_date(videos_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group videos by date."""
    grouped = {}
    
    for video_data in videos_data:
        timestamp = video_data.get('timestamp')
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
        grouped[date_key].append(video_data)
    
    return grouped


def create_summary_stats(videos_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics for processed videos."""
    total_videos = len(videos_data)
    
    # Count successful vs failed processing
    successful = sum(1 for vid in videos_data if 'error' not in vid)
    failed = total_videos - successful
    
    # Count videos with different metadata types
    with_gps = sum(1 for vid in videos_data if vid.get('location'))
    with_ocr = sum(1 for vid in videos_data if vid.get('ocr_text', '').strip())
    with_entities = sum(1 for vid in videos_data if vid.get('entities'))
    
    # Total duration
    total_duration = sum(vid.get('duration', 0) for vid in videos_data if 'error' not in vid)
    
    # Date range
    timestamps = [vid.get('timestamp') for vid in videos_data if vid.get('timestamp')]
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
    for vid in videos_data:
        all_tags.extend(vid.get('tags', []))
    
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'total_videos': total_videos,
        'successful_processing': successful,
        'failed_processing': failed,
        'success_rate': successful / total_videos if total_videos > 0 else 0,
        'videos_with_gps': with_gps,
        'videos_with_ocr': with_ocr,
        'videos_with_entities': with_entities,
        'total_duration_seconds': total_duration,
        'total_duration_formatted': format_duration(total_duration),
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


def check_gps_in_video(video_path: Path) -> Dict[str, Any]:
    """
    Check if a video file contains GPS metadata.
    
    Returns:
        Dict with 'has_gps' bool and 'location' if found
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
             '-show_format', str(video_path)],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return {"has_gps": False, "error": "ffprobe failed"}
        
        data = json.loads(result.stdout)
        tags = data.get('format', {}).get('tags', {})
        
        # Check for GPS in various tag names
        for key in ['location', 'location-eng', 'LOCATION', 'GPS', 'gps']:
            if key in tags:
                location = parse_gps_location(tags[key])
                if location:
                    return {
                        "has_gps": True,
                        "location": location,
                        "raw_location": tags[key]
                    }
        
        return {"has_gps": False}
        
    except Exception as e:
        return {"has_gps": False, "error": str(e)}
