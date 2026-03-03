#!/usr/bin/env python3
"""
Batch processing script for large video collections.
"""

import argparse
import logging
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from memqa.mem_processor.video.video_processor import VideoProcessor
from memqa.mem_processor.video.utils import (
    setup_directories, find_videos_in_directory, create_summary_stats,
    ProgressTracker, save_json_file, load_json_file
)
from memqa.mem_processor.video.config import (
    DEFAULT_PATHS, PROCESSING_CONFIG, OPENAI_CONFIG, VLLM_CONFIG, 
    OCR_CONFIG, GEOCODING_CONFIG, OUTPUT_CONFIG, VIDEO_PROCESSING_CONFIG
)


class BatchProcessor:
    """Batch processor for large video collections with resume capability and concurrent processing."""
    
    def __init__(self, 
                 processor: VideoProcessor,
                 output_dir: str = DEFAULT_PATHS["output_dir"],
                 checkpoint_interval: int = 10,
                 max_concurrent_requests: int = 10,
                 chunk_size: int = 20):
        """
        Initialize batch processor.
        
        Args:
            processor: VideoProcessor instance
            output_dir: Directory for output files
            checkpoint_interval: Save checkpoint every N videos
            max_concurrent_requests: Maximum concurrent requests for VLLM (recommended: 10-20 for videos)
            chunk_size: Size of chunks for concurrent processing
        """
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.checkpoint_interval = checkpoint_interval
        self.max_concurrent_requests = max_concurrent_requests
        self.chunk_size = chunk_size
        
        # Determine if we should use concurrent processing
        self.use_concurrent = (
            processor.provider == "vllm" and 
            max_concurrent_requests > 1
        )
        
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # State management
        self.is_running = True
        self.checkpoint_file = self.output_dir / "processing_checkpoint.json"
        self.results_file = self.output_dir / "batch_results.json"
        self.stats_file = self.output_dir / "processing_stats.json"
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for batch processing."""
        log_file = self.logs_dir / "batch_processing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    def _save_checkpoint(self, 
                        processed_files: List[str],
                        results: List[Dict[str, Any]],
                        current_index: int,
                        total_files: int):
        """Save processing checkpoint."""
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": processed_files,
            "current_index": current_index,
            "total_files": total_files,
            "results_count": len(results)
        }
        
        save_json_file(checkpoint_data, self.checkpoint_file)
        save_json_file(results, self.results_file)
        self.logger.info(f"Checkpoint saved: {current_index}/{total_files} processed")
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load previous checkpoint if exists."""
        if self.checkpoint_file.exists():
            return load_json_file(self.checkpoint_file)
        return None
    
    def _update_stats(self, results: List[Dict[str, Any]]):
        """Update processing statistics."""
        stats = create_summary_stats(results)
        save_json_file(stats, self.stats_file)
    
    def _process_video_chunk_concurrent(self, 
                                      video_chunk: List[Path], 
                                      recursive: bool = True) -> List[Dict[str, Any]]:
        """Process a chunk of videos concurrently using ThreadPoolExecutor."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            # Submit all videos in the chunk
            future_to_video = {
                executor.submit(self.processor.process_single_video, video_file): video_file 
                for video_file in video_chunk
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_video):
                video_file = future_to_video[future]
                result = future.result()  # Let exceptions propagate
                results.append(result)
        
        return results
    
    def _process_video_chunk_sequential(self, 
                                      video_chunk: List[Path], 
                                      recursive: bool = True) -> List[Dict[str, Any]]:
        """Process a chunk of videos sequentially (fallback for non-VLLM providers)."""
        results = []
        
        for video_file in video_chunk:
            if not self.is_running:
                break
                
            result = self.processor.process_single_video(video_file)  # Let exceptions propagate
            results.append(result)
        
        return results
    
    def _split_into_chunks(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split a list into chunks of specified size."""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    def _format_eta(self, eta_seconds: float) -> str:
        """Format ETA in a human-readable format."""
        if eta_seconds <= 0:
            return "unknown"
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:  # Less than 1 hour
            minutes = eta_seconds / 60
            return f"{minutes:.1f}m"
        else:  # 1 hour or more
            hours = eta_seconds / 3600
            if hours < 24:
                return f"{hours:.1f}h"
            else:  # More than 24 hours
                days = hours / 24
                return f"{days:.1f}d"
    
    def process_batch(self,
                     input_dir: str,
                     resume: bool = True,
                     recursive: bool = True,
                     extensions: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of videos with checkpoint/resume capability and optional concurrent processing.
        
        Args:
            input_dir: Input directory containing videos
            resume: Whether to resume from previous checkpoint
            recursive: Process directories recursively
            extensions: Video file extensions to process
            
        Returns:
            List of processing results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all videos
        video_files = find_videos_in_directory(input_path, recursive, extensions)
        total_files = len(video_files)
        
        self.logger.info(f"Found {total_files} videos to process")
        
        # Check for previous checkpoint
        checkpoint = None
        results = []
        start_index = 0
        processed_files = set()
        
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                self.logger.info("Resuming from previous checkpoint...")
                processed_files = set(checkpoint.get("processed_files", []))
                results = load_json_file(self.results_file) or []
                start_index = checkpoint.get("current_index", 0)
                
                # Filter out already processed files
                remaining_files = [f for f in video_files if str(f) not in processed_files]
                self.logger.info(f"Resuming: {len(processed_files)} already processed, {len(remaining_files)} remaining")
                video_files = remaining_files
        
        if not video_files:
            self.logger.info("No videos to process")
            return results
        
        # Log processing mode
        if self.use_concurrent:
            self.logger.info(f"Using concurrent processing with {self.max_concurrent_requests} max concurrent requests")
            self.logger.info(f"Processing {len(video_files)} videos in chunks of {self.chunk_size}")
        else:
            self.logger.info(f"Using sequential processing (provider: {self.processor.provider})")
        
        # Initialize progress tracking
        progress = ProgressTracker(len(video_files))
        
        # Start status reporting thread
        status_thread = threading.Thread(target=self._status_reporter, args=(progress,))
        status_thread.daemon = True
        status_thread.start()
        
        # Process videos
        current_index = start_index
        start_time = datetime.now()
        
        if self.use_concurrent:
            # Split files into chunks for concurrent processing
            file_chunks = self._split_into_chunks(video_files, self.chunk_size)
            
            for chunk_idx, video_chunk in enumerate(file_chunks):
                if not self.is_running:
                    self.logger.info("Processing interrupted by user")
                    break
                
                self.logger.info(f"Processing chunk {chunk_idx + 1}/{len(file_chunks)} ({len(video_chunk)} files)")
                
                # Process chunk concurrently
                chunk_results = self._process_video_chunk_concurrent(video_chunk, recursive)
                results.extend(chunk_results)
                
                # Update processed files set
                for result in chunk_results:
                    if "video_path" in result:
                        processed_files.add(result["video_path"])
                
                # Update progress
                progress.update(len(chunk_results))
                current_index += len(chunk_results)
                
                # Log progress
                progress_info = progress.get_progress()
                eta_formatted = self._format_eta(progress_info['estimated_remaining_seconds'])
                self.logger.info(f"Progress: {progress_info['completed']}/{progress_info['total']} ({progress_info['percentage']:.1f}%) - ETA: {eta_formatted}")
                
                # Save checkpoint after each chunk
                self._save_checkpoint(list(processed_files), results, current_index, total_files)
                self._update_stats(results)
        
        else:
            # Sequential processing (original logic)
            for i, video_path in enumerate(video_files):
                if not self.is_running:
                    self.logger.info("Processing interrupted by user")
                    break
                
                # Process single video
                result = self.processor.process_single_video(video_path)  # Let exceptions propagate
                results.append(result)
                processed_files.add(str(video_path))
                current_index += 1
                
                # Update progress
                progress.update()
                
                # Save checkpoint periodically
                if (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(list(processed_files), results, current_index, total_files)
                    self._update_stats(results)
        
        # Final save
        self._save_checkpoint(list(processed_files), results, current_index, total_files)
        self._update_stats(results)
        
        # Clean up checkpoint if completed
        if current_index >= total_files:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            self.logger.info("Processing completed successfully")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Batch processing finished: {len(results)} videos processed in {processing_time:.1f}s")
        if self.use_concurrent and processing_time > 0:
            self.logger.info(f"Average throughput: {len(results) / processing_time:.2f} videos/second")
        
        return results
    
    def _status_reporter(self, progress: ProgressTracker):
        """Background thread to report processing status."""
        while self.is_running:
            time.sleep(30)  # Report every 30 seconds
            
            if progress.completed_items > 0:
                status = progress.get_progress()
                self.logger.info(
                    f"Progress: {status['completed']}/{status['total']} "
                    f"({status['percentage']:.1f}%) - "
                    f"ETA: {status['estimated_remaining_seconds']/60:.1f} minutes"
                )
    
    def create_html_report(self, results: List[Dict[str, Any]], output_file: str = None) -> str:
        """Create an HTML report of processing results."""
        if output_file is None:
            output_file = self.output_dir / "processing_report.html"
        
        stats = create_summary_stats(results)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Video Processing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
        .videos-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }}
        .video-card {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
        .tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 10px 0; }}
        .tag {{ background-color: #e0e0e0; padding: 2px 8px; border-radius: 3px; font-size: 12px; }}
        .gps-badge {{ background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; }}
        .no-gps-badge {{ background-color: #f44336; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; }}
        .error {{ color: red; }}
        .duration {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Video Processing Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <h3>Total Videos</h3>
            <p>{stats['total_videos']}</p>
        </div>
        <div class="stat-box">
            <h3>Success Rate</h3>
            <p>{stats['success_rate']:.1%}</p>
        </div>
        <div class="stat-box">
            <h3>With GPS</h3>
            <p>{stats['videos_with_gps']}</p>
        </div>
        <div class="stat-box">
            <h3>With OCR Text</h3>
            <p>{stats['videos_with_ocr']}</p>
        </div>
        <div class="stat-box">
            <h3>Total Duration</h3>
            <p>{stats.get('total_duration_formatted', 'N/A')}</p>
        </div>
    </div>
    
    <h2>Top Tags</h2>
    <div class="tags">
        {' '.join([f'<span class="tag">{tag} ({count})</span>' for tag, count in stats['top_tags']])}
    </div>
    
    <h2>Processing Results</h2>
    <div class="videos-grid">
"""
        
        for result in results[:100]:  # Limit to first 100 for HTML
            if 'error' in result:
                html_content += f"""
        <div class="video-card">
            <h4 class="error">Error: {Path(result['video_path']).name}</h4>
            <p class="error">{result['error']}</p>
        </div>
"""
            else:
                tags_html = ' '.join([f'<span class="tag">{tag}</span>' for tag in result.get('tags', [])])
                gps_badge = '<span class="gps-badge">GPS</span>' if result.get('has_gps') else '<span class="no-gps-badge">No GPS</span>'
                html_content += f"""
        <div class="video-card">
            <h4>{Path(result['video_path']).name} {gps_badge}</h4>
            <p class="duration">Duration: {result.get('duration_formatted', 'N/A')}</p>
            <p><strong>Caption:</strong> {result.get('caption', 'N/A')}</p>
            <div class="tags">{tags_html}</div>
            <p><strong>Location:</strong> {result.get('location_name', 'N/A')}</p>
            <p><strong>City:</strong> {result.get('city', 'N/A')}</p>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved to: {output_file}")
        return str(output_file)


def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description="Batch process videos for personal memory retrieval")
    
    parser.add_argument("input_dir", help="Input directory containing videos")
    
    # Vision model configuration
    parser.add_argument("--provider", choices=["vllm", "openai", "none"], default="openai",
                       help="Vision model provider (vllm, openai)")
    parser.add_argument("--vllm-endpoint", default=None,
                       help="VLLM API endpoint URL (overrides config)")
    parser.add_argument("--model", default=None,
                       help="Model name (overrides config)")
    parser.add_argument("--api-key", default=None,
                       help="API key (overrides config)")
    parser.add_argument("--output_dir", default="output/video",
                       help="Output directory")
    parser.add_argument("--cache_dir", default=None,
                       help="Cache directory (default: output_dir/cache)")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                       help="Save checkpoint every N videos")
    parser.add_argument("--max-concurrent", type=int, default=10, 
                       help="Maximum concurrent requests (for VLLM provider, recommended: 10-20 for videos)")
    parser.add_argument("--chunk-size", type=int, default=20,
                       help="Number of videos to process in each concurrent chunk")
    parser.add_argument("--num-frames", type=int, default=8,
                       help="Number of frames to extract from each video")
    parser.add_argument("--timeout", type=int, default=120,
                       help="Request timeout in seconds (default: 120)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from previous checkpoint")
    parser.add_argument("--recursive", action="store_true", default=True,
                       help="Process directories recursively")
    parser.add_argument("--create-html-report", action="store_true",
                       help="Create HTML report of results")
    parser.add_argument("--additional_log_name", default=None)
    
    args = parser.parse_args()

    # Setup directories
    setup_directories(args.output_dir, ["output", "cache", "logs", "frames"])
    
    # Load base configuration
    config = {
        "ocr_config": OCR_CONFIG,
        "geocoding_config": GEOCODING_CONFIG,
        "processing_config": PROCESSING_CONFIG,
        "output_config": OUTPUT_CONFIG,
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir if args.cache_dir else str(Path(args.output_dir) / "cache"),
        "provider": args.provider,
        "frame_extraction": {
            "num_frames": args.num_frames,
            "strategy": "uniform",
        },
    }
    
    # Update config based on provider
    if args.provider == "vllm":
        config.update(VLLM_CONFIG)
        if args.vllm_endpoint:
            config["endpoint"] = args.vllm_endpoint
    elif args.provider == "openai":
        config.update(OPENAI_CONFIG)
    elif args.provider == "none":
        pass
    else:
        raise ValueError("Invalid provider specified. Choose 'vllm', 'openai', or 'none'.")
    
    if args.model:
        config["model"] = args.model
    if args.api_key:
        config["api_key"] = args.api_key
    
    # Set timeout if provided
    if args.timeout:
        config["timeout"] = args.timeout
    
    # Initialize processor
    processor = VideoProcessor(config)
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        processor=processor,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        max_concurrent_requests=args.max_concurrent,
        chunk_size=args.chunk_size
    )
    
    # Process videos
    results = batch_processor.process_batch(
        input_dir=args.input_dir,
        resume=not args.no_resume,
        recursive=args.recursive
    )
    
    print(f"\nBatch processing complete!")
    print(f"Processed {len(results)} videos")
    print(f"Results saved to {batch_processor.results_file}")
    
    # Create HTML report if requested
    if args.create_html_report:
        report_file = batch_processor.create_html_report(results)
        print(f"HTML report saved to: {report_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
