#!/usr/bin/env python3
"""
Batch processing script for large email collections.
"""

import argparse
import logging
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from .email_processor import EmailProcessor
    from .utils import (
        setup_directories,
        find_emails_in_directory,
        create_summary_stats,
        ProgressTracker,
        save_json_file,
        load_json_file,
    )
    from .config import (
        DEFAULT_PATHS,
        PROCESSING_CONFIG,
        OPENAI_CONFIG,
        VLLM_CONFIG,
        OUTPUT_CONFIG,
        HTML_PARSING_CONFIG,
        ATTACHMENT_CONFIG,
    )
except ImportError:  # pragma: no cover
    from email_processor import EmailProcessor
    from utils import (
        setup_directories,
        find_emails_in_directory,
        create_summary_stats,
        ProgressTracker,
        save_json_file,
        load_json_file,
    )
    from config import (
        DEFAULT_PATHS,
        PROCESSING_CONFIG,
        OPENAI_CONFIG,
        VLLM_CONFIG,
        OUTPUT_CONFIG,
        HTML_PARSING_CONFIG,
        ATTACHMENT_CONFIG,
    )


class EmailBatchProcessor:
    """Batch processor for large email collections with resume capability and concurrent processing."""
    
    def __init__(self, 
                 processor: EmailProcessor,
                 output_dir: str = DEFAULT_PATHS["output_dir"],
                 checkpoint_interval: int = 50,
                 max_concurrent_requests: int = 50,
                 chunk_size: int = 100):
        """
        Initialize batch processor.
        
        Args:
            processor: EmailProcessor instance
            output_dir: Directory for output files
            checkpoint_interval: Save checkpoint every N emails
            max_concurrent_requests: Maximum concurrent requests for VLLM (recommended: 50+)
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
    
    def _load_previous_results(self) -> List[Dict[str, Any]]:
        """Load previous results if they exist."""
        if self.results_file.exists():
            results = load_json_file(self.results_file)
            return results if results else []
        return []
    
    def _process_email_chunk_concurrent(self, 
                                      email_chunk: List[Path], 
                                      use_cache: bool = True) -> List[Dict[str, Any]]:
        """Process a chunk of emails concurrently using ThreadPoolExecutor."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            # Submit all emails in the chunk
            future_to_email = {
                executor.submit(self.processor.process_email, email_file, use_cache): email_file 
                for email_file in email_chunk
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_email):
                email_file = future_to_email[future]
                result = future.result()  # Let exceptions propagate
                results.append(result)
        
        return results
    
    def _process_email_chunk_sequential(self, 
                                      email_chunk: List[Path], 
                                      use_cache: bool = True) -> List[Dict[str, Any]]:
        """Process a chunk of emails sequentially (fallback for non-VLLM providers)."""
        results = []
        
        for email_file in email_chunk:
            if not self.is_running:
                break
                
            result = self.processor.process_email(email_file, use_cache=use_cache)  # Let exceptions propagate
            results.append(result)
        
        return results
    
    def _split_into_chunks(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split a list into chunks of specified size."""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    def process_email_list(self, 
                          email_files: List[Path],
                          use_cache: bool = True,
                          resume: bool = True) -> Dict[str, Any]:
        """
        Process a list of email files with optional concurrent processing.
        
        Args:
            email_files: List of email file paths to process
            use_cache: Whether to use caching
            resume: Whether to resume from previous checkpoint
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = datetime.now()
        
        # Load checkpoint if resuming
        processed_files = set()
        results = []
        start_index = 0
        
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                processed_files = set(checkpoint.get("processed_files", []))
                results = self._load_previous_results()
                start_index = checkpoint.get("current_index", 0)
                self.logger.info(f"Resuming from checkpoint: {start_index}/{len(email_files)}")
        
        # Filter out already processed files
        remaining_files = [f for f in email_files if str(f) not in processed_files]
        
        if not remaining_files:
            self.logger.info("All files already processed!")
            return {
                "status": "completed",
                "total_files": len(email_files),
                "results": results,
                "processing_time": 0
            }
        
        # Log processing mode
        if self.use_concurrent:
            self.logger.info(f"Using concurrent processing with {self.max_concurrent_requests} max concurrent requests")
            self.logger.info(f"Processing {len(remaining_files)} emails in chunks of {self.chunk_size}")
        else:
            self.logger.info(f"Using sequential processing (provider: {self.processor.provider})")
        
        # Initialize progress tracker
        progress = ProgressTracker(len(remaining_files))
        
        if self.use_concurrent:
            # Split files into chunks for concurrent processing
            file_chunks = self._split_into_chunks(remaining_files, self.chunk_size)
            
            for chunk_idx, email_chunk in enumerate(file_chunks):
                if not self.is_running:
                    self.logger.info("Processing interrupted by user")
                    break
                
                self.logger.info(f"Processing chunk {chunk_idx + 1}/{len(file_chunks)} ({len(email_chunk)} files)")
                
                # Process chunk concurrently
                chunk_results = self._process_email_chunk_concurrent(email_chunk, use_cache)
                results.extend(chunk_results)
                
                # Update processed files set
                for result in chunk_results:
                    if "file_path" in result:
                        processed_files.add(result["file_path"])
                
                # Update progress
                progress.update(len(chunk_results))
                
                # Log progress
                progress_info = progress.get_progress_info()
                eta_formatted = self._format_eta(progress_info['eta_seconds'])
                self.logger.info(f"Progress: {progress} - ETA: {eta_formatted}")
                
                # Save checkpoint after each chunk
                current_index = start_index + len(results)
                self._save_checkpoint(
                    list(processed_files),
                    results,
                    current_index,
                    len(email_files)
                )
        
        else:
            # Sequential processing (original logic)
            for i, email_file in enumerate(remaining_files):
                if not self.is_running:
                    self.logger.info("Processing interrupted by user")
                    break
                
                # Process single email
                result = self.processor.process_email(email_file, use_cache=use_cache)  # Let exceptions propagate
                results.append(result)
                processed_files.add(str(email_file))
                
                # Update progress
                progress.update()
                
                if (len(results) % 10 == 0) or (len(results) % self.checkpoint_interval == 0):
                    progress_info = progress.get_progress_info()
                    eta_formatted = self._format_eta(progress_info['eta_seconds'])
                    self.logger.info(f"Progress: {progress} - ETA: {eta_formatted}")
                
                # Save checkpoint periodically
                if len(results) % self.checkpoint_interval == 0:
                    self._save_checkpoint(
                        list(processed_files),
                        results,
                        start_index + i + 1,
                        len(email_files)
                    )
        
        # Final save
        self._save_checkpoint(
            list(processed_files),
            results,
            len(email_files),
            len(email_files)
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate summary statistics
        stats = create_summary_stats(results)
        stats["processing_time_seconds"] = processing_time
        stats["files_per_second"] = len(results) / processing_time if processing_time > 0 else 0
        
        save_json_file(stats, self.stats_file)
        
        self.logger.info(f"Processing completed: {len(results)} emails in {processing_time:.1f}s")
        self.logger.info(f"Success rate: {stats['success_rate']:.1%}")
        if self.use_concurrent:
            self.logger.info(f"Average throughput: {stats['files_per_second']:.2f} files/second")
        
        return {
            "status": "completed" if self.is_running else "interrupted",
            "total_files": len(email_files),
            "processed_files": len(results),
            "results": results,
            "statistics": stats,
            "processing_time": processing_time
        }
    
    def process_directory(self, 
                         directory: Path,
                         extensions: List[str] = None,
                         use_cache: bool = True,
                         resume: bool = True) -> Dict[str, Any]:
        """
        Process all emails in a directory.
        
        Args:
            directory: Directory containing email files
            extensions: List of file extensions to process
            use_cache: Whether to use caching
            resume: Whether to resume from previous checkpoint
            
        Returns:
            Dictionary with processing results and statistics
        """
        self.logger.info(f"Scanning directory: {directory}")
        
        # Find all email files
        email_files = find_emails_in_directory(directory, extensions)
        
        if not email_files:
            self.logger.warning(f"No email files found in {directory}")
            return {
                "status": "no_files",
                "total_files": 0,
                "results": [],
                "processing_time": 0
            }
        
        self.logger.info(f"Found {len(email_files)} email files")
        
        # Process the files
        return self.process_email_list(email_files, use_cache, resume)



def main():
    """Main function for command-line batch processing."""
    parser = argparse.ArgumentParser(description="Batch Email Processing for Personal Memory Retrieval")
    parser.add_argument("input_dir", help="Directory containing email files")
    parser.add_argument("--output_dir", default="output/email", help="Output directory")
    parser.add_argument("--vllm-endpoint", default=None,
                       help="VLLM API endpoint URL (overrides config)")
    parser.add_argument("--provider", choices=["openai", "vllm", "none"], default="openai",
                       help="LLM provider to use")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Save checkpoint every N files")
    parser.add_argument("--max-concurrent", type=int, default=50, 
                       help="Maximum concurrent requests (for VLLM provider, recommended: 50+)")
    parser.add_argument("--chunk-size", type=int, default=100,
                       help="Number of files to process in each concurrent chunk")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Request timeout in seconds (default: 60)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--extensions", nargs="+", default=[".html", ".eml"], help="File extensions to process")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    setup_directories(str(output_dir))
    
    # Build configuration
    if args.provider == "openai":
        config = OPENAI_CONFIG.copy()
    elif args.provider == "vllm":
        config = VLLM_CONFIG.copy()
        if args.vllm_endpoint:
            config["endpoint"] = args.vllm_endpoint
    
    if args.model:
        config["model"] = args.model
    
    # Set timeout if provided
    if args.timeout:
        config["timeout"] = args.timeout
    
    config.update({
        "output_dir": str(output_dir),
        "html_config": HTML_PARSING_CONFIG,
        "attachment_config": ATTACHMENT_CONFIG
    })
    
    # Initialize processor and batch processor
    processor = EmailProcessor(config)
    batch_processor = EmailBatchProcessor(
        processor, 
        str(output_dir),
        args.checkpoint_interval,
        args.max_concurrent,
        args.chunk_size
    )
    
    
    # Process directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    result = batch_processor.process_directory(
        input_dir,
        extensions=args.extensions,
        use_cache=not args.no_cache,
        resume=not args.no_resume
    )
    
    print(f"\nProcessing completed!")
    print(f"Status: {result['status']}")
    print(f"Total files: {result['total_files']}")
    print(f"Processed: {result['processed_files']}")
    print(f"Time: {result['processing_time']:.1f}s")
    
    if result.get('statistics'):
        stats = result['statistics']
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Processing speed: {stats['files_per_second']:.1f} files/second")
        
        if stats.get('categories'):
            print(f"Top categories: {list(stats['categories'].keys())[:5]}")


if __name__ == "__main__":
    main()
