#!/usr/bin/env python3
"""
Email Preprocessing for Personal Memory Retrieval

This module processes email files (HTML, EML, MSG) to extract comprehensive metadata.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re

import requests
from bs4 import BeautifulSoup

try:
    from .config import (
        DEFAULT_PATHS, PROMPTS, HTML_PARSING_CONFIG, ATTACHMENT_CONFIG
    )
    from .utils import (
        extract_text_from_html,
        extract_email_metadata_from_html,
        parse_email_date,
        extract_attachments_info,
        clean_email_address,
        extract_sender_name,
        get_file_hash,
        save_json_file,
        load_json_file,
    )
except ImportError:  # pragma: no cover
    # Support running this file as a script: `python email_processor.py ...`
    from config import (
        DEFAULT_PATHS, PROMPTS, HTML_PARSING_CONFIG, ATTACHMENT_CONFIG
    )
    from utils import (
        extract_text_from_html,
        extract_email_metadata_from_html,
        parse_email_date,
        extract_attachments_info,
        clean_email_address,
        extract_sender_name,
        get_file_hash,
        save_json_file,
        load_json_file,
    )


from openai import OpenAI



class EmailProcessor:
    """Main class for processing emails and extracting comprehensive metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the EmailProcessor with a configuration dictionary.
        
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
        
        # Processing settings
        self.html_config = self.config.get("html_config", HTML_PARSING_CONFIG)
        self.attachment_config = self.config.get("attachment_config", ATTACHMENT_CONFIG)
        
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
        log_file = self.logs_dir / "email_processor.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _get_cache_path(self, file_path: Path, cache_type: str) -> Path:
        """Get cache file path for a given email and cache type."""
        file_hash = get_file_hash(file_path)
        return self.cache_dir / f"{file_hash}_{cache_type}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load data from cache if it exists."""
        if cache_path.exists():
            try:
                return load_json_file(cache_path)
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save data to cache."""
        try:
            save_json_file(data, cache_path)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_path}: {e}")

    def read_email_file(self, email_path: Path) -> Tuple[str, str]:
        """
        Read email file and return content and format.
        
        Returns:
            Tuple of (content, format) where format is 'html', 'eml', or 'msg'
        """
        try:
            # Determine file format
            file_format = email_path.suffix.lower().lstrip('.')
            
            if file_format in ['html', 'htm']:
                with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return content, 'html'
            
            elif file_format == 'eml':
                # For .eml files, we'd need email library parsing
                # For now, treat as text and extract HTML if present
                with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return content, 'eml'
            
            elif file_format == 'msg':
                # For .msg files, we'd need python-msg-extractor or similar
                # For now, raise an error
                raise NotImplementedError("MSG format not yet supported")
            
            else:
                raise ValueError(f"Unsupported email format: {file_format}")
                
        except Exception as e:
            self.logger.error(f"Failed to read email file {email_path}: {e}")
            raise

    def extract_email_metadata(self, email_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from email file."""
        try:
            content, file_format = self.read_email_file(email_path)
            
            if file_format == 'html':
                # Extract from HTML structure
                soup = BeautifulSoup(content, 'html.parser')
                metadata = extract_email_metadata_from_html(soup)
                
                # Parse the clean text
                clean_text, _ = extract_text_from_html(content, self.html_config)
                
            elif file_format == 'eml':
                # Basic EML parsing - would need proper email library for full support
                metadata = self._parse_eml_headers(content)
                clean_text = self._extract_text_from_eml(content)
                
            else:
                raise ValueError(f"Unsupported format for metadata extraction: {file_format}")
            
            # Parse date
            timestamp = None
            if metadata.get("date"):
                timestamp = parse_email_date(metadata["date"])
            
            # Extract file metadata
            file_stats = email_path.stat()
            
            return {
                "file_path": str(email_path),
                "file_name": email_path.name,
                "file_size": file_stats.st_size,
                "file_modified": datetime.fromtimestamp(file_stats.st_mtime),
                "file_format": file_format,
                "timestamp": timestamp,
                "text_content": clean_text,
                "text_length": len(clean_text),
                "attachments": extract_attachments_info(content) if self.attachment_config.get("process_attachments") else []
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {email_path}: {e}")
            return {
                "file_path": str(email_path),
                "error": str(e),
                "timestamp": None
            }

    def _parse_eml_headers(self, eml_content: str) -> Dict[str, str]:
        """Basic EML header parsing."""
        headers = {}
        lines = eml_content.split('\n')
        
        current_header = None
        current_value = ""
        
        for line in lines:
            if line.strip() == "":
                break  # End of headers
                
            if line.startswith(' ') or line.startswith('\t'):
                # Continuation of previous header
                if current_header:
                    current_value += " " + line.strip()
            else:
                # Save previous header
                if current_header:
                    headers[current_header.lower()] = current_value.strip()
                
                # Parse new header
                if ':' in line:
                    current_header, current_value = line.split(':', 1)
                    current_header = current_header.strip()
                    current_value = current_value.strip()
        
        # Save last header
        if current_header:
            headers[current_header.lower()] = current_value.strip()
        
        # Map to standard format
        return {
            "subject": headers.get("subject", ""),
            "sender": headers.get("from", ""),
            "recipient": headers.get("to", ""),
            "date": headers.get("date", ""),
            "cc": headers.get("cc", ""),
            "reply_to": headers.get("reply-to", "")
        }

    def _extract_text_from_eml(self, eml_content: str) -> str:
        """Extract text content from EML file."""
        # Find the start of the body (after empty line)
        lines = eml_content.split('\n')
        body_start = 0
        
        for i, line in enumerate(lines):
            if line.strip() == "":
                body_start = i + 1
                break
        
        body_content = '\n'.join(lines[body_start:])
        
        # If it contains HTML, parse it
        if '<html' in body_content.lower() or '<body' in body_content.lower():
            text, _ = extract_text_from_html(body_content, self.html_config)
            return text
        
        # Otherwise, clean up plain text
        return re.sub(r'\s+', ' ', body_content).strip()

    def call_llm_api(self, prompt: str, system_prompt: str = None) -> str:
        """Call LLM API with the given prompt."""
        if self.provider == "none":
            return ""
        
        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            model = self.config.get("model")
            max_tokens_value = self.config.get("max_tokens", 1000)
            
            # Check if using newer reasoning models (GPT-5, o1, o3)
            is_newer_model = any(x in model.lower() for x in ["gpt-5", "o1", "o3"])
            
            kwargs = {
                "model": model,
                "messages": messages,
                "timeout": self.config.get("timeout", 30)
            }
            
            if is_newer_model:
                # Newer models use max_completion_tokens and don't support custom temperature
                # Reasoning models need 3x tokens (reasoning + output)
                kwargs["max_completion_tokens"] = max_tokens_value * 3
            else:
                # Older models support temperature and use max_tokens
                kwargs["max_tokens"] = max_tokens_value
                kwargs["temperature"] = self.config.get("temperature", 0.2)
            
            response = self.openai_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        
        elif self.provider == "vllm":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self.config.get("model"),
                "messages": messages,
                "max_tokens": self.config.get("max_tokens"),
                "temperature": self.config.get("temperature")
            }
            
            response = requests.post(
                self.config.get("endpoint"),
                headers={"Content-Type": "application/json", 
                         "Authorization": f"Bearer {self.config.get('api_key')}"},
                json=data,
                timeout=self.config.get("timeout", 30)
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")

    def analyze_email_content(self, email_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze email content using LLM."""
        if self.provider == "none" or not email_metadata.get("text_content"):
            return {}
        
        text_content = email_metadata["text_content"]
        timestamp = email_metadata.get("timestamp", "Unknown")
        sender = email_metadata.get("sender_name", "Unknown")
        recipient = email_metadata.get("recipient_email", "Unknown")
        
        analysis = {}
        
        # Generate summary
        summary_prompt = PROMPTS["short_summary"].format(
            timestamp=timestamp,
            sender=sender,
            email_body=text_content
        )
        
        analysis["short_summary"] = self.call_llm_api(summary_prompt)

        long_rewrite_prompt = PROMPTS["long_rewrite"].format(
            timestamp=timestamp,
            sender=sender,
            email_body=text_content
        )
        analysis["long_rewrite"] = self.call_llm_api(long_rewrite_prompt)

        # Classify email
        classification_prompt = PROMPTS["classification"] + f"\n\nEmail content:\n{text_content}"
        classification_response = self.call_llm_api(classification_prompt)
        if classification_response:
            analysis["classification"] = classification_response.strip()
        
        return analysis


    def process_email(self, email_path: Path, use_cache: bool = True) -> Dict[str, Any]:
        """Process a single email file and return comprehensive metadata."""
        self.logger.info(f"Processing email: {email_path}")
        
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(email_path, "processed")
            cached_result = self._load_from_cache(cache_path)
            if cached_result:
                self.logger.debug(f"Loaded from cache: {email_path}")
                return cached_result
        
        # Extract basic metadata
        metadata = self.extract_email_metadata(email_path)
        
        if "error" in metadata:
            return {
                "status": "error",
                "file_path": str(email_path),
                "error": metadata["error"],
                "processed_at": datetime.now().isoformat()
            }
        
        # Analyze content with LLM
        analysis = self.analyze_email_content(metadata)
        
        # Combine all results
        result = {
            "status": "success",
            "processed_at": datetime.now().isoformat(),
            **metadata,
            **analysis
        }
        
        # Remove large text content from output if configured
        if not self.config.get("include_html_content", False):
            result.pop("text_content", None)
        
        # Save to cache
        if use_cache:
            self._save_to_cache(cache_path, result)
        
        return result
