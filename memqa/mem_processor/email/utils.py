#!/usr/bin/env python3
"""
Utility functions for email processing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Comment


def setup_directories(base_dir: str) -> Dict[str, Path]:
    """Create necessary directories for processing."""
    base_path = Path(base_dir)
    directories = {
        "output": base_path,
        "cache": base_path / "cache",
        "logs": base_path / "logs", 
        "temp": base_path / "temp"
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        
    return directories


def find_emails_in_directory(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Find all email files in a directory recursively."""
    if extensions is None:
        extensions = ['.html', '.eml', '.msg']
    
    email_files = []
    for ext in extensions:
        email_files.extend(directory.rglob(f"*{ext}"))
    
    # Sort by modification time
    email_files.sort(key=lambda x: x.stat().st_mtime)
    return email_files


def extract_text_from_html(html_content: str, config: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Extract clean text from HTML email content.
    
    Returns:
        Tuple of (clean_text, metadata)
    """
    if config is None:
        config = {
            "remove_scripts": True,
            "remove_styles": True, 
            "remove_images": True,
            "remove_links": False,
            "preserve_structure": True,
            "max_text_length": 50000
        }
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract metadata first
    metadata = extract_email_metadata_from_html(soup)
    
    # Remove unwanted elements
    if config.get("remove_scripts", True):
        for script in soup(["script", "style"]):
            script.decompose()
    
    if config.get("remove_images", True):
        for img in soup("img"):
            img.decompose()
    
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # Handle links
    if config.get("remove_links", False):
        for link in soup("a"):
            link.unwrap()  # Keep text but remove link
    
    # Get clean text
    text = soup.get_text(separator=' ', strip=True)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Limit text length
    max_length = config.get("max_text_length", 50000)
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text, metadata


def extract_email_metadata_from_html(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract email metadata from HTML structure."""
    import html
    
    metadata = {
        "subject": "",
        "sender": "",
        "recipient": "",
        "date": "",
        "cc": "",
        "bcc": "",
        "reply_to": ""
    }
    
    # Get raw HTML content for regex patterns
    html_content = str(soup)
    
    # Method 1: Look for Mozilla/Thunderbird style HTML table patterns
    # Handle patterns like: <b>From: </b>noreply &lt;noreply@interieur.gouv.fr&gt;
    table_patterns = {
        'subject': [
            r'<b>Subject:\s*</b>(.*?)(?:</td>|</tr>|<br>|<table|$)',
            r'<td><b>Subject:\s*</b>(.*?)</td>',
            r'<b>Subject:\s*</b>([^<]*?)(?:<[^/]|$)',
        ],
        'from': [
            r'<b>From:\s*</b>(.*?)(?:</td>|</tr>|<br>|<table|$)',
            r'<td><b>From:\s*</b>(.*?)</td>',
            r'<b>From:\s*</b>([^<]*?)(?:<[^/]|$)',
        ],
        'to': [
            r'<b>To:\s*</b>(.*?)(?:</td>|</tr>|<br>|<table|$)',
            r'<td><b>To:\s*</b>(.*?)</td>',
            r'<b>To:\s*</b>([^<]*?)(?:<[^/]|$)',
        ],
        'date': [
            r'<b>Date:\s*</b>(.*?)(?:</td>|</tr>|<br>|<table|$)',
            r'<td><b>Date:\s*</b>(.*?)</td>',
            r'<b>Date:\s*</b>([^<]*?)(?:<[^/]|$)',
        ],
        'cc': [
            r'<b>CC:\s*</b>(.*?)(?:</td>|</tr>|<br>|<table|$)',
            r'<td><b>CC:\s*</b>(.*?)</td>',
            r'<b>CC:\s*</b>([^<]*?)(?:<[^/]|$)',
        ],
        'reply_to': [
            r'<b>Reply-To:\s*</b>(.*?)(?:</td>|</tr>|<br>|<table|$)',
            r'<td><b>Reply-To:\s*</b>(.*?)</td>',
            r'<b>Reply-To:\s*</b>([^<]*?)(?:<[^/]|$)',
        ]
    }
    
    # Apply patterns
    for field, patterns in table_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                # Decode HTML entities
                extracted = html.unescape(extracted)
                # Handle remaining entities manually
                extracted = extracted.replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&amp;', '&')
                
                # Map to metadata keys
                if field == 'from':
                    metadata["sender"] = extracted
                elif field == 'to':
                    metadata["recipient"] = extracted
                elif field == 'subject':
                    metadata["subject"] = extracted
                elif field == 'date':
                    metadata["date"] = extracted
                elif field == 'cc':
                    metadata["cc"] = extracted
                elif field == 'reply_to':
                    metadata["reply_to"] = extracted
                break  # Use first matching pattern
    
    # Method 2: Look for Mozilla/Thunderbird style headers in table structures
    header_tables = soup.find_all("table", class_=lambda x: x and ("moz-header" in x or "header" in x))
    
    for table in header_tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 2:
                header_name = cells[0].get_text(strip=True).lower().rstrip(':')
                header_value = cells[1].get_text(strip=True)
                
                # Only update if we haven't already found a value from Method 1
                if "subject" in header_name and not metadata["subject"]:
                    metadata["subject"] = header_value
                elif "from" in header_name and not metadata["sender"]:
                    metadata["sender"] = header_value
                elif "to" in header_name and not metadata["recipient"]:
                    metadata["recipient"] = header_value
                elif "date" in header_name and not metadata["date"]:
                    metadata["date"] = header_value
                elif "cc" in header_name and not metadata["cc"]:
                    metadata["cc"] = header_value
                elif "reply-to" in header_name and not metadata["reply_to"]:
                    metadata["reply_to"] = header_value
    
    # Method 3: Check title for subject if still empty
    if not metadata["subject"]:
        title = soup.find("title")
        if title:
            metadata["subject"] = title.get_text(strip=True)
    
    # Method 4: Text-based fallback search with validation
    if not metadata["sender"] or not metadata["recipient"]:
        for text in soup.stripped_strings:
            text_lower = text.lower()
            
            # Only set if we get meaningful content with email validation
            if text_lower.startswith('from:') and not metadata["sender"]:
                potential_sender = text[5:].strip()
                if potential_sender and len(potential_sender) > 3 and '@' in potential_sender:
                    metadata["sender"] = potential_sender
            elif text_lower.startswith('to:') and not metadata["recipient"]:
                potential_recipient = text[3:].strip()
                if potential_recipient and len(potential_recipient) > 3 and '@' in potential_recipient:
                    metadata["recipient"] = potential_recipient
    
    return metadata


def parse_email_date(date_string: str) -> Optional[datetime]:
    """Parse email date string into datetime object."""
    if not date_string:
        return None
    
    # Common email date formats
    formats = [
        "%m/%d/%Y, %I:%M %p",  # 5/10/2024, 2:46 PM
        "%Y-%m-%d %H:%M:%S",   # 2024-05-10 14:46:00
        "%d %b %Y %H:%M:%S",   # 10 May 2024 14:46:00
        "%a, %d %b %Y %H:%M:%S",  # Fri, 10 May 2024 14:46:00
        "%Y-%m-%dT%H:%M:%S",   # ISO format
        "%Y-%m-%d",            # Date only
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string.strip(), fmt)
        except ValueError:
            continue
    
    # Try to parse with timezone info (remove it first)
    cleaned_date = re.sub(r'\s*[+-]\d{4}.*$', '', date_string.strip())
    for fmt in formats:
        try:
            return datetime.strptime(cleaned_date, fmt)
        except ValueError:
            continue
    
    return None


def extract_attachments_info(html_content: str) -> List[Dict[str, str]]:
    """Extract information about email attachments from HTML."""
    attachments = []
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Look for attachment indicators
    # This is very email client specific - might need customization
    attachment_patterns = [
        r'attachment[s]?[:\s]*([^\n]+)',
        r'attached[:\s]*([^\n]+)',
        r'file[s]?[:\s]*([^\n]+)'
    ]
    
    text = soup.get_text()
    for pattern in attachment_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.strip()) > 3:  # Avoid false positives
                attachments.append({
                    "name": match.strip(),
                    "type": "inferred",
                    "size": "unknown"
                })
    
    return attachments


def clean_email_address(email_str: str) -> str:
    """Clean and extract email address from various formats."""
    if not email_str:
        return ""
    
    # Extract email from "Name <email@domain.com>" format
    email_match = re.search(r'<([^>]+)>', email_str)
    if email_match:
        return email_match.group(1).strip()
    
    # Extract email from "email@domain.com (Name)" format
    email_match = re.search(r'([^\s()]+@[^\s()]+)', email_str)
    if email_match:
        return email_match.group(1).strip()
    
    return email_str.strip()


def extract_sender_name(sender_str: str) -> str:
    """Extract sender name from email sender field."""
    if not sender_str:
        return ""
    
    # Handle "Name <email@domain.com>" format
    if '<' in sender_str and '>' in sender_str:
        name_part = sender_str.split('<')[0].strip()
        if name_part:
            return name_part.strip('"\'')
    
    # Handle "email@domain.com (Name)" format
    if '(' in sender_str and ')' in sender_str:
        name_match = re.search(r'\(([^)]+)\)', sender_str)
        if name_match:
            return name_match.group(1).strip()
    
    # If it's just an email, extract the local part
    if '@' in sender_str:
        local_part = sender_str.split('@')[0]
        # Convert common email formats to readable names
        return local_part.replace('.', ' ').replace('_', ' ').title()
    
    return sender_str.strip()


def get_file_hash(file_path: Path) -> str:
    """Generate MD5 hash for a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_json_file(data: Any, file_path: Path):
    """Save data to JSON file with proper error handling."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logging.error(f"Failed to save JSON file {file_path}: {e}")
        raise


def load_json_file(file_path: Path) -> Optional[Any]:
    """Load data from JSON file with proper error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load JSON file {file_path}: {e}")
        return None


def create_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics from processing results."""
    if not results:
        return {"total_emails": 0, "error_count": 0, "success_count": 0}
    
    total = len(results)
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = total - success_count
    
    # Category distribution
    categories = {}
    for result in results:
        if result.get("status") == "success" and "classification" in result:
            cat = result["classification"]
            if isinstance(cat, str):
                cats = [c.strip() for c in cat.split(',')]
            else:
                cats = [str(cat)]
            
            for c in cats:
                categories[c] = categories.get(c, 0) + 1
    
    # Date range
    dates = []
    for result in results:
        if result.get("status") == "success" and "timestamp" in result:
            try:
                if isinstance(result["timestamp"], str):
                    dates.append(datetime.fromisoformat(result["timestamp"]))
                elif isinstance(result["timestamp"], datetime):
                    dates.append(result["timestamp"])
            except:
                continue
    
    date_range = {}
    if dates:
        dates.sort()
        date_range = {
            "earliest": dates[0].isoformat(),
            "latest": dates[-1].isoformat(),
            "span_days": (dates[-1] - dates[0]).days
        }
    
    return {
        "total_emails": total,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": success_count / total if total > 0 else 0,
        "categories": categories,
        "date_range": date_range,
        "processing_timestamp": datetime.now().isoformat()
    }


class ProgressTracker:
    """Track processing progress with timing information."""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time
        
    def update(self, increment: int = 1):
        """Update progress counter."""
        self.processed_items += increment
        self.last_update = datetime.now()
        
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information."""
        elapsed = (self.last_update - self.start_time).total_seconds()
        rate = self.processed_items / elapsed if elapsed > 0 else 0
        
        remaining = self.total_items - self.processed_items
        eta_seconds = remaining / rate if rate > 0 else 0
        
        return {
            "processed": self.processed_items,
            "total": self.total_items,
            "percentage": (self.processed_items / self.total_items * 100) if self.total_items > 0 else 0,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
            "eta_seconds": eta_seconds
        }
        
    def __str__(self) -> str:
        info = self.get_progress_info()
        return f"{info['processed']}/{info['total']} ({info['percentage']:.1f}%) - {info['rate_per_second']:.2f}/s"
