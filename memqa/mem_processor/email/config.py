#!/usr/bin/env python3
"""
Configuration file for email processing setup.
Imports common configurations from global_config and defines email-specific settings.
"""

from pathlib import Path

# Import common configurations from global config
from memqa.global_config import (
    OPENAI_CONFIG,
    VLLM_TEXT_CONFIG as VLLM_CONFIG,
    PROCESSING_CONFIG,
    EMAIL_PROCESSING_CONFIG,
    OUTPUT_CONFIG,
    EMAIL_CATEGORIES,
    DEFAULT_PATHS,
    LOGGING_CONFIG,
)

# Update processing config with email-specific settings
PROCESSING_CONFIG = {
    **PROCESSING_CONFIG,
    **EMAIL_PROCESSING_CONFIG,
    "vllm_timeout": 60,  # Email-specific timeout
}

# ============================================================================
# EMAIL-SPECIFIC CONFIGURATIONS
# ============================================================================

# HTML Parsing Configuration
HTML_PARSING_CONFIG = {
    "remove_scripts": True,
    "remove_styles": True,
    "remove_images": True,
    "remove_links": False,  # Keep link text but can ignore URLs
    "preserve_structure": True,
    "max_text_length": 50000,  # Limit text length for processing
    "encoding": "utf-8"
}

# Attachment Processing Configuration
ATTACHMENT_CONFIG = {
    "process_attachments": False,
    "max_attachment_size": 10 * 1024 * 1024,  # 10MB
    "allowed_attachment_types": ['.pdf', '.doc', '.docx', '.txt', '.jpg', '.png'],
    "extract_text_from_pdf": False,  # Can be enabled if needed
    "save_attachment_metadata": True
}

# ============================================================================
# EMAIL-SPECIFIC PROMPTS
# ============================================================================
PROMPTS = {
    "short_summary": """
Task: Generate a concise, privacy-preserving, anonymized summary of the provided email.
Your output will be used in a public dataset, so strict adherence to privacy rules is paramount. 
Any leakage of Personally Identifiable Information (PII) or sensitive data is unacceptable.

Email Details:
Date: {timestamp}
From: {sender}
Email Content:
---
{email_body}
---

Summary Requirements (1-2 sentences):
- Capture the primary purpose or main topic of the email.
- Highlight essential action items or key information relevant to the user.
- Include important dates or relevant numerical values (excluding sensitive financial data).
- The summary should serve as a quick, effective reminder for personal memory retrieval.

STRICT PRIVACY AND REDACTION RULES:
- Absolutely no Personally Identifiable Information (PII) or sensitive data is permitted.
- Specifically, you MUST NOT mention or include:
    - Any real names of individuals (e.g., sender, recipient, or people mentioned in the body) — use general placeholders if necessary (e.g., "[PERSON]", "[SENDER]", "[RECIPIENT]", "[COLLEAGUE]").
    - Phone numbers.
    - Private residential addresses including shipping addresses. 
    - Specific identification numbers (e.g., passport, driver's license, loyalty program IDs, customer IDs, account numbers, social security numbers/tax IDs, birthdays, one-time passwords/TOTP).
    - Real email addresses.
    - IP addresses or other device-specific identifiers.
    - Full URLs or direct links of any kind.
    - Any specific financial account details (e.g., credit card numbers, full bank account numbers, specific subscription plan names if directly tied to an individual's account).
    - Transactional numbers, order IDs, confirmation numbers, or tracking numbers that could be traced.
    - Any other unique or highly specific personal details that could, directly or indirectly, lead to the re-identification of any person mentioned in the email or the user (e.g., specific medical conditions).

Exceptions:
- Please note public addresses of known establishments like specific restaurants, hotels, or public landmarks are allowed and are encouraged to be kept.
- Company names, product names, and general service references are allowed and should be included if relevant to the summary.
    """,

"long_rewrite": """
Task: Generate a privacy-preserving version of an email.
Your output will be used in a public dataset, so strict adherence to privacy rules is paramount. 
Any leakage of Personally Identifiable Information (PII) or sensitive data is unacceptable.
Email Details:
Date: {timestamp}
From: {sender}
Email Content:
---
{email_body}
---
Please rewrite the email based on the above content.
This rewritten should be within 5 sentences long and focus on:
- Remove Real email addresses or real names. Use general but informative references.
- Rewrite the subject if it contains privacy leaking info. Rewrite the subject to better summarize the content.
- Focus on the semantics of the message, removing any sensitive or irrelevant detail. 
- Output only the summary text — no bullet points, labels, or explanations.
- If contains shopping list, or detailed invoice, list all the relevant products, services being provided. Note that the public address like the store/hotel's location and store/hotel's name is not sensitive content and should be included in details as they are important informations.

STRICT PRIVACY AND REDACTION RULES:
- Absolutely no Personally Identifiable Information (PII) or sensitive data is permitted.
- Specifically, you MUST NOT mention or include:
    - Any real names of individuals (e.g., sender, recipient, or people mentioned in the body) — use general placeholders if necessary (e.g., "[PERSON]", "[SENDER]", "[RECIPIENT]", "[COLLEAGUE]").
    - Phone numbers.
    - Private residential addresses including shipping addresses. 
    - Specific identification numbers (e.g., passport, driver's license, loyalty program IDs, customer IDs, account numbers, social security numbers/tax IDs, birthdays, one-time passwords/TOTP).
    - Real email addresses.
    - IP addresses or other device-specific identifiers.
    - Full URLs or direct links of any kind.
    - Any specific financial account details (e.g., credit card numbers, full bank account numbers, specific subscription plan names if directly tied to an individual's account).
    - Transactional numbers, order IDs, confirmation numbers, or tracking numbers that could be traced.
    - Any other unique or highly specific personal details that could, directly or indirectly, lead to the re-identification of any person mentioned in the email or the user (e.g., specific medical conditions).

Exceptions:
- Please note public addresses of known establishments like specific restaurants, hotels, or public landmarks are allowed and are encouraged to be kept.
- Company names, product names, and general service references are allowed and should be included if relevant to the summary.

Output Format:
Your output must strictly follow this exact structure. Populate the `Date`, `From`, and `Subject` fields using privacy-preserving information derived from the email context:

Date: [e.g., '2023-01-15']
From: 
Subject: [Rewritten Subject that accurately summarizes content and adheres to all privacy rules]
Content: [The 3-5 sentence detailed abstract, adhering to all content and privacy requirements.]
    """,

    "classification": """Classify this email into one or more of these categories:
- personal: Family, friends, personal correspondence
- work: Business, professional, work-related
- finance: Banking, payments, invoices, financial
- travel: Bookings, hotels, flights, travel-related
- shopping: Orders, purchases, deliveries, e-commerce
- notifications: Alerts, reminders, updates, newsletters
- social: Social media notifications, community
- education: Academic, courses, research, learning
- health: Medical, health-related, appointments
- utility: Bills, services, subscriptions, utilities

Return only the most relevant category name(s) as a comma-separated list.""",

}