#!/usr/bin/env python3
"""
Oracle baseline for Memory QA.
Given ground-truth evidence IDs, ask the model to answer.
"""

import argparse
import base64
import json
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests
from openai import OpenAI
from tqdm import tqdm

from memqa.mem_processor.video.utils import extract_frames
from memqa.qa_agent_baselines.oracle.config import ORACLE_CONFIG, PROMPTS
from memqa.qa_agent_baselines.MMRag.llm_utils import (
    TokenUsage,
    _messages_to_responses_input,
    _extract_response_text,
    _should_use_responses,
    _responses_supports_temperature,
)


VIDEO_EXTENSIONS = tuple(
    sorted({f".{ext.lstrip('.').lower()}" for ext in ORACLE_CONFIG["video_extensions"]})
)
IMAGE_EXTENSIONS = tuple(
    sorted({f".{ext.lstrip('.').lower()}" for ext in ORACLE_CONFIG["image_extensions"]})
)
BATCH_FIELD_KEYS = {
    "type",
    "timestamp",
    "location",
    "short_caption",
    "caption",
    "ocr",
    "tags",
}


class OracleLLM:
    def __init__(self, provider: str, model_config: Dict[str, Any]):
        self.provider = provider
        self.config = model_config
        self.openai_client = None
        self.vllm_local = None

        if self.provider == "openai":
            self.openai_client = OpenAI(api_key=self.config.get("api_key"))
        elif self.provider == "vllm_local":
            try:
                from vllm import LLM  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "vllm is not installed. Install vllm to use vllm_local."
                ) from exc
            self.vllm_local = LLM(
                model=self.config.get("model"),
                tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
                gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.9),
                max_model_len=self.config.get("max_model_len"),
                trust_remote_code=True,
            )

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        if self.provider == "openai":
            return self._chat_openai(messages)
        if self.provider == "vllm":
            return self._chat_vllm_http(messages)
        if self.provider == "vllm_local":
            return self._chat_vllm_local(messages)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def chat_with_usage(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[TokenUsage]]:
        if self.provider == "openai":
            return self._chat_openai_with_usage(messages)
        if self.provider == "vllm":
            return self._chat_vllm_http_with_usage(messages)
        if self.provider == "vllm_local":
            return self._chat_vllm_local_with_usage(messages)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _chat_openai(self, messages: List[Dict[str, Any]]) -> str:
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        model = self.config.get("model")
        if not model:
            raise ValueError("Model is required for OpenAI provider")
        max_tokens_value = self.config.get("max_tokens")
        is_newer_model = any(x in model.lower() for x in ["gpt-5", "o1", "o3"])
        if _should_use_responses(model, self.config):
            input_items, instructions = _messages_to_responses_input(messages)
            kwargs: Dict[str, Any] = {
                "model": model,
                "input": input_items,
            }
            if instructions:
                kwargs["instructions"] = instructions
            if max_tokens_value is not None:
                kwargs["max_output_tokens"] = max_tokens_value
            if (
                self.config.get("temperature") is not None
                and _responses_supports_temperature(model)
            ):
                kwargs["temperature"] = self.config.get("temperature")
            response = self.openai_client.responses.create(**kwargs)
            return _extract_response_text(response)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if is_newer_model:
            kwargs["max_completion_tokens"] = (
                max_tokens_value * 3 if max_tokens_value else 3000
            )
        else:
            kwargs["max_tokens"] = max_tokens_value
            kwargs["temperature"] = self.config.get("temperature")

        response = self.openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    def _chat_openai_with_usage(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[TokenUsage]]:
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        model = self.config.get("model")
        if not model:
            raise ValueError("Model is required for OpenAI provider")
        max_tokens_value = self.config.get("max_tokens")
        is_newer_model = any(x in model.lower() for x in ["gpt-5", "o1", "o3"])
        if _should_use_responses(model, self.config):
            input_items, instructions = _messages_to_responses_input(messages)
            kwargs: Dict[str, Any] = {
                "model": model,
                "input": input_items,
            }
            if instructions:
                kwargs["instructions"] = instructions
            if max_tokens_value is not None:
                kwargs["max_output_tokens"] = max_tokens_value
            if (
                self.config.get("temperature") is not None
                and _responses_supports_temperature(model)
            ):
                kwargs["temperature"] = self.config.get("temperature")
            response = self.openai_client.responses.create(**kwargs)
            content = _extract_response_text(response)
            usage = None
            resp_usage = getattr(response, "usage", None)
            if resp_usage:
                usage = TokenUsage(
                    prompt_tokens=getattr(resp_usage, "input_tokens", 0),
                    completion_tokens=getattr(resp_usage, "output_tokens", 0),
                    total_tokens=getattr(resp_usage, "total_tokens", 0),
                )
            return content, usage

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if is_newer_model:
            kwargs["max_completion_tokens"] = (
                max_tokens_value * 3 if max_tokens_value else 3000
            )
        else:
            kwargs["max_tokens"] = max_tokens_value
            kwargs["temperature"] = self.config.get("temperature")

        response = self.openai_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return content, usage

    def _chat_vllm_http(self, messages: List[Dict[str, Any]]) -> str:
        endpoint = self.config.get("endpoint")
        if not endpoint:
            raise ValueError("VLLM endpoint is required for vllm provider")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.get('api_key')}",
        }
        data = {
            "model": self.config.get("model"),
            "messages": messages,
            "max_tokens": self.config.get("max_tokens"),
            "temperature": self.config.get("temperature"),
        }
        max_retries = int(self.config.get("max_retries", 0) or 0)
        request_delay = float(self.config.get("request_delay", 0.0) or 0.0)

        attempt = 0
        while True:
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=data,
                    timeout=self.config.get("timeout"),
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                if request_delay > 0:
                    time.sleep(request_delay * (2 ** (attempt - 1)))

    def _chat_vllm_http_with_usage(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[TokenUsage]]:
        endpoint = self.config.get("endpoint")
        if not endpoint:
            raise ValueError("VLLM endpoint is required for vllm provider")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.get('api_key')}",
        }
        data = {
            "model": self.config.get("model"),
            "messages": messages,
            "max_tokens": self.config.get("max_tokens"),
            "temperature": self.config.get("temperature"),
        }
        max_retries = int(self.config.get("max_retries", 0) or 0)
        request_delay = float(self.config.get("request_delay", 0.0) or 0.0)

        attempt = 0
        while True:
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=data,
                    timeout=self.config.get("timeout"),
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                usage = None
                if "usage" in result and result["usage"]:
                    usage = TokenUsage(
                        prompt_tokens=result["usage"].get("prompt_tokens", 0),
                        completion_tokens=result["usage"].get("completion_tokens", 0),
                        total_tokens=result["usage"].get("total_tokens", 0),
                    )
                return content, usage
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                if request_delay > 0:
                    time.sleep(request_delay * (2 ** (attempt - 1)))

    def _chat_vllm_local(self, messages: List[Dict[str, Any]]) -> str:
        if not self.vllm_local:
            raise RuntimeError("vllm local engine not initialized")
        try:
            from vllm import SamplingParams  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed. Install vllm to use vllm_local."
            ) from exc

        prompt = messages_to_text_prompt(messages)
        sampling_params = SamplingParams(
            temperature=self.config.get("temperature", 0.2),
            max_tokens=self.config.get("max_tokens", 512),
        )
        outputs = self.vllm_local.generate([prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()

    def _chat_vllm_local_with_usage(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[TokenUsage]]:
        if not self.vllm_local:
            raise RuntimeError("vllm local engine not initialized")
        try:
            from vllm import SamplingParams  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed. Install vllm to use vllm_local."
            ) from exc
        prompt = messages_to_text_prompt(messages)
        sampling_params = SamplingParams(
            temperature=self.config.get("temperature", 0.2),
            max_tokens=self.config.get("max_tokens", 512),
        )
        outputs = self.vllm_local.generate([prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            return "", None
        output = outputs[0]
        text = output.outputs[0].text.strip()
        prompt_tokens = (
            len(output.prompt_token_ids) if hasattr(output, "prompt_token_ids") else 0
        )
        completion_tokens = (
            len(output.outputs[0].token_ids)
            if hasattr(output.outputs[0], "token_ids")
            else 0
        )
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return text, usage


def messages_to_text_prompt(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            raise ValueError("vllm_local only supports text-only prompts.")
        role = message.get("role", "user")
        lines.append(f"{role.upper()}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_qa_list(qa_data: Any) -> List[Dict[str, Any]]:
    if isinstance(qa_data, list):
        return qa_data
    if isinstance(qa_data, dict) and "qas" in qa_data:
        qas = qa_data["qas"]
        if isinstance(qas, list):
            return qas
    raise ValueError("Unsupported QA schema. Expected list or dict with 'qas'.")


def apply_niah_pools(
    qas: List[Dict[str, Any]],
    niah_field: str,
    strict: bool = False,
) -> None:
    missing = 0
    for qa in qas:
        pool = qa.get(niah_field)
        if isinstance(pool, list) and pool:
            qa["evidence_ids"] = pool
            continue
        missing += 1
    if missing:
        message = f"{missing} entries missing {niah_field}; using existing evidence_ids"
        if strict:
            raise ValueError(message)
        print(f"Warning: {message}", file=sys.stderr)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_preserve(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def extract_evidence_ids(qa: Dict[str, Any]) -> List[str]:
    evidence_ids = qa.get("evidence_ids", [])
    if not isinstance(evidence_ids, list):
        return []
    return dedupe_preserve([str(item) for item in evidence_ids if item])


def classify_evidence_id(evidence_id: str) -> str:
    lowered = evidence_id.lower()
    if lowered.startswith("email"):
        return "email"
    if lowered.endswith(VIDEO_EXTENSIONS):
        return "video"
    if lowered.endswith(IMAGE_EXTENSIONS):
        return "image"
    return "unknown"


def build_batch_index(
    items: List[Dict[str, Any]], path_key: str
) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in items:
        raw_path = item.get(path_key)
        if not raw_path:
            continue
        base_id = Path(raw_path).stem
        mapping[base_id] = item
    return mapping


def resolve_media_file(
    root: Path, evidence_id: str, extensions: Tuple[str, ...]
) -> Optional[Path]:
    candidate = root / evidence_id
    if candidate.exists():
        return candidate
    base = Path(evidence_id).stem
    for ext in extensions:
        path = root / f"{base}{ext}"
        if path.exists():
            return path
    return None


def list_media_candidates(
    root: Path, evidence_id: str, extensions: Tuple[str, ...]
) -> List[Path]:
    candidates = [root / evidence_id]
    base = Path(evidence_id).stem
    candidates.extend(root / f"{base}{ext}" for ext in extensions)
    return candidates


def resolve_media_evidence(
    evidence_id: str,
    image_root: Path,
    video_root: Path,
) -> Tuple[Optional[str], Optional[Path]]:
    evidence_type = classify_evidence_id(evidence_id)
    video_path = resolve_media_file(video_root, evidence_id, VIDEO_EXTENSIONS)
    image_path = resolve_media_file(image_root, evidence_id, IMAGE_EXTENSIONS)

    if evidence_type == "video" and video_path:
        return "video", video_path
    if evidence_type == "image" and image_path:
        return "image", image_path
    if evidence_type == "unknown":
        if video_path and not image_path:
            return "video", video_path
        if image_path and not video_path:
            return "image", image_path
        if video_path and image_path:
            return "video", video_path
    if video_path:
        return "video", video_path
    if image_path:
        return "image", image_path
    return None, None


def encode_image_to_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def format_batch_evidence(
    item: Dict[str, Any],
    evidence_id: str,
    evidence_type: str,
    batch_fields: Optional[List[str]] = None,
) -> str:
    active_fields = set(batch_fields or ORACLE_CONFIG.get("batch_fields", []))
    timestamp = item.get("timestamp", "")
    location = item.get("location_name", "")
    short_caption = item.get("short_caption", "")
    caption = item.get("caption", "")
    ocr_text = item.get("ocr_text", "")
    tags = item.get("tags", [])
    tags_text = ", ".join(tags) if isinstance(tags, list) else str(tags)

    lines = [f"ID: {evidence_id}"]
    if "type" in active_fields:
        lines.append(f"Type: {evidence_type}")
    if "timestamp" in active_fields:
        lines.append(f"Timestamp: {timestamp}")
    if "location" in active_fields:
        lines.append(f"Location: {location}")
    if "short_caption" in active_fields:
        lines.append(f"Short Caption: {short_caption}")
    if "caption" in active_fields:
        lines.append(f"Caption: {caption}")
    if "ocr" in active_fields:
        lines.append(f"OCR: {ocr_text}")
    if "tags" in active_fields:
        lines.append(f"Tags: {tags_text}")
    return "\n".join(lines) + "\n"


def format_email_evidence(item: Dict[str, Any], evidence_id: str) -> str:
    summary = item.get("short_summary", "")
    detail = item.get("detail", "")
    timestamp = item.get("timestamp", "")
    return (
        f"ID: {evidence_id}\n"
        f"Timestamp: {timestamp}\n"
        f"Summary: {summary}\n"
        f"Detail: {detail}\n"
    )


def build_text_evidence_block(evidence_chunks: List[str]) -> str:
    blocks = []
    for idx, chunk in enumerate(evidence_chunks, start=1):
        blocks.append(f"Evidence {idx}:\n{chunk}")
    return "\n".join(blocks)


def build_text_messages(
    question: str, evidence_chunks: List[str]
) -> List[Dict[str, Any]]:
    evidence_text = build_text_evidence_block(evidence_chunks)
    return [
        {"role": "system", "content": PROMPTS["ORACLE_SYSTEM"]},
        {
            "role": "user",
            "content": PROMPTS["ORACLE_USER_TEXT"].format(
                question=question, evidence=evidence_text
            ),
        },
    ]


def build_multimodal_messages(
    question: str,
    evidence_items: List[Dict[str, Any]],
    text_chunks: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": PROMPTS["ORACLE_USER_MULTIMODAL"].format(question=question),
        },
    ]

    if text_chunks:
        content.append({"type": "text", "text": build_text_evidence_block(text_chunks)})

    for idx, item in enumerate(evidence_items, start=1):
        evidence_id = item["id"]
        evidence_type = item["type"]

        metadata_text = ""
        if "metadata" in item:
            meta = item["metadata"]
            parts = []
            if meta.get("location"):
                parts.append(f"Location: {meta['location']}")
            if meta.get("timestamp"):
                parts.append(f"Timestamp: {meta['timestamp']}")
            if parts:
                metadata_text = f" [{', '.join(parts)}]"

        content.append(
            {
                "type": "text",
                "text": f"Evidence {idx} ({evidence_type} id={evidence_id}{metadata_text})",
            }
        )
        for image_b64 in item["images_b64"]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )

    return [
        {"role": "system", "content": PROMPTS["ORACLE_SYSTEM"]},
        {"role": "user", "content": content},
    ]


def load_email_index(email_file: Path) -> Dict[str, Dict[str, Any]]:
    entries = load_json(email_file)
    if not isinstance(entries, list):
        raise ValueError(f"Email file should be a list: {email_file}")
    return {entry.get("id"): entry for entry in entries if entry.get("id")}


def build_model_config(
    provider: str, has_multimodal: bool, args: argparse.Namespace
) -> Dict[str, Any]:
    if provider == "openai":
        base = dict(ORACLE_CONFIG["openai"])
    elif provider in {"vllm", "vllm_local"}:
        base = dict(
            ORACLE_CONFIG["vllm_vl"] if has_multimodal else ORACLE_CONFIG["vllm_text"]
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if args.model:
        base["model"] = args.model
    if args.api_key:
        base["api_key"] = args.api_key
    if args.vllm_endpoint:
        base["endpoint"] = args.vllm_endpoint
    if args.max_tokens is not None:
        base["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        base["temperature"] = args.temperature
    if args.timeout is not None:
        base["timeout"] = args.timeout
    if args.max_retries is not None:
        base["max_retries"] = args.max_retries
    if args.request_delay is not None:
        base["request_delay"] = args.request_delay
    return base


def collect_text_evidence(
    evidence_ids: List[str],
    email_index: Dict[str, Dict[str, Any]],
    image_batch_index: Dict[str, Dict[str, Any]],
    video_batch_index: Dict[str, Dict[str, Any]],
    batch_fields: Optional[List[str]] = None,
) -> List[str]:
    evidence_chunks: List[str] = []
    for evidence_id in evidence_ids:
        evidence_type = classify_evidence_id(evidence_id)
        if evidence_type == "email":
            email_item = email_index.get(evidence_id)
            if not email_item:
                print(
                    f"Warning: email evidence not found: {evidence_id}", file=sys.stderr
                )
                continue
            evidence_chunks.append(format_email_evidence(email_item, evidence_id))
            continue

        base_id = Path(evidence_id).stem
        item = image_batch_index.get(base_id)
        resolved_type = "image"
        if not item:
            item = video_batch_index.get(base_id)
            resolved_type = "video"
        if not item:
            print(f"Warning: batch evidence not found: {evidence_id}", file=sys.stderr)
            continue
        evidence_chunks.append(
            format_batch_evidence(item, base_id, resolved_type, batch_fields)
        )

    return evidence_chunks


def collect_multimodal_evidence(
    evidence_ids: List[str],
    email_index: Dict[str, Dict[str, Any]],
    image_root: Path,
    video_root: Path,
    num_frames: int,
    max_total_frames: Optional[int],
    frame_strategy: str,
    image_batch_index: Optional[Dict[str, Dict[str, Any]]] = None,
    video_batch_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    multimodal_items: List[Dict[str, Any]] = []
    text_chunks: List[str] = []
    total_frames_used = 0
    video_candidates = 0
    per_video_frames: Optional[int] = None

    if max_total_frames is not None:
        for evidence_id in evidence_ids:
            evidence_type = classify_evidence_id(evidence_id)
            if evidence_type == "email":
                continue
            resolved_type, media_path = resolve_media_evidence(
                evidence_id, image_root, video_root
            )
            if resolved_type == "video" and media_path:
                video_candidates += 1
        if video_candidates > 0:
            per_video_frames = max_total_frames // video_candidates
            if per_video_frames <= 0:
                per_video_frames = 1

    for evidence_id in evidence_ids:
        evidence_type = classify_evidence_id(evidence_id)
        if evidence_type == "email":
            email_item = email_index.get(evidence_id)
            if not email_item:
                print(
                    f"Warning: email evidence not found: {evidence_id}", file=sys.stderr
                )
                continue
            text_chunks.append(format_email_evidence(email_item, evidence_id))
            continue

        resolved_type, media_path = resolve_media_evidence(
            evidence_id, image_root, video_root
        )
        if not media_path or not resolved_type:
            image_candidates = list_media_candidates(
                image_root, evidence_id, IMAGE_EXTENSIONS
            )
            video_candidates = list_media_candidates(
                video_root, evidence_id, VIDEO_EXTENSIONS
            )
            candidates_text = ", ".join(
                str(path) for path in (image_candidates[:2] + video_candidates[:2])
            )
            print(
                "Warning: media evidence not found: "
                f"{evidence_id} (image root {image_root}, video root {video_root}, "
                f"candidates: {candidates_text})",
                file=sys.stderr,
            )
            continue

        if resolved_type == "image":
            image_b64 = encode_image_to_base64(media_path)
            base_id = Path(evidence_id).stem

            metadata = {}
            if image_batch_index and base_id in image_batch_index:
                batch_item = image_batch_index[base_id]
                metadata["location"] = batch_item.get("location_name", "")
                metadata["timestamp"] = batch_item.get("timestamp", "")

            multimodal_items.append(
                {
                    "type": "image",
                    "id": base_id,
                    "images_b64": [image_b64],
                    "metadata": metadata,
                }
            )
            continue

        if resolved_type == "video":
            frame_budget = num_frames
            if max_total_frames is not None:
                if per_video_frames is not None:
                    frame_budget = min(frame_budget, per_video_frames)
                remaining = max_total_frames - total_frames_used
                if remaining <= 0:
                    print(
                        f"Warning: video frame budget exhausted; skipping {evidence_id}",
                        file=sys.stderr,
                    )
                    continue
                frame_budget = min(frame_budget, remaining)

            temp_dir = Path(tempfile.mkdtemp())
            frame_paths = extract_frames(
                media_path,
                num_frames=frame_budget,
                output_dir=temp_dir,
                strategy=frame_strategy,
            )
            if not frame_paths:
                print(
                    f"Warning: no frames extracted for {evidence_id}", file=sys.stderr
                )
                continue
            total_frames_used += len(frame_paths)
            images_b64 = [encode_image_to_base64(path) for path in frame_paths]
            base_id = Path(evidence_id).stem

            metadata = {}
            if video_batch_index and base_id in video_batch_index:
                batch_item = video_batch_index[base_id]
                metadata["location"] = batch_item.get("location_name", "")
                metadata["timestamp"] = batch_item.get("timestamp", "")

            multimodal_items.append(
                {
                    "type": "video",
                    "id": base_id,
                    "images_b64": images_b64,
                    "metadata": metadata,
                }
            )
            continue

    return multimodal_items, text_chunks


def gather_evidence_types(evidence_ids: List[str]) -> Dict[str, bool]:
    has_email = False
    has_video = False
    has_image = False
    has_unknown = False
    for evidence_id in evidence_ids:
        evidence_type = classify_evidence_id(evidence_id)
        if evidence_type == "email":
            has_email = True
        elif evidence_type == "video":
            has_video = True
        elif evidence_type == "image":
            has_image = True
        else:
            has_unknown = True
    return {
        "email": has_email,
        "video": has_video,
        "image": has_image,
        "unknown": has_unknown,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle QA baseline")
    parser.add_argument("--qa-file", required=True, help="Path to QA annotations JSON")
    parser.add_argument(
        "--use-niah-pools",
        action="store_true",
        help="Use niah_evidence_ids (or --niah-field) as evidence_ids",
    )
    parser.add_argument(
        "--niah-field",
        default="niah_evidence_ids",
        help="Field name for NIAH evidence pools (default: niah_evidence_ids)",
    )
    parser.add_argument(
        "--niah-strict",
        action="store_true",
        help="Error if any entry is missing the NIAH evidence field",
    )
    parser.add_argument(
        "--media-source",
        choices=["batch_results", "raw"],
        default=ORACLE_CONFIG["media_source"],
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        default=ORACLE_CONFIG["no_evidence"],
        help="Do not provide evidence to the model",
    )
    parser.add_argument(
        "--image-batch-results",
        default=ORACLE_CONFIG["image_batch_results"],
        help="Path to image batch_results.json",
    )
    parser.add_argument(
        "--video-batch-results",
        default=ORACLE_CONFIG["video_batch_results"],
        help="Path to video batch_results.json",
    )
    parser.add_argument(
        "--image-root",
        default=ORACLE_CONFIG["image_root"],
        help="Root directory for raw images",
    )
    parser.add_argument(
        "--video-root",
        default=ORACLE_CONFIG["video_root"],
        help="Root directory for raw videos",
    )
    parser.add_argument(
        "--email-file",
        default=ORACLE_CONFIG["email_file"],
        help="Path to merged_emails.json",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "vllm", "vllm_local"],
        default=ORACLE_CONFIG["provider"],
    )
    parser.add_argument("--model", default=None, help="Model name (overrides config)")
    parser.add_argument("--api-key", default=None, help="API key (overrides config)")
    parser.add_argument(
        "--vllm-endpoint", default=None, help="VLLM endpoint URL (overrides config)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Max tokens (overrides config)"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Temperature (overrides config)"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout seconds (overrides config)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=ORACLE_CONFIG["max_retries"],
        help="Max retries for vLLM HTTP requests",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=ORACLE_CONFIG["request_delay"],
        help="Base delay in seconds for retry backoff",
    )
    parser.add_argument(
        "--max-evidence-items", type=int, default=ORACLE_CONFIG["max_evidence_items"]
    )
    parser.add_argument(
        "--batch-fields",
        default=None,
        help=(
            "Comma/space-separated fields for batch_results image/video evidence. "
            "Use 'all' or 'none'. Allowed: type,timestamp,location,short_caption,caption,ocr,tags."
        ),
    )
    parser.add_argument("--num-frames", type=int, default=ORACLE_CONFIG["num_frames"])
    parser.add_argument(
        "--max-total-frames",
        type=int,
        default=32,
        help="Cap total frames extracted across all videos (default: num-frames)",
    )
    parser.add_argument("--frame-strategy", default=ORACLE_CONFIG["frame_strategy"])
    parser.add_argument("--max-workers", type=int, default=ORACLE_CONFIG["max_workers"])
    parser.add_argument("--output-file", default=ORACLE_CONFIG["output_file"])
    args = parser.parse_args()
    if args.batch_fields is not None:
        args.batch_fields = parse_batch_fields(args.batch_fields)
    return args


def normalize_batch_field(field: str) -> str:
    normalized = field.strip().lower().replace("-", "_")
    if normalized == "shortcaption":
        return "short_caption"
    if normalized == "ocr_text":
        return "ocr"
    return normalized


def parse_batch_fields(value: str) -> List[str]:
    cleaned = value.strip().lower()
    if cleaned in {"all", "default"}:
        return list(ORACLE_CONFIG.get("batch_fields", []))
    if cleaned in {"none", "id_only", "id-only"}:
        return []
    tokens = [tok for tok in cleaned.replace(",", " ").split() if tok]
    fields = [normalize_batch_field(tok) for tok in tokens]
    unknown = sorted({field for field in fields if field not in BATCH_FIELD_KEYS})
    if unknown:
        allowed = ", ".join(sorted(BATCH_FIELD_KEYS))
        raise ValueError(f"Unknown batch field(s): {', '.join(unknown)}. Allowed: {allowed}")
    return fields


def process_single_qa(
    qa: Dict[str, Any],
    args: argparse.Namespace,
    llm_getter: Callable[[], OracleLLM],
    email_index: Dict[str, Dict[str, Any]],
    image_batch_index: Dict[str, Dict[str, Any]],
    video_batch_index: Dict[str, Dict[str, Any]],
    image_root: Path,
    video_root: Path,
) -> Optional[Dict[str, Any]]:
    qa_id = qa.get("id") or qa.get("qa_id")
    question = qa.get("question")
    if not qa_id or not question:
        return None

    evidence_ids = [] if args.no_evidence else extract_evidence_ids(qa)
    if args.max_evidence_items and evidence_ids:
        evidence_ids = evidence_ids[: args.max_evidence_items]

    if args.media_source == "batch_results":
        evidence_chunks = collect_text_evidence(
            evidence_ids,
            email_index,
            image_batch_index,
            video_batch_index,
            args.batch_fields,
        )
        messages = build_text_messages(question, evidence_chunks)
    else:
        max_total_frames = (
            args.max_total_frames
            if args.max_total_frames is not None
            else args.num_frames
        )
        multimodal_items, text_chunks = collect_multimodal_evidence(
            evidence_ids,
            email_index,
            image_root,
            video_root,
            args.num_frames,
            max_total_frames,
            args.frame_strategy,
            image_batch_index,
            video_batch_index,
        )
        if multimodal_items:
            messages = build_multimodal_messages(
                question, multimodal_items, text_chunks
            )
        else:
            messages = build_text_messages(question, text_chunks)

    llm = llm_getter()
    try:
        answer, usage = llm.chat_with_usage(messages)
    except Exception as exc:
        print(f"Error: QA {qa_id} failed: {exc}", file=sys.stderr)
        raise
    result: Dict[str, Any] = {"id": qa_id, "answer": answer}
    if usage:
        result["prompt_tokens"] = usage.prompt_tokens
        result["completion_tokens"] = usage.completion_tokens
        result["total_tokens"] = usage.total_tokens
    return result


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_file)
    if output_path.exists():
        print(f"Output exists, skipping inference: {output_path}")
        return 0

    qa_file = Path(args.qa_file)
    qa_data = load_json(qa_file)
    qas = load_qa_list(qa_data)
    if args.use_niah_pools:
        apply_niah_pools(qas, args.niah_field, strict=args.niah_strict)

    all_evidence_ids = [
        evidence_id
        for qa in qas
        for evidence_id in extract_evidence_ids(qa)
        if evidence_id
    ]
    evidence_types = gather_evidence_types(all_evidence_ids)
    if args.no_evidence:
        evidence_types = {
            "email": False,
            "video": False,
            "image": False,
            "unknown": False,
        }

    if args.provider == "vllm_local" and args.max_workers > 1:
        raise ValueError("vllm_local does not support concurrent execution.")

    has_multimodal = args.media_source == "raw" and (
        evidence_types["image"] or evidence_types["video"] or evidence_types["unknown"]
    )

    if args.provider == "vllm_local" and has_multimodal:
        raise ValueError("vllm_local does not support raw image/video inputs.")

    llm_config = build_model_config(args.provider, has_multimodal, args)
    thread_local = threading.local()

    def get_llm() -> OracleLLM:
        if not hasattr(thread_local, "llm"):
            thread_local.llm = OracleLLM(args.provider, llm_config)
        return thread_local.llm

    email_index: Dict[str, Dict[str, Any]] = {}
    if evidence_types["email"]:
        email_index = load_email_index(Path(args.email_file))

    image_batch_index: Dict[str, Dict[str, Any]] = {}
    video_batch_index: Dict[str, Dict[str, Any]] = {}

    if evidence_types["image"] or evidence_types["unknown"]:
        image_batch_data = load_json(Path(args.image_batch_results))
        if not isinstance(image_batch_data, list):
            raise ValueError("image batch_results must be a list")
        image_batch_index = build_batch_index(image_batch_data, "image_path")
    if evidence_types["video"] or evidence_types["unknown"]:
        video_batch_data = load_json(Path(args.video_batch_results))
        if not isinstance(video_batch_data, list):
            raise ValueError("video batch_results must be a list")
        video_batch_index = build_batch_index(video_batch_data, "video_path")

    results: List[Dict[str, Any]] = []

    if args.max_workers and args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {
                executor.submit(
                    process_single_qa,
                    qa,
                    args,
                    get_llm,
                    email_index,
                    image_batch_index,
                    video_batch_index,
                    Path(args.image_root),
                    Path(args.video_root),
                ): idx
                for idx, qa in enumerate(qas)
            }
            ordered_results: List[Tuple[int, Dict[str, Any]]] = []
            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="Oracle QA",
            ):
                idx = future_map[future]
                result = future.result()
                if result:
                    ordered_results.append((idx, result))
            ordered_results.sort(key=lambda item: item[0])
            results.extend([item for _, item in ordered_results])
    else:
        for qa in tqdm(qas, desc="Oracle QA"):
            result = process_single_qa(
                qa,
                args,
                get_llm,
                email_index,
                image_batch_index,
                video_batch_index,
                Path(args.image_root),
                Path(args.video_root),
            )
            if result:
                results.append(result)

    write_jsonl(Path(args.output_file), results)

    total_prompt = sum(r.get("prompt_tokens", 0) for r in results)
    total_completion = sum(r.get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    num_samples = len(results)

    run_stats = {
        "num_samples": num_samples,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "avg_prompt_tokens": round(total_prompt / num_samples, 1) if num_samples else 0,
        "avg_completion_tokens": round(total_completion / num_samples, 1)
        if num_samples
        else 0,
        "avg_total_tokens": round(total_tokens / num_samples, 1) if num_samples else 0,
    }

    stats_path = output_path.parent / f"{output_path.stem}_run_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(run_stats, f, indent=2)
    print(f"Token stats written to: {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
