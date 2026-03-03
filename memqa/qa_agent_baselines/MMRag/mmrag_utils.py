#!/usr/bin/env python3
"""Shared helpers for MMRag baselines."""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from memqa.mem_processor.video.utils import extract_frames
from memqa.qa_agent_baselines.MMRag.config import PROMPTS
from memqa.qa_agent_baselines.MMRag.llm_utils import TokenUsage
from memqa.retrieve import RetrievalItem


def encode_image_to_base64(path: Path) -> str:
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def build_text_evidence(items: List[RetrievalItem]) -> List[str]:
    return [item.text for item in items if item.text]


def build_text_messages(
    question: str, evidence_chunks: List[str]
) -> List[Dict[str, Any]]:
    evidence_text = "\n".join(
        [f"Evidence {idx + 1}:\n{chunk}" for idx, chunk in enumerate(evidence_chunks)]
    )
    return [
        {"role": "system", "content": PROMPTS["SYSTEM"]},
        {
            "role": "user",
            "content": PROMPTS["USER_TEXT"].format(
                question=question, evidence=evidence_text
            ),
        },
    ]


def build_multimodal_messages(
    question: str,
    items: List[RetrievalItem],
    text_chunks: Optional[List[str]],
    num_frames: int,
    max_total_frames: Optional[int],
    frame_strategy: str,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": PROMPTS["USER_MULTIMODAL"].format(question=question)}
    ]
    total_frames_used = 0
    video_candidates = 0
    per_video_frames: Optional[int] = None

    if text_chunks:
        evidence_text = "\n".join(
            [f"Evidence {idx + 1}:\n{chunk}" for idx, chunk in enumerate(text_chunks)]
        )
        content.append({"type": "text", "text": evidence_text})

    if max_total_frames is not None:
        for item in items:
            if item.modality == "video" and item.video_path:
                video_candidates += 1
        if video_candidates > 0:
            per_video_frames = max_total_frames // video_candidates
            if per_video_frames <= 0:
                per_video_frames = 1

    for idx, item in enumerate(items, start=1):
        content.append(
            {
                "type": "text",
                "text": f"Evidence {idx} ({item.modality} id={item.item_id})",
            }
        )
        if item.modality == "image" and item.image_path:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_to_base64(item.image_path)}"
                    },
                }
            )
        if item.modality == "video" and item.video_path:
            frame_budget = num_frames
            if max_total_frames is not None:
                if per_video_frames is not None:
                    frame_budget = min(frame_budget, per_video_frames)
                remaining = max_total_frames - total_frames_used
                if remaining <= 0:
                    print(
                        f"Warning: video frame budget exhausted; skipping {item.item_id}",
                        file=sys.stderr,
                    )
                    continue
                frame_budget = min(frame_budget, remaining)

            temp_dir = Path(tempfile.mkdtemp())
            frame_paths = extract_frames(
                item.video_path,
                num_frames=frame_budget,
                output_dir=temp_dir,
                strategy=frame_strategy,
            )
            total_frames_used += len(frame_paths)
            for frame_path in frame_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_to_base64(frame_path)}"
                        },
                    }
                )

    return [
        {"role": "system", "content": PROMPTS["SYSTEM"]},
        {"role": "user", "content": content},
    ]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_resume_map(
    output_file: Path, checkpoint_file: Path
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(output_file):
        qa_id = str(row.get("id")) if row.get("id") is not None else ""
        if qa_id:
            results[qa_id] = row
    for row in load_jsonl(checkpoint_file):
        qa_id = str(row.get("id")) if row.get("id") is not None else ""
        if qa_id:
            results[qa_id] = row
    return results


def is_failed_record(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return True
    status = str(record.get("status", "")).lower()
    if status in {"failed", "fail", "error"}:
        return True
    if record.get("error"):
        return True
    answer = record.get("answer")
    if answer is None:
        return True
    if isinstance(answer, str) and not answer.strip():
        return True
    return False


def merge_token_usage(
    base: Optional[TokenUsage], extra: Optional[TokenUsage]
) -> Optional[TokenUsage]:
    if extra is None:
        return base
    if base is None:
        return TokenUsage(
            prompt_tokens=extra.prompt_tokens,
            completion_tokens=extra.completion_tokens,
            total_tokens=extra.total_tokens,
        )
    return TokenUsage(
        prompt_tokens=base.prompt_tokens + extra.prompt_tokens,
        completion_tokens=base.completion_tokens + extra.completion_tokens,
        total_tokens=base.total_tokens + extra.total_tokens,
    )


def build_agentic_critic_messages(
    question: str, items: List[RetrievalItem], draft: str
) -> List[Dict[str, Any]]:
    evidence_chunks = build_text_evidence(items)
    evidence_text = "\n".join(
        [f"Evidence {idx + 1}:\n{chunk}" for idx, chunk in enumerate(evidence_chunks)]
    )
    return [
        {"role": "system", "content": PROMPTS["AGENTIC_CRITIC_SYSTEM"]},
        {
            "role": "user",
            "content": PROMPTS["AGENTIC_CRITIC_USER"].format(
                question=question, evidence=evidence_text, draft=draft
            ),
        },
    ]


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _normalize_support(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if "fully" in text:
        return "fully"
    if "partial" in text:
        return "partially"
    if "none" in text or "no support" in text or "contradict" in text:
        return "none"
    if text in {"full", "complete"}:
        return "fully"
    if text in {"partial", "part"}:
        return "partially"
    if text in {"no", "none"}:
        return "none"
    return None


def parse_agentic_critic_response(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    payload: Dict[str, Any] = {}
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        blob = _extract_json_object(cleaned)
        if blob:
            try:
                payload = json.loads(blob)
            except json.JSONDecodeError:
                payload = {}

    final_answer = payload.get("final_answer") or payload.get("answer")
    if isinstance(final_answer, list):
        final_answer = ", ".join(str(item) for item in final_answer)
    if final_answer is not None:
        final_answer = str(final_answer).strip()

    support = _normalize_support(payload.get("support") or payload.get("grounding"))

    utility = payload.get("utility")
    utility_score: Optional[int] = None
    if utility is not None:
        try:
            utility_score = int(float(utility))
        except (TypeError, ValueError):
            utility_score = None
    if utility_score is not None:
        utility_score = max(1, min(5, utility_score))

    return {
        "final_answer": final_answer,
        "support": support,
        "utility": utility_score,
        "raw": cleaned if cleaned else text,
    }
