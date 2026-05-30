#!/usr/bin/env python3
"""
Extract a {"id","question","answer"} JSON blob from agent trace outputs.

This is used by runners where the CLI emits a JSON (OpenClaw) or JSONL event stream
(opencode), but we want a uniform `output/answer.json` for evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_with_recovery(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = _find_first_json_object(text)
        if candidate:
            return json.loads(candidate)
        raise


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_json_blob(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty text")

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    candidate = _find_first_json_object(text)
    if candidate:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("could not parse JSON object from text")


def _find_first_json_object(text: str) -> Optional[str]:
    in_string = False
    escape = False
    depth = 0
    start_idx: Optional[int] = None

    for idx, ch in enumerate(text):
        if start_idx is None:
            if ch == "{":
                start_idx = idx
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]

    return None


def normalize_answer(
    obj: Dict[str, Any],
    *,
    expected_id: str,
    question_file: Optional[Path],
) -> Dict[str, str]:
    question_text = ""
    if question_file is not None and question_file.exists():
        q = load_json(question_file)
        question_text = str(q.get("question", ""))

    out = {
        "id": str(expected_id),
        "question": str(question_text or obj.get("question") or ""),
        "answer": str(obj.get("answer") or ""),
    }

    missing = [k for k in ("id", "question", "answer") if not out.get(k)]
    if missing:
        raise ValueError(f"missing required fields after normalization: {missing}")
    return out


def extract_openclaw(trace: Path) -> str:
    data = load_json_with_recovery(trace)

    def payload_text(payload: Any) -> Optional[str]:
        if isinstance(payload, str) and payload.strip():
            return payload
        if not isinstance(payload, dict):
            return None
        for key in ("text", "content", "answer", "value"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        nested = payload.get("data")
        if isinstance(nested, dict):
            for key in ("text", "content", "answer", "value"):
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return None

    result = data.get("result", {}) if isinstance(data, dict) else {}
    payloads = []
    if isinstance(data, dict):
        for key in ("payloads", "output"):
            value = data.get(key)
            if isinstance(value, list):
                payloads.extend(value)
    if isinstance(result, dict):
        for key in ("payloads", "output"):
            value = result.get(key)
            if isinstance(value, list):
                payloads.extend(value)

    texts = []
    if isinstance(payloads, list):
        for payload in payloads:
            text = payload_text(payload)
            if text:
                texts.append(text)
    if not texts:
        raise ValueError("no textual payload found in OpenClaw trace")
    return texts[-1]


def extract_claude_code(trace: Path) -> str:
    """
    Claude Code may emit either a single JSON object or a stream-json JSONL trace.
    Prefer the final structured_output payload when present; otherwise fall back
    to the final result text, then assistant text.
    """
    def maybe_structured_blob(value: Any) -> Optional[str]:
        if not isinstance(value, dict):
            return None
        if "answer" not in value:
            return None
        return json.dumps(value, ensure_ascii=False)

    def maybe_claude_result_text(event: Any) -> Optional[str]:
        if not isinstance(event, dict):
            return None
        structured = maybe_structured_blob(
            event.get("structured_output") or event.get("structuredOutput")
        )
        if structured:
            return structured
        result = _content_to_text(event.get("result"))
        if result:
            return result
        message = event.get("message")
        if isinstance(message, dict):
            structured = maybe_structured_blob(
                message.get("structured_output") or message.get("structuredOutput")
            )
            if structured:
                return structured
            result = _content_to_text(message.get("result"))
            if result:
                return result
        return None

    def maybe_structured_tool_input(event: Any) -> Optional[str]:
        if not isinstance(event, dict):
            return None
        message = event.get("message")
        if not isinstance(message, dict):
            return None
        for item in message.get("content") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tool_use" or item.get("name") != "StructuredOutput":
                continue
            structured = maybe_structured_blob(item.get("input"))
            if structured:
                return structured
        return None

    try:
        data = load_json(trace)
    except json.JSONDecodeError:
        events = list(load_jsonl(trace))
        for event in reversed(events):
            text = maybe_structured_tool_input(event)
            if text:
                return text
        for event in reversed(events):
            if event.get("type") == "result" and event.get("subtype") == "success":
                text = maybe_claude_result_text(event)
                if text:
                    return text

        texts = []
        for event in events:
            text = _find_assistant_text(event)
            if text:
                texts.append(text)
        if texts:
            return texts[-1]
        raise ValueError("could not locate assistant text in Claude Code JSONL output")

    # Sometimes the CLI may print the structured output object directly.
    if isinstance(data, dict) and {"id", "question", "answer"} <= set(data.keys()):
        return json.dumps(data, ensure_ascii=False)

    text = maybe_claude_result_text(data)
    if text:
        return text

    def content_to_text(content: Any) -> Optional[str]:
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    t = _maybe_text(block.get("text"))
                    if t:
                        parts.append(t)
                elif isinstance(block, str) and block.strip():
                    parts.append(block)
            if parts:
                return "\n".join(parts).strip()
        return None

    # Anthropic-style message: {role, content:[{type:"text", text:"..."}], usage:{...}}
    if isinstance(data, dict):
        if data.get("role") == "assistant":
            text = content_to_text(data.get("content"))
            if text:
                return text
        for key in ("message", "result", "data", "output"):
            nested = data.get(key)
            if isinstance(nested, dict):
                if nested.get("role") == "assistant":
                    text = content_to_text(nested.get("content"))
                    if text:
                        return text
                text = content_to_text(nested.get("content") or nested.get("text"))
                if text:
                    return text
                # Some wrappers embed the assistant message under nested.message
                msg = nested.get("message")
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    text = content_to_text(msg.get("content"))
                    if text:
                        return text

        # Generic fallbacks.
        text = content_to_text(data.get("content") or data.get("text"))
        if text:
            return text

    raise ValueError("could not locate assistant text in Claude Code JSON output")


def _maybe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    return None


def _content_to_text(value: Any) -> Optional[str]:
    text = _maybe_text(value)
    if text:
        return text
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, dict):
                block_text = _maybe_text(block.get("text"))
                if block_text:
                    parts.append(block_text)
            elif isinstance(block, str) and block.strip():
                parts.append(block)
        if parts:
            return "\n".join(parts).strip()
    return None


def _find_assistant_text(event: Any) -> Optional[str]:
    if not isinstance(event, dict):
        return None

    # opencode trace format: {"type":"text", "part":{"type":"text", "text":"..."}}
    if event.get("type") == "text":
        part = event.get("part")
        if isinstance(part, dict):
            text = _maybe_text(part.get("text"))
            if text:
                return text

    # Common patterns.
    role = event.get("role")
    if role == "assistant":
        for key in ("content", "text", "message"):
            text = _content_to_text(event.get(key))
            if text:
                return text

    message = event.get("message")
    if isinstance(message, dict) and message.get("role") == "assistant":
        for key in ("content", "text", "message"):
            text = _content_to_text(message.get(key))
            if text:
                return text

    data = event.get("data")
    if isinstance(data, dict):
        if data.get("role") == "assistant":
            for key in ("content", "text", "message"):
                text = _content_to_text(data.get(key))
                if text:
                    return text
        message = data.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            for key in ("content", "text", "message"):
                text = _content_to_text(message.get(key))
                if text:
                    return text

    # Look for a "final"/"output" field.
    for key in ("final", "output", "result"):
        candidate = event.get(key)
        if isinstance(candidate, dict):
            if candidate.get("role") == "assistant":
                for k2 in ("content", "text", "message"):
                    text = _content_to_text(candidate.get(k2))
                    if text:
                        return text
        text = _content_to_text(candidate)
        if text:
            return text

    return None


def _find_opencode_write_answer(event: Any) -> Optional[str]:
    """
    opencode traces may include a tool call that writes `output/answer.json`.
    Prefer the written JSON content if present, since some agents write the file
    but do not print the JSON as assistant text.
    """
    if not isinstance(event, dict):
        return None
    if event.get("type") != "tool_use":
        return None
    part = event.get("part")
    if not isinstance(part, dict):
        return None
    if part.get("tool") != "write":
        return None
    state = part.get("state")
    if not isinstance(state, dict):
        return None
    inp = state.get("input")
    if not isinstance(inp, dict):
        return None
    file_path = inp.get("filePath")
    content = inp.get("content")
    if not isinstance(file_path, str) or not isinstance(content, str):
        return None
    if not file_path.replace("\\", "/").endswith("/output/answer.json"):
        return None
    if not content.strip():
        return None
    return content


def extract_opencode(trace: Path) -> str:
    last_text = None
    last_written = None
    for event in load_jsonl(trace):
        written = _find_opencode_write_answer(event)
        if written:
            last_written = written
        text = _find_assistant_text(event)
        if text:
            last_text = text
    if last_written is not None:
        return last_written
    if last_text is None:
        raise ValueError("no assistant message found in opencode JSONL trace")
    return last_text


def _pi_message_text(message: Any) -> Optional[str]:
    if not isinstance(message, dict):
        return None
    if message.get("role") != "assistant":
        return None
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = _maybe_text(block.get("text"))
            if text:
                parts.append(text)
    if not parts:
        return None
    return "\n".join(parts).strip()


def extract_pi(trace: Path) -> str:
    """
    Extract the final assistant text from a `pi -p --mode json` JSONL trace.

    pi emits events: session, agent_start, turn_start, message_start, message_update,
    message_end, turn_end, agent_end. The most authoritative final-answer source is the
    last assistant `message_end` event (whose `message.content` carries the completed
    text blocks). `agent_end.messages[-1]` is used as a fallback.
    """
    last_text: Optional[str] = None
    final_text: Optional[str] = None

    for event in load_jsonl(trace):
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "message_end":
            text = _pi_message_text(event.get("message"))
            if text:
                last_text = text
        elif etype == "agent_end":
            messages = event.get("messages")
            if isinstance(messages, list):
                for msg in reversed(messages):
                    text = _pi_message_text(msg)
                    if text:
                        final_text = text
                        break

    if final_text:
        return final_text
    if last_text:
        return last_text
    raise ValueError("no assistant message found in pi JSONL trace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract normalized answer.json from traces")
    parser.add_argument(
        "--format",
        required=True,
        choices=["openclaw", "opencode", "claude_code", "pi"],
        help="Trace format to parse",
    )
    parser.add_argument("--trace", required=True, help="Path to trace file (json or jsonl)")
    parser.add_argument("--out", required=True, help="Path to write answer.json")
    parser.add_argument("--expected-id", required=True, help="Expected question id")
    parser.add_argument(
        "--question-file",
        default=None,
        help="Path to question.json (used to fill missing question text)",
    )
    args = parser.parse_args()

    trace = Path(args.trace)
    out = Path(args.out)
    qfile = Path(args.question_file) if args.question_file else None

    try:
        if args.format == "openclaw":
            raw_text = extract_openclaw(trace)
        elif args.format == "claude_code":
            raw_text = extract_claude_code(trace)
        elif args.format == "pi":
            raw_text = extract_pi(trace)
        else:
            raw_text = extract_opencode(trace)
    except (ValueError, json.JSONDecodeError) as exc:
        print(
            f"ERROR: failed to extract answer from {str(args.trace)!r}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        blob = extract_json_blob(raw_text)
    except ValueError:
        # Fallback: model returned natural language instead of JSON.
        # Treat the entire raw text as the answer; normalize_answer will
        # fill id from --expected-id and question from --question-file.
        print(
            f"WARNING: JSON extraction failed for {str(args.trace)!r}; "
            "treating raw text as answer.",
            file=sys.stderr,
        )
        blob = {"answer": raw_text}
    normalized = normalize_answer(blob, expected_id=args.expected_id, question_file=qfile)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
