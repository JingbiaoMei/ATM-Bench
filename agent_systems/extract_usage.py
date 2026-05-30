#!/usr/bin/env python3
"""
Extract token usage / cost from agent trace outputs into a normalized usage.json.

Supported trace formats:
  - codex     JSONL from `codex exec --json`
  - opencode  JSONL from `opencode run --format json`
  - openclaw  JSON from `openclaw agent --json`
  - claude    JSON or JSONL from `claude -p --output-format json|stream-json`

Output schema (best-effort; fields may be null if not found):
{
  "agent": "codex|opencode|openclaw|claude_code",
  "model_tag": "<label used in run dir>",
  "model": "<model id/name if found>",
  "input_tokens": 123,
  "input_tokens_uncached": 45,
  "cache_creation_input_tokens": 56,
  "cache_read_input_tokens": 22,
  "output_tokens": 456,
  "total_tokens": 579,
  "cost_usd": 0.0123,
  "records_found": 1
}
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


TokenCounts = Tuple[Optional[int], Optional[int], Optional[int]]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def load_json_with_recovery(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = _find_first_json_object(text)
        if candidate:
            return json.loads(candidate)
        raise


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return int(value)
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _canonical_tokens(d: Dict[str, Any]) -> TokenCounts:
    # Common providers.
    in_tok = _as_int(d.get("input_tokens"))
    out_tok = _as_int(d.get("output_tokens"))
    tot_tok = _as_int(d.get("total_tokens"))

    # OpenAI-style.
    if in_tok is None and out_tok is None:
        in_tok = _as_int(d.get("prompt_tokens"))
        out_tok = _as_int(d.get("completion_tokens"))
        if tot_tok is None:
            tot_tok = _as_int(d.get("total_tokens"))

    # camelCase fallbacks.
    if in_tok is None and out_tok is None:
        in_tok = _as_int(d.get("inputTokens") or d.get("promptTokens"))
        out_tok = _as_int(d.get("outputTokens") or d.get("completionTokens"))
        if tot_tok is None:
            tot_tok = _as_int(d.get("totalTokens"))

    # OpenCode step token format (observed in opencode JSONL traces):
    # {"tokens": {"total": ..., "input": ..., "output": ..., "cache": {"read": ...}, ...}}
    if in_tok is None and out_tok is None and tot_tok is None:
        in_tok = _as_int(d.get("input"))
        out_tok = _as_int(d.get("output"))
        tot_tok = _as_int(d.get("total"))
        cache = d.get("cache")
        if isinstance(cache, dict):
            cache_read = _as_int(cache.get("read"))
            if cache_read is not None and in_tok is not None:
                # OpenCode's `input` field contains only NON-cached prompt tokens;
                # `cache.read` holds the cached portion.  Verified against real
                # traces: total == input + cache.read + output.  Adding cache.read
                # here makes in_tok reflect total prompt tokens (cached + uncached)
                # so that in_tok + out_tok == tot_tok.
                in_tok += cache_read

    if tot_tok is None:
        if in_tok is not None and out_tok is not None:
            tot_tok = in_tok + out_tok
    return in_tok, out_tok, tot_tok


def _is_usage_like(d: Dict[str, Any]) -> bool:
    keys = set(d.keys())
    tokenish = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "promptTokens",
        "completionTokens",
    }
    if keys & tokenish:
        return True

    # OpenCode step token format: {"total": ..., "input": ..., "output": ...}
    if {"total", "input", "output"}.issubset(keys):
        if any(_as_int(d.get(k)) is not None for k in ("total", "input", "output")):
            return True

    return False


def _extract_cost_usd(d: Dict[str, Any]) -> Optional[float]:
    for key in ("cost_usd", "total_cost_usd", "cost", "total_cost", "usd", "price_usd"):
        if key in d:
            val = _as_float(d.get(key))
            if val is not None:
                return val
    return None


def _openclaw_model(obj: Any) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    meta = obj.get("meta")
    if not isinstance(meta, dict):
        return _maybe_model(obj)
    agent_meta = meta.get("agentMeta")
    if not isinstance(agent_meta, dict):
        return _maybe_model(obj)
    provider = agent_meta.get("provider")
    model = agent_meta.get("model")
    if isinstance(provider, str) and provider.strip() and isinstance(model, str) and model.strip():
        return f"{provider.strip()}/{model.strip()}"
    if isinstance(model, str) and model.strip():
        return model.strip()
    return _maybe_model(obj)


def _pick_id(event: Dict[str, Any]) -> Optional[str]:
    for key in ("response_id", "request_id", "completion_id", "message_id", "messageID", "id"):
        v = event.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for parent_key in ("response", "request", "message", "part", "data", "result", "output"):
        v = event.get(parent_key)
        if isinstance(v, dict):
            for key in ("response_id", "request_id", "completion_id", "message_id", "messageID", "id"):
                vv = v.get(key)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
    return None


@dataclass(frozen=True)
class UsageRecord:
    group_id: Optional[str]
    input_tokens: Optional[int]
    input_tokens_uncached: Optional[int]
    cache_creation_input_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    cost_usd: Optional[float]

    @property
    def score(self) -> int:
        # Prefer records with a total, then larger totals.
        return int(self.total_tokens or 0)


def _extract_usage_fields(d: Dict[str, Any]) -> Tuple[
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
]:
    in_tok, out_tok, tot_tok = _canonical_tokens(d)
    explicit_tot_tok = (
        _as_int(d.get("total_tokens"))
        or _as_int(d.get("totalTokens"))
        or _as_int(d.get("total"))
    )

    uncached_in = _as_int(d.get("input_tokens"))
    if uncached_in is None:
        uncached_in = _as_int(d.get("prompt_tokens"))
    if uncached_in is None:
        uncached_in = _as_int(d.get("inputTokens") or d.get("promptTokens"))
    if uncached_in is None:
        uncached_in = _as_int(d.get("input"))

    cache_create = _as_int(d.get("cache_creation_input_tokens"))
    if cache_create is None:
        cache_create = _as_int(d.get("cacheCreationInputTokens"))
    if cache_create is None:
        cache_create = _as_int(d.get("cache_creation_tokens"))
    if cache_create is None:
        cache_info = d.get("cache_creation")
        if isinstance(cache_info, dict):
            cache_create = sum(
                v
                for v in (
                    _as_int(cache_info.get("ephemeral_5m_input_tokens")),
                    _as_int(cache_info.get("ephemeral_1h_input_tokens")),
                )
                if v is not None
            ) or None

    cache = d.get("cache") if isinstance(d.get("cache"), dict) else None
    if cache_create is None and cache is not None:
        # OpenCode-style traces expose billed cache writes under cache.write.
        cache_create = _as_int(cache.get("write"))

    cache_read = _as_int(d.get("cache_read_input_tokens"))
    if cache_read is None:
        cache_read = _as_int(d.get("cacheReadInputTokens"))
    if cache_read is None and cache is not None:
        cache_read = _as_int(cache.get("read"))

    effective_in = uncached_in
    if effective_in is None and (cache_create is not None or cache_read is not None):
        effective_in = 0
    if effective_in is not None:
        effective_in += cache_create or 0
        effective_in += cache_read or 0

    if effective_in is not None:
        in_tok = effective_in
    # Some runtimes (notably Claude) omit an explicit total token field while also
    # exposing cache read/write components. In that case, recompute total from the
    # cache-inclusive prompt tokens plus output tokens.
    if explicit_tot_tok is None and in_tok is not None and out_tok is not None:
        tot_tok = in_tok + out_tok
    else:
        tot_tok = explicit_tot_tok if explicit_tot_tok is not None else tot_tok

    return in_tok, uncached_in, cache_create, cache_read, out_tok, tot_tok


def _iter_usage_dicts(obj: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(obj, dict):
        if _is_usage_like(obj):
            yield obj
        for v in obj.values():
            yield from _iter_usage_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_usage_dicts(v)


def _records_from_json(obj: Any) -> List[UsageRecord]:
    records: List[UsageRecord] = []
    for d in _iter_usage_dicts(obj):
        in_tok, uncached_in, cache_create, cache_read, out_tok, tot_tok = _extract_usage_fields(d)
        if in_tok is None and out_tok is None and tot_tok is None:
            continue
        records.append(
            UsageRecord(
                group_id=None,
                input_tokens=in_tok,
                input_tokens_uncached=uncached_in,
                cache_creation_input_tokens=cache_create,
                cache_read_input_tokens=cache_read,
                output_tokens=out_tok,
                total_tokens=tot_tok,
                cost_usd=_extract_cost_usd(d),
            )
        )
    return records


def _records_from_jsonl(events: Iterable[Dict[str, Any]]) -> List[UsageRecord]:
    records: List[UsageRecord] = []
    for event in events:
        group_id = _pick_id(event)
        for d in _iter_usage_dicts(event):
            in_tok, uncached_in, cache_create, cache_read, out_tok, tot_tok = _extract_usage_fields(d)
            if in_tok is None and out_tok is None and tot_tok is None:
                continue
            records.append(
                UsageRecord(
                    group_id=group_id,
                    input_tokens=in_tok,
                    input_tokens_uncached=uncached_in,
                    cache_creation_input_tokens=cache_create,
                    cache_read_input_tokens=cache_read,
                    output_tokens=out_tok,
                    total_tokens=tot_tok,
                    cost_usd=_extract_cost_usd(d),
                )
            )
    return records


def _records_from_opencode_jsonl(events: Iterable[Dict[str, Any]]) -> List[UsageRecord]:
    records: List[UsageRecord] = []
    for event in events:
        if event.get("type") != "step_finish":
            continue
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        tokens = part.get("tokens")
        if not isinstance(tokens, dict):
            continue

        in_tok, uncached_in, cache_create, cache_read, out_tok, tot_tok = _extract_usage_fields(tokens)
        if in_tok is None and out_tok is None and tot_tok is None:
            continue

        # OpenCode step_finish.part.tokens reflects the billed token usage for the
        # just-completed model call. Sum one record per unique step id rather than
        # collapsing the whole question to the max step total.
        step_id = _pick_id(part) or _pick_id(event)
        records.append(
            UsageRecord(
                group_id=step_id,
                input_tokens=in_tok,
                input_tokens_uncached=uncached_in,
                cache_creation_input_tokens=cache_create,
                cache_read_input_tokens=cache_read,
                output_tokens=out_tok,
                total_tokens=tot_tok,
                cost_usd=_extract_cost_usd(part) or _extract_cost_usd(tokens),
            )
        )
    return records


def _records_from_claude_jsonl(events: List[Dict[str, Any]]) -> List[UsageRecord]:
    final_event: Optional[Dict[str, Any]] = None
    for event in events:
        if event.get("type") == "result" and event.get("subtype") == "success":
            usage = event.get("usage")
            if isinstance(usage, dict):
                final_event = event

    if final_event is None:
        return _records_from_jsonl(events)

    usage = final_event.get("usage")
    if not isinstance(usage, dict):
        return _records_from_jsonl(events)

    in_tok, uncached_in, cache_create, cache_read, out_tok, tot_tok = _extract_usage_fields(usage)
    if in_tok is None and out_tok is None and tot_tok is None:
        return _records_from_jsonl(events)

    cost_usd = _extract_cost_usd(final_event)
    if cost_usd is None:
        cost_usd = _extract_cost_usd(usage)

    return [
        UsageRecord(
            group_id=final_event.get("session_id"),
            input_tokens=in_tok,
            input_tokens_uncached=uncached_in,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            cost_usd=cost_usd,
        )
    ]


def _normalize_pi_usage(usage: Any) -> Optional[Dict[str, Any]]:
    """
    Pi exposes usage on each `message_end` event as
        {"input": N, "output": N, "cacheRead": N, "cacheWrite": N,
         "totalTokens": N, "cost": {..., "total": F}}
    Reshape into the keys recognised by `_extract_usage_fields`.
    """
    if not isinstance(usage, dict):
        return None
    cost_total: Optional[float] = None
    cost = usage.get("cost")
    if isinstance(cost, dict):
        cost_total = _as_float(cost.get("total"))
    elif cost is not None:
        cost_total = _as_float(cost)
    normalized: Dict[str, Any] = {
        "input": usage.get("input"),
        "output": usage.get("output"),
        "total": usage.get("totalTokens") or usage.get("total"),
        "cache_creation_input_tokens": usage.get("cacheWrite"),
        "cache_read_input_tokens": usage.get("cacheRead"),
    }
    if cost_total is not None:
        normalized["cost_usd"] = cost_total
    return normalized


def _pi_message_model(message: Any) -> Optional[str]:
    if not isinstance(message, dict):
        return None
    provider = message.get("provider")
    model = message.get("model")
    if isinstance(provider, str) and provider.strip() and isinstance(model, str) and model.strip():
        return f"{provider.strip()}/{model.strip()}"
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _records_from_pi_jsonl(events: Iterable[Dict[str, Any]]) -> List[UsageRecord]:
    """
    Iterate pi JSONL events and build one usage record per assistant `message_end`.
    Each assistant turn produces a single `message_end` whose `message.usage` reflects
    that turn's billed tokens; summing them gives total cost for the session.
    """
    records: List[UsageRecord] = []
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        normalized = _normalize_pi_usage(usage)
        if normalized is None:
            continue
        in_tok, uncached_in, cache_create, cache_read, out_tok, tot_tok = _extract_usage_fields(normalized)
        if in_tok is None and out_tok is None and tot_tok is None:
            continue
        cost_usd = _as_float(normalized.get("cost_usd"))
        group_id = message.get("responseId") or _pick_id(message) or _pick_id(event)
        records.append(
            UsageRecord(
                group_id=group_id,
                input_tokens=in_tok,
                input_tokens_uncached=uncached_in,
                cache_creation_input_tokens=cache_create,
                cache_read_input_tokens=cache_read,
                output_tokens=out_tok,
                total_tokens=tot_tok,
                cost_usd=cost_usd,
            )
        )
    return records


def _pi_model(events: List[Dict[str, Any]]) -> Optional[str]:
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        if event.get("type") == "message_end":
            model = _pi_message_model(event.get("message"))
            if model:
                return model
        if event.get("type") == "agent_end":
            messages = event.get("messages")
            if isinstance(messages, list):
                for msg in reversed(messages):
                    model = _pi_message_model(msg)
                    if model:
                        return model
    return None


def _records_from_openclaw_json(obj: Any) -> List[UsageRecord]:
    if not isinstance(obj, dict):
        return []

    meta = obj.get("meta")
    if not isinstance(meta, dict):
        return _records_from_json(obj)

    agent_meta = meta.get("agentMeta")
    if not isinstance(agent_meta, dict):
        return _records_from_json(obj)

    usage = agent_meta.get("usage")
    last_call = agent_meta.get("lastCallUsage")
    usage_dict = usage if isinstance(usage, dict) else last_call if isinstance(last_call, dict) else None
    if not isinstance(usage_dict, dict):
        return _records_from_json(obj)

    uncached_in = _as_int(usage_dict.get("input"))
    cache_create = _as_int(usage_dict.get("cacheWrite"))
    cache_read = _as_int(usage_dict.get("cacheRead"))
    output = _as_int(usage_dict.get("output"))

    effective_input = uncached_in
    if effective_input is None and (cache_create is not None or cache_read is not None):
        effective_input = 0
    if effective_input is not None:
        effective_input += cache_create or 0
        effective_input += cache_read or 0

    total = None
    if effective_input is not None and output is not None:
        total = effective_input + output
    else:
        total = _as_int(usage_dict.get("total"))

    if effective_input is None and output is None and total is None:
        return _records_from_json(obj)

    return [
        UsageRecord(
            group_id=None,
            input_tokens=effective_input,
            input_tokens_uncached=uncached_in,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
            output_tokens=output,
            total_tokens=total,
            cost_usd=None,
        )
    ]


def _aggregate(
    records: List[UsageRecord],
) -> Tuple[
    TokenCounts,
    Tuple[Optional[int], Optional[int], Optional[int]],
    Optional[float],
    int,
]:
    if not records:
        return (None, None, None), (None, None, None), None, 0

    grouped: Dict[str, UsageRecord] = {}
    ungrouped: List[UsageRecord] = []

    for r in records:
        if r.group_id:
            prev = grouped.get(r.group_id)
            if prev is None or r.score >= prev.score:
                grouped[r.group_id] = r
        else:
            ungrouped.append(r)

    selected = list(grouped.values())

    if ungrouped:
        totals = [r.total_tokens for r in ungrouped if r.total_tokens is not None]
        # Detect cumulative running-total records: some agents (e.g. OpenCode)
        # emit a token-count event after every LLM step where the count is a
        # running total of the entire session, not just that step.  Two signals:
        #   1. totals are monotonically non-decreasing (sorted order)
        #   2. sum > max * 1.2  — the non-max records contribute at least 20%
        #      of the max, ruling out a single negligible duplicate.
        # When both hold, treat the whole sequence as one cumulative series and
        # keep only the highest-scoring record (which usually has the final total).
        if len(totals) >= 2 and totals == sorted(totals) and sum(totals) > max(totals) * 1.2:
            best = max(ungrouped, key=lambda r: r.score)
            selected.append(best)
        else:
            # Deduplicate identical ungrouped records to avoid double-counting.
            seen = set()
            for r in ungrouped:
                key = (r.input_tokens, r.output_tokens, r.total_tokens, r.cost_usd)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(r)

    in_sum: Optional[int] = 0
    out_sum: Optional[int] = 0
    tot_sum: Optional[int] = 0
    cost_sum: Optional[float] = 0.0

    any_in = any(r.input_tokens is not None for r in selected)
    any_uncached = any(r.input_tokens_uncached is not None for r in selected)
    any_cache_create = any(r.cache_creation_input_tokens is not None for r in selected)
    any_cache_read = any(r.cache_read_input_tokens is not None for r in selected)
    any_out = any(r.output_tokens is not None for r in selected)
    any_tot = any(r.total_tokens is not None for r in selected)
    any_cost = any(r.cost_usd is not None for r in selected)

    uncached_sum: Optional[int] = None
    cache_create_sum: Optional[int] = None
    cache_read_sum: Optional[int] = None

    if not any_in:
        in_sum = None
    if not any_uncached:
        uncached_sum = None
    else:
        uncached_sum = 0
    if not any_cache_create:
        cache_create_sum = None
    else:
        cache_create_sum = 0
    if not any_cache_read:
        cache_read_sum = None
    else:
        cache_read_sum = 0
    if not any_out:
        out_sum = None
    if not any_tot:
        tot_sum = None
    if not any_cost:
        cost_sum = None

    if in_sum is not None:
        in_sum = sum(r.input_tokens or 0 for r in selected)
    if uncached_sum is not None:
        uncached_sum = sum(r.input_tokens_uncached or 0 for r in selected)
    if cache_create_sum is not None:
        cache_create_sum = sum(r.cache_creation_input_tokens or 0 for r in selected)
    if cache_read_sum is not None:
        cache_read_sum = sum(r.cache_read_input_tokens or 0 for r in selected)
    if out_sum is not None:
        out_sum = sum(r.output_tokens or 0 for r in selected)
    if tot_sum is not None:
        tot_sum = sum(r.total_tokens or 0 for r in selected)
    if cost_sum is not None:
        cost_sum = sum(r.cost_usd or 0.0 for r in selected)

    if tot_sum is None and in_sum is not None and out_sum is not None:
        tot_sum = in_sum + out_sum

    return (in_sum, out_sum, tot_sum), (uncached_sum, cache_create_sum, cache_read_sum), cost_sum, len(selected)


def _maybe_model(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for key in ("model", "model_id", "modelId"):
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for parent_key in ("message", "response", "result", "data", "output"):
            v = obj.get(parent_key)
            if isinstance(v, dict):
                m = _maybe_model(v)
                if m:
                    return m
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract token usage from traces")
    parser.add_argument(
        "--format",
        required=True,
        choices=["codex", "opencode", "openclaw", "claude_code", "pi"],
        help="Trace format to parse",
    )
    parser.add_argument("--trace", required=True, help="Path to trace file (json or jsonl)")
    parser.add_argument("--out", required=True, help="Path to write usage.json")
    parser.add_argument("--agent", default="", help="Agent name for metadata")
    parser.add_argument("--model-tag", default="", help="Model tag label for metadata")
    parser.add_argument("--model", default="", help="Model id/name override")
    args = parser.parse_args()

    trace = Path(args.trace)
    out = Path(args.out)

    if args.format == "codex":
        records = _records_from_jsonl(load_jsonl(trace))
        obj_for_model: Any = {}
        try:
            # Use the last event for model detection (best-effort).
            for obj_for_model in load_jsonl(trace):
                pass
        except Exception:
            obj_for_model = {}
        model = args.model or _maybe_model(obj_for_model)
    elif args.format == "opencode":
        events = list(load_jsonl(trace))
        records = _records_from_opencode_jsonl(events)
        model = args.model or _maybe_model(events[-1] if events else {})
    elif args.format == "claude_code":
        try:
            obj = load_json(trace)
            records = _records_from_json(obj)
            model = args.model or _maybe_model(obj)
        except json.JSONDecodeError:
            events = list(load_jsonl(trace))
            records = _records_from_claude_jsonl(events)
            model = args.model or _maybe_model(events[-1] if events else {})
    elif args.format == "pi":
        events = list(load_jsonl(trace))
        records = _records_from_pi_jsonl(events)
        model = args.model or _pi_model(events)
    else:
        try:
            obj = load_json_with_recovery(trace)
            records = _records_from_openclaw_json(obj)
            model = args.model or _openclaw_model(obj)
        except json.JSONDecodeError:
            records = []
            model = args.model or ""

    (in_tok, out_tok, tot_tok), (uncached_in, cache_create, cache_read), cost_usd, selected = _aggregate(records)

    usage = {
        "agent": args.agent or args.format,
        "model_tag": args.model_tag or "",
        "model": model or "",
        "input_tokens": in_tok,
        "input_tokens_uncached": uncached_in,
        "cache_creation_input_tokens": cache_create,
        "cache_read_input_tokens": cache_read,
        "output_tokens": out_tok,
        "total_tokens": tot_tok,
        "cost_usd": cost_usd,
        "records_found": len(records),
        "records_selected": selected,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(usage, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
