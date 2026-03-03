#!/usr/bin/env python3
"""LLM client wrapper for MMRag baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI


@dataclass
class TokenUsage:
    """Token usage statistics from LLM response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


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


def _messages_to_responses_input(
    messages: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], str]:
    instructions_parts: List[str] = []
    input_items: List[Dict[str, Any]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")

        if role == "system":
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                if text_parts:
                    instructions_parts.append("\n".join(text_parts).strip())
            elif content is not None:
                instructions_parts.append(str(content).strip())
            continue

        input_content: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    input_content.append({"type": "input_text", "text": str(part)})
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    input_content.append(
                        {"type": "input_text", "text": str(part.get("text", ""))}
                    )
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    if image_url:
                        input_content.append(
                            {"type": "input_image", "image_url": str(image_url)}
                        )
                else:
                    text_value = part.get("text")
                    if text_value is not None:
                        input_content.append(
                            {"type": "input_text", "text": str(text_value)}
                        )
        else:
            input_content.append({"type": "input_text", "text": str(content)})

        input_items.append({"role": role, "content": input_content})

    instructions = "\n\n".join([p for p in instructions_parts if p])
    return input_items, instructions


def _extract_response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text).strip()
    output = getattr(response, "output", None)
    if isinstance(output, list):
        texts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                item = getattr(item, "__dict__", {}) or {}
            if item.get("type") != "message":
                continue
            if item.get("role") != "assistant":
                continue
            content = item.get("content") or []
            for part in content:
                if not isinstance(part, dict):
                    part = getattr(part, "__dict__", {}) or {}
                part_type = part.get("type")
                if part_type in {"output_text", "text"} and part.get("text"):
                    texts.append(str(part["text"]))
        if texts:
            return "\n".join(texts).strip()
    return ""


def _should_use_responses(model: str | None, config: Dict[str, Any]) -> bool:
    if config.get("use_responses"):
        return True
    if not model:
        return False
    return "gpt-5-pro" in model.lower()


def _responses_supports_temperature(model: str | None) -> bool:
    if not model:
        return True
    return "gpt-5-pro" not in model.lower()


class LLMClient:
    def __init__(
        self,
        provider: str,
        model_config: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.provider = provider
        self.config = dict(model_config or {})
        if kwargs:
            self.config.update(kwargs)
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
        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=self.config.get("timeout"),
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

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
