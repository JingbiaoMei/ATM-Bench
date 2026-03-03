#!/usr/bin/env python3
"""Reranker implementations for multimodal memory search."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from memqa.retrieve.utils import RetrievalItem


@dataclass
class RerankResult:
    item: RetrievalItem
    score: float


def batched(values: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for idx in range(0, len(values), batch_size):
        yield values[idx : idx + batch_size]


def load_qwen3_vl_reranker(model_id: str, **kwargs: Any):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Qwen3-VL reranker") from exc

    script_path = hf_hub_download(model_id, "scripts/qwen3_vl_reranker.py")
    spec = importlib.util.spec_from_file_location("qwen3_vl_reranker", script_path)
    if not spec or not spec.loader:
        raise RuntimeError("Failed to load Qwen3-VL reranker script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3VLReranker(model_name_or_path=model_id, **kwargs)


class BaseReranker:
    def rerank(self, query: str, candidates: List[RetrievalItem]) -> List[RerankResult]:
        raise NotImplementedError


class NoopReranker(BaseReranker):
    def rerank(self, query: str, candidates: List[RetrievalItem]) -> List[RerankResult]:
        return [RerankResult(item=item, score=0.0) for item in candidates]


class TextReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        max_length: int = 8192,
        instruction: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.instruction = (
            instruction
            or "Given a query and a document, judge whether the document answers the query."
        )
        self._mode = self._detect_mode(model_name)
        if self._mode == "causal_lm":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)
            self._yes_id, self._no_id = self._resolve_yes_no_token_ids()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def _detect_mode(self, model_name: str) -> str:
        if "qwen3-reranker" in model_name.lower():
            return "causal_lm"
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            return "sequence_cls"
        architectures = [name.lower() for name in (config.architectures or [])]
        if any("causallm" in name for name in architectures):
            return "causal_lm"
        return "sequence_cls"

    def _resolve_yes_no_token_ids(self) -> tuple[int, int]:
        yes_ids = self.tokenizer.encode(" yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode(" no", add_special_tokens=False)
        if not yes_ids or not no_ids:
            raise ValueError("Tokenizer missing yes/no token ids for reranking.")
        # Use the first token for each; Qwen-style rerankers typically use single-token yes/no.
        return yes_ids[0], no_ids[0]

    def _build_causal_prompts(self, queries: List[str], docs: List[str]) -> List[str]:
        prompts = []
        for query, doc in zip(queries, docs):
            prompts.append(
                f"{self.instruction}\n\n"
                f"Query: {query}\n"
                f"Document: {doc}\n"
                "Answer: "
            )
        return prompts

    def _score_sequence_cls(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        with torch.no_grad():
            output = self.model(**inputs)
        logits = output.logits
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(-1)
        elif logits.dim() == 2:
            index = self._resolve_positive_label_index(logits.size(1))
            logits = logits[:, index]
        scores = logits.tolist()
        if isinstance(scores, float):
            scores = [scores]
        return [float(score) for score in scores]

    def _resolve_positive_label_index(self, num_labels: int) -> int:
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict):
            lowered = {idx: label.lower() for idx, label in id2label.items()}
            for idx, label in lowered.items():
                if "relevant" in label or "positive" in label or "pos" == label:
                    return int(idx)
            if "label_1" in lowered.values() and 1 in lowered:
                return 1
        return num_labels - 1

    def _score_causal_lm(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        with torch.no_grad():
            output = self.model(**inputs)
        logits = output.logits  # [batch, seq, vocab]
        attention_mask = inputs["attention_mask"]
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[batch_indices, last_indices]
        yes_scores = last_logits[:, self._yes_id]
        no_scores = last_logits[:, self._no_id]
        scores = (yes_scores - no_scores).tolist()
        if isinstance(scores, float):
            scores = [scores]
        return [float(score) for score in scores]

    def rerank(self, query: str, candidates: List[RetrievalItem]) -> List[RerankResult]:
        if not candidates:
            return []
        results: List[RerankResult] = []
        for chunk in batched(candidates, self.batch_size):
            queries = [query for _ in chunk]
            docs = [item.text for item in chunk]
            if self._mode == "causal_lm":
                prompts = self._build_causal_prompts(queries, docs)
                inputs = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            else:
                inputs = self.tokenizer(
                    queries,
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            if self._mode == "causal_lm":
                scores = self._score_causal_lm(inputs)
            else:
                scores = self._score_sequence_cls(inputs)
            for item, score in zip(chunk, scores):
                results.append(RerankResult(item=item, score=float(score)))
        return results


class Qwen3VLReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        instruction: Optional[str] = None,
        num_frames: int = 8,
        max_frames: int = 8,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.batch_size = batch_size
        kwargs: Dict[str, Any] = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.reranker = load_qwen3_vl_reranker(model_name, **kwargs)
        self.instruction = (
            instruction
            or "Given a search query, retrieve relevant candidates that answer the query."
        )
        self.num_frames = num_frames
        self.max_frames = max_frames
        if device:
            self.reranker.device = torch.device(device)

    def rerank(self, query: str, candidates: List[RetrievalItem]) -> List[RerankResult]:
        if not candidates:
            return []
        results: List[RerankResult] = []
        for chunk in batched(candidates, self.batch_size):
            documents = []
            for item in chunk:
                payload: Dict[str, Any] = {"text": item.text}
                if item.modality == "image" and item.image_path:
                    payload["image"] = str(item.image_path)
                if item.modality == "video" and item.video_path:
                    payload["video"] = str(item.video_path)
                    payload["max_frames"] = self.max_frames
                    payload["fps"] = 1
                documents.append(payload)
            inputs = {
                "query": {"text": query},
                "documents": documents,
                "instruction": self.instruction,
                "fps": 1,
                "max_frames": self.max_frames,
            }
            scores = self.reranker.process(inputs)
            for item, score in zip(chunk, scores):
                results.append(RerankResult(item=item, score=float(score)))
        return results
