#!/usr/bin/env python3
"""
Local HuggingFace embedding wrapper for HippoRAG 2.

This module provides a local embedding model implementation that uses
HuggingFace transformers directly, following the same pattern as MMRAG's
TextRetriever but adapted to HippoRAG's embedding interface.

Usage:
    Instead of using --embedding-endpoint with VLLM, use:
    --embedding-mode local --embedding-model Qwen/Qwen3-Embedding-0.6B
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Pool the last token embedding (for decoder-only models like Qwen)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


class LocalHuggingFaceEmbedder:
    """
    Local HuggingFace embedding model for HippoRAG 2.

    This class provides the same interface as HippoRAG's embedding models
    but runs locally using HuggingFace transformers instead of requiring
    an API endpoint.

    Supports:
    - Qwen/Qwen3-Embedding-0.6B
    - Qwen/Qwen3-Embedding-4B
    - Any HuggingFace model that supports AutoModel/AutoTokenizer

    Example:
        embedder = LocalHuggingFaceEmbedder(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            batch_size=8,
            device="cuda",
        )
        embeddings = embedder.embed(["Hello world", "Test query"])
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        max_length: int = 8192,
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the local HuggingFace embedder.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
            batch_size: Batch size for encoding
            max_length: Maximum sequence length for tokenization
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            normalize: Whether to L2-normalize embeddings (recommended)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading local embedding model: {model_name}")
        logger.info(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension
        with torch.no_grad():
            dummy = self.tokenizer(
                ["test"], return_tensors="pt", padding=True, truncation=True
            )
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            outputs = self.model(**dummy)
            self._embedding_dim = outputs.last_hidden_state.shape[-1]

        logger.info(f"Embedding dimension: {self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts."""
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}

        with torch.no_grad():
            outputs = self.model(**batch)

        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed texts and return numpy array.

        Args:
            texts: Single text or list of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self._embedding_dim))

        all_embeddings: List[torch.Tensor] = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(
                iterator, desc="Encoding", total=len(texts) // self.batch_size + 1
            )

        for i in iterator:
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings.numpy()

    def __call__(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Alias for embed()."""
        return self.embed(texts, show_progress=show_progress)


class HippoRAGLocalEmbedder:
    """
    Wrapper that adapts LocalHuggingFaceEmbedder to HippoRAG's embedding interface.

    HippoRAG expects embedding models with specific method signatures. This wrapper
    provides compatibility while using our local HuggingFace implementation.

    The interface mimics HippoRAG's:
    - VLLMEmbedding (for VLLM endpoints)
    - TransformersEmbedding (for sentence-transformers)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        max_length: int = 8192,
        device: Optional[str] = None,
        normalize: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the HippoRAG-compatible local embedder.

        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            device: Device to use
            normalize: Whether to L2-normalize embeddings
            cache_dir: Optional cache directory (unused, for compatibility)
        """
        self.embedder = LocalHuggingFaceEmbedder(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            normalize=normalize,
        )
        self.model_name = model_name

        # HippoRAG compatibility attributes
        self.base_url = None  # No endpoint for local
        self.url = None  # No endpoint for local

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedder.embedding_dim

    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Embed texts (HippoRAG interface).

        Args:
            texts: Single text or list of texts
            **kwargs: Ignored, for compatibility

        Returns:
            numpy array of embeddings
        """
        return self.embedder.embed(texts)

    def encode(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode texts (alternative interface name).

        Args:
            texts: Single text or list of texts
            **kwargs: Ignored, for compatibility

        Returns:
            numpy array of embeddings
        """
        return self.embedder.embed(texts)

    def batch_encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Batch encode texts (HippoRAG batch interface).

        Args:
            texts: List of texts to encode
            batch_size: Optional batch size override
            show_progress: Whether to show progress bar
            **kwargs: Ignored, for compatibility

        Returns:
            numpy array of embeddings
        """
        if batch_size is not None:
            original_batch_size = self.embedder.batch_size
            self.embedder.batch_size = batch_size
            result = self.embedder.embed(texts, show_progress=show_progress)
            self.embedder.batch_size = original_batch_size
            return result
        return self.embedder.embed(texts, show_progress=show_progress)

    def __call__(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> np.ndarray:
        """Alias for embed()."""
        return self.embed(texts, **kwargs)


def create_local_embedder(
    model_name: str,
    batch_size: int = 8,
    max_length: int = 8192,
    device: Optional[str] = None,
    normalize: bool = True,
) -> HippoRAGLocalEmbedder:
    """
    Factory function to create a local embedder for HippoRAG.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        device: Device to use
        normalize: Whether to L2-normalize embeddings

    Returns:
        HippoRAGLocalEmbedder instance
    """
    return HippoRAGLocalEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        normalize=normalize,
    )
