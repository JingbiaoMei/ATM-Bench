#!/usr/bin/env python3
"""Retriever implementations for multimodal memory search."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from memqa.retrieve.utils import (
    RetrievalItem,
    build_cache_key,
    extract_first_frame,
    get_cache_path,
    load_index,
    save_index,
)


@dataclass
class RetrievalResult:
    item: RetrievalItem
    score: float


def batched(values: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for idx in range(0, len(values), batch_size):
        yield values[idx : idx + batch_size]


def load_qwen3_vl_embedder(model_id: str, **kwargs: Any):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Qwen3-VL embedder") from exc

    script_path = hf_hub_download(model_id, "scripts/qwen3_vl_embedding.py")
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", script_path)
    if not spec or not spec.loader:
        raise RuntimeError("Failed to load Qwen3-VL embedding script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder(model_name_or_path=model_id, **kwargs)


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


class BaseRetriever:
    def __init__(
        self,
        cache_dir: Path,
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.items: List[RetrievalItem] = []
        self.embeddings: Optional[torch.Tensor] = None

    def build_index(
        self,
        items: List[RetrievalItem],
        cache_config: Dict[str, Any],
        force_rebuild: bool = False,
    ) -> None:
        self.items = items
        if not items:
            self.embeddings = None
            return
        cache_key = build_cache_key(cache_config)
        cache_path = get_cache_path(self.cache_dir, cache_key)
        if force_rebuild:
            print(f"[RetrievalCache] Force rebuild enabled. Cache dir: {self.cache_dir}")
        else:
            mtime_info = {
                "image_batch_mtime": cache_config.get("image_batch_mtime"),
                "video_batch_mtime": cache_config.get("video_batch_mtime"),
                "email_mtime": cache_config.get("email_mtime"),
                "qa_mtime": cache_config.get("qa_mtime"),
            }
            print(
                "[RetrievalCache] Cache key: "
                f"{cache_key} (retriever={cache_config.get('retriever')}, "
                f"media_source={cache_config.get('media_source')}, "
                f"text_model={cache_config.get('text_embedding_model')}, "
                f"vl_model={cache_config.get('vl_embedding_model')}, "
                f"mtimes={mtime_info})"
            )
            if cache_path.exists():
                try:
                    cached = load_index(cache_path)
                except Exception as exc:
                    cached = None
                    print(
                        f"[RetrievalCache] Cache load failed: {cache_path} ({exc})"
                    )
                if cached:
                    embeddings, cached_items, _ = cached
                    self.items = cached_items
                    self.embeddings = torch.tensor(embeddings).to(self.device)
                    print(f"[RetrievalCache] Cache hit: {cache_path}")
                    return
                print(f"[RetrievalCache] Cache miss (unusable): {cache_path}")
            else:
                cache_dir_exists = self.cache_dir.exists()
                exists_flag = cache_path.exists()
                is_file_flag = cache_path.is_file() if exists_flag else False
                print(
                    f"[RetrievalCache] Cache miss (not found): {cache_path} "
                    f"(exists={exists_flag}, is_file={is_file_flag}, "
                    f"cache_dir={self.cache_dir}, cache_dir_exists={cache_dir_exists})"
                )
                if cache_dir_exists:
                    matches = list(self.cache_dir.glob(f"{cache_key}*.pkl"))
                    if matches:
                        print(
                            "[RetrievalCache] Similar cache files: "
                            + ", ".join(str(path) for path in matches)
                        )
                    else:
                        pkl_count = len(list(self.cache_dir.glob("*.pkl")))
                        print(
                            f"[RetrievalCache] Cache dir has {pkl_count} .pkl files"
                        )
        embeddings = self.encode_items(items)
        self.embeddings = embeddings
        save_index(cache_path, embeddings.detach().cpu().numpy(), items, cache_config)

    def encode_items(self, items: List[RetrievalItem]) -> torch.Tensor:
        raise NotImplementedError

    def encode_query(self, query: str) -> torch.Tensor:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Retrieve top-k items for a query.

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            List of RetrievalResult objects sorted by score (descending)
        """
        if self.embeddings is None or not self.items:
            return []

        query_embedding = self.encode_query(query)
        # Compute similarity scores
        scores = torch.matmul(self.embeddings, query_embedding.T).squeeze(-1)

        # Get top-k indices
        if top_k >= len(self.items):
            top_indices = torch.argsort(scores, descending=True)
        else:
            top_indices = torch.argsort(scores, descending=True)[:top_k]

        # Build results
        return [
            RetrievalResult(item=self.items[idx], score=float(scores[idx]))
            for idx in top_indices.tolist()
        ]


class SentenceTransformerRetriever(BaseRetriever):
    """Retriever wrapper for sentence-transformers models (e.g. all-MiniLM-L6-v2)."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(cache_dir, batch_size, device)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=self.device.type)
        self.model.eval()

    def encode_items(self, items: List[RetrievalItem]) -> torch.Tensor:
        texts = [item.text for item in items]
        if not texts:
            return torch.empty((0, self.model.get_sentence_embedding_dimension())).to(
                self.device
            )

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device.type,
        )
        # Ensure it's on the correct device and normalized
        return F.normalize(embeddings, p=2, dim=1)

    def encode_query(self, query: str) -> torch.Tensor:
        embedding = self.model.encode(
            [query], convert_to_tensor=True, device=self.device.type
        )
        return F.normalize(embedding, p=2, dim=1)


class TextRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        batch_size: int = 16,
        max_length: int = 8192,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(cache_dir, batch_size, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
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
        return F.normalize(embeddings, p=2, dim=1)

    def encode_items(self, items: List[RetrievalItem]) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        texts = [item.text for item in items]
        total = math.ceil(len(texts) / self.batch_size) if texts else 0
        for chunk in tqdm(
            batched(texts, self.batch_size),
            total=total,
            desc="Encoding text",
        ):
            embeddings.append(self._encode_texts(chunk))
        return torch.cat(embeddings, dim=0)

    def encode_query(self, query: str) -> torch.Tensor:
        return self._encode_texts([query])


class Qwen3VLRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        batch_size: int = 4,
        instruction: Optional[str] = None,
        num_frames: int = 8,
        max_frames: int = 8,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(cache_dir, batch_size, device)
        kwargs: Dict[str, Any] = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.embedder = load_qwen3_vl_embedder(
            model_name,
            num_frames=num_frames,
            max_frames=max_frames,
            **kwargs,
        )
        try:
            import decord
        except ImportError:
            warnings.warn(
                "decord is not installed. Video processing will fall back to torchvision, "
                "which is slower and deprecated. Install decord>=0.6.0 for better performance."
            )
        self.instruction = instruction
        self.num_frames = num_frames
        self.max_frames = max_frames

    def _build_input(self, item: RetrievalItem) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": item.text}
        if self.instruction:
            payload["instruction"] = self.instruction
        if item.modality == "image" and item.image_path:
            payload["image"] = str(item.image_path)
        if item.modality == "video" and item.video_path:
            payload["video"] = str(item.video_path)
            payload["max_frames"] = self.max_frames
            payload["fps"] = 1
        return payload

    def encode_items(self, items: List[RetrievalItem]) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        total = math.ceil(len(items) / self.batch_size) if items else 0
        for chunk in tqdm(
            batched(items, self.batch_size),
            total=total,
            desc="Encoding multimodal",
        ):
            payloads = [self._build_input(item) for item in chunk]
            embeddings.append(self.embedder.process(payloads))
        return torch.cat(embeddings, dim=0)

    def encode_query(self, query: str) -> torch.Tensor:
        payload: Dict[str, Any] = {"text": query}
        if self.instruction:
            payload["instruction"] = self.instruction
        return self.embedder.process([payload])


class VistaRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str,
        weights_path: str,
        cache_dir: Path,
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(cache_dir, batch_size, device)
        try:
            from visual_bge.modeling import Visualized_BGE
        except ImportError as exc:
            raise RuntimeError(
                "visual_bge is required. Install FlagEmbedding visual_bge per BAAI/bge-visualized."
            ) from exc
        self.model = Visualized_BGE(
            model_name_bge=model_name, model_weight=weights_path
        )
        self.model.eval()

    def _encode(self, text: Optional[str], image_path: Optional[Path]) -> torch.Tensor:
        try:
            with torch.no_grad():
                return self.model.encode(
                    text=text, image=str(image_path) if image_path else None
                )
        except Exception as exc:
            if image_path:
                import logging

                logging.error(
                    f"VISTA encode failed for image={image_path}, text={text[:100] if text else None}...: {exc}"
                )
            raise

    def encode_items(self, items: List[RetrievalItem]) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        failed_items = []
        total = math.ceil(len(items) / self.batch_size) if items else 0
        for i, chunk in tqdm(
            enumerate(batched(items, self.batch_size)),
            total=total,
            desc="Encoding VISTA",
        ):
            chunk_embeddings = []
            for item in chunk:
                image_path = item.image_path
                if item.modality == "video" and item.video_path:
                    frame_path = extract_first_frame(item.video_path)
                    image_path = frame_path
                try:
                    chunk_embeddings.append(self._encode(item.text, image_path))
                except Exception as exc:
                    import logging
                    import traceback

                    logging.error(
                        f"VISTA encode failed: {image_path} (text: {item.text[:50] if item.text else 'None'}...)"
                    )
                    logging.error(traceback.format_exc())
                    failed_items.append(
                        str(image_path) if image_path else str(item.video_path)
                    )
            embeddings.extend(chunk_embeddings)
            if i > 0 and i % 50 == 0:
                torch.cuda.empty_cache()

        if failed_items:
            import logging

            logging.warning(
                f"Skipped {len(failed_items)} items that failed VISTA encoding:"
            )
            for f in failed_items[:10]:
                logging.warning(f"  - {f}")

        stacked = torch.vstack([emb.cpu() for emb in embeddings]).to(self.device)
        return F.normalize(stacked, p=2, dim=1)

    def encode_query(self, query: str) -> torch.Tensor:
        embedding = self._encode(query, None)
        return F.normalize(embedding.to(self.device), p=2, dim=1)


class ClipRetriever(BaseRetriever):
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(cache_dir, batch_size, device)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.frames_dir = cache_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._image_dim: Optional[int] = None

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return F.normalize(outputs, p=2, dim=1)

    def _encode_images(self, image_paths: List[Optional[Path]]) -> torch.Tensor:
        valid_items: List[Tuple[int, Path]] = [
            (idx, path)
            for idx, path in enumerate(image_paths)
            if path and path.exists()
        ]
        if not valid_items:
            return torch.zeros(
                (len(image_paths), self._get_image_dim()), device=self.device
            )

        image_embeddings = torch.zeros(
            (len(image_paths), self._get_image_dim()), device=self.device
        )
        total = math.ceil(len(valid_items) / self.batch_size) if valid_items else 0
        for chunk in tqdm(
            batched(valid_items, self.batch_size),
            total=total,
            desc="Encoding images",
        ):
            indices = [item[0] for item in chunk]
            images = [self._load_image(item[1]) for item in chunk]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            outputs = F.normalize(outputs, p=2, dim=1)
            for output_idx, item_idx in enumerate(indices):
                image_embeddings[item_idx] = outputs[output_idx]
        return image_embeddings

    def _get_image_dim(self) -> int:
        if self._image_dim is None:
            dummy = torch.zeros((1, 3, 224, 224), device=self.device)
            with torch.no_grad():
                self._image_dim = self.model.get_image_features(
                    pixel_values=dummy
                ).shape[-1]
        return self._image_dim

    def _resolve_image_path(self, item: RetrievalItem) -> Optional[Path]:
        if item.modality == "image":
            return item.image_path
        if item.modality == "video" and item.video_path:
            return extract_first_frame(item.video_path, self.frames_dir)
        return None

    def encode_items(self, items: List[RetrievalItem]) -> torch.Tensor:
        texts = [item.text for item in items]
        image_paths = [self._resolve_image_path(item) for item in items]
        text_embeddings = []
        total = math.ceil(len(texts) / self.batch_size) if texts else 0
        for chunk in tqdm(
            batched(texts, self.batch_size),
            total=total,
            desc="Encoding clip text",
        ):
            text_embeddings.append(self._encode_texts(chunk))
        text_embeddings_tensor = torch.cat(text_embeddings, dim=0)
        image_embeddings_tensor = self._encode_images(image_paths)
        combined = torch.cat([text_embeddings_tensor, image_embeddings_tensor], dim=1)
        return F.normalize(combined, p=2, dim=1)

    def encode_query(self, query: str) -> torch.Tensor:
        text_embedding = self._encode_texts([query])
        combined = torch.cat([text_embedding, text_embedding], dim=1)
        return F.normalize(combined, p=2, dim=1)
