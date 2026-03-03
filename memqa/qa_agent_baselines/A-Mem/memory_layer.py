import json
import os
import pickle
import uuid
import requests
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from nltk.tokenize import word_tokenize
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def simple_tokenize(text: str) -> List[str]:
    try:
        return word_tokenize(text)
    except Exception:
        return text.lower().split()


def extract_json(payload: str) -> Optional[Dict[str, Any]]:
    cleaned = payload.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    fragment = cleaned[start : end + 1]
    try:
        return json.loads(fragment)
    except Exception:
        return None


class LLMController:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.openai_client = None
        self.vllm_local = None

        if self.provider == "openai":
            self.openai_client = OpenAI(api_key=self.api_key)
        elif self.provider == "vllm_local":
            try:
                from vllm import LLM  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "vllm is not installed. Install vllm to use vllm_local."
                ) from exc
            self.vllm_local = LLM(
                model=self.model,
                trust_remote_code=True,
            )

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        image_input: Optional[str] = None,
    ) -> str:
        if image_input:
            import base64
            import mimetypes
            from pathlib import Path

            user_content = [{"type": "text", "text": prompt}]

            if image_input.startswith("http"):
                user_content.append(
                    {"type": "image_url", "image_url": {"url": image_input}}
                )
            elif Path(image_input).exists():
                mime_type, _ = mimetypes.guess_type(image_input)
                if not mime_type:
                    mime_type = "image/jpeg"
                try:
                    with open(image_input, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        }
                    )
                except Exception:
                    pass

            messages = [{"role": "user", "content": user_content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        if self.provider == "openai":
            if not self.openai_client:
                raise RuntimeError("OpenAI client not initialized")
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1000,
            }
            if response_format:
                kwargs["response_format"] = response_format
            response = self.openai_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        if self.provider == "vllm":
            if not self.api_base:
                raise RuntimeError("API base URL required for vllm")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1000,
            }
            if response_format:
                data["response_format"] = response_format

            response = requests.post(
                self.api_base,
                headers=headers,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        if not self.vllm_local:
            raise RuntimeError("vllm local engine not initialized")
        try:
            from vllm import SamplingParams  # type: ignore
        except ImportError as exc:
            raise RuntimeError("vllm is not installed") from exc
        prompt_text = "\n".join(
            [
                "SYSTEM: You must respond with a JSON object.",
                f"USER: {prompt}",
                "ASSISTANT:",
            ]
        )
        sampling_params = SamplingParams(temperature=temperature, max_tokens=1000)
        outputs = self.vllm_local.generate([prompt_text], sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()


@dataclass
class MemoryNote:
    content: str
    id: str
    keywords: List[str]
    context: str
    tags: List[str]
    links: List[Union[int, str]]
    timestamp: str
    image_path: Optional[str] = None  # Store raw media path if available
    modality: Optional[str] = None

    @staticmethod
    def analyze_content(
        content: str, llm: LLMController, image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        import sys
        from pathlib import Path

        current_dir = Path(__file__).resolve().parent
        config_path = current_dir / "config.py"

        import importlib.util

        spec = importlib.util.spec_from_file_location("amem_config", config_path)
        if spec and spec.loader:
            amem_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(amem_config)
            AMEM_PROMPTS = amem_config.AMEM_PROMPTS
        else:
            raise RuntimeError("Could not load config.py")

        if image_path:
            prompt = AMEM_PROMPTS["NOTE_CONSTRUCTION_MULTIMODAL"].format(
                content=content
            )
        else:
            prompt = AMEM_PROMPTS["NOTE_CONSTRUCTION"].format(content=content)

        response = llm.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "context": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["keywords", "context", "tags"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=0.7,
            image_input=image_path,
        )
        parsed = extract_json(response)
        if not parsed:
            return {"keywords": [], "context": "General", "tags": []}
        return {
            "keywords": parsed.get("keywords") or [],
            "context": parsed.get("context") or "General",
            "tags": parsed.get("tags") or [],
        }


class SimpleEmbeddingRetriever:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[str]) -> None:
        if not documents:
            return
        new_embeddings = self.model.encode(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.corpus.extend(documents)

    def search(self, query: str, k: int) -> List[int]:
        if not self.corpus or self.embeddings is None:
            return []
        query_embedding = self.model.encode([query])[0]
        scores = cosine_similarity([query_embedding], self.embeddings)[0]
        k = min(k, len(scores))
        return np.argsort(scores)[-k:][::-1].tolist()

    def search_with_scores(self, query: str, k: int) -> Tuple[List[int], List[float]]:
        if not self.corpus or self.embeddings is None:
            return [], []
        query_embedding = self.model.encode([query])[0]
        scores = cosine_similarity([query_embedding], self.embeddings)[0]
        k = min(k, len(scores))
        indices = np.argsort(scores)[-k:][::-1].tolist()
        return indices, [float(scores[idx]) for idx in indices]

    def save(self, state_path: str, embeddings_path: str) -> None:
        state = {"corpus": self.corpus, "model_name": self.model_name}
        with open(state_path, "wb") as handle:
            pickle.dump(state, handle)
        if self.embeddings is not None:
            np.save(embeddings_path, self.embeddings)

    def load(self, state_path: str, embeddings_path: str) -> "SimpleEmbeddingRetriever":
        with open(state_path, "rb") as handle:
            state = pickle.load(handle)
        self.corpus = state.get("corpus", [])
        self.model_name = state.get("model_name", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        return self


class HybridRetriever:
    """Hybrid BM25 + Embedding retriever matching original A-Mem paper.

    Combines lexical (BM25) and semantic (embedding) scores with configurable weight.
    Formula: score = alpha * BM25_score + (1 - alpha) * semantic_score
    Default alpha=0.5 matches the original A-Mem implementation.
    """

    def __init__(self, model_name: str, alpha: float = 0.5) -> None:
        self.model_name = model_name
        self.alpha = alpha
        self.model = SentenceTransformer(model_name)
        self.corpus: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None

    def add_documents(self, documents: List[str]) -> None:
        if not documents:
            return
        for doc in documents:
            self.corpus.append(doc)
            self.tokenized_corpus.append(simple_tokenize(doc.lower()))

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        new_embeddings = self.model.encode(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, k: int) -> List[int]:
        if not self.corpus or self.embeddings is None or self.bm25 is None:
            return []

        tokenized_query = simple_tokenize(query.lower())
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        if bm25_max > bm25_min:
            bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-6)
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]

        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores

        k = min(k, len(hybrid_scores))
        return np.argsort(hybrid_scores)[-k:][::-1].tolist()

    def search_with_scores(self, query: str, k: int) -> Tuple[List[int], List[float]]:
        if not self.corpus or self.embeddings is None or self.bm25 is None:
            return [], []

        tokenized_query = simple_tokenize(query.lower())
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        if bm25_max > bm25_min:
            bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-6)
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]

        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores

        k = min(k, len(hybrid_scores))
        indices = np.argsort(hybrid_scores)[-k:][::-1].tolist()
        return indices, [float(hybrid_scores[idx]) for idx in indices]

    def save(self, state_path: str, embeddings_path: str) -> None:
        state = {
            "corpus": self.corpus,
            "tokenized_corpus": self.tokenized_corpus,
            "model_name": self.model_name,
            "alpha": self.alpha,
        }
        with open(state_path, "wb") as handle:
            pickle.dump(state, handle)
        if self.embeddings is not None:
            np.save(embeddings_path, self.embeddings)

    def load(self, state_path: str, embeddings_path: str) -> "HybridRetriever":
        with open(state_path, "rb") as handle:
            state = pickle.load(handle)
        self.corpus = state.get("corpus", [])
        self.tokenized_corpus = state.get("tokenized_corpus", [])
        self.model_name = state.get("model_name", self.model_name)
        self.alpha = state.get("alpha", self.alpha)
        self.model = SentenceTransformer(self.model_name)
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        return self


class AgenticMemorySystem:
    def __init__(
        self,
        model_name: str,
        llm_backend: str,
        llm_model: str,
        evo_threshold: int,
        api_key: Optional[str],
        api_base: Optional[str],
        alpha: float = 0.5,
        use_hybrid: bool = True,
        follow_links: bool = True,
        timeout: int = 120,
    ) -> None:
        self.memories: Dict[str, MemoryNote] = {}
        self.memory_ids: List[str] = []
        self.alpha = alpha
        self.use_hybrid = use_hybrid
        self.follow_links = follow_links
        self.model_name = model_name

        if use_hybrid:
            self.retriever: Union[HybridRetriever, SimpleEmbeddingRetriever] = (
                HybridRetriever(model_name, alpha)
            )
        else:
            self.retriever = SimpleEmbeddingRetriever(model_name)

        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base, timeout=timeout)
        self.evo_threshold = evo_threshold
        self.evo_count = 0

    def _memory_doc(self, note: MemoryNote) -> str:
        keywords = ", ".join(note.keywords)
        tags = ", ".join(note.tags)
        return f"content:{note.content} context:{note.context} keywords:{keywords} tags:{tags}"

    def analyze_and_create_note(
        self,
        content: str,
        timestamp: Optional[str],
        note_id: Optional[str] = None,
        image_path: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> MemoryNote:
        note_uuid = note_id or str(uuid.uuid4())
        analysis = MemoryNote.analyze_content(
            content, self.llm_controller, image_path=image_path
        )
        return MemoryNote(
            content=content,
            id=note_uuid,
            keywords=analysis.get("keywords", []),
            context=analysis.get("context", "General"),
            tags=analysis.get("tags", []),
            links=[],
            timestamp=timestamp or datetime.now().strftime("%Y%m%d%H%M"),
            image_path=image_path,
            modality=modality,
        )

    def ingest_note(self, note: MemoryNote, disable_evolution: bool = False) -> None:
        if not disable_evolution:
            self._process_memory(note)
        self.memories[note.id] = note
        self.memory_ids.append(note.id)
        self.retriever.add_documents([self._memory_doc(note)])

    def add_note(
        self,
        content: str,
        timestamp: Optional[str],
        note_id: Optional[str] = None,
        disable_evolution: bool = False,
        image_path: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> str:
        note = self.analyze_and_create_note(
            content,
            timestamp,
            note_id=note_id,
            image_path=image_path,
            modality=modality,
        )
        self.ingest_note(note, disable_evolution)
        return note.id

    def _process_memory(self, note: MemoryNote) -> None:
        neighbor_indices = self.retriever.search(note.content, 5)
        neighbor_text = self._format_neighbor_text(neighbor_indices)

        import importlib.util
        from pathlib import Path

        current_dir = Path(__file__).resolve().parent
        config_path = current_dir / "config.py"
        spec = importlib.util.spec_from_file_location("amem_config", config_path)
        if spec and spec.loader:
            amem_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(amem_config)
            AMEM_PROMPTS = amem_config.AMEM_PROMPTS
        else:
            raise RuntimeError("Could not load config.py")

        prompt = AMEM_PROMPTS["MEMORY_EVOLUTION"].format(
            context=note.context,
            content=note.content,
            keywords=", ".join(note.keywords)
            if isinstance(note.keywords, list)
            else note.keywords,
            nearest_neighbors_memories=neighbor_text,
            neighbor_number=len(neighbor_indices),
        )
        response = self.llm_controller.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "should_evolve": {"type": "boolean"},
                            "actions": {"type": "array", "items": {"type": "string"}},
                            "suggested_connections": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "tags_to_update": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "new_context_neighborhood": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "new_tags_neighborhood": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "required": [
                            "should_evolve",
                            "actions",
                            "suggested_connections",
                            "tags_to_update",
                            "new_context_neighborhood",
                            "new_tags_neighborhood",
                        ],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=0.7,
        )
        parsed = extract_json(response)
        if not parsed:
            return
        if not parsed.get("should_evolve"):
            return
        actions = parsed.get("actions", [])
        if "strengthen" in actions:
            connections = parsed.get("suggested_connections", [])
            connection_indices: List[int] = []
            for idx in connections:
                if isinstance(idx, int) and 0 <= idx < len(self.memory_ids):
                    connection_indices.append(idx)
            if connection_indices:
                note.links.extend(connection_indices)
            tags_update = parsed.get("tags_to_update")
            if tags_update:
                note.tags = tags_update
        if "update_neighbor" in actions:
            new_contexts = parsed.get("new_context_neighborhood", [])
            new_tags = parsed.get("new_tags_neighborhood", [])
            for idx, neighbor_index in enumerate(neighbor_indices):
                if neighbor_index >= len(self.memory_ids):
                    continue
                neighbor_id = self.memory_ids[neighbor_index]
                neighbor = self.memories.get(neighbor_id)
                if not neighbor:
                    continue
                if idx < len(new_contexts):
                    neighbor.context = new_contexts[idx]
                if idx < len(new_tags):
                    neighbor.tags = new_tags[idx]
                self.memories[neighbor_id] = neighbor

    def find_related_memories(
        self, query: str, k: int, follow_links: Optional[bool] = None
    ) -> Tuple[str, List[int]]:
        if not self.memories:
            return "", []

        if follow_links is None:
            follow_links = self.follow_links

        indices = self.retriever.search(query, k)

        if follow_links:
            linked_indices: set[int] = set()
            for idx in indices:
                if idx >= len(self.memory_ids):
                    continue
                memory = self.memories.get(self.memory_ids[idx])
                if memory and memory.links:
                    for link_id in memory.links:
                        link_idx = self._resolve_link_index(link_id)
                        if link_idx is not None and link_idx not in indices:
                            linked_indices.add(link_idx)

            for link_idx in linked_indices:
                if link_idx not in indices:
                    indices.append(link_idx)

        lines: List[str] = []
        for idx in indices:
            if idx >= len(self.memory_ids):
                continue
            memory = self.memories.get(self.memory_ids[idx])
            if not memory:
                continue
            lines.append(f"{memory.timestamp}: {memory.content}")
        return "\n".join(lines), indices

    def find_related_memories_raw(
        self, query: str, k: int, follow_links: Optional[bool] = None
    ) -> str:
        text, _ = self.find_related_memories(query, k, follow_links=follow_links)
        return text

    def _resolve_link_index(self, link_id: Union[int, str]) -> Optional[int]:
        if isinstance(link_id, int):
            return link_id if 0 <= link_id < len(self.memory_ids) else None
        if isinstance(link_id, str):
            try:
                return self.memory_ids.index(link_id)
            except ValueError:
                return None
        return None

    def _format_neighbor_text(self, neighbor_indices: List[int]) -> str:
        lines: List[str] = []
        for idx in neighbor_indices:
            if idx >= len(self.memory_ids):
                continue
            memory = self.memories.get(self.memory_ids[idx])
            if not memory:
                continue
            lines.append(
                "memory index:"
                f"{idx}\t talk start time:{memory.timestamp}\t memory content: "
                f"{memory.content}\t memory context: {memory.context}\t "
                f"memory keywords: {memory.keywords}\t memory tags: {memory.tags}"
            )
        return "\n".join(lines)
