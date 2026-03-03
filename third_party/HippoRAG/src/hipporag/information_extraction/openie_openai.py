import ast
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)
DEFAULT_PARSE_RETRIES = 1


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response):
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    if match is not None:
        parsed = _safe_parse_json_like(match.group())
        if isinstance(parsed, dict):
            entities = parsed.get("named_entities")
            if isinstance(entities, list):
                return entities
    list_payload = _extract_list_payload(real_response, "named_entities")
    parsed_list = _safe_parse_list(list_payload)
    if isinstance(parsed_list, list):
        return parsed_list
    return None


def _normalize_json_like(text: str) -> str:
    if not text:
        return text
    out = []
    i = 0
    in_str = None
    escape = False
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if _is_token(text, i, "null"):
            out.append("None")
            i += 4
            continue
        if _is_token(text, i, "true"):
            out.append("True")
            i += 4
            continue
        if _is_token(text, i, "false"):
            out.append("False")
            i += 5
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _is_token(text: str, idx: int, token: str) -> bool:
    end = idx + len(token)
    if text[idx:end] != token:
        return False
    prev = text[idx - 1] if idx > 0 else ""
    nxt = text[end] if end < len(text) else ""
    if (prev.isalnum() or prev == "_") or (nxt.isalnum() or nxt == "_"):
        return False
    return True


def _safe_parse_json_like(text: str) -> Optional[Any]:
    if text is None:
        return None
    candidates = [text]
    try:
        fixed = fix_broken_generated_json(text)
        if fixed and fixed != text:
            candidates.append(fixed)
    except Exception:
        pass
    for candidate in candidates:
        for parser in ("json", "ast", "json_norm", "ast_norm"):
            try:
                if parser == "json":
                    return json.loads(candidate)
                if parser == "ast":
                    return ast.literal_eval(candidate)
                if parser == "json_norm":
                    return json.loads(_normalize_json_like(candidate))
                if parser == "ast_norm":
                    return ast.literal_eval(_normalize_json_like(candidate))
            except Exception:
                continue
    return None


def _safe_parse_list(list_payload: Optional[str]) -> Optional[List[Any]]:
    if not list_payload:
        return None
    parsed = _safe_parse_json_like(list_payload)
    if isinstance(parsed, list):
        return parsed
    return None


def _extract_list_payload(text: str, key: str) -> Optional[str]:
    if not text:
        return None
    for quote in ('"', "'"):
        key_token = f"{quote}{key}{quote}"
        idx = text.find(key_token)
        if idx == -1:
            continue
        start = text.find("[", idx)
        if start == -1:
            continue
        return _extract_balanced_list(text, start)
    return None


def _extract_balanced_list(text: str, start: int) -> Optional[str]:
    depth = 0
    in_str = None
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.max_workers = None
        self.parse_retries = DEFAULT_PARSE_RETRIES

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        last_error = None
        attempts = self.parse_retries + 1
        for attempt in range(attempts):
            try:
                # LLM INFERENCE
                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=ner_input_message,
                )
                metadata['cache_hit'] = cache_hit
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                extracted_entities = _extract_ner_from_response(real_response)
                if extracted_entities is None:
                    raise ValueError("Failed to parse named_entities from response.")
                unique_entities = list(dict.fromkeys(extracted_entities))
                return NerRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    unique_entities=unique_entities,
                    metadata=metadata
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "NER parse error for chunk %s (attempt %d/%d): %s",
                    chunk_key,
                    attempt + 1,
                    attempts,
                    e,
                )
                metadata.update({'error': str(e), 'parse_retry': attempt + 1})
                if metadata.get("cache_hit") and attempt < attempts - 1:
                    logger.warning(
                        "Cache hit for chunk %s; skipping additional NER retries.",
                        chunk_key,
                    )
                    break
                if attempt < attempts - 1:
                    continue

        metadata.update({'error': str(last_error)})
        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,  # Store the error message in metadata
            unique_entities=[],
            metadata=metadata  # Store the error message in metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is not None:
                parsed = _safe_parse_json_like(match.group())
                if isinstance(parsed, dict):
                    triples = parsed.get("triples")
                    if isinstance(triples, list):
                        return triples
            list_payload = _extract_list_payload(real_response, "triples")
            parsed_list = _safe_parse_list(list_payload)
            if isinstance(parsed_list, list):
                return parsed_list
            return None

        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        last_error = None
        attempts = self.parse_retries + 1
        for attempt in range(attempts):
            try:
                # LLM INFERENCE
                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=messages,
                )
                metadata['cache_hit'] = cache_hit
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                extracted_triples = _extract_triples_from_response(real_response)
                if extracted_triples is None:
                    raise ValueError("Failed to parse triples from response.")
                triplets = filter_invalid_triples(triples=extracted_triples)
                return TripleRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    metadata=metadata,
                    triples=triplets
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "Exception for chunk %s (attempt %d/%d): %s",
                    chunk_key,
                    attempt + 1,
                    attempts,
                    e,
                )
                metadata.update({'error': str(e), 'parse_retry': attempt + 1})
                if metadata.get("cache_hit") and attempt < attempts - 1:
                    logger.warning(
                        "Cache hit for chunk %s; skipping additional triple retries.",
                        chunk_key,
                    )
                    break
                if attempt < attempts - 1:
                    continue

        metadata.update({'error': str(last_error)})
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=[]
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                                chunk_passages[ner_result.chunk_id],
                                ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict
