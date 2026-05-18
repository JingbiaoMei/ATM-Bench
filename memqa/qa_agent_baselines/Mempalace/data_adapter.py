#!/usr/bin/env python3
"""Adapter: convert ATM-Bench RetrievalItems into MemPalace palace format.

Each RetrievalItem becomes a virtual "file" that is chunked into MemPalace drawers
using the upstream chunk_text() function (CHUNK_SIZE=800, CHUNK_OVERLAP=100).

Wing assignment follows modality: email→"email", image→"image", video→"video".
Room assignment uses the item ID for fine-grained filtering.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from memqa.retrieve import RetrievalItem


def build_palace_from_items(
    items: List[RetrievalItem],
    palace_path: str,
    collection_name: Optional[str] = None,
    agent: str = "atmbench",
    batch_size: int = 1000,
    build_closets: bool = True,
) -> Dict[str, Any]:
    """Ingest all RetrievalItems into a MemPalace palace using batched upserts.

    Uses _build_drawer_metadata + batched collection.upsert (aligned with
    MemPalace's process_file approach) for much faster ingestion than
    one-by-one add_drawer. Batch size matches MemPalace's
    DRAWER_UPSERT_BATCH_SIZE=1000.

    Creates the palace collection if needed, then adds drawers for each item.
    When build_closets is enabled, also creates the regex closet collection
    that MemPalace's original process_file() builds after drawer ingestion.
    Returns stats about the ingestion.
    """
    from mempalace.palace import (
        build_closet_lines,
        get_closets_collection,
        get_collection,
        purge_file_closets,
        upsert_closet_lines,
    )
    from mempalace.miner import (
        _build_drawer_metadata,
        _extract_entities_for_metadata,
        chunk_text,
    )

    collection = get_collection(
        palace_path, collection_name=collection_name, create=True
    )
    closets_col = get_closets_collection(palace_path) if build_closets else None

    total_items = 0
    total_drawers = 0
    total_closets = 0
    skipped = 0

    batch_docs: List[str] = []
    batch_ids: List[str] = []
    batch_metas: List[Dict[str, Any]] = []

    def flush_batch():
        if not batch_docs:
            return
        collection.upsert(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_metas,
        )
        batch_docs.clear()
        batch_ids.clear()
        batch_metas.clear()

    for item in items:
        wing = item.modality
        room = item.item_id.replace("/", "_").replace(".", "_")
        source_file = f"atmbench://{item.item_id}"
        content = item.text.strip()

        if not content:
            skipped += 1
            continue

        chunks = chunk_text(content, source_file)
        if not chunks:
            skipped += 1
            continue

        total_items += 1
        drawer_ids: List[str] = []

        for chunk in chunks:
            drawer_id = f"drawer_{wing}_{room}_{hashlib.sha256((source_file + str(chunk['chunk_index'])).encode()).hexdigest()[:24]}"
            metadata = _build_drawer_metadata(
                wing=wing,
                room=room,
                source_file=source_file,
                chunk_index=chunk["chunk_index"],
                agent=agent,
                content=chunk["content"],
                source_mtime=None,
            )
            batch_docs.append(chunk["content"])
            batch_ids.append(drawer_id)
            batch_metas.append(metadata)
            drawer_ids.append(drawer_id)
            total_drawers += 1

            if len(batch_docs) >= batch_size:
                flush_batch()

        if closets_col is not None and drawer_ids:
            closet_lines = build_closet_lines(
                source_file, drawer_ids, content, wing, room
            )
            closet_id_base = (
                f"closet_{wing}_{room}_"
                f"{hashlib.sha256(source_file.encode()).hexdigest()[:24]}"
            )
            closet_meta = {
                "wing": wing,
                "room": room,
                "source_file": source_file,
                "drawer_count": len(drawer_ids),
            }
            entities = _extract_entities_for_metadata(content)
            if entities:
                closet_meta["entities"] = entities
            purge_file_closets(closets_col, source_file)
            total_closets += upsert_closet_lines(
                closets_col, closet_id_base, closet_lines, closet_meta
            )

    flush_batch()

    return {
        "total_items": total_items,
        "total_drawers": total_drawers,
        "total_closets": total_closets,
        "skipped_empty": skipped,
    }


def source_file_to_item_id(source_file: str) -> str:
    """Extract item_id from MemPalace source_file metadata.

    source_file format: "atmbench://<item_id>"
    """
    if source_file.startswith("atmbench://"):
        return source_file[len("atmbench://") :]
    return Path(source_file).stem


def search_results_to_item_ids(
    results: List[Dict[str, Any]],
) -> List[str]:
    """Extract deduplicated item IDs from search results, preserving rank order."""
    seen = set()
    ordered = []
    for hit in results:
        source = hit.get("source_file", "")
        item_id = source_file_to_item_id(source)
        if item_id and item_id not in seen:
            seen.add(item_id)
            ordered.append(item_id)
    return ordered


def search_results_to_scores(
    results: List[Dict[str, Any]],
) -> List[float]:
    """Extract scores aligned with search_results_to_item_ids()."""
    seen = set()
    scores = []
    for hit in results:
        source = hit.get("source_file", "")
        item_id = source_file_to_item_id(source)
        if item_id and item_id not in seen:
            seen.add(item_id)
            scores.append(hit.get("similarity", 0.0))
    return scores
