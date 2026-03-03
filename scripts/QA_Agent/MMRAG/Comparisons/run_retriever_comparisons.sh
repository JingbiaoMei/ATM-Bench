#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "ATMBench MMRAG - Retriever/Reranker Comparisons"
echo "=============================================="
echo "Start time: $(date)"
echo ""

run_script() {
    local script="$1"
    local name
    name="$(basename "${script}" .sh)"

    echo ""
    echo ">>> Running: ${name}"
    echo ">>> Start: $(date)"
    local start_time
    start_time=$(date +%s)

    bash "${script}"

    local end_time
    end_time=$(date +%s)
    local elapsed
    elapsed=$((end_time - start_time))
    echo ">>> Completed: ${name} (${elapsed}s)"
    echo ""
}

echo "=== Text Embedding (no rerank) ==="
run_script "${SCRIPT_DIR}/text_embed/allminilm_l6_qwen3vl8b.sh"
run_script "${SCRIPT_DIR}/text_embed/qwen3emb_0.6b_qwen3vl8b.sh"
run_script "${SCRIPT_DIR}/text_embed/qwen3emb_4b_qwen3vl8b.sh"

echo "=== Text Embedding + Rerank ==="
run_script "${SCRIPT_DIR}/text_embed_rerank/allminilm_l6_qwen3rerank0.6b_qwen3vl8b.sh"
run_script "${SCRIPT_DIR}/text_embed_rerank/qwen3emb_0.6b_qwen3rerank0.6b_qwen3vl8b.sh"
run_script "${SCRIPT_DIR}/text_embed_rerank/qwen3emb_0.6b_qwen3rerank4b_qwen3vl8b.sh"

echo "=== VL Embedding ==="
run_script "${SCRIPT_DIR}/vl_embed/qwen3vlemb2b_notext_qwen3vl8b.sh"
run_script "${SCRIPT_DIR}/vl_embed/qwen3vlemb2b_withtext_qwen3vl8b.sh"

echo ""
echo "=============================================="
echo "All retriever/reranker comparison runs complete"
echo "End time: $(date)"
echo "=============================================="
