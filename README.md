[English](README.md) | [中文](README_zh.md)

# ATM-Bench: Long-Term Personalized Referential Memory QA

[![arXiv](https://img.shields.io/badge/arXiv-2603.01990-b31b1b.svg)](https://arxiv.org/abs/2603.01990)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/datasets/Jingbiao/ATM-Bench)

Official code for ATM-Bench: a benchmark for long-term multimodal personalized AI memory QA and retrieval.

**ATM-Bench** is the first benchmark for **multimodal, multi-source personalized referential memory QA** over long time horizons (~4 years) with **evidence-grounded** retrieval and answering.

<video src="https://atmbench.github.io/static/videos/ATM-Bench-demo.mp4" controls width="100%"></video>

> **Paper:** [According to Me: Long-Term Personalized Referential Memory QA](https://arxiv.org/abs/2603.01990)  
> **Project Page:** [https://atmbench.github.io/](https://atmbench.github.io/)

## Table of Contents

- [ATM-Bench: Long-Term Personalized Referential Memory QA](#atm-bench-long-term-personalized-referential-memory-qa)
  - [Table of Contents](#table-of-contents)
  - [🗓️ Timeline](#️-timeline)
  - [🤖 General-Purpose Agent Results](#-general-purpose-agent-results)
  - [🧠 Memory-System Baseline Results](#-memory-system-baseline-results)
  - [📊 Oracle and NIAH Results](#-oracle-and-niah-results)
    - [Oracle on ATM-Bench-Hard](#oracle-on-atm-bench-hard)
    - [NIAH on ATM-Bench-Hard](#niah-on-atm-bench-hard)
  - [📋 Overview](#-overview)
  - [Memory Ingestion](#memory-ingestion)
    - [Memory Preprocessing](#memory-preprocessing)
    - [Memory Organization](#memory-organization)
  - [NIAH Evaluation Setup](#niah-evaluation-setup)
  - [🚀 Quick Start](#-quick-start)
    - [Download Dataset](#download-dataset)
    - [Installation](#installation)
    - [API Keys](#api-keys)
    - [Prepare Memory Files](#prepare-memory-files)
    - [Quick commands (MMRAG + Oracle)](#quick-commands-mmrag--oracle)
    - [Baseline Compatibility and Environments](#baseline-compatibility-and-environments)
  - [📁 Repository Structure](#-repository-structure)
  - [📚 Documentation](#-documentation)
  - [📖 Citation](#-citation)
  - [🔗 Links](#-links)
  - [📝 License](#-license)

<a id="timeline"></a>
## 🗓️ Timeline

- **2026-03-03:** arXiv paper release ([2603.01990](https://arxiv.org/abs/2603.01990))
- **2026-03-04:** Initial codebase release, including baseline implementations for MMRAG, Oracle, NIAH, and four ported third-party baselines (A-Mem, HippoRAG2, mem0, MemoryOS).
- **2026-03-12:** Initial General-Purpose Agent benchmark results release for Claude Code, Codex, and OpenCode.
- **2026-03-12:** ATM-Bench data release on Hugging Face ([ATM-Bench](https://huggingface.co/datasets/Jingbiao/ATM-Bench)).
- **2026-03-13:** Fixed Opencode Token Accounting and updated OpenClaw results.
- **2026-05-15:** Released the MemPalace port and added memory-system comparison results.
- **Coming soon:** General-Purpose Agents benchmarking support, including OpenClaw.

<a id="General-Purpose-Agent-results"></a>
## 🤖 General-Purpose Agent Results

Initial General-Purpose Agent results on ATM-Bench-Hard are summarized below. The QS score here uses `gpt-5-mini` as the primary judge. `Tokens/QS` shows the token cost per percentage point of QS, so lower is more efficient.

| Agent | Model | QS (Acc.) ↑ | Total Tokens ↓ | Tokens/QS ↓ |
|-------|-------|------------:|---------------:|-------------:|
| Claude Code | Claude Opus 4.6 | 33.80% | 4.93M | 0.146M |
| Codex | GPT-5.2 | 39.70% | 15.46M | 0.389M |
| Codex | GPT-5.4* | 29.60% | 14.29M | 0.483M |
| OpenCode | GLM-5 | 27.00% | 16.89M | 0.626M |
| OpenCode | Qwen3.5-397B-A17B | 24.50% | 12.06M | 0.492M |
| OpenCode | Kimi K2.5 | 30.30% | 8.46M | 0.279M |
| OpenCode | MiniMax M2.5 | 22.90% | 14.5M | 0.633M |
| OpenCode | MiniMax M2.7 | 27.80% | 13.48M | 0.485M |
| OpenClaw 🦞 | Kimi K2.5 | 25.40% | 9.63M | 0.379M |

* `GPT-5.4` results may be unreliable because the Codex service was unstable during evaluation.

The coding agents still struggle on ATM-Bench-Hard, although they perform much better than various agentic memory baselines.

<a id="memory-system-baseline-results"></a>
## 🧠 Memory-System Baseline Results

Memory-system baselines below use `Qwen3-VL-8B-Instruct-FP8` as the answerer, `Qwen3-VL-2B-Instruct` as the memory processor. ATM-Bench-Hard uses the `atm-bench-hard` release set, the results may differ from the original preprint.

| System | Index Time (hr) ↓ | ATM-Bench QS ↑ | ATM-Bench Recall@10 ↑ | ATM-Bench-Hard QS ↑ | ATM-Bench-Hard Recall@10 ↑ |
|--------|------------------:|---------------:|----------------------:|--------------------:|---------------------------:|
| [A-Mem](https://github.com/WujiangXu/A-mem) | 12.6 | 44.8 | 66.4 | 9.9 | 31.7 |
| [mem0](https://github.com/mem0ai/mem0) | 16.7 | 43.5 | 61.9 | 9.2 | 23.7 |
| [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | 36.6 | 47.2 | 59.2 | 13.7 | 32.7 |
| [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG) | 1.5 | 42.9 | 66.4 | 9.4 | 31.9 |
| [MemPalace](https://github.com/MemPalace/mempalace) | 0.5 | 56.8 | 76.4 | 9.7 | 28.3 |

<a id="oracle-and-niah-results"></a>
## 📊 Oracle and NIAH Results


### Oracle on ATM-Bench-Hard

QS is reported with `gpt-5-mini` as the primary judge.

| Model | Setting | QS |
|-------|---------|----|
| GPT-5 | Raw | 72.12% |
| Qwen3-VL-8B-Instruct | Raw | 40.14% |
| Qwen3-VL-8B-Instruct | SGM | 27.98% |
| Qwen3-VL-8B-Instruct | D | 21.69% |

### NIAH on ATM-Bench-Hard

For NIAH, we compare the `Qwen3-VL-8B-Instruct` SGM and Raw settings at different haystack sizes.

| Model | Setting | QS | Avg. Context Tokens |
|-------|---------|----|---------------------|
| Qwen3-VL-8B-Instruct | Raw, Oracle | 40.14% | 5.7k |
| Qwen3-VL-8B-Instruct | Raw, NIAH-25 | 25.43% | 15.9k |
| Qwen3-VL-8B-Instruct | Raw, NIAH-50 | 24.87% | 29.0k |
| Qwen3-VL-8B-Instruct | Raw, NIAH-100 | 10.90% | 56.0k |
| Qwen3-VL-8B-Instruct | SGM, Oracle | 27.98% | 4.6k |
| Qwen3-VL-8B-Instruct | SGM, NIAH-25 | 16.33% | 12.5k |
| Qwen3-VL-8B-Instruct | SGM, NIAH-50 | 15.77% | 23.9k |
| Qwen3-VL-8B-Instruct | SGM, NIAH-100 | 12.66% | 45.8k |


<a id="overview"></a>
## 📋 Overview

Existing long-term memory benchmarks focus primarily on dialogue history, failing to capture realistic personalized references grounded in lived experience. ATM-Bench addresses this gap with:

- 🖼️ **Multimodal and multi-source data:** Images, videos, emails
- 📅 **Long-term horizon:** ~4 years of personal memory
- 🎯 **Referential queries:** Resolving personalized references (e.g., "Show me the moments where Grace was trying to be sneaky...")
- 🔍 **Evidence-grounded:** Human-annotated QA pairs with ground-truth memory evidence
- 🧩 **Multi-evidence reasoning:** Queries requiring evidence from multiple sources
- ⚡ **Conflicting evidence:** Handling contradictory information

![ATM-Bench Overview](docs/images/ATM-Bench-Demo.png)

<a id="memory-ingestion"></a>
## Memory Ingestion

**Memory Ingestion** is decomposed into:

1. **Memory preprocessing** (how each memory item is represented)
2. **Memory organization** (how items are structured/linked)

<p align="center">
  <img src="docs/images/ATM-Method.png" alt="ATM Method" width="50%" />
</p>

### Memory Preprocessing
We compare two preprocessing representations:

- **Descriptive Memory (DM):** each memory item is represented as one natural-language description.
- **Schema-Guided Memory (SGM):** each memory item is represented with fixed text-based key-value fields under a schema.

In SGM, schema fields are modality-aware. For example:

- **Image/Video memory:** `time`, `location`, `entities`, `ocr`, `tags`
- **Email memory:** `time`, `summary`, `body`

DM and SGM contain the same underlying information but use different formats.

In this codebase, DM is implemented as caption/description-style text, while SGM is implemented as schema-based key-value text fields.

### Memory Organization
For organization of the memory store:

- **Piled Memory:** items are stored without explicit links.
- **Linked Memory:** items are linked with inferred relations (graph structure); agentic systems can additionally update existing items during organization.

<a id="niah-evaluation-setup"></a>
## NIAH Evaluation Setup

In addition to end-to-end retrieval + generation evaluation, we provide **NIAH (Needle In A Haystack)**:

- Each question is paired with a fixed evidence pool (`niah_evidence_ids`) that contains all ground-truth items.
- The rest of the pool is filled with realistic distractors.
- This isolates answer generation/reasoning quality from retrieval quality.

See:
- [`docs/niah.md`](docs/niah.md)


<a id="quick-start"></a>
## 🚀 Quick Start

### Download Dataset

ATM-Bench is hosted on Hugging Face at [`Jingbiao/ATM-Bench`](https://huggingface.co/datasets/Jingbiao/ATM-Bench). A one-shot script downloads the full released dataset and stages the files where the evaluation scripts expect them.

**Full download (~3.3 GB)** — includes QA, NIAH pools, preprocessed memory, emails, raw images, raw videos, and the GPS reverse-geocoding cache:

```bash
bash scripts/download_data.sh
```

This populates:

```
data/atm-bench/atm-bench.json
data/atm-bench/atm-bench-hard.json
data/atm-bench/niah/...
data/raw_memory/email/emails.json                   # emails
data/raw_memory/image/...                           # raw images
data/raw_memory/video/...                           # raw videos
data/raw_memory/geocoding_cache/...                 # GPS reverse-geocoding cache
output/image/qwen3vl2b/batch_results.json           # preprocessed image memory
output/video/qwen3vl2b/batch_results.json           # preprocessed video memory
```

The HF files `data/processed_memory/{image,video}_batch_results.json` are automatically renamed/copied into `output/image/qwen3vl2b/batch_results.json` and `output/video/qwen3vl2b/batch_results.json` by the script.

The script uses the `huggingface_hub` Python package (installed automatically if missing). If the dataset is private, run `huggingface-cli login` first.

### Installation

```bash
conda create -n atmbench python=3.11 -y
conda activate atmbench
pip install -r requirements.txt
pip install -e .
```

### API Keys

Set via environment variables:
```bash
export OPENAI_API_KEY="your-key"
export VLLM_API_KEY="your-key"
```

Or use local key files (gitignored):
- `api_keys/.openai_key`
- `api_keys/.vllm_key`

### Prepare Memory Files

Before running baselines, the image/video `batch_results.json` files must exist under `output/{image,video}/qwen3vl2b/`. You have two options:

**Option A (recommended): download the preprocessed memory from Hugging Face.**

If you already ran `bash scripts/download_data.sh` above, the preprocessed memory files are already staged at:

- `output/image/qwen3vl2b/batch_results.json`
- `output/video/qwen3vl2b/batch_results.json`

Nothing more to do — you can skip straight to the Quick commands.

**Option B: regenerate the memory files from raw images/videos.**

Only needed if you want to re-run preprocessing (for example, to try a different VLM or your own raw memory). Requires raw images under `data/raw_memory/image/` and videos under `data/raw_memory/video/`:

```bash
# Optional but recommended: preload reverse-geocoding cache
# Cache files are keyed by media filename stem, so the cache bundle must match
# the current image/video filenames.
bash scripts/memory_processor/image/copy_gps_cache.sh output/image/qwen3vl2b/cache
bash scripts/memory_processor/video/copy_gps_cache.sh output/video/qwen3vl2b/cache

# Generate memory itemization results
bash scripts/memory_processor/image/memory_itemize/run_qwen3vl2b.sh
bash scripts/memory_processor/video/memory_itemize/run_qwen3vl2b.sh
```


### Quick commands (MMRAG + Oracle)

```bash
# MMRAG (runs both ATM-bench and ATM-bench-hard)
#   Needs: `bash scripts/download_data.sh`
#        + a running vLLM endpoint at http://127.0.0.1:8000/v1/chat/completions
#          serving Qwen/Qwen3-VL-8B-Instruct-FP8 (override with VLLM_ENDPOINT /
#          ANSWERER_MODEL env vars).
bash scripts/QA_Agent/MMRAG/run.sh

# Oracle with Qwen3-VL-8B on raw images/videos (local upper bound)
#   Needs: `bash scripts/download_data.sh`
#        + a running vLLM endpoint serving Qwen/Qwen3-VL-8B-Instruct-FP8.
bash scripts/QA_Agent/Oracle/run_oracle_qwen3vl8b_raw.sh

# Oracle with GPT-5 on raw images/videos (no local GPU / vLLM)
#   Needs: `bash scripts/download_data.sh`
#        + OPENAI_API_KEY set in the environment or api_keys/.openai_key.
bash scripts/QA_Agent/Oracle/run_oracle_gpt5.sh
```

### Baseline Compatibility and Environments

- Core baselines (`MMRAG`, `Oracle`, `NIAH`) are tested in the main `atmbench` environment.
- Third-party memory-system baselines in this repo include:
  - `A-Mem`
  - `HippoRAG2`
  - `mem0`
  - `MemoryOS`
  - `MemPalace`
- `MemoryOS` and `MemPalace` are strongly recommended to run in separate conda environments. `MemoryOS` uses a FAISS / sentence-transformers stack, while `MemPalace` uses ChromaDB / ONNX-backed local embeddings; isolating them avoids dependency collisions with the core baseline environment and each other.
- `A-Mem`, `HippoRAG2`, and `mem0` are tested to be compatible with the core baseline environment, but separate environments are still safer for reproducibility and dependency isolation.
- Setup references for the vendored baselines are under `third_party/`:
  - `third_party/A-mem/`
  - `third_party/HippoRAG/`
  - `third_party/mem0/`
  - `third_party/MemoryOS/`
- `MemPalace` ships as a PyPI package (`mempalace==3.3.5`) and is installed via `memqa/qa_agent_baselines/Mempalace/requirements.txt` — no `third_party/` vendoring.
- OpenClaw support is planned; We will shortly release the evaluation setup for all General-Purpose Agents (Claude Code, Codex, OpenCode, OpenClaw) on ATM-Bench.

For detailed setup, data layout, and reproducibility settings, see:
- [`docs/README.md`](docs/README.md)
- [`docs/data.md`](docs/data.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/baseline.md`](docs/baseline.md)
- [`docs/niah.md`](docs/niah.md)

<a id="repository-structure"></a>
## 📁 Repository Structure

```
ATMBench/
├── memqa/              # Core memory QA implementation
├── scripts/            # Experiment scripts
├── docs/               # Documentation
├── data/               # Data directory (user-provided)
├── third_party/        # Vendored agentic memory systems
└── output/             # Experiment outputs (gitignored)
```

<a id="documentation"></a>
## 📚 Documentation

- [`docs/README.md`](docs/README.md) - Getting started guide
- [`docs/data.md`](docs/data.md) - Data format and preparation
- [`docs/baseline.md`](docs/baseline.md) - Baseline implementations
- [`docs/niah.md`](docs/niah.md) - NIAH protocol and usage
- [`docs/metrics.md`](docs/metrics.md) - Evaluation metrics
- [`docs/reproducibility.md`](docs/reproducibility.md) - Reproduction instructions
- [`docs/repo_structure.md`](docs/repo_structure.md) - Repository organization

<a id="citation"></a>
## 📖 Citation

If you use ATM-Bench in your research, please cite:

```bibtex
@article{mei2026atm,
  title={According to Me: Long-Term Personalized Referential Memory QA},
  author={Mei, Jingbiao and Chen, Jinghong and Yang, Guangyu and Hou, Xinyu and Li, Margaret and Byrne, Bill},
  journal={arXiv preprint arXiv:2603.01990},
  year={2026},
  url={https://arxiv.org/abs/2603.01990},
  doi={10.48550/arXiv.2603.01990}
}
```

<a id="links"></a>
## 🔗 Links

- 📄 **Paper:** https://arxiv.org/abs/2603.01990
- 🤗 **Dataset:** https://huggingface.co/datasets/Jingbiao/ATM-Bench
- 💻 **Code:** https://github.com/JingbiaoMei/ATM-Bench
- 🐛 **Issues:** https://github.com/JingbiaoMei/ATM-Bench/issues

<a id="license"></a>
## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
