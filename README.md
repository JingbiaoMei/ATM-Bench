# ATM-Bench: Long-Term Personalized Referential Memory QA

[![arXiv](https://img.shields.io/badge/arXiv-2603.01990-b31b1b.svg)](https://arxiv.org/abs/2603.01990)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official code for ATM-Bench: a benchmark for long-term multimodal personalized AI memory QA and retrieval.

**ATM-Bench** is the first benchmark for **multimodal, multi-source personalized referential memory QA** over long time horizons (~4 years) with **evidence-grounded** retrieval and answering.

> **Paper:** [According to Me: Long-Term Personalized Referential Memory QA](https://arxiv.org/abs/2603.01990)  
> **Project Page:** [https://atmbench.github.io/](https://atmbench.github.io/)

## Table of Contents

- [ATM-Bench: Long-Term Personalized Referential Memory QA](#atm-bench-long-term-personalized-referential-memory-qa)
  - [Table of Contents](#table-of-contents)
  - [🗓️ Timeline](#️-timeline)
  - [🤖 General-Purpose Agent Results](#-general-purpose-agent-results)
  - [📊 Oracle and NIAH Results](#-oracle-and-niah-results)
    - [Oracle on ATM-Bench-Hard](#oracle-on-atm-bench-hard)
    - [NIAH on ATM-Bench-Hard](#niah-on-atm-bench-hard)
  - [📋 Overview](#-overview)
  - [Memory Ingestion](#memory-ingestion)
    - [Memory Preprocessing](#memory-preprocessing)
    - [Memory Organization](#memory-organization)
  - [NIAH Evaluation Setup](#niah-evaluation-setup)
  - [🚀 Quick Start](#-quick-start)
    - [Installation](#installation)
    - [API Keys](#api-keys)
    - [Generate Memory Files First](#generate-memory-files-first)
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
- **2026-03-12:** ATM-Bench data release on Hugging Face ([Jingbiao/ATM-Bench](https://huggingface.co/datasets/Jingbiao/ATM-Bench)).
- **Coming soon:** General-Purpose Agents benchmarking support, including OpenClaw.

<a id="General-Purpose-Agent-results"></a>
## 🤖 General-Purpose Agent Results

Initial General-Purpose Agent results on ATM-Bench-Hard are summarized below. The QS score here uses `gpt-5-mini` as the primary judge.

| Agent | Model | QS | Total Tokens |
|-------|-------|----|--------------|
| Claude Code | Claude Opus 4.6 | 0.338 | 4.93M |
| Codex | GPT-5.2 | 0.397 | 15.46M |
| OpenCode | GLM-5 | 0.297 | 0.95M |
| OpenCode | Kimi K2.5 | 0.303 | 0.86M |
| OpenCode | MiniMax M2.5 | 0.229 | 1.06M |

The coding agents still struggle on ATM-Bench-Hard, although they perform much better than various agentic memory baselines.

<a id="oracle-and-niah-results"></a>
## 📊 Oracle and NIAH Results


### Oracle on ATM-Bench-Hard

QS is reported with `gpt-5-mini` as the primary judge.

| Model | Setting | QS |
|-------|---------|----|
| GPT-5 | Reasoning Low + Raw | 72.12% |
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

### Generate Memory Files First

Before running `MMRAG` or `Oracle`, generate the image/video `batch_results.json` files:

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
bash scripts/QA_Agent/MMRAG/run.sh

# Oracle (upper bound; raw multimodal evidence)
bash scripts/QA_Agent/Oracle/run_oracle_qwen3vl8b_raw.sh

```

### Baseline Compatibility and Environments

- Core baselines (`MMRAG`, `Oracle`, `NIAH`) are tested in the main `atmbench` environment.
- Third-party memory-system baselines in this repo include:
  - `A-Mem`
  - `HippoRAG2`
  - `mem0`
  - `MemoryOS`
- `MemoryOS` is strongly recommended to run in a separate conda environment.
- `A-Mem`, `HippoRAG2`, and `mem0` are tested to be compatible with the core baseline environment, but separate environments are still safer for reproducibility and dependency isolation.
- Setup references for these baselines are under `third_party/`:
  - `third_party/A-mem/`
  - `third_party/HippoRAG/`
  - `third_party/mem0/`
  - `third_party/MemoryOS/`
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
