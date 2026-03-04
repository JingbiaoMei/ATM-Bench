# ATM-Bench: Long-Term Personalized Referential Memory QA

[![arXiv](https://img.shields.io/badge/arXiv-2603.01990-b31b1b.svg)](https://arxiv.org/abs/2603.01990)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**ATM-Bench** is the first benchmark for **multimodal, multi-source personalized referential memory QA** over long time horizons (~4 years) with **evidence-grounded** retrieval and answering.

> **Paper:** [According to Me: Long-Term Personalized Referential Memory QA](https://arxiv.org/abs/2603.01990)  

## 🗓️ Timeline

- **2026-03-03:** arXiv paper release ([2603.01990](https://arxiv.org/abs/2603.01990))
- **2026-03-04:** Initial codebase release, including baseline implementations for MMRAG, Oracle, and NIAH.
- **Coming soon:** ATM-Bench data release
- **Coming soon:** Implementations for benchmarking on OpenClaw, Codex, and OpenCode

## 📋 Overview

Existing long-term memory benchmarks focus primarily on dialogue history, failing to capture realistic personalized references grounded in lived experience. ATM-Bench addresses this gap with:

- 🖼️ **Multimodal and multi-source data:** Images, videos, emails
- 📅 **Long-term horizon:** ~4 years of personal memory
- 🎯 **Referential queries:** Resolving personalized references (e.g., "the restaurant I visited last month")
- 🔍 **Evidence-grounded:** Human-annotated QA pairs with ground-truth memory evidence
- 🧩 **Multi-evidence reasoning:** Queries requiring evidence from multiple sources
- ⚡ **Conflicting evidence:** Handling contradictory information

![ATM-Bench Overview](docs/images/ATM-Bench-Demo.png)

## Memory Ingestion

**Memory Ingestion** is decomposed into:

1. **Memory preprocessing** (how each memory item is represented)
2. **Memory organization** (how items are structured/linked)

![ATM Method](docs/images/ATM-Method.png)
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

## NIAH Evaluation Setup

In addition to end-to-end retrieval+generation evaluation, we provide **NIAH (Needle In A Haystack)**:

- Each question is paired with a fixed evidence pool (`niah_evidence_ids`) that contains all ground-truth items.
- The rest of the pool is filled with realistic distractors.
- This isolates answer generation/reasoning quality from retrieval quality.

See:
- [`docs/niah.md`](docs/niah.md)
- [`memqa/qa_agent_baselines/NIAH/README.md`](memqa/qa_agent_baselines/NIAH/README.md)
- [`scripts/QA_Agent/NIAH/README.md`](scripts/QA_Agent/NIAH/README.md)


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

### Run Baselines

```bash
# MMRAG (runs both ATM-bench and ATM-bench-hard)
bash scripts/QA_Agent/MMRAG/run.sh

# Oracle (upper bound)
bash scripts/QA_Agent/Oracle/run_oracle_qwen3vl8b.sh

```

For detailed setup, data layout, and reproducibility settings, see:
- [`docs/README.md`](docs/README.md)
- [`docs/data.md`](docs/data.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/baseline.md`](docs/baseline.md)
- [`docs/niah.md`](docs/niah.md)

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

## 📚 Documentation

- [`docs/README.md`](docs/README.md) - Getting started guide
- [`docs/data.md`](docs/data.md) - Data format and preparation
- [`docs/baseline.md`](docs/baseline.md) - Baseline implementations
- [`docs/niah.md`](docs/niah.md) - NIAH protocol and usage
- [`docs/metrics.md`](docs/metrics.md) - Evaluation metrics
- [`docs/reproducibility.md`](docs/reproducibility.md) - Reproduction instructions
- [`docs/repo_structure.md`](docs/repo_structure.md) - Repository organization

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

## 🔗 Links

- 📄 **Paper:** https://arxiv.org/abs/2603.01990
- 💻 **Code:** https://github.com/JingbiaoMei/ATM-Bench
- 🐛 **Issues:** https://github.com/JingbiaoMei/ATM-Bench/issues

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
