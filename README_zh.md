[English](README.md) | [中文](README_zh.md)

<a id="atm-bench-zh"></a>
# ATM-Bench：长期个性化参照记忆问答

[![arXiv](https://img.shields.io/badge/arXiv-2603.01990-b31b1b.svg)](https://arxiv.org/abs/2603.01990)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/datasets/Jingbiao/ATM-Bench)

ATM-Bench 官方代码：面向长期多模态个性化 AI 记忆问答与检索的基准测试。

**ATM-Bench** 是首个针对**多模态、多来源个性化参照记忆问答**的基准，涵盖约 4 年的长时间跨度，支持**基于证据的**检索与回答。

<video src="https://atmbench.github.io/static/videos/ATM-Bench-demo.mp4" controls width="100%"></video>

> **论文：** [According to Me: Long-Term Personalized Referential Memory QA](https://arxiv.org/abs/2603.01990)
> **项目主页：** [https://atmbench.github.io/](https://atmbench.github.io/)

<a id="table-of-contents-zh"></a>
## 目录

- [ATM-Bench：长期个性化参照记忆问答](#atm-bench-zh)
  - [目录](#table-of-contents-zh)
  - [时间线](#timeline-zh)
  - [通用智能体结果](#general-purpose-agent-results-zh)
  - [Oracle 与 NIAH 结果](#oracle-and-niah-results-zh)
  - [概述](#overview-zh)
  - [记忆摄入](#memory-ingestion-zh)
  - [NIAH 评估设置](#niah-evaluation-setup-zh)
  - [快速开始](#quick-start-zh)
  - [仓库结构](#repository-structure-zh)
  - [文档](#documentation-zh)
  - [引用](#citation-zh)
  - [链接](#links-zh)
  - [许可证](#license-zh)

<a id="timeline-zh"></a>
## 时间线

- **2026-03-03：** arXiv 论文发布（[2603.01990](https://arxiv.org/abs/2603.01990)）
- **2026-03-04：** 初始代码发布，包含 MMRAG、Oracle、NIAH 基线实现，以及四个移植的第三方基线（A-Mem、HippoRAG2、mem0、MemoryOS）。
- **2026-03-12：** 首批通用智能体基准结果发布，涵盖 Claude Code、Codex 和 OpenCode。
- **2026-03-12：** ATM-Bench 数据集在 Hugging Face 发布（[Jingbiao/ATM-Bench](https://huggingface.co/datasets/Jingbiao/ATM-Bench)）。
- **2026-03-13：** 修复 Opencode Token 统计并更新 OpenClaw 结果。
- **即将推出：** 通用智能体基准支持，包括 OpenClaw。

<a id="general-purpose-agent-results-zh"></a>
## 通用智能体结果

ATM-Bench-Hard 上的初始通用智能体结果如下。QS 分数使用 `gpt-5-mini` 作为主要评判模型。

| 智能体 | 模型 | QS | 总 Token 数 |
|--------|------|----|-------------|
| Claude Code | Claude Opus 4.6 | 0.338 | 4.93M |
| Codex | GPT-5.2 | 0.397 | 15.46M |
| OpenCode | GLM-5 | 0.270 | 16.89M |
| OpenCode | Kimi K2.5 | 0.303 | 8.46M |
| OpenCode | MiniMax M2.5 | 0.229 | 14.5M |
| OpenClaw | Kimi K2.5 | 0.254 | 9.63M |

编程智能体在 ATM-Bench-Hard 上仍然表现不佳，但显著优于各种智能体记忆基线。

<a id="oracle-and-niah-results-zh"></a>
## Oracle 与 NIAH 结果

### ATM-Bench-Hard 上的 Oracle 结果

QS 使用 `gpt-5-mini` 作为主要评判模型。

| 模型 | 设置 | QS |
|------|------|----|
| GPT-5 | Raw | 72.12% |
| Qwen3-VL-8B-Instruct | Raw | 40.14% |
| Qwen3-VL-8B-Instruct | SGM | 27.98% |
| Qwen3-VL-8B-Instruct | D | 21.69% |

### ATM-Bench-Hard 上的 NIAH 结果

对于 NIAH，我们比较了 `Qwen3-VL-8B-Instruct` 在不同 haystack 规模下的 SGM 和 Raw 设置。

| 模型 | 设置 | QS | 平均上下文 Token 数 |
|------|------|----|---------------------|
| Qwen3-VL-8B-Instruct | Raw, Oracle | 40.14% | 5.7k |
| Qwen3-VL-8B-Instruct | Raw, NIAH-25 | 25.43% | 15.9k |
| Qwen3-VL-8B-Instruct | Raw, NIAH-50 | 24.87% | 29.0k |
| Qwen3-VL-8B-Instruct | Raw, NIAH-100 | 10.90% | 56.0k |
| Qwen3-VL-8B-Instruct | SGM, Oracle | 27.98% | 4.6k |
| Qwen3-VL-8B-Instruct | SGM, NIAH-25 | 16.33% | 12.5k |
| Qwen3-VL-8B-Instruct | SGM, NIAH-50 | 15.77% | 23.9k |
| Qwen3-VL-8B-Instruct | SGM, NIAH-100 | 12.66% | 45.8k |

<a id="overview-zh"></a>
## 概述

现有的长期记忆基准主要关注对话历史，无法捕捉基于真实生活经验的个性化参照。ATM-Bench 通过以下特性填补了这一空白：

- **多模态与多来源数据：** 图像、视频、邮件
- **长时间跨度：** 约 4 年的个人记忆
- **参照性查询：** 解析个性化参照（如"展示 Grace 试图偷偷摸摸的那些瞬间……"）
- **基于证据：** 人工标注的问答对，配有真实记忆证据
- **多证据推理：** 需要来自多个来源的证据的查询
- **冲突证据：** 处理矛盾信息

![ATM-Bench 概述](docs/images/ATM-Bench-Demo.png)

<a id="memory-ingestion-zh"></a>
## 记忆摄入

**记忆摄入**分为两个步骤：

1. **记忆预处理**（每条记忆项的表示方式）
2. **记忆组织**（记忆项的结构化/关联方式）

<p align="center">
  <img src="docs/images/ATM-Method.png" alt="ATM 方法" width="50%" />
</p>

### 记忆预处理

我们比较了两种预处理表示：

- **描述式记忆（DM）：** 每条记忆项用一段自然语言描述表示。
- **模式引导记忆（SGM）：** 每条记忆项用固定的文本键值字段和模式表示。

在 SGM 中，模式字段与模态相关。例如：

- **图像/视频记忆：** `time`、`location`、`entities`、`ocr`、`tags`
- **邮件记忆：** `time`、`summary`、`body`

DM 和 SGM 包含相同的底层信息，但使用不同的格式。

在本代码库中，DM 以描述/标题风格的文本实现，SGM 以基于模式的键值文本字段实现。

### 记忆组织

记忆存储的组织方式：

- **堆叠记忆：** 记忆项无显式关联地存储。
- **链接记忆：** 记忆项通过推断的关系链接（图结构）；智能体系统还可以在组织过程中更新现有记忆项。

<a id="niah-evaluation-setup-zh"></a>
## NIAH 评估设置

除了端到端的检索+生成评估外，我们还提供了 **NIAH（大海捞针）** 评估：

- 每个问题配有固定的证据池（`niah_evidence_ids`），包含所有真实记忆项。
- 池中其余部分由真实干扰项填充。
- 这将答案生成/推理质量与检索质量隔离开来。

参见：
- [`docs/niah.md`](docs/niah.md)

<a id="quick-start-zh"></a>
## 快速开始

### 安装

```bash
conda create -n atmbench python=3.11 -y
conda activate atmbench
pip install -r requirements.txt
pip install -e .
```

### API 密钥

通过环境变量设置：
```bash
export OPENAI_API_KEY="your-key"
export VLLM_API_KEY="your-key"
```

或使用本地密钥文件（已 gitignore）：
- `api_keys/.openai_key`
- `api_keys/.vllm_key`

### 首先生成记忆文件

在运行 `MMRAG` 或 `Oracle` 之前，先生成图像/视频的 `batch_results.json` 文件：

```bash
# 可选但推荐：预加载反向地理编码缓存
# 缓存文件以媒体文件名为键，因此缓存包必须与当前图像/视频文件名匹配。
bash scripts/memory_processor/image/copy_gps_cache.sh output/image/qwen3vl2b/cache
bash scripts/memory_processor/video/copy_gps_cache.sh output/video/qwen3vl2b/cache

# 生成记忆项化结果
bash scripts/memory_processor/image/memory_itemize/run_qwen3vl2b.sh
bash scripts/memory_processor/video/memory_itemize/run_qwen3vl2b.sh
```

### 快速命令（MMRAG + Oracle）

```bash
# MMRAG（同时运行 ATM-bench 和 ATM-bench-hard）
bash scripts/QA_Agent/MMRAG/run.sh

# Oracle（上界；原始多模态证据）
bash scripts/QA_Agent/Oracle/run_oracle_qwen3vl8b_raw.sh
```

### 基线兼容性与环境

- 核心基线（`MMRAG`、`Oracle`、`NIAH`）在主 `atmbench` 环境中测试。
- 本仓库中的第三方记忆系统基线包括：
  - `A-Mem`
  - `HippoRAG2`
  - `mem0`
  - `MemoryOS`
- 强烈建议在独立的 conda 环境中运行 `MemoryOS`。
- `A-Mem`、`HippoRAG2` 和 `mem0` 经测试与核心基线环境兼容，但为确保可复现性和依赖隔离，仍建议使用独立环境。
- 这些基线的设置参考位于 `third_party/` 下：
  - `third_party/A-mem/`
  - `third_party/HippoRAG/`
  - `third_party/mem0/`
  - `third_party/MemoryOS/`
- OpenClaw 支持已在规划中；我们将很快发布所有通用智能体（Claude Code、Codex、OpenCode、OpenClaw）在 ATM-Bench 上的评估设置。

详细的设置、数据布局和可复现性设置，请参见：
- [`docs/README.md`](docs/README.md)
- [`docs/data.md`](docs/data.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/baseline.md`](docs/baseline.md)
- [`docs/niah.md`](docs/niah.md)

<a id="repository-structure-zh"></a>
## 仓库结构

```
ATMBench/
├── memqa/              # 核心记忆问答实现
├── scripts/            # 实验脚本
├── docs/               # 文档
├── data/               # 数据目录（用户提供）
├── third_party/        # 外部智能体记忆系统
└── output/             # 实验输出（已 gitignore）
```

<a id="documentation-zh"></a>
## 文档

- [`docs/README.md`](docs/README.md) - 入门指南
- [`docs/data.md`](docs/data.md) - 数据格式与准备
- [`docs/baseline.md`](docs/baseline.md) - 基线实现
- [`docs/niah.md`](docs/niah.md) - NIAH 协议与使用
- [`docs/metrics.md`](docs/metrics.md) - 评估指标
- [`docs/reproducibility.md`](docs/reproducibility.md) - 复现说明
- [`docs/repo_structure.md`](docs/repo_structure.md) - 仓库组织

<a id="citation-zh"></a>
## 引用

如果您在研究中使用了 ATM-Bench，请引用：

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

<a id="links-zh"></a>
## 链接

- **论文：** https://arxiv.org/abs/2603.01990
- **数据集：** https://huggingface.co/datasets/Jingbiao/ATM-Bench
- **代码：** https://github.com/JingbiaoMei/ATM-Bench
- **问题反馈：** https://github.com/JingbiaoMei/ATM-Bench/issues

<a id="license-zh"></a>
## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
