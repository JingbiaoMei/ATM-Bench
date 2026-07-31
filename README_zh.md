<a id="atm-bench-zh"></a>

<div align="center">

# ATM-Bench：长期个性化参照记忆问答

**首个针对多模态、多来源个性化参照记忆问答的基准，涵盖约 4 年的长时间跨度，支持基于证据的检索与回答。**

[🇬🇧 English](README.md) • [🇨🇳 中文](README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2603.01990-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.01990)
[![Project Page](https://img.shields.io/badge/🌐_项目主页-atmbench.github.io-1f6feb.svg)](https://atmbench.github.io/)
[![Live Leaderboard](https://img.shields.io/badge/🏆_在线榜单-Live-orange.svg)](https://atmbench.github.io/leaderboard.html)
[![Hugging Face](https://img.shields.io/badge/🤗_HuggingFace-Dataset-FFD21E.svg)](https://huggingface.co/datasets/Jingbiao/ATM-Bench)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[🚀 快速开始](#quick-start-zh) • [🤖 智能体结果](#general-purpose-agent-results-zh) • [🧠 记忆系统](#memory-system-baseline-results-zh) • [📊 Oracle / NIAH](#oracle-and-niah-results-zh) • [🏆 在线榜单](https://atmbench.github.io/leaderboard.html) • [📖 引用](#citation-zh)

</div>

<video src="https://atmbench.github.io/static/videos/ATM-Bench-demo.mp4" controls width="100%"></video>

> 📄 **论文：** [According to Me: Long-Term Personalized Referential Memory QA](https://arxiv.org/abs/2603.01990)  
> 🌐 **项目主页：** [https://atmbench.github.io/](https://atmbench.github.io/)  
> 🏆 **在线榜单：** [https://atmbench.github.io/leaderboard.html](https://atmbench.github.io/leaderboard.html)

<a id="table-of-contents-zh"></a>
## 目录

- [时间线](#timeline-zh)
- [通用智能体结果](#general-purpose-agent-results-zh)
- [记忆系统基线结果](#memory-system-baseline-results-zh)
- [Oracle 与 NIAH 结果](#oracle-and-niah-results-zh)
- [概述](#overview-zh)
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
- **2026-05-15：** 发布 MemPalace 移植版本，并加入记忆系统对比结果。
- **2026-05-27：** 发布 SimpleMem 移植版本，并加入记忆系统对比结果。
- **2026-05-28：** 发布 Pi 智能体基准结果。
- **2026-05-30：** 发布通用智能体基准测试框架（`agent_systems/`）——为 Claude Code、Codex、Pi、OpenCode 和 OpenClaw 提供隔离的、按问题独立运行的运行器。
- **2026-06-07：** 更新更多 NIAH 结果与分析，包括多种多模态回答模型在 SGM 与 Raw 设置下的对比。
- **2026-06-12：** 为通用智能体结果新增每次运行的美元成本估算。
- **2026-07-18：** ATM-Bench-Hard (SGM) 以 [Harbor](https://github.com/harbor-framework/harbor) 数据集形式发布——通过 `harbor run -d atm-bench/atm-bench-hard-sgm`，即可在按问题隔离的 Docker 容器中运行任意 Harbor 智能体，完成全部 31 个困难问题（参见 [`agent_systems/HARBOR.md`](agent_systems/HARBOR.md)）。

<a id="general-purpose-agent-results-zh"></a>
## 通用智能体结果

> 🏆 **最新结果请查看 [ATM-Bench 在线榜单](https://atmbench.github.io/leaderboard.html)。** 下方静态表格可能落后于最新提交。

ATM-Bench-Hard 上的通用智能体结果如下。QS 分数使用 `gpt-5-mini` 作为主要评判模型。

### 成本 – 得分

<a href="https://atmbench.github.io/leaderboard.html">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/price-performance-dark.png">
    <img src="docs/images/price-performance-light.png" width="100%"
      alt="ATM-Bench-Hard 得分对运行成本的散点图，成本为对数轴（$0.26 至 $39.74）。折线连接同一模型的各推理强度档位，菱形表示单一配置运行。GPT-5.6 Sol 以 $12.52 取得 58.8%，Kimi K3-256k 以 $5.54 取得 52.5%，GPT-5.6 Luna 以 $1.01 取得 42.4%。">
  </picture>
</a>

每次运行的得分与花费（按 API 标价等价折算）。折线连接同一模型的各推理强度档位，标记大小表示档位，菱形表示单一配置的系统。
**[榜单页面上的该图可交互](https://atmbench.github.io/leaderboard.html)** — 悬停任意数据点可查看 token 数、每分成本，以及 31 题中实际回答的题数。
此图为快照：它覆盖的配置多于下方表格，且个别「更贵但得分更低」的运行未画入折线。

| 智能体 | 模型 | QS (Acc.) ↑ | 总 Token 数 ↓ | 成本 (USD) ↓ |
|--------|------|-------------:|---------------:|------------:|
| Claude Code | Claude Fable 5 (xhigh) | 56.4% | 2.9M | $15.06 |
| Claude Code | Claude Opus 4.7 (max) | 46.6% | 6.9M | $9.58 |
| Claude Code | Claude Opus 4.6 | 33.8% | 4.9M | $8.01 |
| Claude Code | Claude Opus 4.7 | 39.5% | 5.0M | $7.70 |
| Claude Code | Claude Opus 4.7 (w/o SGM) | 23.1% | 17.0M | $19.84 |
| Claude Code | Claude Opus 4.8 | 41.6% | 4.4M | $7.49 |
| Codex | GPT-5.6 Sol (medium) | 58.8% | 7.5M | $12.52 |
| Codex | GPT-5.2 | 39.7% | 15.5M | — |
| Codex | GPT-5.2 (w/o SGM) | 16.3% | 22.2M | $9.22 |
| Codex | GPT-5.5 | 41.4% | 16.1M | $27.17 |
| Codex | GPT-5.5 (xhigh) | 48.1% | 22.9M | $39.74 |
| OpenCode | GLM-5 | 27.0% | 16.9M | $14.92 |
| OpenCode | Qwen3.5-397B-A17B | 24.5% | 12.1M | $4.93 |
| OpenCode | Kimi K2.5 | 30.3% | 8.5M | $1.81 |
| OpenCode | Kimi K2.5 (w/o SGM) | 6.5% | 21.4M | $6.40 |
| OpenCode | MiniMax M2.5 | 22.9% | 14.5M | $4.43 |
| OpenCode | MiniMax M2.7 | 27.8% | 13.5M | $1.36 |
| OpenClaw 🦞 | Kimi K2.5 | 25.4% | 9.6M | $2.37 |
| Pi | GLM-5.1 | 38.8% | 8.2M | $4.33 |
| Pi | Kimi K2.5 | 37.8% | 9.9M | $2.67 |
| Pi | MiMo v2.5 | 36.1% | 18.2M | $2.06 |
| Pi | MiniMax M3 | 43.2% | 15.6M | $3.39 |
| Pi | Qwen3.6-27B | 38.5% | 7.1M | $2.45 |
| Pi | Qwen3.6-27B (w/o SGM) | 16.6% | 20.8M | $6.29 |

> **成本**为一次完整 ATM-Bench-Hard 运行（31 个问题）的 API 等价估算，使用保存的每次调用 token 计数与 Tokdash 内置的标准短上下文费率计算。该数值不代表 Codex 订阅费用。

* 除非模型标签注明 `max` 或 `xhigh` 等 reasoning effort，编程智能体均使用默认配置。

编程智能体在 ATM-Bench-Hard 上仍然表现不佳，但显著优于各种智能体记忆基线。

要复现这些结果，请参见 [`agent_systems/`](agent_systems/README.md) 下的通用智能体测试框架，它为 Claude Code、Codex、Pi、OpenCode 和 OpenClaw 提供隔离的、按问题独立运行的运行器。

<a id="memory-system-baseline-results-zh"></a>
## 记忆系统基线结果

除特别说明外，下面的记忆系统基线使用 `Qwen3-VL-8B-Instruct-FP8` 作为回答模型，`Qwen3-VL-2B-Instruct` 作为共享的图像/视频描述预处理器。ATM-Bench-Hard 使用 `atm-bench-hard` 发布集合，因此结果可能与原始预印本不同。

| 系统 | 索引时间 (hr) ↓ | ATM-Bench QS ↑ | ATM-Bench Recall@10 ↑ | ATM-Bench-Hard QS ↑ | ATM-Bench-Hard Recall@10 ↑ |
|------|----------------:|---------------:|----------------------:|--------------------:|---------------------------:|
| [A-Mem](https://github.com/WujiangXu/A-mem) | 12.6 | 44.8 | 66.4 | 9.9 | 31.7 |
| [mem0](https://github.com/mem0ai/mem0) | 16.7 | 43.5 | 61.9 | 9.2 | 23.7 |
| [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | 36.6 | 47.2 | 59.2 | 13.7 | 32.7 |
| [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG) | 1.5 | 42.9 | 66.4 | 9.4 | 31.9 |
| [MemPalace](https://github.com/MemPalace/mempalace) | 0.5 | 56.8 | 76.4 | 9.7 | 28.3 |
| [SimpleMem](https://github.com/aiming-lab/SimpleMem) | 15.7 | 27.3 | 23.3 | 3.2 | 7.0 |
| [Memexa](https://github.com/labazhou2024/memexa)（DeepSeek-V4-flash 记忆+回答，Qwen3.6-27B captions） | — | 68.0* | 79.1 | 47.9* | 44.7† |
| **ATM-RAG（本文方法）** | 0.5 | 51.0 | 68.7 | 8.4 | 28.8 |

* `*` 表示 QS 使用 DeepSeek-V4-flash 评判模型，而不是其他行使用的 `gpt-5-mini` 评判模型。`†` 表示 ATM-Bench-Hard Recall 使用固定的 Qwen3-VL-2B captions 汇报，尽管已提交的 Hard QS 运行使用 Qwen3.6-27B captions 回答。

<a id="oracle-and-niah-results-zh"></a>
## Oracle 与 NIAH 结果

我们在包含 31 个问题的 ATM-Bench-Hard 划分上，报告多模态回答模型在 SGM 与 Raw（真实图像/视频）设置下的 Oracle 上限和 NIAH haystack 扫描（k=25/50/100）QS，使用 `gpt-5-mini` 作为评判模型。

完整报告请参见 [ATM-Bench 在线榜单](https://atmbench.github.io/leaderboard.html)。

### SGM

| 模型 | 上下文窗口 | 参数量 | Oracle | NIAH-25 | NIAH-50 | NIAH-100 |
|------|-----------:|--------|-------:|--------:|--------:|---------:|
| Qwen3-VL-8B-Instruct | 256K | 8B LM（总计约 9B） | 28.0 | 16.3 | 15.8 | 12.7 |
| MiniMax-M3 | 1M | 总计 428B / 激活 23B | 60.5 | 45.9 | 55.1 | 43.4 |
| MiMo-V2.5 | 1M | 总计 310B / 激活 15B | 44.6 | 39.1 | 34.5 | 31.8 |
| Kimi-K2.5 | 256K | 总计 1T / 激活 32B | 41.9 | 47.9 | 39.6 | 33.5 |
| Qwen3.6-27B | 262K | 27B LM | 42.8 | 39.2 | 29.6 | 27.4 |
| 约输入上下文深度 | — | — | ≈4.5K | ≈12K | ≈22K | ≈44K |

### Raw（图像/视频）

| 模型 | 上下文窗口 | 参数量 | Oracle | NIAH-25 | NIAH-50 | NIAH-100 |
|------|-----------:|--------|-------:|--------:|--------:|---------:|
| Qwen3-VL-8B-Instruct | 256K | 8B LM（总计约 9B） | 40.1 | 25.4 | 24.9 | 10.9 |
| MiniMax-M3 | 1M | 总计 428B / 激活 23B | 61.8 | 41.8 | 34.2 | 35.2 |
| MiMo-V2.5 | 1M | 总计 310B / 激活 15B | 52.1 | 43.3 | 33.1 | 23.6 |
| Kimi-K2.5 | 256K | 总计 1T / 激活 32B | 57.1 | 45.4 | failed | failed |
| Qwen3.6-27B | 262K | 27B LM | 62.3 | 50.5 | failed | failed |
| 约输入上下文深度 | — | — | ≈6.5K | ≈18K | ≈31K | ≈60K |

> **为什么选择 SGM 而不是 Raw？** Raw 在 Oracle 上限下优于 SGM，但这种优势在现实条件下会消失：随着 haystack 中干扰项增多，Raw 的性能会下降，甚至因 payload/上下文限制而失败；在智能体检索中差距更加明显，所有“w/o SGM”（Raw）智能体的表现都远低于对应的 SGM 运行。SGM 是在现实噪声条件下更稳健的表示方式。

> **failed** 表示请求超过了模型允许的最大图像/视频数量，或 API 服务器的最大上传/payload 大小，导致该规模的输入无法处理。这属于服务限制，而不是模型质量结果。



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

### 下载数据集

ATM-Bench 托管在 Hugging Face [`Jingbiao/ATM-Bench`](https://huggingface.co/datasets/Jingbiao/ATM-Bench)。下载脚本会下载完整发布数据，并自动将文件放到评估脚本期望的位置。

**完整下载（约 3.3 GB）** — 包含 QA、NIAH 池、预处理记忆、emails、原始图像、原始视频以及 GPS 反向地理编码缓存：

```bash
bash scripts/download_data.sh
```

执行后将得到：

```
data/atm-bench/atm-bench.json
data/atm-bench/atm-bench-hard.json
data/atm-bench/niah/...
data/raw_memory/email/emails.json                   # emails
data/raw_memory/image/...                           # 原始图像
data/raw_memory/video/...                           # 原始视频
data/raw_memory/geocoding_cache/...                 # GPS 反向地理编码缓存
output/image/qwen3vl2b/batch_results.json           # 预处理的图像记忆
output/video/qwen3vl2b/batch_results.json           # 预处理的视频记忆
```

HF 上的 `data/processed_memory/{image,video}_batch_results.json` 会被脚本自动重命名并复制到 `output/image/qwen3vl2b/batch_results.json` 与 `output/video/qwen3vl2b/batch_results.json`。

脚本使用 `huggingface_hub` Python 包（若未安装会自动安装）。若数据集为私有，请先运行 `huggingface-cli login`。

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

### 准备记忆文件

运行这些基线之前，`output/{image,video}/qwen3vl2b/` 下必须存在 `batch_results.json`。有两种方式：

**方式 A（推荐）：直接从 Hugging Face 下载预处理记忆。**

若已运行上面的 `bash scripts/download_data.sh`，预处理的记忆文件已就位：

- `output/image/qwen3vl2b/batch_results.json`
- `output/video/qwen3vl2b/batch_results.json`

无需其他操作，可直接跳到下面的快速命令。

**方式 B：从原始图像/视频重新生成记忆文件。**

仅当你需要重新预处理（例如换用其他 VLM 或使用自己的原始记忆）时才需要。需要 `data/raw_memory/image/` 下的原始图像与 `data/raw_memory/video/` 下的视频：

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
#   需要：`bash scripts/download_data.sh`
#        + 本地运行的 vLLM 服务，地址 http://127.0.0.1:8000/v1/chat/completions，
#          模型 Qwen/Qwen3-VL-8B-Instruct-FP8（可通过 VLLM_ENDPOINT / ANSWERER_MODEL
#          环境变量覆盖）。
bash scripts/QA_Agent/MMRAG/run.sh

# Qwen3-VL-8B 原始图像/视频 Oracle（本地上界）
#   需要：`bash scripts/download_data.sh`
#        + 本地 vLLM 服务，模型 Qwen/Qwen3-VL-8B-Instruct-FP8。
bash scripts/QA_Agent/Oracle/run_oracle_qwen3vl8b_raw.sh

# GPT-5 原始图像/视频 Oracle（无需本地 GPU / vLLM）
#   需要：`bash scripts/download_data.sh`
#        + 环境变量 OPENAI_API_KEY 或文件 api_keys/.openai_key。
bash scripts/QA_Agent/Oracle/run_oracle_gpt5.sh
```

### 基线兼容性与环境

- 核心基线（`MMRAG`、`Oracle`、`NIAH`）在主 `atmbench` 环境中测试。
- 本仓库中的第三方记忆系统基线包括：
  - `A-Mem`
  - `HippoRAG2`
  - `mem0`
  - `MemoryOS`
  - `MemPalace`
  - `SimpleMem`
- 强烈建议在独立的 conda 环境中运行 `MemoryOS` 和 `MemPalace`。`MemoryOS` 使用 FAISS / sentence-transformers 依赖栈，`MemPalace` 使用 ChromaDB / ONNX 本地嵌入依赖栈；隔离环境可避免它们与核心基线环境及彼此之间发生依赖冲突。
- `A-Mem`、`HippoRAG2` 和 `mem0` 经测试与核心基线环境兼容，但为确保可复现性和依赖隔离，仍建议使用独立环境。
- `SimpleMem` 通过克隆的上游仓库（LanceDB + Tantivy FTS 依赖栈）运行；详见 [`memqa/qa_agent_baselines/SimpleMem/README.md`](memqa/qa_agent_baselines/SimpleMem/README.md)。固定的上游 commit 为 [`094027eca4c890dc9912be8cee1da04428de8076`](https://github.com/aiming-lab/SimpleMem/commit/094027eca4c890dc9912be8cee1da04428de8076)（由 `scripts/QA_Agent/SimpleMem/run.sh` 校验）。
- 这些基线的设置参考位于 `third_party/` 下：
  - `third_party/A-mem/`
  - `third_party/HippoRAG/`
  - `third_party/mem0/`
  - `third_party/MemoryOS/`
- `MemPalace` 以 PyPI 包形式发布（`mempalace==3.3.5`），通过 `memqa/qa_agent_baselines/Mempalace/requirements.txt` 安装；不需要放入 `third_party/`。
- `SimpleMem` **未**纳入 `third_party/`。请在 ATMBench 旁边克隆上游仓库到固定的 commit，并将 `SIMPLEMEM_DIR` 指向它（默认值为 `../SimpleMem`）：

  ```bash
  git clone https://github.com/aiming-lab/SimpleMem.git ../SimpleMem
  git -C ../SimpleMem checkout 094027eca4c890dc9912be8cee1da04428de8076
  pip install -r ../SimpleMem/requirements.txt
  pip install -r memqa/qa_agent_baselines/SimpleMem/requirements.txt
  ```
- 面向全部五个智能体（Claude Code、Codex、Pi、OpenCode、OpenClaw）的通用智能体评估框架位于 [`agent_systems/`](agent_systems/README.md)。

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

- 📄 **论文：** https://arxiv.org/abs/2603.01990
- 🌐 **项目主页：** https://atmbench.github.io/
- 🏆 **在线榜单：** https://atmbench.github.io/leaderboard.html
- 🤗 **数据集：** https://huggingface.co/datasets/Jingbiao/ATM-Bench
- 💻 **代码：** https://github.com/JingbiaoMei/ATM-Bench
- 🐛 **问题反馈：** https://github.com/JingbiaoMei/ATM-Bench/issues

<a id="license-zh"></a>
## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
