# Third-Party Vendor Directory

This repo vendors selected upstream code under `third_party/` for **reproducibility** and to keep baseline integrations self-contained.

Notes:
- These are **code-only** snapshots (we intentionally omit large artifacts like model outputs, datasets, and PDFs when not required to run the baselines).
- Each component has a `UPSTREAM.md` describing the upstream repo URL + pinned commit.
- If we carry local changes on top of upstream, we include `ATMBench.patch` (a `git diff` against the pinned commit) and the vendored files already include those changes.

## Vendored Components

| Component | Upstream | Pinned Commit | Local Patch? | License |
|---|---|---|---|---|
| `third_party/HippoRAG/` | OSU-NLP-Group/HippoRAG | `d437bfb1805278b81e20c82357ed3f7d90f14901` | Yes (`ATMBench.patch`) | `third_party/HippoRAG/LICENSE` |
| `third_party/MemoryOS/` | BAI-LAB/MemoryOS | `d7a546274877429fa23912baac4cc6ad773b6dd6` | Yes (`ATMBench.patch`) | `third_party/MemoryOS/LICENSE` |
| `third_party/A-mem/` | WujiangXu/A-mem | `5b345e6c377ed13e4e361a8a982aeebacedc9cbd` | No | `third_party/A-mem/LICENSE` |
| `third_party/mem0/` | mem0ai/mem0 | `dba7f0458aeb50aa7078d36eaefa2405afbee620` | No | `third_party/mem0/LICENSE` |

