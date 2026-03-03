# MemoryOS (Vendored)

Upstream:
- Repo: `https://github.com/BAI-LAB/MemoryOS`
- Commit: `d7a546274877429fa23912baac4cc6ad773b6dd6`
- Commit date: `2025-09-20 13:08:48 +0800`

Vendoring:
- Vendored into: `third_party/MemoryOS/`

Local changes:
- Patch file (reference): `third_party/MemoryOS/ATMBench.patch` (generated via `git diff` against the pinned commit).

Scope:
- Included: `memoryos-pypi/` (upstream source layout) and `memoryos/` (import-friendly mirror of the same code) + minimal top-level docs/license.
- Omitted (intentionally): other subprojects and evaluation assets not required for ATMBench baseline runs.
