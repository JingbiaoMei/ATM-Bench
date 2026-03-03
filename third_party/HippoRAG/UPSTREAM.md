# HippoRAG (Vendored)

Upstream:
- Repo: `https://github.com/OSU-NLP-Group/HippoRAG`
- Commit: `d437bfb1805278b81e20c82357ed3f7d90f14901`
- Commit date: `2025-09-04 10:42:42 -0400`

Vendoring:
- Vendored into: `third_party/HippoRAG/`

Local changes:
- This vendored copy includes local modifications used by our baseline integration.
- Patch file (reference): `third_party/HippoRAG/ATMBench.patch` (generated via `git diff` against the pinned commit).

Scope:
- Included: `src/` + minimal top-level packaging/docs files needed for import.
- Omitted (intentionally): large/non-essential directories like `outputs/`, `reproduce/`, and PDFs.
