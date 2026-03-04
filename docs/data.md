# Data

This repo expects you to place benchmark files and raw memory artifacts under
`data/`.

## Layout

- Benchmark QA:
  - `data/atm-bench/atm-bench.json`
  - `data/atm-bench/atm-bench-hard.json`
  - `data/atm-bench/niah/` (NIAH pool files)
- Raw personal memory (user-provided):
  - `data/raw_memory/image/` (raw images)
  - `data/raw_memory/video/` (raw videos)
  - `data/raw_memory/email/merged_emails.json` (optional; see schema below)
  - `data/raw_memory/image/batch_results.json` (generated)
  - `data/raw_memory/video/batch_results.json` (generated)
- Generated artifacts:
  - `data/processed_memory/` (optional; for any future normalized memory store)
  - `output/` (all run outputs; always safe to delete/re-generate)

Note: `data/` and `output/` are gitignored.

## Benchmark Data Release Status

Current status:

- Paper/code release is available.
- Benchmark data release is pending.

Planned release channel:

- Hugging Face (dataset artifacts and versioned files).

Planned metadata to publish with the data release:

- HF dataset/model link
- Versioning scheme (tag/date + git commit)
- `sha256` checksums for released files
- Minimal download instructions (curl / `huggingface_hub`)
- License + citation block

## Schemas (What Scripts Expect)

### QA JSON (`atm-bench*.json`)

The QA files are JSON arrays (or a dict with a `qas` list) of entries containing:

- `id` (string)
- `question` (string)
- `answer` (string)
- `evidence_ids` (list of strings; ground-truth evidence IDs)

For NIAH pool files, each entry additionally contains:

- `niah_evidence_ids` (list of strings; fixed evidence pool, superset of `evidence_ids`)

### Email File (`data/raw_memory/email/merged_emails.json`)

If a QA item includes email evidence IDs (IDs starting with `email...`), Oracle/MMRAG
scripts may load email evidence from a JSON list with entries like:

```json
[
  {
    "id": "email202401010001",
    "timestamp": "2024-01-01 12:34:56",
    "short_summary": "One-line summary",
    "detail": "Longer email content or extracted body"
  }
]
```

If your released benchmark does not include emails, you can omit this file.

### Batch Results (`batch_results.json`)

Text-only evidence for images/videos is read from `batch_results.json` files.
Scripts index entries by `Path(image_path).stem` / `Path(video_path).stem`.

Each entry typically contains:

- `image_path` / `video_path` (string path; used to derive the evidence ID stem)
- `timestamp` (string)
- `location_name` (string)
- `short_caption` (string)
- `caption` (string)
- `ocr_text` (string)
- `tags` (list of strings)

#### GPS / Reverse-Geocoding Cache (location_name)

`location_name` is derived from GPS coordinates via reverse geocoding (default: OpenStreetMap Nominatim).
Public geocoding endpoints are rate-limited (often strict per-IP requests/minute) and can become a
bottleneck for large archives, especially if you run the processors with high concurrency.

The processors cache reverse-geocoding results as JSON files under `<output_dir>/cache/`:

- `*_location_name.json`

If you have a pre-extracted GPS cache bundle (recommended; we plan to ship one with the benchmark artifacts),
copy those cache files into your processor cache directory before running the processors so geocoding calls
are skipped:

```bash
python memqa/utils/copy_gps_info.py <GPS_CACHE_DIR> output/image/qwen3vl2b/cache
python memqa/utils/copy_gps_info.py <GPS_CACHE_DIR> output/video/qwen3vl2b/cache
```

You can also use the convenience wrappers:
- `scripts/memory_processor/image/copy_gps_cache.sh`
- `scripts/memory_processor/video/copy_gps_cache.sh`

## Building NIAH Pools

If you have an MMRAG run that produced `retrieval_recall_details.json`, you can
build/validate NIAH pools via:

```bash
python scripts/QA_Agent/NIAH/build_niah_pools.py \
  --qa-file data/atm-bench/atm-bench-hard.json \
  --retrieval-details <PATH_TO>/retrieval_recall_details.json \
  --pool-sizes 25 50 100 200
```
