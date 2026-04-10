#!/usr/bin/env bash
# Download the full ATM-Bench release from Hugging Face and stage files at the
# paths the evaluation scripts expect. Run from the repository root.
#
#   bash scripts/download_data.sh
#
# This downloads:
#   data/atm-bench/atm-bench.json
#   data/atm-bench/atm-bench-hard.json
#   data/atm-bench/niah/...
#   data/raw_memory/email/emails.json
#   data/raw_memory/image/...
#   data/raw_memory/video/...
#   data/raw_memory/geocoding_cache/...
#   output/image/qwen3vl2b/batch_results.json   (preprocessed image memory)
#   output/video/qwen3vl2b/batch_results.json   (preprocessed video memory)

REPO="Jingbiao/ATM-Bench"

for arg in "$@"; do
  case "$arg" in
    --full)
      echo "[download_data] --full is now the default; continuing."
      ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "[download_data] unknown arg: $arg"
      exit 2
      ;;
  esac
done

python - "$REPO" <<'PY'
import os
import shutil
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[download_data] installing huggingface_hub Python package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.24"])
    from huggingface_hub import snapshot_download

repo_id = sys.argv[1]

patterns = [
    "data/atm-bench/*",
    "data/atm-bench/**/*",
    "data/processed_memory/image_batch_results.json",
    "data/processed_memory/video_batch_results.json",
    "data/raw_memory/email/emails.json",
    "data/raw_memory/image/**",
    "data/raw_memory/video/**",
    "data/raw_memory/geocoding_cache/**",
]

print(f"[download_data] downloading full release from {repo_id}...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=".",
    allow_patterns=patterns,
)

print("[download_data] staging processed memory into output/...")
pairs = [
    ("data/processed_memory/image_batch_results.json", "output/image/qwen3vl2b/batch_results.json"),
    ("data/processed_memory/video_batch_results.json", "output/video/qwen3vl2b/batch_results.json"),
]
for src, dst in pairs:
    if not os.path.exists(src):
        sys.exit(f"[download_data] ERROR: expected file missing after download: {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  {src} -> {dst}")

email = "data/raw_memory/email/emails.json"
if os.path.exists(email):
    print(f"  {email}  (staged)")
else:
    print(f"  WARNING: {email} was not downloaded; MMRAG will fail to load emails.")

print("[download_data] done.")
print("  data/atm-bench/                            (QA + NIAH pools)")
print("  data/raw_memory/email/emails.json          (emails for MMRAG/Oracle)")
print("  output/image/qwen3vl2b/batch_results.json  (preprocessed image memory)")
print("  output/video/qwen3vl2b/batch_results.json  (preprocessed video memory)")
print("  data/raw_memory/image/                     (raw images)")
print("  data/raw_memory/video/                     (raw videos)")
print("  data/raw_memory/geocoding_cache/           (GPS reverse-geocoding cache)")
PY
