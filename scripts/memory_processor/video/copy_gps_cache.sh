#!/usr/bin/env bash

# Copy reverse-geocoding cache files (*_location_name.json) into a target cache directory.
#
# Public reverse-geocoding services are rate-limited; copying a GPS cache bundle first
# allows the processors to skip geocoding API calls.
# Cache files are keyed by the source video filename stem.
#
# Usage:
#   bash scripts/memory_processor/video/copy_gps_cache.sh [SRC_CACHE_DIR] <DST_CACHE_DIR>
#
# Example:
#   bash scripts/memory_processor/video/copy_gps_cache.sh \
#     output/video/qwen3vl2b/cache
#   bash scripts/memory_processor/video/copy_gps_cache.sh \
#     /path/to/other/video_gps_cache \
#     output/video/qwen3vl2b/cache

DEFAULT_SRC_CACHE_DIR="data/raw_memory/geocoding_cache/video"

if [ "$#" -eq 1 ]; then
  SRC_CACHE_DIR="${DEFAULT_SRC_CACHE_DIR}"
  DST_CACHE_DIR="${1}"
else
  SRC_CACHE_DIR="${1:-${DEFAULT_SRC_CACHE_DIR}}"
  DST_CACHE_DIR="${2:-}"
fi

if [ -z "${DST_CACHE_DIR}" ]; then
  echo "Usage: bash scripts/memory_processor/video/copy_gps_cache.sh [SRC_CACHE_DIR] <DST_CACHE_DIR>"
  echo "Default SRC_CACHE_DIR: ${DEFAULT_SRC_CACHE_DIR}"
  exit 1
fi

python memqa/utils/copy_gps_info.py "${SRC_CACHE_DIR}" "${DST_CACHE_DIR}"
