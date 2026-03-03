#!/usr/bin/env bash

# Copy reverse-geocoding cache files (*_location_name.json) into a target cache directory.
#
# Public reverse-geocoding services are rate-limited; copying a GPS cache bundle first
# allows the processors to skip geocoding API calls.
#
# Usage:
#   bash scripts/memory_processor/video/copy_gps_cache.sh <SRC_CACHE_DIR> <DST_CACHE_DIR>
#
# Example:
#   bash scripts/memory_processor/video/copy_gps_cache.sh \
#     <GPS_CACHE_DIR> \
#     output/video/qwen3vl2b/cache

SRC_CACHE_DIR="${1:-}"
DST_CACHE_DIR="${2:-}"

if [ -z "${SRC_CACHE_DIR}" ] || [ -z "${DST_CACHE_DIR}" ]; then
  echo "Usage: bash scripts/memory_processor/video/copy_gps_cache.sh <SRC_CACHE_DIR> <DST_CACHE_DIR>"
  exit 1
fi

python memqa/utils/copy_gps_info.py "${SRC_CACHE_DIR}" "${DST_CACHE_DIR}"
