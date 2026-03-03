#!/bin/bash

# This script runs the processor to obtain the geo location.
# We have provided the extracted geo location info, so you do not need to run this script normally.

python memqa/mem_processor/image/batch_processor.py "./data/raw_memory/image" \
  --output_dir "output/image/geo_only" \
  --provider "none"
