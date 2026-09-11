#!/bin/bash
# SCRATCH_ROOT defaults to the CINECA Leonardo path; override it in your
# environment to run this on a different cluster/filesystem -- see README > Configuration.
export SCRATCH_ROOT="${SCRATCH_ROOT:-/leonardo_scratch/large/userexternal/$USER}"

models=(
    "meta-llama/Llama-3.1-8B"
    "sapienzanlp/Minerva-7B-base-v1.0"
)
for model in "${models[@]}"; do
    echo "Downloading $model"
    uv run download_models.py --model "$model" --hf-home "$SCRATCH_ROOT/.hf_cache"
done