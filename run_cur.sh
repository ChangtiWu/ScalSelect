#!/bin/bash

# CUR Matrix Decomposition for Computing Sample Importance Scores
# This script computes importance scores for all samples using CUR decomposition

# Configuration
FEATURES_DIR=<your_features_directory> # e.g., ./features/llava-15-7b/llava665k-v_features
SV_THRESHOLD=0.9  # Singular value cumulative energy threshold (90%)

# Run CUR decomposition
python scripts/cur.py \
    --features-dir "${FEATURES_DIR}" \
    --sv-threshold ${SV_THRESHOLD} \
    --no-standardize \
    --no-l2-normalize \

echo 'CUR decomposition complete!'
echo "Importance scores saved to: ${FEATURES_DIR}/importance_scores.jsonl"
