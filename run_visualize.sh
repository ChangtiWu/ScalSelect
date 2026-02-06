#!/bin/bash

# Visualize CUR Importance Score Distribution

# Configuration
INPUT=<your_importance_scores_file> # e.g., ./features/llava-15-7b/llava665k-v_features/importance_scores.jsonl
OUTPUT=<your_output_file> # e.g., ./features/llava-15-7b/llava665k-v_features/importance_distribution.pdf

# Visualize
python utils/visualize_scores.py \
    --input ${INPUT} \
    --output ${OUTPUT}

echo "Visualization saved to: ${OUTPUT}"
