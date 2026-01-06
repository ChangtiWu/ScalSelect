#!/bin/bash

# Visualize CUR Importance Score Distribution

# Configuration
INPUT=./features/llava-15-7b/llava665k-v_features_noatten/importance_scores.jsonl
OUTPUT=./features/llava-15-7b/llava665k-v_features_noatten/cur_importance_distribution.png

# Visualize
python utils/visualize_scores.py \
    --input ${INPUT} \
    --output ${OUTPUT}

echo "Visualization saved to: ${OUTPUT}"
