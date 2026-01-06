#!/bin/bash

# Select top-N samples based on the CUR importance scores

# sharegpt
# NOTE: Directly use importance_scores.jsonl from cur.py
IMPORTANCE_SCORES=./features/llava-15-7b/llava665k-v_features_noatten/importance_scores.jsonl
INPUT_FILE=/mnt/project_ai4edu/share/code/RobobrainFactory/data/LLaVA-665K/LF_LLaVA-665K-V.json
OUTPUT_FILE=/mnt/project_ai4edu/share/code/RobobrainFactory/data/LLaVA-665K/LF_LLaVA-665K-V_noatten_100k.json
NUM_SELECTED=100000  # Number of top samples to select

# Select
python scripts/select_sharegpt.py \
    --importance-scores ${IMPORTANCE_SCORES} \
    --input-dataset ${INPUT_FILE} \
    --output-dataset ${OUTPUT_FILE} \
    --num-selected ${NUM_SELECTED}

echo "Data selecting completed!"
echo "Output: ${OUTPUT_FILE}"
