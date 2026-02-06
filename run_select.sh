#!/bin/bash

# Select top-N samples based on the CUR importance scores

# sharegpt
# NOTE: Directly use importance_scores.jsonl from cur.py
IMPORTANCE_SCORES=<your_importance_scores_file> # e.g., ./features/llava-15-7b/llava665k-v_features/importance_scores.jsonl
INPUT_FILE=<your_input_dataset_file> # e.g., /mnt/project_ai4edu/share/code/RobobrainFactory/data/LLaVA-665K/LF_LLaVA-665K-V.json
OUTPUT_FILE=<your_output_dataset_file> # e.g., /mnt/project_ai4edu/share/code/RobobrainFactory/data/LLaVA-665K/LF_LLaVA-665K-V_llava7b_scalselect_100k.json
NUM_SELECTED=100000  # Number of top samples to select

# Select
python scripts/select_sharegpt.py \
    --importance-scores ${IMPORTANCE_SCORES} \
    --input-dataset ${INPUT_FILE} \
    --output-dataset ${OUTPUT_FILE} \
    --num-selected ${NUM_SELECTED}

echo "Data selecting completed!"
echo "Output: ${OUTPUT_FILE}"
