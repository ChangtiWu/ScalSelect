#!/bin/bash

# Multi-turn Conversation Feature Extraction Script
# Extracts vision representations from user-attended vision tokens

# NCCL Settings - prevent timeout on variable-length conversations
export NCCL_TIMEOUT=1800              # 30 minutes (default: 600s)
export NCCL_DEBUG=WARN                # Show warnings for debugging
export TORCH_NCCL_BLOCKING_WAIT=1     # Better error messages (new PyTorch API)
export NCCL_IB_TIMEOUT=22             # InfiniBand timeout

# 配置参数
MODEL_TYPE=llava  # Model type: llava or qwen
MODEL=<your_model_path> # e.g., /mnt/project_ai4edu/share/models/llava-1.5-7b-pretrain
DATASET=<your_dataset_path> # e.g., /mnt/project_ai4edu/share/code/RobobrainFactory/data/LLaVA-665K/LF_LLaVA-665K-V.json
OUTPUT_DIR=<your_output_directory> # e.g., ./features/llava-15-7b/llava665k-v_features
MAX_SAMPLES=-1  # -1 = all samples
NUM_PROCESSES=8  # Number of GPUs
SAMPLE_BATCH_SIZE=1  # Number of samples to process together per device
TORCH_DTYPE=bfloat16
MAX_LENGTH=4096 
CUMULATIVE_THRESHOLD=0.9 

echo "======================================================================"
echo "Multi-turn Conversation Feature Extraction"
echo "======================================================================"
echo "Model: ${MODEL}"
echo "Model type: ${MODEL_TYPE}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "Sample batch size: ${SAMPLE_BATCH_SIZE}"
echo "GPUs: ${NUM_PROCESSES}"
echo "Max length: ${MAX_LENGTH}"
echo "======================================================================"

# Run extraction
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes=${NUM_PROCESSES} \
    --num_machines=1 \
    --mixed_precision=bf16 \
    scripts/feature_extract_sft.py \
    --model "${MODEL}" \
    --model-type "${MODEL_TYPE}" \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-samples ${MAX_SAMPLES} \
    --sample-batch-size ${SAMPLE_BATCH_SIZE} \
    --torch-dtype "${TORCH_DTYPE}" \
    --max-length ${MAX_LENGTH} \
    --cumulative-threshold ${CUMULATIVE_THRESHOLD}

echo ""
echo "======================================================================"
echo "✓ Feature extraction complete!"
echo "Vision representations: ${OUTPUT_DIR}/all_representations.npz"
echo "======================================================================"
