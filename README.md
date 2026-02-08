# ScalSelect

ScalSelect: Scalable Training-Free Multimodal Data Selection for Efficient Visual Instruction Tuning

## Requirements

- Python 3.11.11
- PyTorch
- Transformers (HuggingFace)
- Accelerate (HuggingFace)
- NumPy, Matplotlib, Pillow, tqdm

## Project Structure

```
ScalSelect/
├── scripts/
│   ├── feature_extract.py         # VLM feature extractor (LLaVA & Qwen-VL)
│   ├── feature_extract_sft.py     # Multi-GPU extraction entry point
│   ├── representation_utils.py    # Vision representation extraction & I/O
│   ├── cur.py                     # CUR decomposition & importance scoring
│   └── select_sharegpt.py         # Select top-N samples from dataset
├── utils/
│   ├── visualize_scores.py        # Importance score distribution plots
│   ├── check_dataset_validity.py  # Check dataset for missing images/IDs
│   └── filter_valid_samples.py    # Filter out invalid samples
├── run_extraction.sh              # Step 1: feature extraction
├── run_cur.sh                     # Step 2: CUR decomposition
├── run_select.sh                  # Step 3: sample selection
├── run_visualize.sh               # Optional: visualization
└── accelerate_config.yaml         # Multi-GPU accelerate config
```

## Dataset Format

Input datasets should be in **ShareGPT format** (JSON):

```json
[
  {
    "id": 0,
    "messages": [
      {"role": "user", "content": "<image>\nDescribe this image."},
      {"role": "assistant", "content": "The image shows ..."},
      {"role": "user", "content": "What color is the sky?"},
      {"role": "assistant", "content": "The sky is blue."}
    ],
    "images": ["path/to/image.jpg"],
  },
  ...
]
```

## Supported Models

| Model Family | `--model-type` | Example |
|---|---|---|
| LLaVA | `llava` | `llava-hf/llava-1.5-7b-hf` |
| Qwen-VL | `qwen` | `Qwen/Qwen3-VL-8B-Instruct` |


## Quick Start

### Step 1: Extract Features

Before running `run_extraction.sh`, please configure the base url `base_path` of input images in the `script/feature_extract.py`:
```python
def process_image_paths(images: List[str], base_path: str = None) -> List[str]:
    """Process image paths, add base prefix if needed."""
    if base_path is None:
        base_path = "<your dataset base path>" # e.g. "/mnt/project_ai4edu/share/code/RobobrainFactory/data/"

    return [
        img if img.startswith('/') else base_path + img
        for img in images
    ]
```
Note. The base path must satisfy the following requirement: it must be able to be concatenated with the image paths in your input samples to form absolute paths for the images.


Then, extract vision representations from your dataset using a VLM. Supports **multi-GPU** via HuggingFace Accelerate.

```bash
# Edit run_extraction.sh to set your paths, then:
bash run_extraction.sh
```

Key parameters in `run_extraction.sh`:

| Parameter | Description | Default |
|---|---|---|
| `MODEL` | Path to VLM model (LLaVA / Qwen-VL) | — |
| `MODEL_TYPE` | `llava` or `qwen` | `llava` |
| `DATASET` | Path to ShareGPT-format JSON dataset | — |
| `OUTPUT_DIR` | Directory to save extracted features | — |
| `NUM_PROCESSES` | Number of GPUs | `8` |
| `SAMPLE_BATCH_SIZE` | Samples per batch per GPU | `1` |

Output: `OUTPUT_DIR/all_representations.npz`

### Step 2: Compute Importance Score

Compute importance scores for all samples based on the CUR matrix decomposition.

```bash
# Edit run_cur.sh to set your features directory, then:
bash run_cur.sh
```

Key parameters in `run_cur.sh`:

| Parameter | Description | Default |
|---|---|---|
| `FEATURES_DIR` | Directory containing `all_representations.npz` | — |
| `SV_THRESHOLD` | Cumulative energy threshold for selecting k singular values | `0.9` |

Output: `FEATURES_DIR/importance_scores.jsonl`

### Step 3: Select Top-N Samples

Extract the most important samples from the original dataset.

```bash
# Edit run_select.sh to set your paths, then:
bash run_select.sh
```

Key parameters in `run_select.sh`:

| Parameter | Description | Default |
|---|---|---|
| `IMPORTANCE_SCORES` | Path to `importance_scores.jsonl` | — |
| `INPUT_FILE` | Original dataset file | — |
| `OUTPUT_FILE` | Output selected dataset file | — |
| `NUM_SELECTED` | Number of top samples to select | `100000` |

### (Optional) Visualize Score Distribution

```bash
bash run_visualize.sh
```

