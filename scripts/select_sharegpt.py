"""
Extract Selected Samples from ShareGPT Dataset Based on CUR Importance Scores.

This module reads importance scores (from cur.py), selects top-N samples,
and extracts them from the original ShareGPT dataset.

Usage:
    python select_sharegpt.py --importance-scores importance_scores.jsonl \
                              --input-dataset dataset.json \
                              --output-dataset selected.json \
                              --num-selected 500
"""

import argparse
import json
from pathlib import Path
from typing import List


def load_importance_scores(importance_scores_path: str, num_selected: int = None) -> List[int]:
    """
    Load importance scores and select top-N samples.

    Args:
        importance_scores_path: Path to importance scores JSONL file (from cur.py)
        num_selected: Number of top samples to select (if None, select all)

    Returns:
        List of selected sample IDs
    """
    print(f"Loading importance scores from: {importance_scores_path}")

    importance_samples = []
    with open(importance_scores_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            # Skip metadata line
            if 'metadata' in entry:
                continue
            importance_samples.append(entry)

    total_samples = len(importance_samples)

    print(f"  Total samples: {total_samples}")

    # Select top-N samples (already sorted by importance in cur.py)
    if num_selected is None:
        num_selected = total_samples
        print(f"  Selecting all {num_selected} samples")
    else:
        num_selected = min(num_selected, total_samples)
        print(f"  Selecting top {num_selected} samples by importance")

    selected_ids = [item['sample_id'] for item in importance_samples[:num_selected]]

    print(f"  Selected {len(selected_ids)} samples")
    if num_selected > 0:
        print(f"  Importance range: [{importance_samples[num_selected-1]['importance']:.6f}, {importance_samples[0]['importance']:.6f}]")

    return selected_ids


def extract_selected_samples(
    input_dataset_path: str,
    selected_ids: List[int],
    output_dataset_path: str,
) -> List:
    """
    Extract selected samples from the input ShareGPT JSON dataset and sort by ID.

    Args:
        input_dataset_path: Path to the input JSON file
        selected_ids: List of sample IDs to extract
        output_dataset_path: Path to save the output JSON file

    Returns:
        List containing the selected samples (sorted by ID)
    """
    print(f"\nLoading input dataset from: {input_dataset_path}")

    with open(input_dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Total samples in dataset: {len(data)}")

    # Create ID to sample mapping
    id_to_sample = {}
    for sample in data:
        sample_id = sample.get('id')
        if sample_id is not None:
            id_to_sample[sample_id] = sample

    # Extract selected samples
    print(f"\nExtracting {len(selected_ids)} selected samples...")
    selected_samples = []
    missing_ids = []

    for sample_id in selected_ids:
        if sample_id in id_to_sample:
            selected_samples.append(id_to_sample[sample_id])
        else:
            missing_ids.append(sample_id)

    if missing_ids:
        print(f"  ⚠️  Warning: {len(missing_ids)} IDs not found in dataset")
        if len(missing_ids) <= 10:
            print(f"      Missing IDs: {missing_ids}")

    print(f"  Successfully extracted {len(selected_samples)} samples")

    # Sort by ID for consistent output
    print(f"\nSorting samples by ID...")
    selected_samples.sort(key=lambda x: x.get('id', 0))
    if selected_samples:
        print(f"  ID range: [{selected_samples[0]['id']}, {selected_samples[-1]['id']}]")

    # Save to output
    output_path = Path(output_dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving selected dataset to: {output_dataset_path}")
    with open(output_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(selected_samples, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved {len(selected_samples)} samples (sorted by ID)")

    return selected_samples


def main():
    parser = argparse.ArgumentParser(
        description="Extract top-N samples from ShareGPT dataset based on CUR importance scores"
    )
    parser.add_argument(
        '--importance-scores',
        type=str,
        required=True,
        help='Path to importance scores JSONL file (from cur.py)',
    )
    parser.add_argument(
        '--input-dataset',
        type=str,
        required=True,
        help='Path to input ShareGPT JSON dataset file',
    )
    parser.add_argument(
        '--output-dataset',
        type=str,
        required=True,
        help='Path to output JSON file (selected samples, sorted by ID)',
    )
    parser.add_argument(
        '--num-selected',
        type=int,
        default=None,
        help='Number of top samples to select (default: all samples)',
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Extract Top-N Samples from ShareGPT Dataset (by CUR Importance)")
    print("=" * 70)

    # Load importance scores and select top-N
    selected_ids = load_importance_scores(
        importance_scores_path=args.importance_scores,
        num_selected=args.num_selected,
    )

    # Extract and save selected samples (sorted by ID)
    selected_samples = extract_selected_samples(
        input_dataset_path=args.input_dataset,
        selected_ids=selected_ids,
        output_dataset_path=args.output_dataset,
    )

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Selected samples: {len(selected_samples)}")
    print(f"Output file: {args.output_dataset}")
    print("=" * 70)


if __name__ == '__main__':
    main()
