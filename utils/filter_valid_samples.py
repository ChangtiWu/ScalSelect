#!/usr/bin/env python3
"""
Filter out samples without images from dataset.

Usage:
    python filter_valid_samples.py
"""

import json
from pathlib import Path


def filter_valid_samples(
    input_path: str,
    output_path: str,
    base_image_path: str = "/mnt/project_ai4edu/share/code/RobobrainFactory/data/"
) -> dict:
    """
    Filter dataset to keep only samples with valid images.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        base_image_path: Base path for image files

    Returns:
        dict: Statistics about filtering
    """
    print(f"Loading dataset: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    total = len(data)
    print(f"Total samples: {total:,}\n")

    # Filter valid samples
    valid_samples = []
    skipped_no_images = 0
    skipped_missing_files = 0

    for sample in data:
        # Check if images field exists and is not empty
        images = sample.get('images', [])
        if not isinstance(images, list):
            images = [images] if images else []

        if not images:
            skipped_no_images += 1
            continue

        # Check if image files exist
        images_exist = True
        for img in images:
            img_path = img if img.startswith('/') else base_image_path + img

            if not Path(img_path).exists():
                images_exist = False
                break

        if not images_exist:
            skipped_missing_files += 1
            continue

        # Valid sample - keep it
        valid_samples.append(sample)

    # Save filtered dataset
    print(f"Saving filtered dataset to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(valid_samples, f, indent=2, ensure_ascii=False)

    # Statistics
    stats = {
        'input_total': total,
        'output_total': len(valid_samples),
        'skipped_no_images': skipped_no_images,
        'skipped_missing_files': skipped_missing_files,
        'skipped_total': total - len(valid_samples),
    }

    return stats


def main():
    input_path = "/mnt/project_ai4edu/share/code/RobobrainFactory/data/LLaVA-665K/LF_LLaVA-665K_id.json"
    output_path = "/mnt/project_ai4edu/share/code/RobobrainFactory/data/LLaVA-665K/LF_LLaVA-655K-V.json"

    # Run filtering
    stats = filter_valid_samples(input_path, output_path)

    # Print results
    print("\n" + "="*70)
    print("DATASET FILTERING RESULTS")
    print("="*70)
    print(f"\nInput samples:          {stats['input_total']:>10,}")
    print(f"Output samples:         {stats['output_total']:>10,}")
    print(f"Skipped (no images):    {stats['skipped_no_images']:>10,}")
    print(f"Skipped (missing file): {stats['skipped_missing_files']:>10,}")
    print(f"Total skipped:          {stats['skipped_total']:>10,}")
    print(f"\nRetention rate:         {stats['output_total']/stats['input_total']*100:>9.2f}%")
    print("="*70)

    print(f"\n✓ Filtered dataset saved to: {output_path}")


if __name__ == '__main__':
    main()
