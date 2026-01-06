#!/usr/bin/env python3
"""
Check dataset validity: identify samples without ID or images.

Usage:
    python check_dataset_validity.py
"""

import json
from pathlib import Path
from collections import defaultdict


def check_dataset_validity(
    dataset_path: str,
    base_image_path: str = "/mnt/project_ai4edu/share/code/RobobrainFactory/data/"
) -> dict:
    """
    Check dataset for invalid samples.

    Returns:
        dict: Statistics and invalid sample details
    """
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    total = len(data)
    print(f"Total samples: {total:,}\n")

    # Track invalid samples
    invalid_samples = defaultdict(list)

    for idx, sample in enumerate(data):
        sample_id = sample.get('id')

        # Check 1: No ID
        if sample_id is None:
            invalid_samples['no_id'].append(idx)
            continue

        # Check 2: No images or missing image files
        images = sample.get('images', [])
        if not isinstance(images, list):
            images = [images] if images else []

        if not images:
            invalid_samples['no_images_field'].append(sample_id)
            continue

        # Check if image files exist
        for img in images:
            img_path = img if img.startswith('/') else base_image_path + img

            if not Path(img_path).exists():
                invalid_samples['missing_image_files'].append(sample_id)
                break

    # Calculate statistics
    total_invalid = sum(len(v) for v in invalid_samples.values())
    valid_count = total - total_invalid

    return {
        'total': total,
        'valid': valid_count,
        'invalid': total_invalid,
        'details': dict(invalid_samples),
    }


def main():
    dataset_path = "/mnt/project_ai4edu/share/code/RobobrainFactory/data/LLaVA-665K/LF_LLaVA-665K_200K.json"

    # Run check
    results = check_dataset_validity(dataset_path)

    # Print results
    print("="*70)
    print("DATASET VALIDITY CHECK RESULTS")
    print("="*70)
    print(f"\nTotal samples:        {results['total']:>10,}")
    print(f"Valid samples:        {results['valid']:>10,}  ({results['valid']/results['total']*100:.2f}%)")
    print(f"Invalid samples:      {results['invalid']:>10,}  ({results['invalid']/results['total']*100:.2f}%)")

    if results['details']:
        print("\nInvalid samples breakdown:")
        print("-"*70)
        for reason, samples in sorted(results['details'].items(), key=lambda x: len(x[1]), reverse=True):
            count = len(samples)
            pct = count / results['total'] * 100
            print(f"  {reason:25s}: {count:>7,}  ({pct:5.2f}%)")

            # Show first 5 examples
            if samples:
                examples = samples[:5]
                print(f"    Examples: {examples}")

    print("\n" + "="*70)

    # Save detailed report
    output_path = Path(__file__).parent / "invalid_samples_report.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed report saved to: {output_path}")


if __name__ == '__main__':
    main()
