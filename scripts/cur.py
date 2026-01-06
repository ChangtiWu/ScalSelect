"""
CUR Matrix Decomposition for Computing Sample Importance Scores.

This module implements CUR decomposition to compute importance scores for all samples
from extracted feature representations. The output is sorted by importance (high to low).

Usage:
    python cur.py --features-dir ./output/geometry3k_features_accelerate \
                  --output importance_scores.json \
                  --sv-threshold 0.9
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def load_representations(features_dir: str) -> Tuple[np.ndarray, List]:
    """
    Load all sample representations from the features directory.

    Args:
        features_dir: Path to directory containing all_representations.npz

    Returns:
        Tuple of:
        - matrix: [num_samples, feature_dim] matrix where each row is a sample
        - sample_ids: List of sample IDs corresponding to each row
    """
    features_path = Path(features_dir) / 'all_representations.npz'

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    print(f"Loading representations from: {features_path}")
    data = np.load(features_path)

    # Load matrix and sample IDs directly (fast: 2 array loads vs 625K)
    matrix = data['representations']
    sample_ids_str = data['sample_ids']

    # Convert string IDs to integers
    ordered_ids = []
    for sample_id in sample_ids_str:
        if sample_id.isdigit():
            ordered_ids.append(int(sample_id))
        else:
            # Handle format like "sample_123"
            numeric_id = int(sample_id.split('_')[-1]) if '_' in sample_id else 0
            ordered_ids.append(numeric_id)

    print(f"Loaded {matrix.shape[0]} samples with dimension {matrix.shape[1]}")

    return matrix, ordered_ids


def preprocess_matrix(
    matrix: np.ndarray,
    center: bool = True,
    standardize: bool = True,
    l2_normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess the feature matrix before CUR decomposition.

    This is CRITICAL for large model embeddings to avoid the k=1 problem:
    - Centering: removes the mean direction (MUST enable)
    - Standardization: normalizes variance across features (recommended)
    - L2 normalization: projects to unit sphere, removes anisotropy (recommended)

    Args:
        matrix: Feature matrix (n_samples, n_features) - vision representations only
        center: Whether to center the data (subtract column means). Default: True
        standardize: Whether to standardize the data (divide by column std). Default: True
        l2_normalize: Whether to L2-normalize each row (unit length per sample). Default: True

    Returns:
        Preprocessed matrix with same shape as input
    """
    if not any([center, standardize, l2_normalize]):
        print("⚠️  WARNING: All preprocessing disabled! May encounter k=1 problem.")
        return matrix.copy()

    print(f"\n🔧 Preprocessing matrix ({matrix.shape[0]} samples, {matrix.shape[1]} features)...")

    matrix_processed = matrix.copy()

    if center:
        print(f"  [1/3] Centering the entire matrix...")
        means = matrix_processed.mean(axis=0, keepdims=True)
        matrix_processed -= means
        print(f"       Means range: [{means.min():.4f}, {means.max():.4f}]")

    if standardize:
        print(f"  [2/3] Standardizing the entire matrix...")
        stds = matrix_processed.std(axis=0, keepdims=True)
        stds[stds == 0] = 1
        matrix_processed /= stds
        print(f"       Stds range: [{stds.min():.4f}, {stds.max():.4f}]")

    if l2_normalize:
        print("  [3/3] L2-normalizing each sample (row-wise)...")
        norms = np.linalg.norm(matrix_processed, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix_processed = matrix_processed / norms
        print(f"       All samples normalized to unit length")

    print("  ✓ Preprocessing complete\n")
    return matrix_processed


def select_k_singular_values(
    singular_values: np.ndarray,
    cumulative_threshold: float = 0.9,
) -> int:
    """
    Select k singular values that capture the specified cumulative energy.

    Energy is defined as the sum of squared singular values.

    Args:
        singular_values: Array of singular values in descending order
        cumulative_threshold: Cumulative energy threshold (default: 0.9 for 90%)

    Returns:
        k: Number of singular values to keep
    """
    # Compute squared singular values (energy)
    squared_sv = singular_values ** 2
    total_energy = squared_sv.sum()

    # Compute cumulative energy ratio
    cumulative_energy = np.cumsum(squared_sv) / total_energy

    # Find k where cumulative energy exceeds threshold
    k = np.searchsorted(cumulative_energy, cumulative_threshold) + 1

    # Ensure k is at least 1 and at most the number of singular values
    k = max(1, min(k, len(singular_values)))

    print(f"  Selected k={k} singular values (captures {cumulative_energy[k-1]*100:.2f}% energy)")

    return k


def compute_row_importance_scores(
    matrix: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute importance scores for each row using leverage scores.

    Leverage scores are computed as:
        π_i = ||U_k[i,:]||²
    where U_k is the first k columns of the left singular vectors.

    NOTE: This function expects the matrix to be ALREADY PREPROCESSED.
    Use preprocess_matrix() before calling this function.

    Args:
        matrix: Feature matrix (n_samples, n_features) - MUST be preprocessed
        k: Number of singular values to use

    Returns:
        row_scores: Raw leverage scores for each row (not normalized)
        singular_values: All singular values from SVD
    """
    print(f"\n2️⃣ Computing leverage scores using k={k} singular values...")

    # Perform SVD
    print("  [SVD] Decomposing matrix...")
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Extract first k columns of U
    U_k = U[:, :k]

    # Compute leverage scores: ||U_k[i,:]||² for each row i
    leverage_scores = np.sum(U_k ** 2, axis=1)

    # Return raw leverage scores without normalization
    # (No normalization to avoid dilution when handling large datasets)
    row_scores = leverage_scores

    print(f"  ✓ Computed leverage scores for {len(row_scores)} samples")
    print(f"    - Max score: {row_scores.max():.6f}")
    print(f"    - Min score: {row_scores.min():.6f}")
    print(f"    - Mean score: {row_scores.mean():.6f}")

    return row_scores, S





def cur_row_selection(
    matrix: np.ndarray,
    sample_ids: List,
    sv_threshold: float = 0.9,
    center: bool = True,
    standardize: bool = True,
    l2_normalize: bool = True,
) -> Dict:
    """
    Perform CUR decomposition to compute importance scores for all samples.

    Pipeline:
        1. Preprocess matrix (center + standardize + L2 normalize)
        2. Perform SVD on preprocessed matrix and select k
        3. Compute leverage scores on preprocessed matrix
        4. Return all samples with their importance scores (sorted by score)

    Args:
        matrix: [num_samples, feature_dim] matrix of sample representations
        sample_ids: List of sample IDs corresponding to each row
        sv_threshold: Singular value cumulative energy threshold (default: 0.9)
        center: Whether to center the data (default: True)
        standardize: Whether to standardize the data (default: True)
        l2_normalize: Whether to L2-normalize each row (default: True)

    Returns:
        Dictionary containing:
        - sample_scores: List of {sample_id, score} sorted by score (high to low)
        - k: Number of singular values used
        - num_total: Total number of samples
    """
    print("=" * 70)
    print("CUR Matrix Decomposition - Computing Importance Scores")
    print("=" * 70)
    print(f"Input matrix shape: {matrix.shape}")
    print(f"SV threshold: {sv_threshold}")
    print(f"Preprocessing: center={center}, standardize={standardize}, l2_normalize={l2_normalize}")
    print("=" * 70)

    # Step 0: Preprocess matrix (CRITICAL - must be done BEFORE any SVD)
    print("\n0️⃣ Preprocessing matrix...")
    matrix_preprocessed = preprocess_matrix(
        matrix,
        center=center,
        standardize=standardize,
        l2_normalize=l2_normalize
    )

    # Step 1: Perform SVD on preprocessed matrix and select k
    print("1️⃣ Performing SVD and selecting k...")
    print("  [SVD] Decomposing preprocessed matrix...")
    U, S, Vt = np.linalg.svd(matrix_preprocessed, full_matrices=False)
    k = select_k_singular_values(S, sv_threshold)

    # Step 2: Compute row importance scores on preprocessed matrix
    # (SVD is performed again inside, but on the SAME preprocessed matrix)
    row_scores, _ = compute_row_importance_scores(matrix_preprocessed, k)

    # Step 3: Sort all samples by importance score (high to low)
    print("\n3️⃣ Sorting samples by importance score...")
    sorted_indices = np.argsort(row_scores)[::-1]

    sample_scores = [
        {
            'sample_id': sample_ids[idx],
            'importance': float(row_scores[idx])
        }
        for idx in sorted_indices
    ]

    results = {
        'sample_scores': sample_scores,
        'k': int(k),
        'num_total': len(sample_ids),
        'preprocessing': {
            'center': center,
            'standardize': standardize,
            'l2_normalize': l2_normalize,
        },
    }

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total samples: {results['num_total']}")
    print(f"Top-k singular values: {k}")
    print(f"Preprocessing: center={center}, std={standardize}, L2={l2_normalize}")
    print(f"Top 10 samples by importance:")
    for i, item in enumerate(sample_scores[:10]):
        print(f"  {i+1}. ID={item['sample_id']}, importance={item['importance']:.6f}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CUR Matrix Decomposition - Compute Importance Scores for All Samples"
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        required=True,
        help='Directory containing all_representations.npz',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for sample importance scores (default: save to features_dir/importance_scores.jsonl)',
    )
    parser.add_argument(
        '--sv-threshold',
        type=float,
        default=0.9,
        help='Singular value cumulative energy threshold (default: 0.9)',
    )
    parser.add_argument(
        '--no-center',
        action='store_true',
        help='Disable centering (subtracting column means). Default: enabled',
    )
    parser.add_argument(
        '--no-standardize',
        action='store_true',
        help='Disable standardization (dividing by column std). Default: enabled',
    )
    parser.add_argument(
        '--no-l2-normalize',
        action='store_true',
        help='Disable L2-normalization (row-wise unit length). Default: enabled',
    )

    args = parser.parse_args()

    # Load representations
    matrix, sample_ids = load_representations(args.features_dir)

    # Determine output path: default to features_dir/importance_scores.jsonl
    if args.output is None:
        output_path = Path(args.features_dir) / 'importance_scores.jsonl'
        print(f"\nOutput path not specified, using default: {output_path}")
    else:
        output_path = Path(args.output)

    # Perform CUR decomposition and compute importance scores
    # Use NOT of the no-* flags (default: all enabled)
    results = cur_row_selection(
        matrix=matrix,
        sample_ids=sample_ids,
        sv_threshold=args.sv_threshold,
        center=not args.no_center,
        standardize=not args.no_standardize,
        l2_normalize=not args.no_l2_normalize,
    )

    # Save results in JSONL format (one sample per line)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        # Write each sample score as a separate line
        for item in results['sample_scores']:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nResults saved to: {output_path}")
    print(f"  Samples: {len(results['sample_scores'])} lines")


if __name__ == '__main__':
    main()
