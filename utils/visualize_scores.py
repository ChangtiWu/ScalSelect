"""
Visualize CUR importance score distribution.

Usage:
    python utils/visualize_scores.py \
        --input features/sft_features/importance_scores.jsonl \
        --output figures/cur_distribution.png
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_cur_scores(scores_path: str) -> Tuple[np.ndarray, str]:
    """
    Load CUR importance scores from JSONL file.

    Args:
        scores_path: Path to importance_scores.jsonl

    Returns:
        Tuple of (scores array, metric name)
    """
    scores_list = []
    with open(scores_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            # Skip metadata line
            if 'metadata' in entry:
                continue
            scores_list.append(entry['importance'])

    scores = np.array(scores_list)
    return scores, 'CUR Importance Score'


def visualize_distribution(
    scores: np.ndarray,
    metric_name: str,
    output_path: str,
    title_prefix: str = ""
):
    """
    Create comprehensive score distribution visualization.

    Args:
        scores: Array of scores to visualize
        metric_name: Name of the metric (for labels)
        output_path: Path to save the figure
        title_prefix: Optional prefix for plot titles
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Add main title
    if title_prefix:
        fig.suptitle(f'{title_prefix} Distribution Analysis',
                    fontsize=16, fontweight='bold', y=0.995)

    # 1. Score distribution histogram
    ax = axes[0, 0]
    ax.hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution Histogram', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = (f'Mean: {scores.mean():.4f}\n'
                 f'Std: {scores.std():.4f}\n'
                 f'Median: {np.median(scores):.4f}\n'
                 f'Min: {scores.min():.4f}\n'
                 f'Max: {scores.max():.4f}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=10)

    # 2. Sorted scores (high to low)
    ax = axes[0, 1]
    sorted_scores = np.sort(scores)[::-1]
    ax.plot(range(len(sorted_scores)), sorted_scores, color='coral', linewidth=1.5)
    ax.set_xlabel('Sample Rank', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title('Ranking Curve (High to Low)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add percentile markers
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        idx = int(len(sorted_scores) * p / 100)
        ax.axvline(idx, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.text(idx, ax.get_ylim()[1] * 0.95, f'{p}%',
                fontsize=9, ha='center', color='gray',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 3. Cumulative distribution
    ax = axes[1, 0]
    cumsum = np.cumsum(sorted_scores)
    cumsum_normalized = cumsum / cumsum[-1]
    ax.plot(range(len(cumsum_normalized)), cumsum_normalized * 100,
            color='mediumseagreen', linewidth=2.5)
    ax.set_xlabel('Number of Top Samples', fontsize=12)
    ax.set_ylabel(f'Cumulative {metric_name} (%)', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add reference lines
    for threshold in [50, 80, 90, 95]:
        idx = np.searchsorted(cumsum_normalized * 100, threshold)
        if idx < len(cumsum_normalized):
            ax.axhline(threshold, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axvline(idx, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            pct = (idx / len(scores)) * 100
            ax.text(idx, threshold + 1.5, f'{idx} samples\n({pct:.1f}% of total)',
                    fontsize=8, ha='left', color='darkgreen',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 4. Box plot with violin overlay
    ax = axes[1, 1]

    # Violin plot for density
    parts = ax.violinplot([scores], vert=False, widths=0.7,
                          showmeans=False, showmedians=False)
    for pc in parts['bodies']:
        pc.set_facecolor('lightsteelblue')
        pc.set_alpha(0.6)

    # Box plot overlay
    bp = ax.boxplot([scores], vert=False, patch_artist=True,
                    widths=0.4, showmeans=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(marker='D', markerfacecolor='green', markersize=8))

    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_title('Statistical Summary (Box + Violin Plot)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_yticklabels([''])

    # Add quartile annotations
    q1, median, q3 = np.percentile(scores, [25, 50, 75])
    iqr = q3 - q1
    ax.text(q1, 1.3, f'Q1\n{q1:.4f}', ha='center', fontsize=10)
    ax.text(median, 1.3, f'Median\n{median:.4f}', ha='center', fontsize=10,
            color='red', fontweight='bold')
    ax.text(q3, 1.3, f'Q3\n{q3:.4f}', ha='center', fontsize=10)
    ax.text(scores.mean(), 0.5, f'Mean\n{scores.mean():.4f}', ha='center',
            fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.close()


def print_statistics(scores: np.ndarray, metric_name: str):
    """Print detailed statistics about the scores."""
    print(f"\n{metric_name} Statistics:")
    print("=" * 70)
    print(f"  Total samples: {len(scores)}")
    print(f"  Mean:          {scores.mean():.6f}")
    print(f"  Std:           {scores.std():.6f}")
    print(f"  Median:        {np.median(scores):.6f}")
    print(f"  Min:           {scores.min():.6f}")
    print(f"  Max:           {scores.max():.6f}")
    print(f"\nPercentiles:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(scores, p)
        print(f"  {p:3d}%:         {value:.6f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CUR importance score distribution"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to importance_scores.jsonl file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/cur_distribution.png',
        help='Output figure path (default: figures/cur_distribution.png)',
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CUR Importance Score Distribution Visualization")
    print("=" * 70)

    # Load CUR importance scores
    scores, metric_name = load_cur_scores(args.input)
    title_prefix = 'CUR Importance'

    print(f"Loaded {len(scores)} samples from: {args.input}")

    # Print statistics
    print_statistics(scores, metric_name)

    # Create visualization
    visualize_distribution(scores, metric_name, args.output, title_prefix)

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
