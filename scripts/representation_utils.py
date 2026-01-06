"""
Utility functions for extracting vision representations.

This module provides functions for:
1. Extracting vision representation from top 90% attention tokens (most attended by text)
2. Saving all results to a single file
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def extract_vision_representation(
    first_layer_hidden: torch.Tensor,
    attention_matrix: torch.Tensor,
    text_indices: List[int],
    vision_indices: List[int],
    cumulative_threshold: float = 0.9,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Extract vision representation by averaging embeddings of top attention vision tokens.

    Select vision tokens that account for top 90% (or specified threshold) cumulative
    attention from text tokens, then average their first layer embeddings.

    Args:
        first_layer_hidden: Hidden states from first layer [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        attention_matrix: Attention matrix [batch, num_heads, seq_len, seq_len]
        text_indices: Indices of text tokens in the sequence
        vision_indices: Indices of vision tokens in the sequence
        cumulative_threshold: Threshold for cumulative attention (default 0.9 = top 90%)

    Returns:
        Tuple of:
        - vision_repr: Averaged vision representation [hidden_dim]
        - selected_indices: Indices of selected vision tokens (relative to vision_indices)
    """
    # Remove batch dimension if present
    if first_layer_hidden.dim() == 3:
        first_layer_hidden = first_layer_hidden[0]  # [seq_len, hidden_dim]

    # Average attention across batch and heads
    attention_matrix = attention_matrix.mean(dim=[0, 1])  # [seq_len, seq_len]

    # Extract text-to-vision attention submatrix
    text_to_vision = attention_matrix[text_indices][:, vision_indices]
    # Shape: [num_text_tokens, num_vision_tokens]

    # Average across text tokens to get attention per vision token
    avg_attention_per_vision = text_to_vision.mean(dim=0)  # [num_vision_tokens]

    # Normalize to get probability distribution
    total_attention = avg_attention_per_vision.sum()
    if total_attention > 0:
        attention_probs = avg_attention_per_vision / total_attention
    else:
        # Fallback: uniform distribution
        attention_probs = torch.ones_like(avg_attention_per_vision) / len(avg_attention_per_vision)

    # Sort by attention in descending order
    sorted_attention, sorted_indices = torch.sort(attention_probs, descending=True)

    # Find tokens that account for cumulative threshold
    cumsum = torch.cumsum(sorted_attention, dim=0)

    # Find the index where cumulative sum exceeds threshold
    threshold_mask = cumsum <= cumulative_threshold
    # Include at least one token, and the first token that exceeds threshold
    num_selected = max(1, threshold_mask.sum().item() + 1)
    num_selected = min(num_selected, len(vision_indices))  # Don't exceed total vision tokens

    # Get selected indices (in sorted order)
    selected_sorted_indices = sorted_indices[:num_selected]

    # Map back to original vision token positions in sequence
    selected_vision_positions = [vision_indices[idx.item()] for idx in selected_sorted_indices]

    # Extract embeddings of selected vision tokens and average
    selected_embeddings = first_layer_hidden[selected_vision_positions]  # [num_selected, hidden_dim]
    vision_repr = selected_embeddings.mean(dim=0)  # [hidden_dim]

    # Return relative indices (within vision_indices)
    selected_relative_indices = [idx.item() for idx in selected_sorted_indices]

    return vision_repr, selected_relative_indices




def save_all_representations(
    results: List[Dict],
    output_path: Union[str, Path],
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save all sample representations to a single file.

    Each sample is saved with its ID as the key for easy indexing.

    Args:
        results: List of result dictionaries, each containing:
                 - 'sample_id': Sample ID
                 - 'combined_repr': Combined representation tensor or numpy array
                 - 'vision_repr': Vision representation (optional)
                 - 'text_repr': Text representation (optional)
                 - Other metadata fields
        output_path: Path to save the results
        metadata: Optional metadata dictionary to include in the output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving
    save_data = {
        'metadata': metadata or {},
        'num_samples': len(results),
    }

    # Prepare arrays indexed by sample ID
    # Use dictionary format where key = sample_id, value = representation
    representations_dict = {}

    for result in results:
        sample_id = result.get('sample_id', 'unknown')

        # Handle combined representation
        if 'combined_repr' in result:
            repr_data = result['combined_repr']
            if isinstance(repr_data, torch.Tensor):
                repr_data = repr_data.float().cpu().numpy()

            # Use sample_id as key for easy indexing
            # Convert to string to ensure compatibility
            representations_dict[str(sample_id)] = repr_data

    # Save representations as npz with single matrix + ID array
    npz_path = output_path.with_suffix('.npz')
    if representations_dict:
        # Convert dict to matrix format for fast loading (625K samples -> 2 arrays)
        sample_ids_array = np.array(list(representations_dict.keys()))
        matrix = np.stack(list(representations_dict.values()), axis=0)

        np.savez_compressed(npz_path,
                           representations=matrix,
                           sample_ids=sample_ids_array)
        print(f"Saved {len(representations_dict)} representations to: {npz_path}")
        print(f"Matrix shape: {matrix.shape}, Sample IDs: {len(sample_ids_array)}")

    # Save metadata as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to: {json_path}")

    # Print summary
    if representations_dict:
        sample_repr = next(iter(representations_dict.values()))
        print(f"Representation shape per sample: {sample_repr.shape}")
        print(f"Total samples: {len(representations_dict)}")


def load_representations(
    input_path: Union[str, Path],
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """
    Load saved representations from file.

    Args:
        input_path: Path to the saved file (without extension)

    Returns:
        Tuple of:
        - metadata: Dictionary with sample metadata and information
        - representations: Dictionary mapping sample IDs to representations
                          Access by: representations['sample_<id>']
    """
    input_path = Path(input_path)

    # Load metadata
    json_path = input_path.with_suffix('.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Load numpy arrays
    npz_path = input_path.with_suffix('.npz')
    npz_data = np.load(npz_path)

    # Convert to dictionary for easy access
    representations = {key: npz_data[key] for key in npz_data.files}

    print(f"Loaded {len(representations)} representations from {npz_path}")
    print(f"Sample IDs available: {list(representations)[:5]}...")

    return metadata, representations


def process_sample_representations(
    features: Dict,
    text_indices: List[int],
    vision_indices: List[int],
    cumulative_threshold: float = 0.9,
) -> Dict:
    """
    Process a single sample to extract vision representation.

    Args:
        features: Dictionary containing:
                  - 'first_layer_hidden': Hidden states from first layer (for vision)
                  - 'attention': Attention matrix
        text_indices: Indices of text tokens
        vision_indices: Indices of vision tokens
        cumulative_threshold: Threshold for cumulative attention selection

    Returns:
        Dictionary containing:
        - 'vision_repr': Vision representation [hidden_dim]
        - 'combined_repr': Same as vision_repr (for compatibility) [hidden_dim]
        - 'selected_vision_indices': Indices of selected vision tokens
        - 'num_selected_vision_tokens': Number of selected vision tokens
    """
    first_layer_hidden = features.get('first_layer_hidden')
    attention = features.get('attention')

    if first_layer_hidden is None:
        raise ValueError("first_layer_hidden is required")
    if attention is None:
        raise ValueError("attention is required")

    # Extract vision representation (top 90% attention tokens)
    vision_repr, selected_indices = extract_vision_representation(
        first_layer_hidden=first_layer_hidden,
        attention_matrix=attention,
        text_indices=text_indices,
        vision_indices=vision_indices,
        cumulative_threshold=cumulative_threshold,
    )

    return {
        'vision_repr': vision_repr,
        'combined_repr': vision_repr,  # Only use vision representation
        'selected_vision_indices': selected_indices,
        'num_selected_vision_tokens': len(selected_indices),
    }
