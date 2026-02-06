"""
Multi-turn Conversation Feature Extraction

Strategy:
- For each sample: Process full multi-turn conversation (image + all user/assistant turns)
- Single forward pass: image + user-0 + assistant-0 + user-1 + assistant-1 + ...
- Vision representation: Average of top attended vision tokens (by USER prompts only)

Outputs:
- Vision representations: Saved to NPZ format

Usage:
    accelerate launch --num_processes=8 scripts/feature_extract_sft.py \
        --model /path/to/model \
        --dataset /path/to/dataset.json \
        --output-dir ./output \
        --sample-batch-size 4
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast_object_list
from tqdm import tqdm

from feature_extract import VLMFeatureExtractor
from representation_utils import (
    save_all_representations,
    load_representations,
)


def load_dataset(dataset_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load ShareGPT format dataset."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples and max_samples > 0:
        data = data[:max_samples]

    return data


def process_image_paths(images: List[str], base_path: str = None) -> List[str]:
    """Process image paths, add base prefix if needed."""
    if base_path is None:
        base_path = "<your dataset base path>" # e.g. "/mnt/project_ai4edu/share/code/RobobrainFactory/data/"

    return [
        img if img.startswith('/') else base_path + img
        for img in images
    ]


def process_sample_batch(
    extractor: VLMFeatureExtractor,
    samples: List[Dict],
    cumulative_threshold: float = 0.9,
) -> List[Optional[Dict]]:
    """
    Process a batch of samples with multi-turn conversations.

    For each sample:
    - Extract vision representation (from user->vision attention)

    Args:
        extractor: Feature extractor
        samples: List of samples with full conversations
        cumulative_threshold: Attention threshold for vision token selection

    Returns:
        List of result dicts (one per sample), None for failed samples
    """
    batch_results = []

    # Prepare batch inputs
    images_list = []
    messages_list = []
    sample_ids = []

    for sample in samples:
        sample_id = sample.get('id')
        if sample_id is None:
            batch_results.append(None)
            continue

        # Extract and process images
        images = sample.get('images', [])
        if not isinstance(images, list):
            images = [images] if images else []
        images = process_image_paths(images)

        # Get full conversation messages (user + assistant)
        messages = sample.get('messages', [])
        if not messages:
            batch_results.append(None)
            continue

        images_list.append(images)
        messages_list.append(messages)
        sample_ids.append(sample_id)

    # Skip if no valid samples in batch
    if not sample_ids:
        return [None] * len(samples)

    try:
        # Load all images for this batch (parallel I/O for better performance)
        def load_images_safe(images):
            """Load images and return None if failed."""
            loaded = extractor.load_images(images)
            return loaded if loaded else None

        with ThreadPoolExecutor(max_workers=4) as executor:
            loaded_images_list = list(executor.map(load_images_safe, images_list))

        # Filter out failed samples
        valid_indices = [
            i for i, imgs in enumerate(loaded_images_list)
            if imgs is not None
        ]

        if not valid_indices:
            return [None] * len(samples)

        valid_images = [loaded_images_list[i] for i in valid_indices]
        valid_messages = [messages_list[i] for i in valid_indices]
        valid_sample_ids = [sample_ids[i] for i in valid_indices]

        # Batch inference: Process all samples together
        batch_features = extractor.extract_features_batch(
            images_list=valid_images,
            messages_list=valid_messages,
        )

        # Process each sample's features
        results_map = {}
        for idx, features in enumerate(batch_features):
            sample_id = valid_sample_ids[idx]

            try:
                # Get token indices
                vision_indices = features.get('vision_indices', [])
                user_indices = features.get('user_indices', [])

                if not user_indices or not vision_indices:
                    results_map[sample_id] = None
                    continue

                # Validate required features
                if not all([
                    features.get('attention') is not None,
                    features.get('first_layer_hidden') is not None,
                ]):
                    results_map[sample_id] = None
                    continue

                # Extract vision representation using USER attention only
                vision_repr, selected_indices = extract_user_attended_vision(
                    first_layer_hidden=features['first_layer_hidden'],
                    attention_matrix=features['attention'],
                    user_indices=user_indices,
                    vision_indices=vision_indices,
                    cumulative_threshold=cumulative_threshold,
                )

                # Move to CPU and convert to numpy
                results_map[sample_id] = {
                    'sample_id': sample_id,
                    'combined_repr': vision_repr.cpu().numpy(),
                    'success': True,
                }

            except Exception as e:
                print(f"Warning: Failed to process sample {sample_id}: {e}")
                results_map[sample_id] = None

        # Map results back to original order
        final_results = []
        for sample in samples:
            sample_id = sample.get('id')
            if sample_id in results_map:
                final_results.append(results_map[sample_id])
            else:
                final_results.append(None)

        return final_results

    except Exception as e:
        print(f"Warning: Batch processing failed: {e}")
        return [None] * len(samples)

    finally:
        # Aggressive cleanup to prevent memory buildup
        if 'loaded_images_list' in locals():
            del loaded_images_list
        if 'batch_features' in locals():
            del batch_features
        if 'valid_images' in locals():
            del valid_images
        if 'results_map' in locals():
            del results_map

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations complete


def extract_user_attended_vision(
    first_layer_hidden: torch.Tensor,
    attention_matrix: torch.Tensor,
    user_indices: List[int],
    vision_indices: List[int],
    cumulative_threshold: float = 0.9,
) -> tuple:
    """
    Extract vision representation using USER tokens' attention only.

    Args:
        first_layer_hidden: Hidden states from first layer [batch, seq, hidden] or [seq, hidden]
        attention_matrix: Attention matrix [seq_len, seq_len] (already averaged over heads)
        user_indices: Indices of user tokens
        vision_indices: Indices of vision tokens
        cumulative_threshold: Threshold for cumulative attention

    Returns:
        Tuple of (vision_repr, selected_indices)
    """
    # Remove batch dimension if present
    if first_layer_hidden.dim() == 3:
        first_layer_hidden = first_layer_hidden[0]

    # Attention matrix is already [seq_len, seq_len] (pre-processed)
    # No need to average over batch and heads anymore

    # Extract USER-to-vision attention submatrix
    user_to_vision = attention_matrix[user_indices][:, vision_indices]
    # Shape: [num_user_tokens, num_vision_tokens]

    # Average across user tokens to get attention per vision token
    avg_attention_per_vision = user_to_vision.mean(dim=0)  # [num_vision_tokens]

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

    # Find the first position where cumulative sum exceeds threshold
    exceeds_threshold = cumsum > cumulative_threshold
    if exceeds_threshold.any():
        # Select up to and including the first token that exceeds threshold
        num_selected = exceeds_threshold.nonzero(as_tuple=True)[0][0].item() + 1
    else:
        # If no token exceeds threshold (rare), select all
        num_selected = len(vision_indices)

    # Ensure at least 1 token is selected
    num_selected = max(1, min(num_selected, len(vision_indices)))

    # Get selected indices
    selected_sorted_indices = sorted_indices[:num_selected]

    # Map back to original vision token positions in sequence
    selected_vision_positions = [vision_indices[idx.item()] for idx in selected_sorted_indices]

    # Extract embeddings of selected vision tokens and average
    selected_embeddings = first_layer_hidden[selected_vision_positions]
    vision_repr = selected_embeddings.mean(dim=0)

    # Return relative indices (within vision_indices)
    selected_relative_indices = [idx.item() for idx in selected_sorted_indices]

    return vision_repr, selected_relative_indices


def process_dataset(
    extractor: VLMFeatureExtractor,
    samples: List[Dict],
    sample_batch_size: int,
    cumulative_threshold: float = 0.9,
    process_rank: int = 0,
) -> List[Dict]:
    """
    Process all samples with multi-turn conversation support.

    Args:
        extractor: Feature extractor
        samples: List of all samples
        sample_batch_size: Number of samples to process together
        cumulative_threshold: Attention threshold
        process_rank: Process rank for logging

    Returns:
        List of successful results
    """
    results = []

    pbar = tqdm(
        total=len(samples),
        desc=f"GPU {process_rank}",
        disable=(process_rank != 0),
        mininterval=2.0,
    )

    # Process in batches
    for i in range(0, len(samples), sample_batch_size):
        batch = samples[i:i + sample_batch_size]

        batch_results = process_sample_batch(
            extractor=extractor,
            samples=batch,
            cumulative_threshold=cumulative_threshold,
        )

        # Collect successful results
        for result in batch_results:
            if result and result.get('success'):
                results.append(result)

        pbar.update(len(batch))
        pbar.set_postfix({'success': len(results)})

    pbar.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn conversation feature extraction"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name or path (e.g., Qwen/Qwen2-VL-7B-Instruct, llava-1.5-7b-hf)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to ShareGPT format JSON dataset',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for extracted features',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=-1,
        help='Maximum number of samples to process (-1 for all)',
    )
    parser.add_argument(
        '--sample-batch-size',
        type=int,
        default=4,
        help='Number of samples to process together (default: 4)',
    )
    parser.add_argument(
        '--torch-dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help='Data type for model weights',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=4096,
        help='Maximum sequence length (default: 4096)',
    )
    parser.add_argument(
        '--cumulative-threshold',
        type=float,
        default=0.9,
        help='Cumulative attention threshold for vision tokens (default: 0.9)',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='llava',
        choices=['llava', 'qwen'],
        help='Model type: llava or qwen (default: llava)',
    )

    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print("=" * 70)
        print("Multi-turn Conversation Feature Extraction - Configuration")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Model type: {args.model_type}")
        print(f"Dataset: {args.dataset}")
        print(f"Output: {args.output_dir}")
        print(f"Max samples: {args.max_samples if args.max_samples > 0 else 'All'}")
        print(f"Sample batch size: {args.sample_batch_size}")
        print(f"Num GPUs: {accelerator.num_processes}")
        print(f"Torch dtype: {args.torch_dtype}")
        print(f"Cumulative threshold: {args.cumulative_threshold}")
        print("=" * 70)

    # Load dataset
    if accelerator.is_main_process:
        samples = load_dataset(
            args.dataset,
            max_samples=args.max_samples if args.max_samples > 0 else None
        )
        print(f"\nLoaded {len(samples)} samples")
    else:
        samples = None

    # Broadcast to all processes
    samples = broadcast_object_list([samples])[0]

    # Split across GPUs
    with accelerator.split_between_processes(samples) as local_samples:
        if accelerator.is_main_process:
            print(f"Each GPU processes ~{len(local_samples)} samples\n")

        # Initialize extractor
        extractor = VLMFeatureExtractor(
            model_name=args.model,
            device=accelerator.device,
            torch_dtype=args.torch_dtype,
            max_length=args.max_length,
            model_type=args.model_type,
        )

        # Process samples
        print(f"GPU {accelerator.process_index}: Starting to process {len(list(local_samples))} samples")
        results = process_dataset(
            extractor=extractor,
            samples=list(local_samples),
            sample_batch_size=args.sample_batch_size,
            cumulative_threshold=args.cumulative_threshold,
            process_rank=accelerator.process_index,
        )
        print(f"GPU {accelerator.process_index}: Finished processing, got {len(results)} successful results")

        # Clean up model before gather to reduce memory pressure
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Gather results with timeout protection
    # IMPORTANT: All processes must reach here together
    accelerator.wait_for_everyone()
    all_results = gather_object(results)

    # Free local results after gather
    del results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save (main process only)
    if accelerator.is_main_process:
        print(f"\nSuccessfully processed: {len(all_results)}/{len(samples)} samples")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            'model': args.model,
            'dataset': args.dataset,
            'num_samples': len(all_results),
            'sample_batch_size': args.sample_batch_size,
            'cumulative_threshold': args.cumulative_threshold,
            'extraction_mode': 'multi-turn-conversation',
            'description': 'Vision representation from user attention',
        }

        output_path = output_dir / 'all_representations'

        # Save vision representations (NPZ format)
        save_all_representations(
            results=all_results,
            output_path=output_path,
            metadata=metadata,
        )

        # Verify by loading
        load_representations(output_path)

        print(f"\n✓ Features saved to: {output_path}")
        print("=" * 70)
    else:
        # Non-main processes: free gathered results immediately
        del all_results

    # Final cleanup (all processes)
    # Note: extractor already deleted in the with block
    del samples
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.free_memory()


if __name__ == '__main__':
    main()
