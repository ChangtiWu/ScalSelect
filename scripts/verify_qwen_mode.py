"""
Verify Qwen token identification for vision, user, and assistant tokens.
"""

import sys
sys.path.insert(0, '.')

from PIL import Image
import torch
from feature_extract import VLMFeatureExtractor


def create_test_image(size=(224, 224)):
    """Create a simple test image."""
    import numpy as np
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def verify_token_identification(model_path: str, model_type: str = "qwen"):
    """Verify token identification for the given model."""
    
    print("=" * 70)
    print(f"Token Identification Verification")
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    print("=" * 70)
    
    # Initialize extractor
    extractor = VLMFeatureExtractor(
        model_name=model_path,
        device="cuda",
        torch_dtype="bfloat16",
        max_length=2048,
        model_type=model_type,
    )
    
    # Test data: ShareGPT format
    test_messages = [
        {"role": "user", "content": "<image> What is shown in this image?"},
        {"role": "assistant", "content": "This is a test image with random colors."},
        {"role": "user", "content": "Can you describe it in more detail?"},
        {"role": "assistant", "content": "The image contains various colored pixels arranged randomly."},
    ]
    
    test_image = create_test_image()
    
    print("\n[Input Messages]")
    for msg in test_messages:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    # Extract features
    print("\n[Processing...]")
    results = extractor.extract_features_batch(
        images_list=[[test_image]],
        messages_list=[test_messages],
    )
    
    result = results[0]
    input_ids = result['input_ids'][0]
    vision_indices = result['vision_indices']
    user_indices = result['user_indices']
    
    # Decode tokens for visualization
    tokenizer = extractor.processor.tokenizer
    tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]
    
    print("\n" + "=" * 70)
    print("[Token Analysis]")
    print("=" * 70)
    
    print(f"\nTotal tokens: {len(input_ids)}")
    print(f"Vision tokens: {len(vision_indices)}")
    print(f"User tokens: {len(user_indices)}")
    
    # Create labels to identify assistant tokens
    labels = extractor.create_assistant_labels(input_ids, test_messages)
    assistant_indices = [i for i, l in enumerate(labels) if l != -100]
    print(f"Assistant tokens: {len(assistant_indices)}")
    
    # Visualize token types
    print("\n[Token Sequence Visualization]")
    print("-" * 70)
    
    # Group consecutive tokens by type for cleaner output
    current_type = None
    current_tokens = []
    
    def flush_group():
        if current_tokens:
            preview = ''.join(current_tokens)[:60]
            if len(''.join(current_tokens)) > 60:
                preview += "..."
            print(f"  [{current_type:10}] ({len(current_tokens):4} tokens): {repr(preview)}")
    
    for idx, token in enumerate(tokens):
        if idx in vision_indices:
            token_type = "VISION"
        elif idx in assistant_indices:
            token_type = "ASSISTANT"
        elif idx in user_indices:
            token_type = "USER"
        else:
            token_type = "OTHER"  # System, special tokens, etc.
        
        if token_type != current_type:
            flush_group()
            current_type = token_type
            current_tokens = []
        
        current_tokens.append(token)
    
    flush_group()
    
    # Verify coverage
    print("\n[Coverage Check]")
    print("-" * 70)
    
    all_classified = set(vision_indices) | set(user_indices) | set(assistant_indices)
    unclassified = [i for i in range(len(input_ids)) if i not in all_classified]
    
    print(f"Vision indices range: {min(vision_indices) if vision_indices else 'N/A'} - {max(vision_indices) if vision_indices else 'N/A'}")
    print(f"User indices count: {len(user_indices)}")
    print(f"Assistant indices count: {len(assistant_indices)}")
    print(f"Other/System tokens: {len(unclassified)}")
    
    # Show some unclassified tokens (usually system/special tokens)
    if unclassified:
        print(f"\nOther tokens (first 10):")
        for idx in unclassified[:10]:
            print(f"  [{idx:4}]: {repr(tokens[idx])}")
    
    # Verify no overlap
    print("\n[Overlap Check]")
    vision_set = set(vision_indices)
    user_set = set(user_indices)
    assistant_set = set(assistant_indices)
    
    v_u_overlap = vision_set & user_set
    v_a_overlap = vision_set & assistant_set
    u_a_overlap = user_set & assistant_set
    
    print(f"Vision ∩ User: {len(v_u_overlap)} (should be 0)")
    print(f"Vision ∩ Assistant: {len(v_a_overlap)} (should be 0)")
    print(f"User ∩ Assistant: {len(u_a_overlap)} (should be 0)")
    
    if v_u_overlap or v_a_overlap or u_a_overlap:
        print("\n⚠ WARNING: Token overlap detected!")
    else:
        print("\n✓ No overlap - token classification is correct!")
    
    print("\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--model-type", type=str, default="qwen", choices=["qwen", "llava"])
    args = parser.parse_args()
    
    verify_token_identification(args.model, args.model_type)
