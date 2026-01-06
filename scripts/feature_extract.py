"""
Feature extraction for Vision-Language Models with Multi-turn Conversation Support.

Extracts vision representations from user prompts' attention to vision tokens.

Supports ShareGPT format multi-turn conversations.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class VLMFeatureExtractor:
    """Extract features from Vision-Language Models."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        trust_remote_code: bool = True,
        torch_dtype: str = "auto",
        max_length: int = 4096,
        model_type: str = "llava",
    ):
        """
        Initialize the VLM feature extractor.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on
            trust_remote_code: Whether to trust remote code
            torch_dtype: Data type for model weights
            max_length: Maximum sequence length
            model_type: Model type - "llava" or "qwen"
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model_type = model_type.lower()

        # Map torch_dtype string to actual dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, "auto")

        print(f"Loading model: {model_name}")
        print(f"Model type: {self.model_type}")
        print(f"Device: {device}")
        print(f"Dtype: {torch_dtype}")
        print(f"Max length: {max_length}")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        # Set padding side to left for decoder-only models
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.torch_dtype,
            device_map=device if device == "auto" else None,
            attn_implementation="eager",  # Required for output_attentions
        )

        if device != "auto":
            self.model = self.model.to(device)

        self.model.eval()

        # Disable KV cache
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = False

        # Check if processor supports apply_chat_template
        self.use_chat_template = hasattr(self.processor, 'apply_chat_template')
        if self.use_chat_template:
            print("✓ Using official apply_chat_template")
        else:
            print("⚠ Fallback to manual formatting")

        print("Model loaded successfully!")

    def convert_sharegpt_to_hf_format(
        self,
        messages: List[Dict],
        num_images: int = 0,
    ) -> List[Dict]:
        """
        Convert ShareGPT format to HuggingFace LLaVA format.

        ShareGPT: [{"role": "user", "content": "text"}, ...]
        HF: [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}, ...]

        Args:
            messages: ShareGPT format messages
            num_images: Number of images in this conversation (will add image placeholders)

        Returns:
            HF format messages
        """
        hf_messages = []
        images_added = False

        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            hf_content = []

            # For the first user message, add image placeholders and remove existing <image> tags
            if role == 'user' and num_images > 0 and not images_added:
                hf_content.extend({"type": "image"} for _ in range(num_images))
                images_added = True
                # Remove <image> tags from content to avoid duplication
                content = content.replace('<image>', '').replace('<img>', '').strip()

            # Add text content
            if content:
                hf_content.append({"type": "text", "text": content})

            if hf_content:
                hf_messages.append({"role": role, "content": hf_content})

        return hf_messages

    def create_assistant_labels(
        self,
        input_ids: torch.Tensor,
        messages: List[Dict],
    ) -> torch.Tensor:
        """
        Create labels to identify assistant tokens (used for user_indices calculation).

        Strategy: Tokenize assistant responses and search for them in input_ids.
        This approach is robust to image tokens and special formatting.

        Args:
            input_ids: Input token IDs [seq_len] from batch processing
            messages: List of messages with 'role' and 'content' (ShareGPT format)

        Returns:
            labels: Same shape as input_ids, with -100 for non-assistant positions
        """
        labels = torch.full_like(input_ids, -100)

        try:
            # Handle padding: find actual content in input_ids
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                non_pad_mask = input_ids != pad_token_id
                if non_pad_mask.any():
                    first_non_pad = non_pad_mask.nonzero(as_tuple=True)[0][0].item()
                else:
                    first_non_pad = 0
            else:
                first_non_pad = 0

            # Search for each assistant response in input_ids
            search_start = first_non_pad

            for msg in messages:
                role = msg.get('role')
                content = msg.get('content', '').strip()

                if role == 'assistant' and content:
                    # Tokenize just the assistant's response content
                    # Use add_special_tokens=False to get pure content tokens
                    response_ids = self.processor.tokenizer(
                        content,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )['input_ids'][0]

                    if len(response_ids) == 0:
                        continue

                    # Search for this token sequence in input_ids
                    # Start from where we left off to maintain order
                    found_pos = self._find_token_sequence(
                        input_ids,
                        response_ids,
                        start_pos=search_start
                    )

                    if found_pos is not None:
                        # Mark these positions in labels
                        end_pos = found_pos + len(response_ids)
                        labels[found_pos:end_pos] = input_ids[found_pos:end_pos]
                        # Update search start for next assistant message
                        search_start = end_pos

        except Exception as e:
            print(f"Warning: Failed to create assistant labels: {e}")
            import traceback
            traceback.print_exc()

        return labels

    def _find_token_sequence(
        self,
        input_ids: torch.Tensor,
        pattern: torch.Tensor,
        start_pos: int = 0,
    ) -> int:
        """
        Find a token sequence in input_ids using sliding window.

        Args:
            input_ids: The full token sequence to search in
            pattern: The token pattern to find
            start_pos: Position to start searching from

        Returns:
            Starting position of the pattern, or None if not found
        """
        if len(pattern) == 0 or len(input_ids) == 0:
            return None

        pattern_len = len(pattern)
        max_start = len(input_ids) - pattern_len + 1

        for i in range(start_pos, max_start):
            if torch.equal(input_ids[i:i + pattern_len], pattern):
                return i

        return None

    def identify_token_types(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[List[int], List[int]]:
        """
        Identify vision and text tokens.

        Args:
            input_ids: Input token IDs [batch, seq_len] or [seq_len]

        Returns:
            (vision_indices, text_indices)
        """
        input_ids_flat = input_ids[0] if input_ids.dim() > 1 else input_ids

        vocab_size = self.processor.tokenizer.vocab_size
        vision_token_id = getattr(self.processor.tokenizer, 'vision_token_id', None)
        image_token_id = getattr(self.processor.tokenizer, 'image_token_id', None)

        vision_indices = []
        text_indices = []

        for idx, token_id in enumerate(input_ids_flat):
            token_id_val = token_id.item()
            is_vision = False

            if vision_token_id is not None and token_id_val == vision_token_id:
                is_vision = True
            elif image_token_id is not None and token_id_val == image_token_id:
                is_vision = True
            elif token_id_val >= vocab_size:
                is_vision = True

            if is_vision:
                vision_indices.append(idx)
            else:
                text_indices.append(idx)

        return vision_indices, text_indices

    def extract_features_batch(
        self,
        images_list: List[List[Image.Image]],
        messages_list: List[List[Dict]],
    ) -> List[Dict]:
        """
        Extract features for multi-turn conversations.

        For each sample:
        - Extract vision representation (from user prompts' attention)

        Args:
            images_list: List of image lists
            messages_list: List of conversation messages (ShareGPT format)

        Returns:
            List of dicts with features for each sample
        """
        batch_size = len(messages_list)

        # Build conversation text based on model type
        text_inputs = []
        for idx, messages in enumerate(messages_list):
            if self.model_type == 'qwen':
                # Use chat template for Qwen
                text_input = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                # LLaVA: use apply_chat_template if supported
                if self.use_chat_template:
                    # Convert ShareGPT format to HF format
                    num_images = len(images_list[idx]) if idx < len(images_list) else 0
                    hf_messages = self.convert_sharegpt_to_hf_format(messages, num_images)

                    # Use official chat template
                    text_input = self.processor.apply_chat_template(
                        hf_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                else:
                    # Fallback: Manual format for LLaVA
                    conversation_parts = []
                    for msg in messages:
                        role = msg.get('role', '')
                        content = msg.get('content', '')

                        if role == 'user':
                            conversation_parts.append(f"USER: {content}")
                        elif role == 'assistant':
                            conversation_parts.append(f"ASSISTANT: {content}")

                    text_input = " ".join(conversation_parts)

            text_inputs.append(text_input)

        # Process inputs
        # NOTE: Pass images_list directly (list of lists) so processor knows
        # which images belong to which text in the batch
        # Check if there are any actual images in the batch
        has_images = any(len(imgs) > 0 for imgs in images_list)

        if has_images:
            inputs = self.processor(
                text=text_inputs,
                images=images_list,  # Keep as list of lists, not flattened
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
        else:
            # Text-only processing (no images)
            inputs = self.processor(
                text=text_inputs,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

            # Extract outputs
            batch_input_ids = inputs['input_ids'].cpu().clone()
            batch_first_hidden = outputs.hidden_states[1] if len(outputs.hidden_states) > 1 else None
            batch_attention = outputs.attentions[0] if outputs.attentions else None

        # Cleanup inputs
        for key in list(inputs.keys()):
            if isinstance(inputs[key], torch.Tensor):
                del inputs[key]
        del inputs

        # Pre-process attention: compute mean and extract user-to-vision for each sample
        # This reduces O(n²) memory to O(n) before cloning
        processed_attentions = []
        if batch_attention is not None:
            # Mean over batch and heads once: [batch, heads, seq, seq] -> [batch, seq, seq]
            attention_mean = batch_attention.mean(dim=1)  # Average across heads

            for i in range(batch_size):
                # Only keep the 2D attention matrix for this sample
                processed_attentions.append(attention_mean[i].cpu().float())

            # Free the huge 4D tensor immediately
            del batch_attention, attention_mean
        else:
            processed_attentions = [None] * batch_size

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Process each sample
        results = []
        for i in range(batch_size):
            sample_input_ids = batch_input_ids[i:i+1].clone()
            sample_first_hidden = batch_first_hidden[i:i+1].float().cpu() if batch_first_hidden is not None else None
            sample_attention = processed_attentions[i]  # Already processed and on CPU

            # Identify token types
            vision_indices, text_indices = self.identify_token_types(sample_input_ids)

            # Create labels to identify assistant tokens precisely
            labels = self.create_assistant_labels(
                sample_input_ids[0],
                messages_list[i],
            )

            # User indices = text tokens that are NOT assistant tokens (labels == -100)
            # This ensures we only use USER tokens for vision attention
            assistant_mask = labels != -100
            user_indices = [
                idx for idx in text_indices
                if idx not in vision_indices and not assistant_mask[idx].item()
            ]

            results.append({
                'first_layer_hidden': sample_first_hidden,
                'attention': sample_attention,
                'input_ids': sample_input_ids,
                'vision_indices': vision_indices,
                'user_indices': user_indices,
            })

        # Cleanup
        del batch_first_hidden, processed_attentions, batch_input_ids, outputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def load_images(
        self,
        image_data: Union[str, List[str], np.ndarray, List[np.ndarray], Image.Image, List[Image.Image], dict, List[dict]],
    ) -> List[Image.Image]:
        """Load images from various formats."""
        if not isinstance(image_data, list):
            image_data = [image_data]

        images = []
        for item in image_data:
            try:
                if isinstance(item, Image.Image):
                    images.append(item)
                elif isinstance(item, str):
                    if item and item.strip():
                        images.append(Image.open(item).convert('RGB'))
                elif isinstance(item, np.ndarray):
                    if item.ndim == 2:
                        images.append(Image.fromarray(item, mode='L').convert('RGB'))
                    elif item.ndim == 3:
                        if item.dtype != np.uint8:
                            if item.max() <= 1.0:
                                item = (item * 255).astype(np.uint8)
                            else:
                                item = item.astype(np.uint8)
                        if item.shape[2] == 3:
                            images.append(Image.fromarray(item, mode='RGB'))
                        elif item.shape[2] == 4:
                            images.append(Image.fromarray(item, mode='RGBA').convert('RGB'))
            except Exception as e:
                print(f"Warning: Failed to load image: {e}")
                continue

        return images
