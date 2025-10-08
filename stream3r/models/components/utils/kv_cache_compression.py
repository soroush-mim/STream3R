# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
KV Cache Compression utilities for inference-time memory reduction.

Implements token eviction based on attention scores while protecting:
1. Anchor frame (first frame) tokens
2. Special tokens (camera tokens, register tokens)
"""

import torch


def compute_token_importance(attn_weights, aggregation_method='sum'):
    """
    Compute importance score for each cached token based on attention weights.

    Args:
        attn_weights: [B, num_heads, num_query_tokens, num_cached_tokens]
            Attention weights from current frame queries attending to all cached tokens
        aggregation_method: How to aggregate importance across queries
            - 'sum': Sum attention received from all queries
            - 'max': Max attention received from any query
            - 'mean': Average attention received

    Returns:
        importance_scores: [B, num_heads, num_cached_tokens]
    """
    if aggregation_method == 'sum':
        importance = attn_weights.sum(dim=2)
    elif aggregation_method == 'max':
        importance = attn_weights.max(dim=2)[0]
    elif aggregation_method == 'mean':
        importance = attn_weights.mean(dim=2)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    return importance


def evict_tokens_by_importance(k, v, importance_scores,
                                tokens_per_frame,
                                patch_start_idx,
                                eviction_ratio=0.3):
    """
    Evict low-importance tokens while protecting anchor frame and special tokens.

    Protection rules:
    1. First frame (anchor): ALL tokens protected
    2. Subsequent frames: Special tokens (camera + register) protected
    3. Subsequent frames: Patch tokens can be evicted based on importance

    Args:
        k, v: [B, num_heads, total_tokens, head_dim]
            Key and value tensors from KV cache
        importance_scores: [B, num_heads, total_tokens]
            Importance score for each token (from attention weights)
        tokens_per_frame: Number of tokens per frame (P)
            Includes special tokens + patch tokens
        patch_start_idx: Index where patch tokens start
            Special tokens are [0:patch_start_idx]
        eviction_ratio: Fraction of evictable tokens to remove
            0.3 = remove 30% of the least important evictable tokens

    Returns:
        k_evicted, v_evicted: Reduced KV cache with same format
    """
    B, H, L, D = k.shape

    # If cache is just the anchor frame, don't evict
    if L <= tokens_per_frame:
        return k, v

    # Split anchor frame from rest
    k_anchor = k[:, :, :tokens_per_frame, :]  # [B, H, P, D]
    v_anchor = v[:, :, :tokens_per_frame, :]

    k_rest = k[:, :, tokens_per_frame:, :]  # [B, H, L-P, D]
    v_rest = v[:, :, tokens_per_frame:, :]
    importance_rest = importance_scores[:, :, tokens_per_frame:]  # [B, H, L-P]

    # Within rest, separate special tokens from patch tokens
    num_rest_tokens = k_rest.shape[2]
    num_frames_rest = num_rest_tokens // tokens_per_frame

    # Create mask for evictable tokens (patch tokens only, not special tokens)
    evictable_mask = torch.ones(num_rest_tokens, dtype=torch.bool, device=k.device)

    for frame_idx in range(num_frames_rest):
        frame_start = frame_idx * tokens_per_frame
        frame_special_end = frame_start + patch_start_idx
        # Mark special tokens as non-evictable
        evictable_mask[frame_start:frame_special_end] = False

    # Handle remainder tokens if num_rest_tokens is not perfectly divisible
    remainder = num_rest_tokens % tokens_per_frame
    if remainder > 0:
        # Protect special tokens in incomplete last frame
        last_frame_start = num_frames_rest * tokens_per_frame
        last_frame_special_end = min(last_frame_start + patch_start_idx, num_rest_tokens)
        evictable_mask[last_frame_start:last_frame_special_end] = False

    # Expand mask for batch and heads
    evictable_mask = evictable_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1)

    # Get indices of evictable tokens
    evictable_indices = torch.where(evictable_mask[0, 0])[0]  # Same for all batch/heads
    num_evictable = len(evictable_indices)

    if num_evictable == 0:
        # Nothing to evict, return as is
        return k, v

    # Calculate how many to evict vs keep
    num_to_evict = int(num_evictable * eviction_ratio)
    num_to_keep = num_evictable - num_to_evict

    if num_to_evict == 0:
        return k, v

    # Get importance scores for evictable tokens only
    # Shape: [B, H, num_evictable]
    importance_evictable = torch.gather(
        importance_rest,
        dim=2,
        index=evictable_indices.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
    )

    # Find top-k most important evictable tokens to KEEP
    _, top_k_indices = torch.topk(importance_evictable, k=num_to_keep, dim=2)

    # Map back to original indices in k_rest
    top_k_indices_original = torch.gather(
        evictable_indices.unsqueeze(0).unsqueeze(0).expand(B, H, -1),
        dim=2,
        index=top_k_indices
    )

    # Also keep all non-evictable tokens (special tokens)
    non_evictable_indices = torch.where(~evictable_mask[0, 0])[0]

    # Combine indices to keep
    indices_to_keep = torch.cat([
        non_evictable_indices.unsqueeze(0).unsqueeze(0).expand(B, H, -1),
        top_k_indices_original
    ], dim=2)

    # Sort to maintain temporal order
    indices_to_keep = indices_to_keep.sort(dim=2)[0]

    # Gather selected tokens
    k_rest_evicted = torch.gather(
        k_rest,
        dim=2,
        index=indices_to_keep.unsqueeze(-1).expand(-1, -1, -1, D)
    )
    v_rest_evicted = torch.gather(
        v_rest,
        dim=2,
        index=indices_to_keep.unsqueeze(-1).expand(-1, -1, -1, D)
    )

    # Concatenate anchor + evicted rest
    k_final = torch.cat([k_anchor, k_rest_evicted], dim=2)
    v_final = torch.cat([v_anchor, v_rest_evicted], dim=2)

    return k_final, v_final


def get_cache_statistics(kv_cache_list, tokens_per_frame):
    """
    Get statistics about KV cache for debugging/monitoring.

    Args:
        kv_cache_list: List of [k, v, attn_weights] tuples
        tokens_per_frame: Number of tokens per frame

    Returns:
        Dictionary with cache statistics
    """
    stats = {
        'num_layers': len(kv_cache_list),
        'cache_sizes': [],
        'num_frames': [],
        'total_tokens': 0,
        'avg_tokens_per_layer': 0.0,
    }

    for kv in kv_cache_list:
        if kv[0] is not None:
            cache_size = kv[0].shape[2]  # Number of cached tokens
            stats['cache_sizes'].append(cache_size)
            stats['num_frames'].append(cache_size / tokens_per_frame)
            stats['total_tokens'] += cache_size

    if len(stats['cache_sizes']) > 0:
        stats['avg_tokens_per_layer'] = stats['total_tokens'] / len(stats['cache_sizes'])

    return stats
