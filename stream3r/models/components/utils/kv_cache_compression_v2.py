# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
StreamVGGT-style KV Cache Compression with per-layer budget allocation.

Implements:
1. Cumulative Attention Maps (CAM) per layer
2. Exposure count tracking per layer
3. Variance-based layer importance computation
4. Per-layer token eviction with special token protection
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class LayerWiseKVCompression:
    """
    Manages per-layer KV cache compression using cumulative attention scores.

    This implements the StreamVGGT algorithm:
    - Track cumulative attention maps (CAM) per layer
    - Track exposure counts per layer
    - Compute layer importance via variance
    - Allocate eviction budget per layer
    - Protect special tokens (camera, register) from ALL frames
    """

    def __init__(
        self,
        num_layers: int,
        tokens_per_frame: int,
        patch_start_idx: int,
        total_frames: int,
        budget_ratio: float = 0.5,
        temp: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Initialize layer-wise KV compression manager.

        Args:
            num_layers: Number of decoder layers
            tokens_per_frame: Number of tokens per frame (P)
            patch_start_idx: Index where patch tokens start (special tokens are [0:patch_start_idx])
            total_frames: Total number of frames in the video
            budget_ratio: P in formula - percentage of total frames to keep (0.5 = 50% budget)
            temp: Temperature for softmax in layer importance computation
            device: Device to store tensors
        """
        self.num_layers = num_layers
        self.tokens_per_frame = tokens_per_frame
        self.patch_start_idx = patch_start_idx
        self.total_frames = total_frames
        self.budget_ratio = budget_ratio
        self.temp = temp
        self.device = device

        # Per-layer cumulative attention maps (CAM)
        # Each will be a 1D tensor of shape [num_tokens_in_cache]
        self.cum_attn_maps: List[Optional[torch.Tensor]] = [None] * num_layers

        # Per-layer exposure counts
        # Each will be a 1D tensor of shape [num_tokens_in_cache]
        self.exposure_counts: List[Optional[torch.Tensor]] = [None] * num_layers

        # Track current frame index
        self.current_frame = 0

    def initialize_first_frame(self, attn_maps_layers: List[torch.Tensor]):
        """
        Initialize CAM and exposure for the first frame.
        All tokens get inf attention to prevent eviction.

        Args:
            attn_maps_layers: List of attention maps [B, N, K] for each layer
                N = tokens_per_frame (query tokens from frame 1)
                K = tokens_per_frame (key tokens from frame 1)
        """
        for layer_idx, attn_map in enumerate(attn_maps_layers):
            # attn_map shape: [B, N, K] but for first frame N=K=tokens_per_frame
            # We only need to track key dimension (K)
            B, N, K = attn_map.shape

            # Initialize CAM to +inf for all tokens in first frame
            self.cum_attn_maps[layer_idx] = torch.full(
                (K,), float('inf'), device=self.device, dtype=torch.float32
            )

            # Initialize exposure counts to 1
            self.exposure_counts[layer_idx] = torch.ones(
                K, device=self.device, dtype=torch.int64
            )

        self.current_frame = 1
        print(f"✓ Initialized first frame: CAM and exposure for {self.num_layers} layers")

    def update_cumulative_maps(
        self,
        attn_maps_layers: List[torch.Tensor],
        frame_idx: int
    ):
        """
        Update cumulative attention maps and exposure counts for new frame.

        Args:
            attn_maps_layers: List of attention maps [B, N, K] for each layer
                N = tokens_per_frame (current frame queries)
                K = total_cached_tokens (all previous + current)
            frame_idx: Current frame index (1-indexed)
        """
        for layer_idx, attn_map in enumerate(attn_maps_layers):
            B, N, K = attn_map.shape  # [B, tokens_per_frame, total_cache_size]

            # Get current cumulative map and exposure
            curr_cum = self.cum_attn_maps[layer_idx]
            curr_exp = self.exposure_counts[layer_idx]

            # Create new tensors for updated values
            new_cum = torch.empty(K, device=self.device, dtype=torch.float32)
            new_exp = torch.ones(K, device=self.device, dtype=torch.int64)

            # Sum attention across queries (rows) to get per-token scores
            # Shape: [B, K] -> [K] (reduce batch too)
            token_scores = attn_map.sum(dim=1).mean(dim=0) * float(K)  # Scale by K
            new_cum.copy_(token_scores)

            # Add previous cumulative values for old tokens
            old_size = min(curr_cum.size(0), K)
            if old_size > 0:
                new_cum[:old_size].add_(curr_cum[:old_size])
                new_exp[:old_size].copy_(curr_exp[:old_size]).add_(1)

            # Protect special tokens from current frame by setting CAM to inf
            # Current frame tokens are at indices [-tokens_per_frame:]
            frame_start = K - self.tokens_per_frame
            frame_special_end = frame_start + self.patch_start_idx
            new_cum[frame_start:frame_special_end] = float('inf')

            # Update stored maps
            self.cum_attn_maps[layer_idx] = new_cum
            self.exposure_counts[layer_idx] = new_exp

            # Cleanup
            del curr_cum, curr_exp, token_scores

        self.current_frame = frame_idx
        print(f"✓ Updated CAM for frame {frame_idx}, cache size: {K} tokens")

    def compute_layer_importance(
        self,
        attn_maps_layers: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-layer importance using variance of attention patterns.
        Lower variance = more focused attention = more important layer.

        Args:
            attn_maps_layers: List of attention maps [B, N, K] for each layer

        Returns:
            alphas: [num_layers] - normalized importance weights (sum to 1)
            variances: [num_layers] - raw variance values per layer
        """
        variances = []

        for attn_map in attn_maps_layers:
            # Compute variance of attention distribution
            # Shape: [B, N, K] -> scalar
            attn_map = attn_map.sum(dim=1).squeeze(0)
            var = attn_map.var(unbiased=False)
            variances.append(var)

        variances = torch.stack(variances)  # [num_layers]

        # Apply inverse-variance softmax: lower variance = higher importance
        alphas = F.softmax(-variances / self.temp, dim=0)

        return alphas, variances

    def compute_per_layer_budgets(
        self,
        attn_maps_layers: List[torch.Tensor],
        current_frame_idx: int
    ) -> List[int]:
        """
        Compute how many tokens to remove from each layer based on:
        1. Layer importance (variance-based)
        2. Total budget constraint

        Formula from StreamVGGT line 387-388:
            total_budget = len_frames * L * frame_token_num
            target_tokens_per_layer = (frame_token_num * (S+1)) - (total_budget * P * alphas)

        This gives the TARGET number of tokens to KEEP (not remove).
        Higher alpha (lower variance/more important) = keep more tokens.

        Args:
            attn_maps_layers: List of attention maps for layer importance
            current_frame_idx: Current frame index S (0-indexed for frames processed)

        Returns:
            List of integers indicating tokens to remove per layer
        """
        # Compute layer importance
        alphas, variances = self.compute_layer_importance(attn_maps_layers)

        # Total budget in tokens across all layers
        total_budget = self.total_frames * self.num_layers * self.tokens_per_frame

        current_cache_size = self.tokens_per_frame * (current_frame_idx + 1)

        target_per_layer = current_cache_size - (total_budget * self.budget_ratio * alphas)
        target_per_layer = torch.clamp(target_per_layer, min=0).int().tolist()

        print(f"✓ Layer budgets: alphas={alphas.cpu().numpy().round(3)}, "
              f"variances={variances.cpu().numpy().round(4)}, "
              f"targets={target_per_layer}")

        return target_per_layer

    def get_removal_indices_per_layer(
        self,
        tk_rm_num_per_layer: List[int],
        frame_idx: int
    ) -> List[Optional[torch.Tensor]]:
        """
        Compute indices to remove from each layer based on exposure-normalized importance.

        Args:
            tk_rm_num_per_layer: Number of tokens to remove per layer

        Returns:
            List of 1D tensors with indices to remove (or None if no removal)
        """
        removal_indices = []
        total_evicted = 0

        for layer_idx, num_to_remove in enumerate(tk_rm_num_per_layer):
            if num_to_remove <= 0:
                removal_indices.append(None)
                continue


            # Get CAM and exposure for this layer
            cam = self.cum_attn_maps[layer_idx]
            exp = self.exposure_counts[layer_idx]

            rm_num = num_to_remove - (self.tokens_per_frame*(frame_idx+1) - cam.shape[0])
            if rm_num > 0:

                # Compute exposure-normalized scores
                # Lower score = less important = candidate for removal
                scores = cam / (exp.float() + 1e-6)

                # Find lowest scoring tokens
                _, indices = torch.topk(scores, rm_num, largest=False)

                removal_indices.append(indices)
                total_evicted += rm_num
            else:
                removal_indices.append(None)

        print(f"✓ Computed removal indices: {total_evicted} tokens to evict")
        return removal_indices

    def compact_kv_cache(
        self,
        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        removal_indices_per_layer: List[Optional[torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, None]]:
        """
        Remove tokens from KV cache and update CAM/exposure accordingly.

        Args:
            kv_cache_list: List of (K, V, attn) tuples per layer
                K, V: [B, H, T, D]
                attn: not used (set to None after eviction)
            removal_indices_per_layer: Indices to remove per layer

        Returns:
            Compacted KV cache list with updated (K, V, None)
        """
        compacted_cache = []

        for layer_idx, (K, V, _) in enumerate(kv_cache_list):
            rem_indices = removal_indices_per_layer[layer_idx]

            # If no removal for this layer, keep as is
            if rem_indices is None or rem_indices.numel() == 0:
                compacted_cache.append((K, V, None))
                continue

            # Get dimensions
            B, H, T, D = K.shape
            device = K.device

            # Create keep mask
            keep_mask = torch.ones(T, dtype=torch.bool, device=device)
            keep_mask[rem_indices] = False
            keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

            # Compact K and V
            K_new = K.index_select(2, keep_indices)
            V_new = V.index_select(2, keep_indices)

            # Update CAM and exposure
            cam = self.cum_attn_maps[layer_idx]
            exp = self.exposure_counts[layer_idx]

            cam_new = cam.index_select(0, keep_indices)
            exp_new = exp.index_select(0, keep_indices)

            self.cum_attn_maps[layer_idx] = cam_new
            self.exposure_counts[layer_idx] = exp_new

            compacted_cache.append((K_new, V_new, None))

            # Cleanup
            del K, V, keep_mask, keep_indices, cam, exp

        torch.cuda.empty_cache()
        return compacted_cache

    def process_frame(
        self,
        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        frame_idx: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor, None]]:
        """
        Main entry point: process a new frame and apply eviction if needed.

        Args:
            kv_cache_list: List of (K, V, attn_avg) tuples from aggregator
            frame_idx: Current frame index (0-indexed)

        Returns:
            Compacted KV cache list
        """
        # Extract attention maps from cache
        attn_maps_layers = [kv[2] for kv in kv_cache_list if kv[2] is not None]

        if frame_idx == 0:
            # First frame: initialize CAM and exposure
            self.initialize_first_frame(attn_maps_layers)
            # Don't evict from first frame
            return [(k, v, None) for k, v, _ in kv_cache_list]

        # Update cumulative attention maps
        self.update_cumulative_maps(attn_maps_layers, frame_idx)

        # Skip eviction on last frame
        if frame_idx >= self.total_frames - 1:
            print(f"✓ Last frame {frame_idx}, skipping eviction")
            return [(k, v, None) for k, v, _ in kv_cache_list]

        # Compute per-layer budgets
        tk_rm_num_per_layer = self.compute_per_layer_budgets(attn_maps_layers, frame_idx)

        # Check if any eviction needed
        if all(n == 0 for n in tk_rm_num_per_layer):
            print(f"✓ No eviction needed for frame {frame_idx}")
            return [(k, v, None) for k, v, _ in kv_cache_list]

        # Compute removal indices
        removal_indices = self.get_removal_indices_per_layer(tk_rm_num_per_layer, frame_idx)

        # Compact KV cache
        compacted_cache = self.compact_kv_cache(kv_cache_list, removal_indices)

        return compacted_cache

    def reset(self):
        """Reset all state for a new sequence."""
        self.cum_attn_maps = [None] * self.num_layers
        self.exposure_counts = [None] * self.num_layers
        self.current_frame = 0
        print("✓ Reset compression state")
