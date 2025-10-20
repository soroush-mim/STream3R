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
5. Optional D2O-style token merging with:
   - Nearest neighbor matching using cosine similarity
   - EMA threshold for merge/discard decision
   - Weighted token merging based on similarity
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
        device: str = 'cuda',
        enable_token_merging: bool = False,
        merge_beta: float = 0.7
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
            enable_token_merging: Whether to enable D2O token merging (default: False)
            merge_beta: EMA smoothing constant for token merging threshold (default: 0.7)
        """
        self.num_layers = num_layers
        self.tokens_per_frame = tokens_per_frame
        self.patch_start_idx = patch_start_idx
        self.total_frames = total_frames
        self.budget_ratio = budget_ratio
        self.temp = temp
        self.device = device
        self.enable_token_merging = enable_token_merging
        self.merge_beta = merge_beta

        # Per-layer cumulative attention maps (CAM)
        # Each will be a 1D tensor of shape [num_tokens_in_cache]
        self.cum_attn_maps: List[Optional[torch.Tensor]] = [None] * num_layers

        # Per-layer exposure counts
        # Each will be a 1D tensor of shape [num_tokens_in_cache]
        self.exposure_counts: List[Optional[torch.Tensor]] = [None] * num_layers

        # Track current frame index
        self.current_frame = 0

        # Token merging: EMA threshold per layer
        self.ema_thresholds: List[Optional[float]] = [None] * num_layers

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

    def nearest_neighbor_matching(
        self,
        K_evicted: torch.Tensor,
        K_conserved: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute nearest neighbor matching between evicted and conserved tokens.
        Based on D2O paper Section 4.2.2, Equation 9.

        Args:
            K_evicted: Evicted key tokens [L_e, D]
            K_conserved: Conserved key tokens [L_c, D]

        Returns:
            similarity_matrix: Cosine similarity matrix [L_e, L_c]
            nearest_indices: Index of nearest conserved token for each evicted token [L_e]
        """
        # Compute cosine similarity: u_i,j = (k_i^T * k_j) / (||k_i|| ||k_j||)
        K_evicted_norm = F.normalize(K_evicted, p=2, dim=-1)  # [L_e, D]
        K_conserved_norm = F.normalize(K_conserved, p=2, dim=-1)  # [L_c, D]

        similarity_matrix = torch.matmul(K_evicted_norm, K_conserved_norm.T)  # [L_e, L_c]

        # Find nearest neighbor for each evicted token
        nearest_indices = torch.argmax(similarity_matrix, dim=1)  # [L_e]

        return similarity_matrix, nearest_indices

    def update_ema_threshold(
        self,
        similarity_matrix: torch.Tensor,
        layer_idx: int,
        is_first_frame: bool = False
    ) -> float:
        """
        Update EMA threshold for token merging decision.
        Based on D2O paper Section 4.2.2, Equation 10.

        Args:
            similarity_matrix: Similarity matrix [L_e, L_c] or [L_c]
            layer_idx: Current layer index
            is_first_frame: Whether this is the first eviction (not frame 0, but first
                            frame where eviction actually occurs, typically frame 1)

        Returns:
            Current threshold value

        Note:
            - First eviction: τ_0 = (1/L_e) * Σ Max(U_0[i,:])
            - Subsequent: τ_t = β*Max(U_t[:]) + (1-β)*τ_(t-1)
        """
        if is_first_frame:
            # τ_0 = (1/L_e) * sum(Max(U_t[i,:]))
            if similarity_matrix.dim() == 2:
                max_similarities = torch.max(similarity_matrix, dim=1)[0]  # [L_e]
                threshold = max_similarities.mean().item()
            else:
                threshold = similarity_matrix.max().item()
            self.ema_thresholds[layer_idx] = threshold
        else:
            # τ_t = β*Max(U_t[:]) + (1-β)*τ_{t-1}
            current_max = similarity_matrix.max().item()
            prev_threshold = self.ema_thresholds[layer_idx]
            if prev_threshold is None:
                threshold = current_max
            else:
                threshold = self.merge_beta * current_max + (1 - self.merge_beta) * prev_threshold
            self.ema_thresholds[layer_idx] = threshold

        return threshold
        
    @torch.no_grad()
    def weighted_token_merging(
        self,
        K_conserved: torch.Tensor,
        V_conserved: torch.Tensor,
        K_evicted: torch.Tensor,
        V_evicted: torch.Tensor,
        similarity_matrix: torch.Tensor,
        nearest_indices: torch.Tensor,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge evicted tokens into conserved tokens using weighted merging.
        Based on D2O paper Section 4.2.2, Equations 11 and 12.

        Args:
            K_conserved: Conserved key tokens [L_c, D]
            V_conserved: Conserved value tokens [L_c, D]
            K_evicted: Evicted key tokens [L_e, D]
            V_evicted: Evicted value tokens [L_e, D]
            similarity_matrix: Similarity matrix [L_e, L_c]
            nearest_indices: Nearest conserved token for each evicted token [L_e]
            threshold: EMA threshold for merging decision

        Returns:
            K_merged: Merged key tokens [L_c, D]
            V_merged: Merged value tokens [L_c, D]
        """
        L_c = K_conserved.shape[0]

        # Get max similarity for each evicted token
        max_similarities = torch.max(similarity_matrix, dim=1)[0]  # [L_e]

        # Create mask: only merge tokens above threshold
        merge_mask = max_similarities >= threshold  # [L_e]

        if merge_mask.sum() == 0:
            # No tokens to merge
            return K_conserved.clone(), V_conserved.clone()

        # Filter tokens to merge
        K_evicted_merge = K_evicted[merge_mask]  # [L_merge, D]
        V_evicted_merge = V_evicted[merge_mask]  # [L_merge, D]
        nearest_indices_merge = nearest_indices[merge_mask]  # [L_merge]
        similarity_matrix_merge = similarity_matrix[merge_mask]  # [L_merge, L_c]

        L_merge = K_evicted_merge.shape[0]

        # Create matching mask matrix: m_ij = 1 if j is nearest to i, else 0
        match_mask = torch.zeros(L_merge, L_c, device=K_conserved.device)
        match_mask[torch.arange(match_mask.shape[0], device=K_conserved.device), nearest_indices_merge] = 1.0

        # Compute weighted merging weights (Equation 12)
        # w_ei = sum(exp(u_ij)*m_ij) / (sum(exp(u_ij)*m_ij) + e)
        # w_cj = e / (sum(exp(u_ij)*m_ij) + e)

        exp_sim = torch.exp(similarity_matrix_merge)  # [L_merge, L_c]
        weighted_exp_sim = exp_sim * match_mask  # [L_merge, L_c]

        # Sum over evicted tokens for each conserved token
        sum_weighted = weighted_exp_sim.sum(dim=0)  # [L_c]

        e = torch.e
        w_conserved = e / (sum_weighted + e)  # [L_c]

        # Compute weights for each evicted token
        w_evicted = weighted_exp_sim / (weighted_exp_sim.sum(dim=1, keepdim=True) + e)  # [L_merge, L_c]

        # Clone for merging
        K_merged = K_conserved.clone()
        V_merged = V_conserved.clone()

        # Apply weighted merging (Equation 11)
        # k_cj = w_cj*k_cj + sum(w_ei*k_ei)
        # v_cj = w_cj*v_cj + sum(w_ei*v_ei)

        K_merged = K_merged * w_conserved.unsqueeze(1)  # [L_c, D]
        V_merged = V_merged * w_conserved.unsqueeze(1)  # [L_c, D]

        # Add weighted contributions from evicted tokens
        K_contribution = torch.matmul(w_evicted.T, K_evicted_merge)  # [L_c, D]
        V_contribution = torch.matmul(w_evicted.T, V_evicted_merge)  # [L_c, D]

        K_merged = K_merged + K_contribution
        V_merged = V_merged + V_contribution

        return K_merged, V_merged

    def compact_kv_cache(
        self,
        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        removal_indices_per_layer: List[Optional[torch.Tensor]],
        is_first_frame: bool = False
    ) -> List[Tuple[torch.Tensor, torch.Tensor, None]]:
        """
        Remove tokens from KV cache and update CAM/exposure accordingly.
        Optionally apply D2O token merging.

        Args:
            kv_cache_list: List of (K, V, attn) tuples per layer
                K, V: [B, H, T, D]
                attn: not used (set to None after eviction)
            removal_indices_per_layer: Indices to remove per layer
            is_first_frame: Whether this is the first frame (for EMA threshold init)

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

            # Apply token merging if enabled (D2O)
            if self.enable_token_merging:
                # Reshape K and V for token merging: [B, H, T, D] -> [T, B*H*D]
                K_flat = K.permute(2, 0, 1, 3).reshape(T, -1)  # [T, B*H*D]
                V_flat = V.permute(2, 0, 1, 3).reshape(T, -1)  # [T, B*H*D]

                # Get evicted and conserved tokens
                K_evicted = K_flat[rem_indices]  # [L_e, B*H*D]
                V_evicted = V_flat[rem_indices]  # [L_e, B*H*D]
                K_conserved = K_flat[keep_indices]  # [L_c, B*H*D]
                V_conserved = V_flat[keep_indices]  # [L_c, B*H*D]

                # Perform nearest neighbor matching
                similarity_matrix, nearest_indices = self.nearest_neighbor_matching(
                    K_evicted, K_conserved
                )

                # Update EMA threshold
                threshold = self.update_ema_threshold(
                    similarity_matrix, layer_idx, is_first_frame
                )

                # Perform weighted token merging
                K_merged, V_merged = self.weighted_token_merging(
                    K_conserved, V_conserved,
                    K_evicted, V_evicted,
                    similarity_matrix, nearest_indices,
                    threshold
                )

                # Reshape back to [B, H, T', D]
                T_new = K_merged.shape[0]
                K_new = K_merged.reshape(T_new, B, H, D).permute(1, 2, 0, 3)  # [B, H, T', D]
                V_new = V_merged.reshape(T_new, B, H, D).permute(1, 2, 0, 3)  # [B, H, T', D]


            else:
                # Standard eviction without merging
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
            del K, V, keep_mask, keep_indices

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

        # Determine if this is first eviction (for EMA threshold initialization)
        # Frame 1 is first eviction since frame 0 is protected
        is_first_eviction = (frame_idx == 1)

        # Compact KV cache (with optional token merging)
        compacted_cache = self.compact_kv_cache(kv_cache_list, removal_indices,
                                                is_first_frame=is_first_eviction)

        return compacted_cache

    def reset(self):
        """Reset all state for a new sequence."""
        self.cum_attn_maps = [None] * self.num_layers
        self.exposure_counts = [None] * self.num_layers
        self.ema_thresholds = [None] * self.num_layers
        self.current_frame = 0
        print("✓ Reset compression state")
