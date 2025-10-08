import torch
from stream3r.models.stream3r import STream3R


class StreamSession:
    """
    A causal streaming inference session with KV cache management for STream3R.
    """
    def __init__(
        self,
        model: STream3R,
        mode: str,
        use_kv_compression: bool = False,
        eviction_ratio: float = 0.3,
        importance_aggregation: str = 'sum',
    ):
        """
        Initialize streaming session with optional KV cache compression.

        Args:
            model: STream3R model
            mode: Attention mode ('causal' or 'window')
            use_kv_compression: Enable token eviction for memory reduction
            eviction_ratio: Fraction of evictable tokens to remove (0.3 = remove 30%)
            importance_aggregation: How to compute token importance ('sum', 'max', 'mean')
        """
        self.model = model
        self.mode = mode


        self.aggregator_kv_cache_depth = model.aggregator.depth
        self.camera_head_kv_cache_depth = model.camera_head.trunk_depth
        self.camera_head_iterations = 4

        # KV cache compression settings
        self.use_kv_compression = use_kv_compression
        self.eviction_ratio = eviction_ratio
        self.importance_aggregation = importance_aggregation

        # Calculate tokens per frame based on model configuration
        # This will be set after first forward pass when we know image size
        self.tokens_per_frame = None
        self.patch_start_idx = model.aggregator.patch_start_idx

        if self.mode not in ["causal", "window"]:
            raise ValueError(f"Unsupported attention mode when using kv_cache: {self.mode}")

        self.clear()

    def _clear_predictions(self):
        self.predictions = dict()
    
    def _update_predictions(self, predictions):
        for k in ["pose_enc", "world_points", "world_points_conf", "depth", "depth_conf", "images"]:
            if k in predictions:
                self.predictions[k] = torch.cat(
                    [self.predictions.get(k, torch.empty(0, device=predictions[k].device)), predictions[k]],
                    dim=1
                )

    def _clear_cache(self):
        # Initialize cache with 3-element lists to accommodate [k, v, attn_weights]
        self.aggregator_kv_cache_list = [[None, None, None] for _ in range(self.aggregator_kv_cache_depth)]
        self.camera_head_kv_cache_list = [[[None, None] for _ in range(self.camera_head_kv_cache_depth)] for _ in range(self.camera_head_iterations)]
    
    def _compress_aggregator_cache(self, aggregator_kv_cache_list):
        """
        Apply token eviction to aggregator KV cache using attention scores.

        Args:
            aggregator_kv_cache_list: List of [k, v, attn_weights] tuples

        Returns:
            Compressed cache list with [k, v, None] format
        """
        from stream3r.models.components.utils.kv_cache_compression import (
            compute_token_importance,
            evict_tokens_by_importance
        )

        compressed_cache = []

        for i, kv_tuple in enumerate(aggregator_kv_cache_list):
            k, v, attn_weights = kv_tuple[0], kv_tuple[1], kv_tuple[2]

            # Skip compression if no attention weights or cache too small
            if attn_weights is None or k is None or k.shape[2] <= self.tokens_per_frame:
                compressed_cache.append([k, v, None])
                continue

            # Compute importance scores from attention weights
            importance_scores = compute_token_importance(
                attn_weights,
                aggregation_method=self.importance_aggregation
            )

            # Evict low-importance tokens
            k_compressed, v_compressed = evict_tokens_by_importance(
                k, v, importance_scores,
                tokens_per_frame=self.tokens_per_frame,
                patch_start_idx=self.patch_start_idx,
                eviction_ratio=self.eviction_ratio
            )

            # Don't keep attention weights in cache to save memory
            compressed_cache.append([k_compressed, v_compressed, None])

        return compressed_cache

    def _update_cache(self, aggregator_kv_cache_list, camera_head_kv_cache_list):
        # Calculate tokens_per_frame on first update if not set
        if self.tokens_per_frame is None and "depth" in self.predictions:
            h, w = self.predictions["depth"].shape[2], self.predictions["depth"].shape[3]
            self.tokens_per_frame = h * w // self.model.aggregator.patch_size // self.model.aggregator.patch_size + self.model.aggregator.patch_start_idx

        if self.mode == "causal":
            # Apply compression if enabled
            if self.use_kv_compression and self.tokens_per_frame is not None:
                aggregator_kv_cache_list = self._compress_aggregator_cache(aggregator_kv_cache_list)

            # Store cache (strip attention weights to save memory)
            self.aggregator_kv_cache_list = [[kv[0], kv[1], None] for kv in aggregator_kv_cache_list]
            self.camera_head_kv_cache_list = camera_head_kv_cache_list

        elif self.mode == "window":
            window_size = 5

            # Apply compression first if enabled
            if self.use_kv_compression and self.tokens_per_frame is not None:
                aggregator_kv_cache_list = self._compress_aggregator_cache(aggregator_kv_cache_list)

            # Then apply window-based truncation
            for k in range(2):
                for i in range(self.aggregator_kv_cache_depth):
                    if aggregator_kv_cache_list[i][k] is None:
                        continue

                    h, w = self.predictions["depth"].shape[2], self.predictions["depth"].shape[3]
                    P = h * w // self.model.aggregator.patch_size // self.model.aggregator.patch_size + self.model.aggregator.patch_start_idx
                    anchor_token = aggregator_kv_cache_list[i][k][:, :, :P]
                    window_tokens = aggregator_kv_cache_list[i][k][:, :, max(P, aggregator_kv_cache_list[i][k].size(2)-window_size*P):]
                    self.aggregator_kv_cache_list[i][k] = torch.cat(
                        [
                            anchor_token,
                            window_tokens
                        ],
                        dim=2
                    )
                for i in range(self.camera_head_iterations):
                    for j in range(self.camera_head_kv_cache_depth):
                        anchor_token = camera_head_kv_cache_list[i][j][k][:, :, :1]
                        window_tokens = camera_head_kv_cache_list[i][j][k][:, :, max(1, camera_head_kv_cache_list[i][j][k].size(2)-window_size):]
                        self.camera_head_kv_cache_list[i][j][k] = torch.cat(
                            [
                                anchor_token,
                                window_tokens
                            ],
                            dim=2
                        )
        else:
            raise ValueError(f"Unsupported attention mode when using kv_cache: {self.mode}")

    def _get_cache(self):
        return self.aggregator_kv_cache_list, self.camera_head_kv_cache_list
    
    def get_all_predictions(self):
        return self.predictions
    
    def get_last_prediction(self):
        last_predictions = dict()
        for k in ["pose_enc", "world_points", "world_points_conf", "depth", "depth_conf", "images"]:
            if k in self.predictions:
                last_predictions[k] = self.predictions[k][:, -1:]
        return last_predictions

    def clear(self):
        self._clear_predictions()
        self._clear_cache()

    def forward_stream(self, images):
        aggregator_kv_cache_list, camera_head_kv_cache_list = self._get_cache()

        outputs = self.model(
            images=images, 
            mode=self.mode, 
            aggregator_kv_cache_list=aggregator_kv_cache_list, 
            camera_head_kv_cache_list=camera_head_kv_cache_list, 
        )

        self._update_predictions(outputs)
        self._update_cache(outputs["aggregator_kv_cache_list"], outputs["camera_head_kv_cache_list"])

        return self.get_all_predictions()
