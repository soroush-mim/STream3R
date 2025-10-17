# StreamVGGT-Style Layer-Wise Token Eviction for STREAM3R

## Overview

This implementation adds StreamVGGT's layer-wise token eviction algorithm to STREAM3R, enabling more efficient memory management during long streaming inference.

## Key Differences from Original Implementation

### Original STREAM3R Eviction
- **Global eviction ratio**: Same percentage removed from all layers
- **Immediate attention**: Based only on current frame's attention weights
- **No layer importance**: All layers treated equally
- **No exposure tracking**: No normalization for token lifetime

### New StreamVGGT-Style Eviction
- **Per-layer budgets**: Different removal counts based on layer importance
- **Cumulative attention maps (CAM)**: Tracks long-term token importance
- **Variance-based layer importance**: Lower variance = more focused = more important
- **Exposure count normalization**: Fair comparison across tokens of different ages
- **Special token protection**: Camera and register tokens protected from ALL frames

## Files Modified/Created

### Created Files
1. **`stream3r/models/components/utils/kv_cache_compression_v2.py`**
   - `LayerWiseKVCompression` class implementing the full algorithm
   - Methods for CAM tracking, exposure counting, layer importance, and eviction

2. **`test_layerwise_eviction.py`**
   - Test script comparing old vs new eviction
   - Supports multiple budget ratios (P parameter)
   - Shows per-layer token distribution

### Modified Files
1. **`stream3r/models/components/layers/attention.py`**
   - Line 94-98: Return head-averaged attention maps
   - Format: `attn_avg` shape `[B, N, K]` instead of `[B, num_heads, N, K]`

2. **`stream3r/stream_session.py`**
   - Added `use_layerwise_eviction`, `budget_ratio`, `temp` parameters
   - Initialize `LayerWiseKVCompression` on first frame
   - Call `layerwise_compressor.process_frame()` in `_update_cache()`
   - Reset compressor in `clear()`

## Algorithm Details

### 1. First Frame Initialization
```python
# For each layer:
CAM_l = [+inf, +inf, ..., +inf]  # All tokens
EXPOSURE_l = [1, 1, ..., 1]       # Start with 1
```

### 2. Per Subsequent Frame
```python
# Update cumulative attention
for layer_idx in range(num_layers):
    token_scores = attn_map.sum(dim=1).mean(dim=0) * K  # Sum over queries
    new_cum[old_tokens] += curr_cum[old_tokens]        # Accumulate
    new_exp[old_tokens] += 1                            # Increment exposure

    # Protect special tokens from current frame
    new_cum[frame_start:frame_start+patch_start_idx] = +inf
```

### 3. Layer Importance (Variance-Based)
```python
variances = [attn_map.var() for attn_map in attn_maps_layers]
alphas = softmax(-variances / temp)  # Lower variance = higher importance
```

### 4. Per-Layer Budget Allocation
```python
total_budget = total_frames * num_layers * tokens_per_frame
current_cache_size = tokens_per_frame * (S + 1)  # S = frames processed

# Per layer removal count
tk_rm_num_per_layer = current_cache_size - (total_budget * P * alphas)
```

### 5. Token Removal (Exposure-Normalized)
```python
scores = CAM_l / (EXPOSURE_l + 1e-6)  # Normalize by exposure
indices = topk(scores, k=tk_rm_num_per_layer[l], largest=False)
```

## Usage

### Basic Usage
```python
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession

model = STream3R.from_pretrained("yslan/STream3R").to("cuda")

session = StreamSession(
    model,
    mode="causal",
    num_frames=10,  # Total frames in video
    use_layerwise_eviction=True,  # Enable new algorithm
    budget_ratio=0.5,  # P = 0.5 means keep 50% of total frames
    temp=0.5  # Temperature for layer importance softmax
)

# Process frames
for frame in frames:
    predictions = session.forward_stream(frame)
```

### Parameters

- **`use_layerwise_eviction`**: Enable StreamVGGT-style eviction (default: False)
- **`budget_ratio`** (P): Percentage of total frames to fit in budget (default: 0.5)
  - 0.5 = keep 5 frames worth of tokens if video has 10 frames
  - Lower = more aggressive eviction
- **`temp`**: Temperature for layer importance softmax (default: 0.5)
  - Lower = more extreme allocation (high-importance layers get more budget)
  - Higher = more uniform allocation across layers

### Running Tests

```bash
# Quick test (5 frames)
python test_layerwise_eviction.py quick

# Full comparison test
python test_layerwise_eviction.py

# With local model
python test_layerwise_eviction.py --model_path /path/to/model
```

## Expected Results

With `budget_ratio=0.5` on a 10-frame video:

- **Memory savings**: 30-50% reduction in KV cache size
- **Per-layer variation**: Important layers keep more tokens
- **Special token protection**: Camera/register tokens never evicted
- **Performance**: Minimal impact on reconstruction quality

## Comparison with StreamVGGT Implementation

### Similarities
✅ Cumulative attention map tracking per layer
✅ Exposure count normalization
✅ Variance-based layer importance
✅ Special token protection with +inf scores
✅ Per-layer budget allocation formula

### Differences
- **Attention averaging**: We average across heads immediately (line 96 in attention.py)
- **Budget formula**: Adjusted for already-removed tokens in previous frames
- **Device handling**: Explicit device parameter for tensors

## Debugging Tips

1. **Check attention shapes**: Attention maps should be `[B, N, K]` after averaging
2. **Monitor CAM values**: Should see +inf for special tokens
3. **Verify layer budgets**: Use debug prints in `compute_per_layer_budgets()`
4. **Track token counts**: Use `get_cache_tokens_per_layer()` helper

## Next Steps

To test the implementation:

```bash
cd /root/soroush/STream3R
python test_layerwise_eviction.py quick
```

This will:
1. Load the model
2. Process 5 frames with layer-wise eviction
3. Print per-frame statistics
4. Verify the algorithm works correctly

For full comparison:
```bash
python test_layerwise_eviction.py
```

This will compare:
- No compression
- Old compression (30% eviction)
- Layer-wise (P=0.5)
- Layer-wise (P=0.7)
