# Quick Start: KV Cache Compression

## TL;DR

Token eviction is now implemented! Use it like this:

```python
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession

model = STream3R.from_pretrained("yslan/STream3R").to("cuda")

# WITH compression (new!)
session = StreamSession(
    model,
    mode="causal",
    use_kv_compression=True,    # ← Enable eviction
    eviction_ratio=0.3           # ← Remove 30% of tokens
)

# Use as normal
for image in images:
    predictions = session.forward_stream(image)
```

## What Changed

### 3 Files Modified:
1. **`attention.py`**: Returns attention weights with KV cache
2. **`stream_session.py`**: Applies eviction using attention scores
3. **New `kv_cache_compression.py`**: Core eviction logic

### Key Features:
- ✅ No training required
- ✅ Anchor frame protected
- ✅ Special tokens protected
- ✅ RoPE handled correctly
- ✅ Works with causal and window modes

## Testing

```bash
# Quick test (3 frames)
python test_kv_compression.py quick

# Full test (compare different eviction ratios)
python test_kv_compression.py
```

## Parameters

```python
StreamSession(
    model,
    mode="causal",              # or "window"
    use_kv_compression=False,   # Enable compression
    eviction_ratio=0.3,         # Remove 30% of evictable tokens
    importance_aggregation='sum' # 'sum', 'max', or 'mean'
)
```

## Expected Results

With `eviction_ratio=0.3`:
- **Memory**: -20% to -30% cache reduction
- **Speed**: ~same (slight overhead from manual attention)
- **Quality**: Minimal impact (tokens are chosen by importance)

## Example: Compare Baseline vs Compressed

```python
import torch
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.models.components.utils.load_fn import load_and_preprocess_images

device = "cuda"
model = STream3R.from_pretrained("yslan/STream3R").to(device)
images = load_and_preprocess_images(image_paths).to(device)

# Test 1: Baseline
session_baseline = StreamSession(model, mode="causal", use_kv_compression=False)
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    for i in range(images.shape[0]):
        _ = session_baseline.forward_stream(images[i:i+1])

baseline_memory = torch.cuda.max_memory_allocated() / (1024**2)
print(f"Baseline peak memory: {baseline_memory:.2f} MB")
session_baseline.clear()

# Test 2: With compression
session_compressed = StreamSession(
    model,
    mode="causal",
    use_kv_compression=True,
    eviction_ratio=0.3
)
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    for i in range(images.shape[0]):
        _ = session_compressed.forward_stream(images[i:i+1])

compressed_memory = torch.cuda.max_memory_allocated() / (1024**2)
print(f"Compressed peak memory: {compressed_memory:.2f} MB")
print(f"Memory reduction: {(1 - compressed_memory/baseline_memory)*100:.1f}%")
session_compressed.clear()
```

## Integration with Evaluation Scripts

In your evaluation scripts (e.g., `eval/video_depth/run.py`):

```python
# Find where StreamSession is created, replace:
# session = StreamSession(model, mode=mode)

# With:
session = StreamSession(
    model,
    mode=mode,
    use_kv_compression=True,  # ← Add this
    eviction_ratio=0.3         # ← Add this
)

# Everything else stays the same!
```

## Monitoring Cache Size

```python
def get_cache_size(session):
    """Get total tokens in cache."""
    return sum(kv[0].shape[2] if kv[0] is not None else 0
               for kv in session.aggregator_kv_cache_list)

# After each frame
print(f"Cache size: {get_cache_size(session)} tokens")
```

## Troubleshooting

**Q: Memory not reducing?**
A: Check `use_kv_compression=True` and `eviction_ratio > 0`

**Q: Quality dropped?**
A: Reduce `eviction_ratio` to 0.2 or 0.1

**Q: Slower inference?**
A: Expected ~5-10% overhead from computing attention weights. Trade-off for memory savings.

**Q: Error about cache dimensions?**
A: Make sure all modified files are saved and reloaded

## Files Changed Summary

```
stream3r/
├── models/
│   └── components/
│       ├── layers/
│       │   └── attention.py          [MODIFIED]
│       └── utils/
│           └── kv_cache_compression.py  [NEW]
└── stream_session.py                 [MODIFIED]

test_kv_compression.py                [NEW]
KV_COMPRESSION_README.md              [NEW]
QUICK_START_COMPRESSION.md            [NEW]
```

## Next Steps

1. **Test it**: Run `python test_kv_compression.py quick`
2. **Try different ratios**: Test 0.2, 0.3, 0.5 on your data
3. **Integrate**: Add to your evaluation scripts
4. **Monitor**: Track memory and quality metrics

## Full Documentation

See `KV_COMPRESSION_README.md` for:
- Detailed implementation explanation
- How token protection works
- Advanced usage patterns
- Performance tuning tips
