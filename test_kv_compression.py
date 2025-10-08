"""
Test script for KV cache compression (token eviction).

This script demonstrates:
1. Loading STream3R model
2. Running inference with and without KV compression
3. Comparing memory usage and cache sizes
4. Visualizing the effects of token eviction
"""

import os
import torch
import time
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.models.components.utils.load_fn import load_and_preprocess_images

def get_cache_size_mb(session):
    """Calculate total memory used by KV cache in MB."""
    total_size = 0
    for kv in session.aggregator_kv_cache_list:
        if kv[0] is not None:
            total_size += kv[0].element_size() * kv[0].nelement()
        if kv[1] is not None:
            total_size += kv[1].element_size() * kv[1].nelement()
    return total_size / (1024 ** 2)  # Convert to MB

def get_cache_tokens(session):
    """Get number of cached tokens per layer."""
    token_counts = []
    for kv in session.aggregator_kv_cache_list:
        if kv[0] is not None:
            token_counts.append(kv[0].shape[2])  # seq_len dimension
        else:
            token_counts.append(0)
    return token_counts

def test_compression(
    example_dir="examples/static_room",
    num_frames=10,
    eviction_ratios=[0.0, 0.3, 0.5],
    mode="causal",
    model_path=None
):
    """
    Test KV compression with different eviction ratios.

    Args:
        example_dir: Directory with test images
        num_frames: Number of frames to process
        eviction_ratios: List of eviction ratios to test
        mode: Attention mode ('causal' or 'window')
        model_path: Local path to pretrained model (optional)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading STream3R model...")
    if model_path is not None:
        # Load from local directory
        model = STream3R.from_pretrained(model_path).to(device)
        print(f"Model loaded from {model_path}")
    else:
        # Download from HuggingFace
        model = STream3R.from_pretrained("yslan/STream3R").to(device)
        print("Model loaded from HuggingFace")
    print("Model loaded successfully!")

    # Load images
    print(f"\nLoading images from {example_dir}...")
    image_names = [os.path.join(example_dir, file) for file in sorted(os.listdir(example_dir))]
    image_names = image_names[:num_frames]  # Limit to num_frames
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Loaded {images.shape[0]} frames")

    results = {}
    model.eval()

    for eviction_ratio in eviction_ratios:
        use_compression = eviction_ratio > 0.0

        print(f"\n{'='*60}")
        print(f"Testing with eviction_ratio={eviction_ratio:.1f} (compression={'ON' if use_compression else 'OFF'})")
        print(f"{'='*60}")

        # Create session
        session = StreamSession(
            model,
            mode=mode,
            use_kv_compression=use_compression,
            eviction_ratio=eviction_ratio,
            importance_aggregation='sum'
        )

        # Process frames
        cache_sizes = []
        token_counts_per_frame = []
        inference_times = []

        torch.cuda.reset_peak_memory_stats()
        start_total = time.time()

        with torch.no_grad():
            for i in range(images.shape[0]):
                image = images[i:i+1]

                start = time.time()
                predictions = session.forward_stream(image)
                inference_time = time.time() - start

                # Get cache statistics
                cache_size_mb = get_cache_size_mb(session)
                token_counts = get_cache_tokens(session)

                cache_sizes.append(cache_size_mb)
                token_counts_per_frame.append(token_counts[0])  # Just track first layer
                inference_times.append(inference_time)

                print(f"Frame {i+1}/{images.shape[0]}: "
                      f"Cache={cache_size_mb:.2f}MB, "
                      f"Tokens={token_counts[0]}, "
                      f"Time={inference_time:.3f}s")

        total_time = time.time() - start_total
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

        # Store results
        results[eviction_ratio] = {
            'cache_sizes': cache_sizes,
            'token_counts': token_counts_per_frame,
            'inference_times': inference_times,
            'total_time': total_time,
            'peak_memory_mb': peak_memory,
            'final_cache_mb': cache_sizes[-1] if cache_sizes else 0,
            'avg_inference_time': sum(inference_times) / len(inference_times)
        }

        print(f"\nSummary:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg inference time: {results[eviction_ratio]['avg_inference_time']:.3f}s")
        print(f"  Peak GPU memory: {peak_memory:.2f}MB")
        print(f"  Final cache size: {cache_sizes[-1]:.2f}MB")
        print(f"  Final token count (layer 0): {token_counts_per_frame[-1]}")

        session.clear()

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Eviction Ratio':<15} {'Final Cache (MB)':<20} {'Tokens':<10} {'Avg Time (s)':<15} {'Peak Mem (MB)':<15}")
    print(f"{'-'*60}")

    baseline_cache = results[0.0]['final_cache_mb']
    baseline_tokens = results[0.0]['token_counts'][-1]

    for ratio in eviction_ratios:
        res = results[ratio]
        cache_reduction = (1 - res['final_cache_mb'] / baseline_cache) * 100 if baseline_cache > 0 else 0
        token_reduction = (1 - res['token_counts'][-1] / baseline_tokens) * 100 if baseline_tokens > 0 else 0

        print(f"{ratio:<15.1f} "
              f"{res['final_cache_mb']:<10.2f}(-{cache_reduction:.1f}%) "
              f"{res['token_counts'][-1]:<10}(-{token_reduction:.1f}%) "
              f"{res['avg_inference_time']:<15.3f} "
              f"{res['peak_memory_mb']:<15.2f}")

    return results

def quick_test(model_path=None):
    """Quick test to verify the implementation works."""
    print("Running quick test...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path is not None:
        model = STream3R.from_pretrained(model_path).to(device)
        print(f"Model loaded from {model_path}")
    else:
        model = STream3R.from_pretrained("yslan/STream3R").to(device)
        print("Model loaded from HuggingFace")

    # Load a few images
    example_dir = "examples/static_room"
    image_names = [os.path.join(example_dir, file) for file in sorted(os.listdir(example_dir))[:3]]
    images = load_and_preprocess_images(image_names).to(device)

    # Test with compression
    session = StreamSession(model, mode="causal", use_kv_compression=True, eviction_ratio=0.3)

    with torch.no_grad():
        for i in range(images.shape[0]):
            preds = session.forward_stream(images[i:i+1])
            print(f"Frame {i+1}: depth shape = {preds['depth'].shape}")

    print("✓ Quick test passed!")
    session.clear()

def download_model(save_dir="./pretrained_models"):
    """
    Download the pretrained model to a specific directory.

    Args:
        save_dir: Directory to save the model

    Returns:
        Path to the downloaded model
    """
    import os
    from huggingface_hub import snapshot_download

    print(f"Downloading model to {save_dir}...")
    model_path = snapshot_download(
        repo_id="yslan/STream3R",
        cache_dir=save_dir,
        local_dir=os.path.join(save_dir, "STream3R"),
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to {model_path}")
    return model_path


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    model_path = None
    if "--model_path" in sys.argv:
        idx = sys.argv.index("--model_path")
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]
            print(f"Using model from: {model_path}")

    if "--download" in sys.argv:
        # Download model to specified directory
        idx = sys.argv.index("--download")
        save_dir = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "./pretrained_models"
        model_path = download_model(save_dir)
        print(f"Model saved to: {model_path}")
        print("You can now use it with: --model_path {model_path}")
        sys.exit(0)

    if "quick" in sys.argv:
        # Quick test
        quick_test(model_path=model_path)
    else:
        # Full comparison test
        results = test_compression(
            example_dir="examples/static_room",
            num_frames=10,
            eviction_ratios=[0.0, 0.3, 0.5],
            mode="causal",
            model_path=model_path
        )
