"""
Test script for StreamVGGT-style layer-wise KV compression.

This script demonstrates:
1. Using the new LayerWiseKVCompression algorithm
2. Comparing it with the old eviction method
3. Showing per-layer token allocation and eviction
"""

import os
import torch
import time
import numpy as np
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


def get_cache_tokens_per_layer(session):
    """Get number of cached tokens for each layer."""
    token_counts = []
    for kv in session.aggregator_kv_cache_list:
        if kv[0] is not None:
            token_counts.append(kv[0].shape[2])  # seq_len dimension
        else:
            token_counts.append(0)
    return token_counts


def test_layerwise_vs_old(
    example_dir="examples/static_room",
    budget_ratios=[0.7, 0.5, 0.3],
    model_path=None,
):
    """
    Compare layer-wise eviction with old method.

    Args:
        example_dir: Directory with test images
        budget_ratios: List of budget ratios to test (P parameter)
        model_path: Local path to pretrained model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("\nLoading STream3R model...")
    if model_path is not None:
        model = STream3R.from_pretrained(model_path).to(device)
        print(f"Model loaded from {model_path}")
    else:
        model = STream3R.from_pretrained("yslan/STream3R").to(device)
        print("Model loaded from HuggingFace")

    # Load images
    print(f"\nLoading images from {example_dir}...")
    image_names = [os.path.join(example_dir, file) for file in sorted(os.listdir(example_dir))]
    images = load_and_preprocess_images(image_names).to(device)
    num_frames = images.shape[0]
    print(f"Loaded {num_frames} frames")

    model.eval()

    # Test configurations
    configs = [
        {"name": "No Compression", "use_layerwise": False, "use_old": False},
        # {"name": "Old Compression (30%)", "use_layerwise": False, "use_old": True, "eviction_ratio": 0.3},
    ]

    # Add layer-wise configs
    for budget_ratio in budget_ratios:
        configs.append({
            "name": f"LayerWise (P={budget_ratio})",
            "use_layerwise": True,
            "use_old": False,
            "budget_ratio": budget_ratio
        })

    results = {}

    for config in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")

        # Create session
        session = StreamSession(
            model,
            mode="causal",
            num_frames=num_frames,
            use_kv_compression=config.get("use_old", False),
            eviction_ratio=config.get("eviction_ratio", 0.3),
            use_layerwise_eviction=config.get("use_layerwise", False),
            budget_ratio=config.get("budget_ratio", 0.5),
            temp=1.0
        )

        # Process frames
        cache_sizes = []
        token_counts_per_frame = []  # Track first layer only
        all_layer_counts = []  # Track all layers
        inference_times = []

        torch.cuda.reset_peak_memory_stats()
        start_total = time.time()

        with torch.no_grad():
            for i in range(num_frames):
                print(f"  Frame {i+1}/{num_frames}")
                image = images[i:i+1]

                start = time.time()
                predictions = session.forward_stream(image)
                inference_time = time.time() - start

                # Get cache statistics
                cache_size_mb = get_cache_size_mb(session)
                token_counts = get_cache_tokens_per_layer(session)

                cache_sizes.append(cache_size_mb)
                token_counts_per_frame.append(token_counts[0])  # First layer
                all_layer_counts.append(token_counts)
                inference_times.append(inference_time)

                # Print layer-wise info for layerwise eviction
                if config.get("use_layerwise") and i > 0:
                    print(f"  Frame {i+1}/{num_frames}: Cache={cache_size_mb:.2f}MB, "
                          f"Tokens L0={token_counts[0]}, L{len(token_counts)//2}={token_counts[len(token_counts)//2]}, "
                          f"L{len(token_counts)-1}={token_counts[-1]}, Time={inference_time:.3f}s")
                else:
                    print(f"  Frame {i+1}/{num_frames}: Cache={cache_size_mb:.2f}MB, "
                          f"Tokens={token_counts[0]}, Time={inference_time:.3f}s")

        total_time = time.time() - start_total
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB

        # Store results
        results[config['name']] = {
            'cache_sizes': cache_sizes,
            'token_counts': token_counts_per_frame,
            'all_layer_counts': all_layer_counts,
            'inference_times': inference_times,
            'total_time': total_time,
            'peak_memory_mb': peak_memory,
            'final_cache_mb': cache_sizes[-1],
            'avg_inference_time': sum(inference_times) / len(inference_times)
        }

        print(f"\n  Summary:")
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Avg inference time: {results[config['name']]['avg_inference_time']:.3f}s")
        print(f"    Peak GPU memory: {peak_memory:.2f}MB")
        print(f"    Final cache size: {cache_sizes[-1]:.2f}MB")
        print(f"    Final token count (L0): {token_counts_per_frame[-1]}")

        # Show per-layer distribution for last frame
        if config.get("use_layerwise"):
            final_layer_counts = all_layer_counts[-1]
            print(f"    Per-layer tokens (final): min={min(final_layer_counts)}, "
                  f"max={max(final_layer_counts)}, "
                  f"mean={np.mean(final_layer_counts):.1f}, "
                  f"std={np.std(final_layer_counts):.1f}")

        session.clear()

    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Cache (MB)':<15} {'Tokens':<12} {'Speedup':<12} {'Memory':<15}")
    print(f"{'-'*80}")

    baseline_name = "No Compression"
    baseline = results[baseline_name]

    for name, res in results.items():
        cache_reduction = (1 - res['final_cache_mb'] / baseline['final_cache_mb']) * 100
        token_reduction = (1 - res['token_counts'][-1] / baseline['token_counts'][-1]) * 100
        speedup = baseline['avg_inference_time'] / res['avg_inference_time']

        print(f"{name:<30} "
              f"{res['final_cache_mb']:<7.2f}(-{cache_reduction:>4.1f}%) "
              f"{res['token_counts'][-1]:<6}(-{token_reduction:>3.0f}%) "
              f"{speedup:<12.2f}x "
              f"{res['peak_memory_mb']:<15.2f}")

    return results


def quick_test(model_path=None):
    """Quick test to verify layer-wise eviction works."""
    print("Running quick test of layer-wise eviction...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path is not None:
        model = STream3R.from_pretrained(model_path).to(device)
        print(f"Model loaded from {model_path}")
    else:
        model = STream3R.from_pretrained("yslan/STream3R").to(device)
        print("Model loaded from HuggingFace")

    # Load a few images
    example_dir = "examples/static_room"
    image_names = [os.path.join(example_dir, file) for file in sorted(os.listdir(example_dir))[:5]]
    images = load_and_preprocess_images(image_names).to(device)
    num_frames = images.shape[0]

    # Test with layer-wise eviction
    session = StreamSession(
        model,
        mode="causal",
        num_frames=num_frames,
        use_layerwise_eviction=True,
        budget_ratio=0.5,
        temp=0.5
    )

    model.eval()

    with torch.no_grad():
        for i in range(num_frames):
            preds = session.forward_stream(images[i:i+1])
            token_counts = get_cache_tokens_per_layer(session)
            print(f"Frame {i+1}: depth shape={preds['depth'].shape}, "
                  f"tokens L0={token_counts[0] if token_counts else 'N/A'}")

    print("✓ Quick test passed!")
    session.clear()


if __name__ == "__main__":
    import sys

    model_path = None
    if "--model_path" in sys.argv:
        idx = sys.argv.index("--model_path")
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]
            print(f"Using model from: {model_path}")

    if "quick" in sys.argv:
        # Quick test
        quick_test(model_path=model_path)
    else:
        # Full comparison test
        results = test_layerwise_vs_old(
            example_dir="examples/static_room",
            budget_ratios=[0.3, 0.5],
            model_path=model_path
        )
