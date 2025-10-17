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

def get_cache_tokens(session):
    """Get number of cached tokens per layer."""
    token_counts = []
    for kv in session.aggregator_kv_cache_list:
        if kv[0] is not None:
            token_counts.append(kv[0].shape[2])  # seq_len dimension
        else:
            token_counts.append(0)
    return token_counts

def save_pointmap(predictions, output_dir, prefix="pointmap"):
    """
    Save point cloud from predictions to PLY file.

    Args:
        predictions: Dictionary with 'world_points', 'world_points_conf', 'images'
        output_dir: Directory to save output files
        prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    world_points = predictions['world_points']  # [B, S, H, W, 3]
    world_points_conf = predictions['world_points_conf']  # [B, S, H, W]
    images = predictions['images']  # [B, S, 3, H, W]

    B, S, H, W, _ = world_points.shape

    # Convert to numpy
    points = world_points.cpu().numpy()  # [B, S, H, W, 3]
    conf = world_points_conf.cpu().numpy()  # [B, S, H, W]
    imgs = images.cpu().numpy()  # [B, S, 3, H, W]

    # Process each batch and sequence
    for b in range(B):
        for s in range(S):
            # Get points for this frame
            pts = points[b, s].reshape(-1, 3)  # [H*W, 3]
            confidence = conf[b, s].reshape(-1)  # [H*W]

            # Get RGB colors (convert from [3, H, W] to [H*W, 3])
            rgb = imgs[b, s].transpose(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
            rgb = (rgb * 255).astype(np.uint8)  # Convert to 0-255

            # Filter by confidence threshold
            conf_threshold = 0.5
            valid_mask = confidence > conf_threshold

            pts_filtered = pts[valid_mask]
            rgb_filtered = rgb[valid_mask]

            # Save to PLY
            output_file = os.path.join(output_dir, f"{prefix}_b{b}_frame{s:03d}.ply")
            save_ply(pts_filtered, rgb_filtered, output_file)
            print(f"Saved pointmap: {output_file} ({len(pts_filtered)} points)")

    # Also save combined point cloud for all frames
    all_points = []
    all_colors = []

    for b in range(B):
        for s in range(S):
            pts = points[b, s].reshape(-1, 3)
            confidence = conf[b, s].reshape(-1)
            rgb = imgs[b, s].transpose(1, 2, 0).reshape(-1, 3)
            rgb = (rgb * 255).astype(np.uint8)

            valid_mask = confidence > 0.5
            all_points.append(pts[valid_mask])
            all_colors.append(rgb[valid_mask])

    if len(all_points) > 0:
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)

        combined_file = os.path.join(output_dir, f"{prefix}_combined.ply")
        save_ply(all_points, all_colors, combined_file)
        print(f"Saved combined pointmap: {combined_file} ({len(all_points)} points)")

def save_ply(points, colors, filename):
    """
    Save point cloud to PLY file.

    Args:
        points: [N, 3] numpy array of XYZ coordinates
        colors: [N, 3] numpy array of RGB colors (0-255)
        filename: Output PLY file path
    """
    assert points.shape[0] == colors.shape[0], "Points and colors must have same length"

    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write vertices
        for i in range(len(points)):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
            f.write(f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")

def test_compression(
    example_dir="examples/static_room",
    eviction_ratios=[0.0, 0.3, 0.5],
    mode="causal",
    model_path=None,
    save_pointmaps=False,
    output_dir="./output_pointmaps"
):
    """
    Test KV compression with different eviction ratios.

    Args:
        example_dir: Directory with test images
        eviction_ratios: List of eviction ratios to test
        mode: Attention mode ('causal' or 'window')
        model_path: Local path to pretrained model (optional)
        save_pointmaps: Whether to save point clouds to PLY files
        output_dir: Directory to save pointmap files
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
            num_frames=images.shape[0],
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

        # Save pointmaps if requested
        if save_pointmaps:
            print(f"\nSaving pointmaps...")
            final_predictions = session.get_all_predictions()
            pointmap_dir = os.path.join(output_dir, f"eviction_{eviction_ratio:.1f}")
            save_pointmap(final_predictions, pointmap_dir, prefix=f"pointmap_ratio{eviction_ratio:.1f}")

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

def quick_test(model_path=None, save_pointmaps=False):
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

    # Save pointmaps if requested
    if save_pointmaps:
        print("\nSaving pointmaps...")
        final_predictions = session.get_all_predictions()
        save_pointmap(final_predictions, "./output_pointmaps/quick_test", prefix="quick_test")

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
    save_pointmaps = False
    output_dir = "./output_pointmaps"

    if "--model_path" in sys.argv:
        idx = sys.argv.index("--model_path")
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]
            print(f"Using model from: {model_path}")

    if "--save_pointmaps" in sys.argv:
        save_pointmaps = True
        print("Will save pointmaps to PLY files")

    if "--output_dir" in sys.argv:
        idx = sys.argv.index("--output_dir")
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
            print(f"Output directory: {output_dir}")

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
        quick_test(model_path=model_path, save_pointmaps=save_pointmaps)
    else:
        # Full comparison test
        results = test_compression(
            example_dir="examples/static_room",
            eviction_ratios=[0.0, 0.3, 0.5],
            mode="causal",
            model_path=model_path,
            save_pointmaps=save_pointmaps,
            output_dir=output_dir
        )
