#!/usr/bin/env python3
"""Benchmark script for pose estimation models.

Usage:
    python scripts/benchmark_pose_models.py [--frames 100] [--output results.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.ml.pose_models import get_pose_model_registry


def create_test_frames(num_frames: int, width: int = 640, height: int = 480) -> list:
    """Create synthetic test frames with a simple pattern."""
    frames = []
    for i in range(num_frames):
        # Create gradient pattern for some variety
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(0, 255, width).astype(np.uint8)  # Blue gradient
        frame[:, :, 1] = ((i * 10) % 255)  # Changing green
        frame[:, :, 2] = np.linspace(255, 0, width).astype(np.uint8)  # Red gradient
        frames.append(frame)
    return frames


def benchmark_model(model, frames: list, warm_up: int = 10) -> dict:
    """Benchmark a single model."""
    # Warm up
    print(f"    Warming up ({warm_up} frames)...")
    for frame in frames[:warm_up]:
        model._process_frame_impl(frame)
    
    # Reset state
    model.reset_state()
    
    # Benchmark
    print(f"    Benchmarking ({len(frames)} frames)...")
    times = []
    detections = 0
    
    for frame in frames:
        start = time.perf_counter()
        result = model._process_frame_impl(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        if result is not None:
            detections += 1
    
    # Calculate stats
    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    std_ms = np.std(times)
    fps = 1000 / avg_ms
    
    return {
        "model_id": model.config.model_id,
        "backend": model.config.backend.value,
        "model_size": model.config.model_size.value,
        "num_frames": len(frames),
        "avg_ms": round(avg_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "std_ms": round(std_ms, 2),
        "fps": round(fps, 1),
        "detection_rate": round(detections / len(frames), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark pose estimation models")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to test")
    parser.add_argument("--warm-up", type=int, default=10, help="Warm-up frames")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--models", type=str, nargs="*", help="Specific models to test")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FMS Pose Model Benchmark")
    print("=" * 60)
    
    # Create test frames
    print(f"\nCreating {args.frames} test frames...")
    frames = create_test_frames(args.frames)
    
    # Get registry
    registry = get_pose_model_registry()
    available_models = registry.get_available_models()
    
    print(f"\nAvailable backends: {registry.get_stats()['backends']}")
    print(f"Total models: {len(available_models)}")
    
    # Filter to available and requested models
    models_to_test = [
        m for m in available_models 
        if m.is_available and (not args.models or m.model_id in args.models)
    ]
    
    print(f"Models to benchmark: {len(models_to_test)}")
    
    results = []
    
    for model_info in models_to_test:
        model_id = model_info.model_id
        print(f"\n[{model_id}]")
        print(f"  Backend: {model_info.backend}, Size: {model_info.model_size}")
        
        try:
            # Load model
            print(f"  Loading model...")
            model = registry.get_model(model_id)
            
            if model is None:
                print(f"  ERROR: Failed to load model")
                continue
            
            # Benchmark
            result = benchmark_model(model, frames, args.warm_up)
            results.append(result)
            
            print(f"  Results:")
            print(f"    Avg: {result['avg_ms']:.1f}ms ({result['fps']:.1f} FPS)")
            print(f"    Min/Max: {result['min_ms']:.1f}ms / {result['max_ms']:.1f}ms")
            print(f"    Detection rate: {result['detection_rate']*100:.1f}%")
            
            # Unload to free memory
            registry.unload_model(model_id)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Sort by FPS
    results.sort(key=lambda x: x["fps"], reverse=True)
    
    print(f"\n{'Model':<25} {'Backend':<12} {'Size':<8} {'FPS':>8} {'Avg ms':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['model_id']:<25} {r['backend']:<12} {r['model_size']:<8} {r['fps']:>8.1f} {r['avg_ms']:>10.1f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump({
                "benchmark_config": {
                    "num_frames": args.frames,
                    "warm_up": args.warm_up,
                },
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
