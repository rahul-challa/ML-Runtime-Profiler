#!/usr/bin/env python3
"""
Example usage of the ML Runtime Profiler
Demonstrates different profiling scenarios and configurations.
"""

import sys
from pathlib import Path
from profiler import MLRuntimeProfiler
from compare_onnx import ONNXComparison


def example_basic_profiling():
    """Example of basic profiling with default settings."""
    print("=" * 60)
    print("🔍 EXAMPLE 1: Basic Profiling")
    print("=" * 60)
    
    # Create profiler with DistilBERT
    profiler = MLRuntimeProfiler(
        model_name="distilbert-base-uncased",
        device="cpu"  # Use CPU for this example
    )
    
    # Run profiling
    results = profiler.profile_inference(
        batch_size=4,
        sequence_length=128,
        num_runs=3
    )
    
    # Save and display results
    profiler.save_results("example_basic_profiling.json")
    profiler.print_summary()
    
    return results


def example_gpu_profiling():
    """Example of GPU profiling (if CUDA is available)."""
    print("\n" + "=" * 60)
    print("🔍 EXAMPLE 2: GPU Profiling")
    print("=" * 60)
    
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU example")
        return None
    
    # Create profiler with GPU
    profiler = MLRuntimeProfiler(
        model_name="distilbert-base-uncased",
        device="cuda"
    )
    
    # Run profiling with larger batch size
    results = profiler.profile_inference(
        batch_size=8,
        sequence_length=256,
        num_runs=5
    )
    
    # Save and display results
    profiler.save_results("example_gpu_profiling.json")
    profiler.print_summary()
    
    return results


def example_different_model():
    """Example with a different model (MiniLM)."""
    print("\n" + "=" * 60)
    print("🔍 EXAMPLE 3: Different Model (MiniLM)")
    print("=" * 60)
    
    # Create profiler with MiniLM
    profiler = MLRuntimeProfiler(
        model_name="microsoft/MiniLM-L12-H384-uncased",
        device="cpu"
    )
    
    # Run profiling
    results = profiler.profile_inference(
        batch_size=2,  # Smaller batch size for larger model
        sequence_length=128,
        num_runs=3
    )
    
    # Save and display results
    profiler.save_results("example_minilm_profiling.json")
    profiler.print_summary()
    
    return results


def example_onnx_comparison():
    """Example of ONNX vs PyTorch comparison."""
    print("\n" + "=" * 60)
    print("🔍 EXAMPLE 4: ONNX vs PyTorch Comparison")
    print("=" * 60)
    
    # Create comparison
    comparison = ONNXComparison(
        model_name="distilbert-base-uncased"
    )
    
    # Run comparison
    results = comparison.compare_inference(
        batch_size=4,
        num_runs=10
    )
    
    # Save and display results
    comparison.results = results
    comparison.save_results("example_onnx_comparison.json")
    comparison.print_summary()
    
    return results


def example_batch_size_analysis():
    """Example analyzing different batch sizes."""
    print("\n" + "=" * 60)
    print("🔍 EXAMPLE 5: Batch Size Analysis")
    print("=" * 60)
    
    batch_sizes = [1, 2, 4, 8]
    results_summary = {}
    
    for batch_size in batch_sizes:
        print(f"\n📊 Profiling with batch size: {batch_size}")
        
        profiler = MLRuntimeProfiler(
            model_name="distilbert-base-uncased",
            device="cpu"
        )
        
        results = profiler.profile_inference(
            batch_size=batch_size,
            sequence_length=128,
            num_runs=3
        )
        
        # Store summary statistics
        stats = results["statistics"]
        results_summary[batch_size] = {
            "mean_inference_time": stats["inference"]["mean"],
            "mean_total_time": stats["total"]["mean"],
            "throughput": batch_size / (stats["total"]["mean"] / 1000)  # samples per second
        }
        
        profiler.save_results(f"batch_size_{batch_size}_analysis.json")
    
    # Print batch size comparison
    print("\n📈 BATCH SIZE COMPARISON:")
    print("-" * 50)
    print(f"{'Batch Size':<12} {'Inference (ms)':<15} {'Total (ms)':<12} {'Throughput':<12}")
    print("-" * 50)
    
    for batch_size, metrics in results_summary.items():
        print(f"{batch_size:<12} {metrics['mean_inference_time']:<15.2f} "
              f"{metrics['mean_total_time']:<12.2f} {metrics['throughput']:<12.2f}")
    
    return results_summary


def main():
    """Run all examples."""
    print("🚀 ML Runtime Profiler - Example Usage")
    print("This script demonstrates various profiling scenarios.")
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    try:
        # Run examples
        example_basic_profiling()
        example_gpu_profiling()
        example_different_model()
        example_onnx_comparison()
        example_batch_size_analysis()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("📁 Check the 'results/' directory for output files.")
        print("📊 Use visualize_results.py to create charts and plots.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 