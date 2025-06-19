#!/usr/bin/env python3
"""
ONNX Runtime vs PyTorch Inference Comparison
Compares performance between ONNX Runtime and PyTorch for transformer models.
"""

import argparse
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import onnxruntime as ort
from datetime import datetime


class ONNXComparison:
    """Compare ONNX Runtime vs PyTorch inference performance."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the comparison with a transformer model.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.pytorch_model = None
        self.onnx_model = None
        self.onnx_session = None
        
        print(f"🚀 Initializing ONNX vs PyTorch Comparison")
        print(f"📦 Model: {model_name}")
        
        self._load_models()
    
    def _load_models(self):
        """Load PyTorch model and convert to ONNX."""
        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            print("📥 Loading PyTorch model...")
            self.pytorch_model = AutoModel.from_pretrained(self.model_name)
            self.pytorch_model.eval()
            
            print("🔄 Converting to ONNX...")
            self._convert_to_onnx()
            
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def _convert_to_onnx(self):
        """Convert PyTorch model to ONNX format."""
        # Create dummy input for ONNX conversion
        dummy_input = self.tokenizer(
            "This is a sample text for ONNX conversion.",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Export to ONNX
        onnx_path = f"results/{self.model_name.replace('/', '_')}.onnx"
        Path("results").mkdir(exist_ok=True)
        
        torch.onnx.export(
            self.pytorch_model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "last_hidden_state": {0: "batch_size", 1: "sequence"}
            }
        )
        
        # Load ONNX model
        self.onnx_session = ort.InferenceSession(onnx_path)
        print(f"💾 ONNX model saved to: {onnx_path}")
    
    def _prepare_inputs(self, texts: List[str]) -> Dict[str, Any]:
        """Prepare inputs for both PyTorch and ONNX models."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )
        
        # Prepare PyTorch inputs
        pytorch_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        
        # Prepare ONNX inputs
        onnx_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy()
        }
        
        return pytorch_inputs, onnx_inputs
    
    def _generate_sample_texts(self, num_samples: int) -> List[str]:
        """Generate sample texts for comparison."""
        sample_texts = [
            "This is a sample text for performance comparison.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require careful optimization.",
            "Transformer architectures have revolutionized NLP.",
            "ONNX Runtime provides optimized inference.",
            "PyTorch offers flexible model development.",
            "Performance comparison helps choose the right tool.",
            "Batch processing improves throughput significantly.",
            "Memory usage varies between frameworks.",
            "Inference speed is critical for production deployment."
        ]
        
        # Repeat to get desired number of samples
        texts = []
        for i in range(num_samples):
            text = sample_texts[i % len(sample_texts)]
            texts.append(text)
        
        return texts
    
    def compare_inference(self, batch_size: int = 4, num_runs: int = 10) -> Dict[str, Any]:
        """
        Compare PyTorch vs ONNX Runtime inference performance.
        
        Args:
            batch_size: Number of samples per batch
            num_runs: Number of comparison runs
            
        Returns:
            Dictionary containing comparison results
        """
        print(f"\n🔍 Starting PyTorch vs ONNX comparison...")
        print(f"📊 Batch size: {batch_size}")
        print(f"🔄 Number of runs: {num_runs}")
        
        # Generate sample texts
        sample_texts = self._generate_sample_texts(batch_size * num_runs)
        
        results = {
            "model_name": self.model_name,
            "batch_size": batch_size,
            "num_runs": num_runs,
            "timestamp": datetime.now().isoformat(),
            "pytorch_times": [],
            "onnx_times": [],
            "speedup_ratios": []
        }
        
        # Warmup runs
        print("🔥 Warming up models...")
        warmup_texts = sample_texts[:batch_size]
        pytorch_inputs, onnx_inputs = self._prepare_inputs(warmup_texts)
        
        with torch.no_grad():
            _ = self.pytorch_model(**pytorch_inputs)
        
        _ = self.onnx_session.run(None, onnx_inputs)
        
        # Comparison runs
        for run_idx in range(num_runs):
            print(f"🔄 Run {run_idx + 1}/{num_runs}")
            
            # Prepare batch
            start_idx = run_idx * batch_size
            end_idx = start_idx + batch_size
            batch_texts = sample_texts[start_idx:end_idx]
            
            pytorch_inputs, onnx_inputs = self._prepare_inputs(batch_texts)
            
            # PyTorch inference
            with torch.no_grad():
                start_time = time.time()
                pytorch_output = self.pytorch_model(**pytorch_inputs)
                pytorch_time = time.time() - start_time
            
            # ONNX inference
            start_time = time.time()
            onnx_output = self.onnx_session.run(None, onnx_inputs)
            onnx_time = time.time() - start_time
            
            # Calculate speedup
            speedup = pytorch_time / onnx_time if onnx_time > 0 else 0
            
            results["pytorch_times"].append(pytorch_time * 1000)  # Convert to ms
            results["onnx_times"].append(onnx_time * 1000)  # Convert to ms
            results["speedup_ratios"].append(speedup)
        
        # Calculate statistics
        self._calculate_comparison_statistics(results)
        
        return results
    
    def _calculate_comparison_statistics(self, results: Dict[str, Any]):
        """Calculate statistics for the comparison results."""
        pytorch_times = results["pytorch_times"]
        onnx_times = results["onnx_times"]
        speedup_ratios = results["speedup_ratios"]
        
        results["statistics"] = {
            "pytorch": {
                "mean": np.mean(pytorch_times),
                "std": np.std(pytorch_times),
                "min": np.min(pytorch_times),
                "max": np.max(pytorch_times),
                "median": np.median(pytorch_times)
            },
            "onnx": {
                "mean": np.mean(onnx_times),
                "std": np.std(onnx_times),
                "min": np.min(onnx_times),
                "max": np.max(onnx_times),
                "median": np.median(onnx_times)
            },
            "speedup": {
                "mean": np.mean(speedup_ratios),
                "std": np.std(speedup_ratios),
                "min": np.min(speedup_ratios),
                "max": np.max(speedup_ratios),
                "median": np.median(speedup_ratios)
            }
        }
    
    def save_results(self, filename: str = None):
        """Save comparison results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/onnx_comparison_{timestamp}.json"
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Comparison results saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a summary of comparison results."""
        if not hasattr(self, 'results') or not self.results:
            print("❌ No comparison results available. Run compare_inference() first.")
            return
        
        stats = self.results["statistics"]
        
        print("\n" + "="*70)
        print("📊 ONNX RUNTIME vs PYTORCH COMPARISON SUMMARY")
        print("="*70)
        print(f"🤖 Model: {self.results['model_name']}")
        print(f"📦 Batch Size: {self.results['batch_size']}")
        print(f"🔄 Number of Runs: {self.results['num_runs']}")
        print(f"⏰ Timestamp: {self.results['timestamp']}")
        
        print("\n⏱️  INFERENCE TIMES (milliseconds):")
        print("-" * 50)
        print(f"PyTorch:")
        print(f"  Mean: {stats['pytorch']['mean']:.2f} ± {stats['pytorch']['std']:.2f}")
        print(f"  Range: {stats['pytorch']['min']:.2f} - {stats['pytorch']['max']:.2f}")
        print(f"  Median: {stats['pytorch']['median']:.2f}")
        
        print(f"\nONNX Runtime:")
        print(f"  Mean: {stats['onnx']['mean']:.2f} ± {stats['onnx']['std']:.2f}")
        print(f"  Range: {stats['onnx']['min']:.2f} - {stats['onnx']['max']:.2f}")
        print(f"  Median: {stats['onnx']['median']:.2f}")
        
        print(f"\n🚀 SPEEDUP ANALYSIS:")
        print(f"  Mean Speedup: {stats['speedup']['mean']:.2f}x")
        print(f"  Best Speedup: {stats['speedup']['max']:.2f}x")
        print(f"  Worst Speedup: {stats['speedup']['min']:.2f}x")
        
        # Performance recommendation
        if stats['speedup']['mean'] > 1.1:
            print(f"\n✅ ONNX Runtime is {stats['speedup']['mean']:.2f}x faster on average")
        elif stats['speedup']['mean'] < 0.9:
            print(f"\n⚠️  PyTorch is {1/stats['speedup']['mean']:.2f}x faster on average")
        else:
            print(f"\n🔄 Performance is similar between frameworks")
        
        print("="*70)


def main():
    """Main function to run the ONNX comparison with CLI arguments."""
    parser = argparse.ArgumentParser(description="ONNX Runtime vs PyTorch Inference Comparison")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                       help="HuggingFace model name (default: distilbert-base-uncased)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for inference (default: 4)")
    parser.add_argument("--num-runs", type=int, default=10,
                       help="Number of comparison runs (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename for results (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Create and run comparison
    comparison = ONNXComparison(model_name=args.model)
    
    try:
        results = comparison.compare_inference(
            batch_size=args.batch_size,
            num_runs=args.num_runs
        )
        
        comparison.results = results
        comparison.save_results(args.output)
        comparison.print_summary()
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 