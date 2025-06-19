#!/usr/bin/env python3
"""
ML Runtime Profiler for Transformer Inference
A lightweight profiling tool that analyzes inference latency in transformer-based models.
"""

import argparse
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import torch.profiler
from transformers import AutoTokenizer, AutoModel
import numpy as np
from datetime import datetime


class MLRuntimeProfiler:
    """Main profiler class for analyzing transformer model inference performance."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = "cpu"):
        """
        Initialize the profiler with a transformer model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.results = {}
        
        print(f"🚀 Initializing ML Runtime Profiler")
        print(f"📦 Model: {model_name}")
        print(f"💻 Device: {device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            print("📥 Loading model...")
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _generate_sample_texts(self, num_samples: int, sequence_length: int) -> List[str]:
        """Generate sample texts for profiling."""
        sample_texts = [
            "This is a sample text for machine learning inference profiling.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require careful performance analysis.",
            "Transformer architectures have revolutionized natural language processing.",
            "Profiling helps identify bottlenecks in model inference.",
            "GPU acceleration can significantly improve inference speed.",
            "Attention mechanisms are computationally expensive but effective.",
            "Batch processing can improve throughput in production systems.",
            "Memory usage is a critical consideration for large models.",
            "Optimization techniques can reduce inference latency."
        ]
        
        # Repeat and truncate to get desired number of samples
        texts = []
        for i in range(num_samples):
            text = sample_texts[i % len(sample_texts)]
            # Truncate or pad to approximate sequence length
            words = text.split()
            if len(words) > sequence_length // 5:  # Rough estimate
                words = words[:sequence_length // 5]
            texts.append(" ".join(words))
        
        return texts
    
    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        start_time = time.time()
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        tokenization_time = time.time() - start_time
        
        return inputs, tokenization_time
    
    def profile_inference(self, batch_size: int = 4, sequence_length: int = 128, 
                         num_runs: int = 5) -> Dict[str, Any]:
        """
        Profile model inference with detailed timing analysis.
        
        Args:
            batch_size: Number of samples per batch
            sequence_length: Maximum sequence length
            num_runs: Number of profiling runs
            
        Returns:
            Dictionary containing profiling results
        """
        print(f"\n🔍 Starting inference profiling...")
        print(f"📊 Batch size: {batch_size}")
        print(f"📏 Sequence length: {sequence_length}")
        print(f"🔄 Number of runs: {num_runs}")
        
        # Generate sample texts
        sample_texts = self._generate_sample_texts(batch_size, sequence_length)
        
        # Prepare batches
        batches = [sample_texts[i:i + batch_size] for i in range(0, len(sample_texts), batch_size)]
        if not batches:
            batches = [sample_texts]
        
        results = {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "num_runs": num_runs,
            "timestamp": datetime.now().isoformat(),
            "runs": []
        }
        
        # Warmup run
        print("🔥 Warming up model...")
        with torch.no_grad():
            for batch in batches[:1]:
                inputs, _ = self._tokenize_batch(batch)
                _ = self.model(**inputs)
        
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.synchronize()
        
        # Profiling runs
        for run_idx in range(num_runs):
            print(f"🔄 Run {run_idx + 1}/{num_runs}")
            
            run_results = {
                "run_id": run_idx,
                "batch_times": [],
                "tokenization_times": [],
                "inference_times": [],
                "total_times": []
            }
            
            for batch_idx, batch in enumerate(batches):
                # Tokenization timing
                inputs, tokenization_time = self._tokenize_batch(batch)
                
                # Inference timing with torch.profiler
                with torch.no_grad():
                    if run_idx == 0:  # Only profile the first run for detailed analysis
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                            with_stack=True,
                            profile_memory=True
                        ) as prof:
                            start_time = time.time()
                            outputs = self.model(**inputs)
                            if torch.cuda.is_available() and self.device == "cuda":
                                torch.cuda.synchronize()
                            inference_time = time.time() - start_time
                    else:
                        start_time = time.time()
                        outputs = self.model(**inputs)
                        if torch.cuda.is_available() and self.device == "cuda":
                            torch.cuda.synchronize()
                        inference_time = time.time() - start_time
                
                total_time = tokenization_time + inference_time
                
                run_results["batch_times"].append(batch_idx)
                run_results["tokenization_times"].append(tokenization_time * 1000)  # Convert to ms
                run_results["inference_times"].append(inference_time * 1000)  # Convert to ms
                run_results["total_times"].append(total_time * 1000)  # Convert to ms
            
            results["runs"].append(run_results)
            
            # Save detailed profiler output for first run
            if run_idx == 0:
                prof_dir = Path("results")
                prof_dir.mkdir(exist_ok=True)
                prof.export_chrome_trace(f"results/trace_run_{run_idx}.json")
                
                # Extract key metrics from profiler
                key_averages = prof.key_averages()
                attention_times = []
                embedding_times = []
                
                for event in key_averages:
                    if "attention" in event.key.lower():
                        attention_times.append(event.cpu_time_total / 1000)  # Convert to ms
                    elif "embedding" in event.key.lower():
                        embedding_times.append(event.cpu_time_total / 1000)  # Convert to ms
                
                results["detailed_analysis"] = {
                    "attention_layer_avg_time": np.mean(attention_times) if attention_times else 0,
                    "embedding_avg_time": np.mean(embedding_times) if embedding_times else 0,
                    "total_cpu_time": sum(event.cpu_time_total for event in key_averages) / 1000,
                    "total_cuda_time": sum(event.cuda_time_total for event in key_averages) / 1000,
                }
        
        # Calculate aggregate statistics
        self._calculate_statistics(results)
        
        self.results = results
        return results
    
    def _calculate_statistics(self, results: Dict[str, Any]):
        """Calculate aggregate statistics from profiling runs."""
        all_tokenization_times = []
        all_inference_times = []
        all_total_times = []
        
        for run in results["runs"]:
            all_tokenization_times.extend(run["tokenization_times"])
            all_inference_times.extend(run["inference_times"])
            all_total_times.extend(run["total_times"])
        
        results["statistics"] = {
            "tokenization": {
                "mean": np.mean(all_tokenization_times),
                "std": np.std(all_tokenization_times),
                "min": np.min(all_tokenization_times),
                "max": np.max(all_tokenization_times),
                "median": np.median(all_tokenization_times)
            },
            "inference": {
                "mean": np.mean(all_inference_times),
                "std": np.std(all_inference_times),
                "min": np.min(all_inference_times),
                "max": np.max(all_inference_times),
                "median": np.median(all_inference_times)
            },
            "total": {
                "mean": np.mean(all_total_times),
                "std": np.std(all_total_times),
                "min": np.min(all_total_times),
                "max": np.max(all_total_times),
                "median": np.median(all_total_times)
            }
        }
    
    def save_results(self, filename: str = None):
        """Save profiling results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/profiling_results_{timestamp}.json"
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a summary of profiling results."""
        if not self.results:
            print("❌ No profiling results available. Run profile_inference() first.")
            return
        
        stats = self.results["statistics"]
        
        print("\n" + "="*60)
        print("📊 ML RUNTIME PROFILER RESULTS SUMMARY")
        print("="*60)
        print(f"🤖 Model: {self.results['model_name']}")
        print(f"💻 Device: {self.results['device']}")
        print(f"📦 Batch Size: {self.results['batch_size']}")
        print(f"📏 Sequence Length: {self.results['sequence_length']}")
        print(f"🔄 Number of Runs: {self.results['num_runs']}")
        print(f"⏰ Timestamp: {self.results['timestamp']}")
        
        print("\n⏱️  TIMING STATISTICS (milliseconds):")
        print("-" * 40)
        print(f"Tokenization:")
        print(f"  Mean: {stats['tokenization']['mean']:.2f} ± {stats['tokenization']['std']:.2f}")
        print(f"  Range: {stats['tokenization']['min']:.2f} - {stats['tokenization']['max']:.2f}")
        
        print(f"\nInference:")
        print(f"  Mean: {stats['inference']['mean']:.2f} ± {stats['inference']['std']:.2f}")
        print(f"  Range: {stats['inference']['min']:.2f} - {stats['inference']['max']:.2f}")
        
        print(f"\nTotal:")
        print(f"  Mean: {stats['total']['mean']:.2f} ± {stats['total']['std']:.2f}")
        print(f"  Range: {stats['total']['min']:.2f} - {stats['total']['max']:.2f}")
        
        if "detailed_analysis" in self.results:
            detailed = self.results["detailed_analysis"]
            print(f"\n🔍 DETAILED ANALYSIS:")
            print(f"  Attention Layer Avg: {detailed['attention_layer_avg_time']:.2f} ms")
            print(f"  Embedding Avg: {detailed['embedding_avg_time']:.2f} ms")
            print(f"  Total CPU Time: {detailed['total_cpu_time']:.2f} ms")
            print(f"  Total CUDA Time: {detailed['total_cuda_time']:.2f} ms")
        
        print("="*60)


def main():
    """Main function to run the profiler with CLI arguments."""
    parser = argparse.ArgumentParser(description="ML Runtime Profiler for Transformer Inference")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                       help="HuggingFace model name (default: distilbert-base-uncased)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for inference (default: 4)")
    parser.add_argument("--sequence-length", type=int, default=128,
                       help="Maximum sequence length (default: 128)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run inference on (default: cpu)")
    parser.add_argument("--num-runs", type=int, default=5,
                       help="Number of profiling runs (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename for results (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    
    # Create and run profiler
    profiler = MLRuntimeProfiler(model_name=args.model, device=args.device)
    
    try:
        results = profiler.profile_inference(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_runs=args.num_runs
        )
        
        profiler.save_results(args.output)
        profiler.print_summary()
        
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 