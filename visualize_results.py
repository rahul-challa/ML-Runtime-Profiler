#!/usr/bin/env python3
"""
Visualization script for ML Runtime Profiler results
Generates charts and plots from profiling data.
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


class ResultsVisualizer:
    """Visualize profiling and comparison results."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory containing result files
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path("results/plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_profiling_results(self, filename: str) -> Dict[str, Any]:
        """Load profiling results from JSON file."""
        filepath = self.results_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_timing_distribution(self, results: Dict[str, Any], save_plot: bool = True):
        """Plot timing distribution for different components."""
        stats = results["statistics"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Timing Distribution - {results["model_name"]}', fontsize=16)
        
        # Tokenization times
        tokenization_times = []
        for run in results["runs"]:
            tokenization_times.extend(run["tokenization_times"])
        
        axes[0].hist(tokenization_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(stats["tokenization"]["mean"], color='red', linestyle='--', 
                       label=f'Mean: {stats["tokenization"]["mean"]:.2f}ms')
        axes[0].set_title('Tokenization Times')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Inference times
        inference_times = []
        for run in results["runs"]:
            inference_times.extend(run["inference_times"])
        
        axes[1].hist(inference_times, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].axvline(stats["inference"]["mean"], color='red', linestyle='--',
                       label=f'Mean: {stats["inference"]["mean"]:.2f}ms')
        axes[1].set_title('Inference Times')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        # Total times
        total_times = []
        for run in results["runs"]:
            total_times.extend(run["total_times"])
        
        axes[2].hist(total_times, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[2].axvline(stats["total"]["mean"], color='red', linestyle='--',
                       label=f'Mean: {stats["total"]["mean"]:.2f}ms')
        axes[2].set_title('Total Times')
        axes[2].set_xlabel('Time (ms)')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"timing_distribution_{results['model_name'].replace('/', '_')}.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved timing distribution plot: {filename}")
        
        plt.show()
    
    def plot_run_progression(self, results: Dict[str, Any], save_plot: bool = True):
        """Plot timing progression across runs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Run Progression Analysis - {results["model_name"]}', fontsize=16)
        
        # Prepare data
        run_data = []
        for run_idx, run in enumerate(results["runs"]):
            for batch_idx, (token_time, inf_time, total_time) in enumerate(
                zip(run["tokenization_times"], run["inference_times"], run["total_times"])
            ):
                run_data.append({
                    'run': run_idx + 1,
                    'batch': batch_idx + 1,
                    'tokenization': token_time,
                    'inference': inf_time,
                    'total': total_time
                })
        
        df = pd.DataFrame(run_data)
        
        # Tokenization progression
        token_means = df.groupby('run')['tokenization'].mean()
        axes[0, 0].plot(token_means.index, token_means.values, 'o-', color='skyblue', linewidth=2)
        axes[0, 0].set_title('Tokenization Time by Run')
        axes[0, 0].set_xlabel('Run')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Inference progression
        inf_means = df.groupby('run')['inference'].mean()
        axes[0, 1].plot(inf_means.index, inf_means.values, 'o-', color='lightgreen', linewidth=2)
        axes[0, 1].set_title('Inference Time by Run')
        axes[0, 1].set_xlabel('Run')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total time progression
        total_means = df.groupby('run')['total'].mean()
        axes[1, 0].plot(total_means.index, total_means.values, 'o-', color='lightcoral', linewidth=2)
        axes[1, 0].set_title('Total Time by Run')
        axes[1, 0].set_xlabel('Run')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot of all times
        all_times = []
        labels = []
        for run in results["runs"]:
            all_times.extend(run["total_times"])
            labels.extend([f"Run {i+1}"] * len(run["total_times"]))
        
        axes[1, 1].boxplot([run["total_times"] for run in results["runs"]], 
                          labels=[f"Run {i+1}" for i in range(len(results["runs"]))])
        axes[1, 1].set_title('Total Time Distribution by Run')
        axes[1, 1].set_xlabel('Run')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"run_progression_{results['model_name'].replace('/', '_')}.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved run progression plot: {filename}")
        
        plt.show()
    
    def plot_detailed_analysis(self, results: Dict[str, Any], save_plot: bool = True):
        """Plot detailed analysis if available."""
        if "detailed_analysis" not in results:
            print("⚠️  No detailed analysis available for plotting")
            return
        
        detailed = results["detailed_analysis"]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Detailed Analysis - {results["model_name"]}', fontsize=16)
        
        # Component breakdown
        components = ['Attention', 'Embedding']
        times = [detailed['attention_layer_avg_time'], detailed['embedding_avg_time']]
        colors = ['lightblue', 'lightgreen']
        
        axes[0].bar(components, times, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_title('Component Breakdown')
        axes[0].set_ylabel('Time (ms)')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (component, time) in enumerate(zip(components, times)):
            axes[0].text(i, time + 0.1, f'{time:.2f}ms', ha='center', va='bottom')
        
        # CPU vs CUDA breakdown
        cpu_cuda_data = {
            'CPU': detailed['total_cpu_time'],
            'CUDA': detailed['total_cuda_time']
        }
        
        axes[1].pie(cpu_cuda_data.values(), labels=cpu_cuda_data.keys(), autopct='%1.1f%%',
                   colors=['lightcoral', 'lightblue'], startangle=90)
        axes[1].set_title('CPU vs CUDA Time Distribution')
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"detailed_analysis_{results['model_name'].replace('/', '_')}.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved detailed analysis plot: {filename}")
        
        plt.show()
    
    def plot_onnx_comparison(self, results: Dict[str, Any], save_plot: bool = True):
        """Plot ONNX vs PyTorch comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'ONNX vs PyTorch Comparison - {results["model_name"]}', fontsize=16)
        
        stats = results["statistics"]
        
        # Bar chart comparison
        frameworks = ['PyTorch', 'ONNX Runtime']
        mean_times = [stats['pytorch']['mean'], stats['onnx']['mean']]
        std_times = [stats['pytorch']['std'], stats['onnx']['std']]
        
        bars = axes[0, 0].bar(frameworks, mean_times, yerr=std_times, 
                             color=['lightblue', 'lightgreen'], alpha=0.7, 
                             capsize=5, edgecolor='black')
        axes[0, 0].set_title('Mean Inference Times')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, mean_times):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{time:.2f}ms', ha='center', va='bottom')
        
        # Speedup distribution
        speedup_ratios = results["speedup_ratios"]
        axes[0, 1].hist(speedup_ratios, bins=15, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].axvline(stats['speedup']['mean'], color='red', linestyle='--',
                          label=f'Mean: {stats["speedup"]["mean"]:.2f}x')
        axes[0, 1].set_title('Speedup Distribution')
        axes[0, 1].set_xlabel('Speedup Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time progression
        run_numbers = list(range(1, len(results["pytorch_times"]) + 1))
        axes[1, 0].plot(run_numbers, results["pytorch_times"], 'o-', 
                       label='PyTorch', color='lightblue', linewidth=2)
        axes[1, 0].plot(run_numbers, results["onnx_times"], 's-', 
                       label='ONNX Runtime', color='lightgreen', linewidth=2)
        axes[1, 0].set_title('Inference Times by Run')
        axes[1, 0].set_xlabel('Run')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[1, 1].boxplot([results["pytorch_times"], results["onnx_times"]], 
                          labels=['PyTorch', 'ONNX Runtime'])
        axes[1, 1].set_title('Time Distribution Comparison')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f"onnx_comparison_{results['model_name'].replace('/', '_')}.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved ONNX comparison plot: {filename}")
        
        plt.show()
    
    def create_summary_report(self, profiling_file: str = None, onnx_file: str = None):
        """Create a comprehensive summary report with all visualizations."""
        print("📊 Creating comprehensive summary report...")
        
        if profiling_file:
            try:
                profiling_results = self.load_profiling_results(profiling_file)
                print(f"📈 Visualizing profiling results: {profiling_file}")
                
                self.plot_timing_distribution(profiling_results)
                self.plot_run_progression(profiling_results)
                self.plot_detailed_analysis(profiling_results)
                
            except Exception as e:
                print(f"❌ Error processing profiling file: {e}")
        
        if onnx_file:
            try:
                onnx_results = self.load_profiling_results(onnx_file)
                print(f"📈 Visualizing ONNX comparison: {onnx_file}")
                
                self.plot_onnx_comparison(onnx_results)
                
            except Exception as e:
                print(f"❌ Error processing ONNX file: {e}")
        
        print("✅ Summary report completed!")


def main():
    """Main function to run the visualizer with CLI arguments."""
    parser = argparse.ArgumentParser(description="Visualize ML Runtime Profiler results")
    parser.add_argument("--profiling-file", type=str, default=None,
                       help="Profiling results JSON file")
    parser.add_argument("--onnx-file", type=str, default=None,
                       help="ONNX comparison results JSON file")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory (default: results)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save plots to files")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ResultsVisualizer(results_dir=args.results_dir)
    
    # If no files specified, try to find recent files
    if not args.profiling_file and not args.onnx_file:
        results_dir = Path(args.results_dir)
        if results_dir.exists():
            profiling_files = list(results_dir.glob("profiling_results_*.json"))
            onnx_files = list(results_dir.glob("onnx_comparison_*.json"))
            
            if profiling_files:
                args.profiling_file = profiling_files[-1].name  # Most recent
                print(f"📁 Found profiling file: {args.profiling_file}")
            
            if onnx_files:
                args.onnx_file = onnx_files[-1].name  # Most recent
                print(f"📁 Found ONNX file: {args.onnx_file}")
    
    # Create summary report
    visualizer.create_summary_report(
        profiling_file=args.profiling_file,
        onnx_file=args.onnx_file
    )
    
    return 0


if __name__ == "__main__":
    exit(main()) 