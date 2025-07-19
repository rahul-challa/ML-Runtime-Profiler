# ML Runtime Profiler for Transformer Inference

A lightweight Python-based profiling tool that analyzes inference latency in transformer-based models using PyTorch and HuggingFace Transformers.

## Goal

Demonstrate understanding of performance bottlenecks in ML workloads by building a comprehensive runtime profiler that analyzes inference latency in transformer-based models like DistilBERT or MiniLM.

## Features

- **Model Loading**: Supports any HuggingFace transformer model
- **Detailed Profiling**: Uses `torch.profiler` to measure:
  - Time taken by attention layers
  - Time spent on tokenization and embedding
  - GPU vs CPU execution time
  - Memory usage patterns
- **Batch Processing**: Configurable batch sizes and sequence lengths
- **Device Support**: CPU and CUDA (GPU) inference
- **Visualization**: TensorBoard integration for detailed traces
- **ONNX Comparison**: Optional comparison between ONNX Runtime and PyTorch
- **CLI Interface**: Easy-to-use command-line arguments

## Tools & Technologies

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **HuggingFace Transformers** - Pre-trained models
- **torch.profiler** - Performance profiling
- **TensorBoard** - Visualization
- **ONNX Runtime** - Optimized inference engine
- **NumPy** - Numerical computations

## Project Structure

```
ML Runtime Profiler/
├── profiler.py          # Main profiler script
├── compare_onnx.py      # ONNX vs PyTorch comparison
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── results/            # Output directory (created automatically)
    ├── profiling_results_*.json
    ├── onnx_comparison_*.json
    └── trace_*.json    # TensorBoard traces
```

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd "ML Runtime Profiler"

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Run basic profiling with default settings
python profiler.py

# Profile with custom parameters
python profiler.py --model distilbert-base-uncased --batch-size 8 --device cuda --num-runs 10

# Compare ONNX vs PyTorch performance
python compare_onnx.py --model distilbert-base-uncased --batch-size 4 --num-runs 20
```

### 3. CLI Arguments

#### Main Profiler (`profiler.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `distilbert-base-uncased` | HuggingFace model name |
| `--batch-size` | int | `4` | Batch size for inference |
| `--sequence-length` | int | `128` | Maximum sequence length |
| `--device` | str | `cpu` | Device (`cpu` or `cuda`) |
| `--num-runs` | int | `5` | Number of profiling runs |
| `--output` | str | `auto` | Output filename |

#### ONNX Comparison (`compare_onnx.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `distilbert-base-uncased` | HuggingFace model name |
| `--batch-size` | int | `4` | Batch size for inference |
| `--num-runs` | int | `10` | Number of comparison runs |
| `--output` | str | `auto` | Output filename |

## Output Analysis

### Profiling Results

The profiler generates detailed JSON reports containing:

- **Timing Statistics**: Mean, std, min, max, median for tokenization, inference, and total times
- **Detailed Analysis**: Attention layer times, embedding times, CPU vs CUDA breakdown
- **Memory Usage**: Peak memory consumption and allocation patterns
- **TensorBoard Traces**: Chrome trace format for detailed visualization

### Sample Output

```
ML RUNTIME PROFILER RESULTS SUMMARY
============================================================
 Model: distilbert-base-uncased
 Device: cuda
 Batch Size: 4
 Sequence Length: 128
 Number of Runs: 5
 Timestamp: 2024-01-15T10:30:45.123456

TIMING STATISTICS (milliseconds):
----------------------------------------
Tokenization:
  Mean: 2.45 ± 0.12
  Range: 2.31 - 2.67

Inference:
  Mean: 15.23 ± 0.89
  Range: 14.12 - 16.45

Total:
  Mean: 17.68 ± 0.91
  Range: 16.43 - 18.12

DETAILED ANALYSIS:
  Attention Layer Avg: 8.45 ms
  Embedding Avg: 2.12 ms
  Total CPU Time: 3.21 ms
  Total CUDA Time: 12.02 ms
============================================================
```

### ONNX Comparison Results

```
ONNX RUNTIME vs PYTORCH COMPARISON SUMMARY
======================================================================
Model: distilbert-base-uncased
Batch Size: 4
Number of Runs: 10
Timestamp: 2024-01-15T10:35:22.654321

INFERENCE TIMES (milliseconds):
--------------------------------------------------
PyTorch:
  Mean: 15.23 ± 0.89
  Range: 14.12 - 16.45
  Median: 15.18

ONNX Runtime:
  Mean: 12.45 ± 0.67
  Range: 11.89 - 13.21
  Median: 12.38

SPEEDUP ANALYSIS:
  Mean Speedup: 1.22x
  Best Speedup: 1.35x
  Worst Speedup: 1.08x

ONNX Runtime is 1.22x faster on average
======================================================================
```

## Skills Demonstrated

### ML Systems Performance Understanding
- Profiling attention mechanisms and embedding layers
- Analyzing GPU vs CPU performance characteristics
- Understanding memory allocation patterns
- Identifying computational bottlenecks

### Familiarity with Inference Engines
- PyTorch native inference
- ONNX Runtime optimization
- Model format conversion
- Performance comparison methodologies

### GPU-Aware Engineering
- CUDA synchronization
- Memory management
- Device-specific optimizations
- Profiling GPU operations

## Advanced Usage

### TensorBoard Visualization

```bash
# Start TensorBoard to view detailed traces
tensorboard --logdir results/

# Open browser to http://localhost:6006
```

### Custom Model Analysis

```python
from profiler import MLRuntimeProfiler

# Create profiler with custom model
profiler = MLRuntimeProfiler(
    model_name="microsoft/MiniLM-L12-H384-uncased",
    device="cuda"
)

# Run profiling
results = profiler.profile_inference(
    batch_size=8,
    sequence_length=256,
    num_runs=10
)

# Save and analyze results
profiler.save_results("custom_analysis.json")
profiler.print_summary()
```

### Batch Size Optimization

```bash
# Test different batch sizes
for batch_size in 1 2 4 8 16; do
    python profiler.py --batch-size $batch_size --num-runs 5
done
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 2`
   - Use CPU: `--device cpu`

2. **Model Download Issues**
   - Check internet connection
   - Verify model name in HuggingFace Hub

3. **ONNX Conversion Errors**
   - Ensure PyTorch version compatibility
   - Try different opset versions

### Performance Tips

- **GPU Memory**: Monitor with `nvidia-smi`
- **Batch Size**: Larger batches often improve throughput
- **Sequence Length**: Shorter sequences are faster
- **Warmup**: First run includes model compilation time

## Future Enhancements

- [ ] Support for more model architectures (GPT, T5, etc.)
- [ ] Memory profiling and optimization suggestions
- [ ] Integration with other inference engines (TensorRT, OpenVINO)
- [ ] Real-time monitoring dashboard
- [ ] Automated performance regression testing
- [ ] Support for quantized models

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the profiler!

## License

This project is open source and available under the MIT License. 
