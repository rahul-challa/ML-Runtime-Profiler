# Quick Start Guide

Get up and running with the ML Runtime Profiler in minutes!

## 🚀 Installation

### Option 1: Quick Install (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Test installation
python test_installation.py
```

### Option 2: Development Install
```bash
# Install in development mode
pip install -e .

# Test installation
python test_installation.py
```

## 🎯 Quick Examples

### 1. Basic Profiling
```bash
# Profile DistilBERT on CPU
python profiler.py

# Profile with custom settings
python profiler.py --model distilbert-base-uncased --batch-size 8 --device cuda --num-runs 10
```

### 2. ONNX Comparison
```bash
# Compare ONNX Runtime vs PyTorch
python compare_onnx.py --model distilbert-base-uncased --batch-size 4 --num-runs 20
```

### 3. Run All Examples
```bash
# Run comprehensive examples
python example_usage.py
```

### 4. Visualize Results
```bash
# Create charts and plots
python visualize_results.py
```

## 📊 Understanding Output

### Profiling Results
The profiler generates detailed timing analysis:
- **Tokenization Time**: Time to convert text to tokens
- **Inference Time**: Time for model forward pass
- **Total Time**: Complete end-to-end time
- **Detailed Analysis**: Attention layers, embeddings, CPU vs GPU breakdown

### Sample Output
```
📊 ML RUNTIME PROFILER RESULTS SUMMARY
============================================================
🤖 Model: distilbert-base-uncased
💻 Device: cuda
📦 Batch Size: 4
📏 Sequence Length: 128
🔄 Number of Runs: 5

⏱️  TIMING STATISTICS (milliseconds):
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
============================================================
```

## 🔧 Common Use Cases

### Performance Optimization
```bash
# Test different batch sizes
for batch_size in 1 2 4 8 16; do
    python profiler.py --batch-size $batch_size --num-runs 5
done

# Compare CPU vs GPU
python profiler.py --device cpu --num-runs 5
python profiler.py --device cuda --num-runs 5
```

### Model Comparison
```bash
# Compare different models
python profiler.py --model distilbert-base-uncased --num-runs 5
python profiler.py --model microsoft/MiniLM-L12-H384-uncased --num-runs 5
```

### Production Optimization
```bash
# Test ONNX optimization
python compare_onnx.py --model distilbert-base-uncased --batch-size 8 --num-runs 50
```

## 📁 Output Files

After running the profiler, check the `results/` directory:

```
results/
├── profiling_results_20240115_103045.json    # Main profiling results
├── onnx_comparison_20240115_103522.json      # ONNX comparison results
├── trace_run_0.json                          # TensorBoard trace (first run)
├── distilbert-base-uncased.onnx              # ONNX model file
└── plots/                                    # Generated charts
    ├── timing_distribution_distilbert-base-uncased.png
    ├── run_progression_distilbert-base-uncased.png
    └── onnx_comparison_distilbert-base-uncased.png
```

## 🎨 Visualization

### TensorBoard Traces
```bash
# Start TensorBoard
tensorboard --logdir results/

# Open browser to http://localhost:6006
```

### Generated Plots
```bash
# Create all visualizations
python visualize_results.py

# View plots in results/plots/ directory
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python profiler.py --batch-size 2 --device cuda
   ```

2. **Model Download Issues**
   ```bash
   # Check internet connection
   # Try different model
   python profiler.py --model microsoft/MiniLM-L12-H384-uncased
   ```

3. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --upgrade
   ```

### Performance Tips

- **GPU Memory**: Monitor with `nvidia-smi`
- **Batch Size**: Larger batches often improve throughput
- **Warmup**: First run includes model compilation time
- **Sequence Length**: Shorter sequences are faster

## 📈 Next Steps

1. **Explore Examples**: Run `python example_usage.py`
2. **Custom Analysis**: Modify `profiler.py` for your needs
3. **Visualization**: Use `visualize_results.py` for insights
4. **Production**: Integrate profiling into your ML pipeline

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Run `python test_installation.py` to diagnose issues
- Review example outputs in the `results/` directory

Happy Profiling! 🚀 