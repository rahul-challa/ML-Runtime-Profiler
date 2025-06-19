#!/usr/bin/env python3
"""
Test script to verify ML Runtime Profiler installation
Checks dependencies and basic functionality.
"""

import sys
import importlib
from pathlib import Path


def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    required_packages = [
        "torch",
        "transformers",
        "numpy",
        "matplotlib",
        "seaborn"
    ]
    
    optional_packages = [
        "onnxruntime",
        "tensorboard"
    ]
    
    all_good = True
    
    # Test required packages
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            all_good = False
    
    # Test optional packages
    print("\n🔍 Testing optional packages...")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"⚠️  {package}: {e} (optional)")
    
    return all_good


def test_cuda():
    """Test CUDA availability."""
    print("\n🔍 Testing CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA not available (CPU-only mode)")
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")


def test_model_download():
    """Test if we can download a small model."""
    print("\n🔍 Testing model download...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        print("📥 Downloading test model (distilbert-base-uncased)...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        
        print("✅ Model downloaded successfully!")
        
        # Test basic inference
        print("🧪 Testing basic inference...")
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Inference successful! Output shape: {outputs.last_hidden_state.shape}")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False
    
    return True


def test_profiler_import():
    """Test if the profiler can be imported."""
    print("\n🔍 Testing profiler import...")
    
    try:
        from profiler import MLRuntimeProfiler
        print("✅ MLRuntimeProfiler imported successfully!")
        
        # Test basic instantiation
        profiler = MLRuntimeProfiler(model_name="distilbert-base-uncased", device="cpu")
        print("✅ Profiler instantiated successfully!")
        
    except Exception as e:
        print(f"❌ Error importing profiler: {e}")
        return False
    
    return True


def test_file_structure():
    """Test if all required files exist."""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "profiler.py",
        "compare_onnx.py",
        "requirements.txt",
        "README.md"
    ]
    
    all_good = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            all_good = False
    
    return all_good


def main():
    """Run all tests."""
    print("🧪 ML Runtime Profiler - Installation Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Package imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: CUDA availability
    test_cuda()
    tests_passed += 1  # This test doesn't fail the installation
    
    # Test 3: File structure
    if test_file_structure():
        tests_passed += 1
    
    # Test 4: Profiler import
    if test_profiler_import():
        tests_passed += 1
    
    # Test 5: Model download
    if test_model_download():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Installation is complete.")
        print("\n🚀 You can now run:")
        print("   python profiler.py --help")
        print("   python example_usage.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Try running:")
        print("   pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    exit(main()) 