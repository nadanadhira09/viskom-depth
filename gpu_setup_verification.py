"""
GPU SETUP & VERIFICATION SCRIPT
================================
Deteksi GPU, install CUDA PyTorch jika diperlukan,
dan verifikasi setup untuk optimization pipeline.
"""

import subprocess
import sys
import torch
import onnxruntime as ort
import psutil
from pathlib import Path

print("=" * 70)
print("🖥️  GPU SETUP & VERIFICATION")
print("=" * 70)

# ============================================================================
# STEP 1: Check System Resources
# ============================================================================
print("\n[1/5] Checking system resources...")
print(f"  • CPU Cores: {psutil.cpu_count()}")
print(f"  • RAM Total: {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"  • RAM Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")

# ============================================================================
# STEP 2: Check PyTorch CUDA Support
# ============================================================================
print("\n[2/5] Checking PyTorch CUDA support...")
print(f"  • PyTorch Version: {torch.__version__}")
print(f"  • CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  ✅ GPU DETECTED!")
    print(f"  • GPU Name: {torch.cuda.get_device_name()}")
    print(f"  • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"  • CUDA Version: {torch.version.cuda}")
    print(f"  • cuDNN Version: {torch.backends.cudnn.version()}")
    GPU_AVAILABLE = True
else:
    print(f"  ❌ GPU NOT DETECTED (CPU-only mode)")
    GPU_AVAILABLE = False

# ============================================================================
# STEP 3: Check ONNX Runtime Backends
# ============================================================================
print("\n[3/5] Checking ONNX Runtime backends...")
print(f"  • ONNX Runtime Version: {ort.__version__}")
available_providers = ort.get_available_providers()
print(f"  • Available Providers: {available_providers}")

if "CUDAExecutionProvider" in available_providers:
    print("  ✅ CUDA Provider available for ONNX Runtime")
    ONNX_CUDA = True
else:
    print("  ⚠️  CUDA Provider NOT available for ONNX Runtime")
    ONNX_CUDA = False

# ============================================================================
# STEP 4: Recommendation & Optional GPU Installation
# ============================================================================
print("\n[4/5] Optimization recommendations...")

if GPU_AVAILABLE and ONNX_CUDA:
    print("\n  ✅ FULL GPU SETUP COMPLETE!")
    print("  Expected speedup multiplier: 3-5x")
    print("  Target achievable: YES (12.78x possible)")
    STATUS = "FULL_GPU"

elif GPU_AVAILABLE and not ONNX_CUDA:
    print("\n  ⚠️  GPU detected but ONNX Runtime CUDA not available")
    print("  Recommendation: Reinstall ONNX Runtime with CUDA support")
    print("  Command: pip install onnxruntime-gpu")
    STATUS = "PARTIAL_GPU"

else:
    print("\n  ❌ GPU NOT AVAILABLE (CPU-only)")
    print("  Alternatives:")
    print("    1. Use CPU optimization: Batch + ONNX optimization")
    print("    2. Expected speedup: 1.3-1.5x (batch) + 1.5x (ONNX) = 1.95-2.25x")
    print("    3. To achieve 12.78x: Need GPU hardware")
    STATUS = "CPU_ONLY"

# ============================================================================
# STEP 5: Offer GPU Installation if Not Available
# ============================================================================
print("\n[5/5] Setup options...")

if not GPU_AVAILABLE:
    print("\n  To enable GPU optimization:")
    print("  " + "-" * 60)
    print("  1. Uninstall CPU PyTorch:")
    print("     pip uninstall torch torchvision torchaudio -y")
    print("\n  2. Install CUDA PyTorch (CUDA 11.8):")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n  3. Install ONNX Runtime GPU:")
    print("     pip install onnxruntime-gpu")
    print("\n  4. Verify:")
    print("     python gpu_setup_verification.py")
    print("  " + "-" * 60)
    
    user_input = input("\n  Would you like to install GPU support now? (y/n): ").strip().lower()
    
    if user_input == 'y':
        print("\n  Installing GPU support...")
        try:
            print("  [1/3] Removing CPU PyTorch...")
            subprocess.run([sys.executable, "-m", "pip", "uninstall", 
                          "torch", "torchvision", "torchaudio", "-y"],
                         capture_output=True, timeout=120)
            
            print("  [2/3] Installing CUDA PyTorch...")
            subprocess.run([sys.executable, "-m", "pip", "install", 
                          "torch", "torchvision", "torchaudio",
                          "--index-url", "https://download.pytorch.org/whl/cu118"],
                         capture_output=False, timeout=120)
            
            print("  [3/3] Installing ONNX Runtime GPU...")
            subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu"],
                         capture_output=False, timeout=120)
            
            print("\n  ✅ GPU installation complete!")
            print("  Please run this script again to verify.")
            sys.exit(0)
        except Exception as e:
            print(f"  ❌ Installation failed: {e}")
            sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("📊 SETUP SUMMARY")
print("=" * 70)

summary = {
    "FULL_GPU": {
        "status": "✅ READY FOR FULL OPTIMIZATION",
        "speedup": "3-5x (GPU)",
        "target": "12.78x (WITH Batch + ONNX Opt)",
        "next": "Run: python batch_inference_optimization.py"
    },
    "PARTIAL_GPU": {
        "status": "⚠️  PARTIAL GPU (need ONNX fix)",
        "speedup": "Limited",
        "target": "Reinstall onnxruntime-gpu",
        "next": "pip install onnxruntime-gpu"
    },
    "CPU_ONLY": {
        "status": "❌ CPU-ONLY (No GPU Hardware)",
        "speedup": "1.95-2.25x (Batch + ONNX Opt max)",
        "target": "12.78x not achievable without GPU",
        "next": "Run CPU optimizations OR install GPU hardware"
    }
}

info = summary[STATUS]
print(f"\nStatus: {info['status']}")
print(f"Speedup Potential: {info['speedup']}")
print(f"Target Achievable: {info['target']}")
print(f"Next Command: {info['next']}")

print("\n" + "=" * 70)
print("✅ VERIFICATION COMPLETE")
print("=" * 70)
