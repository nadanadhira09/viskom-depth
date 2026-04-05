"""
check_deployment_strategy.py
=============================
Hardware acceleration check dan deployment strategy recommendation
"""

import sys
import torch
import cv2
import numpy as np
from pathlib import Path

print("="*80)
print("HARDWARE ACCELERATION & DEPLOYMENT STRATEGY")
print("="*80)

# 1. Check CUDA availability
print("\n[1] CUDA & GPU CHECK")
print("-" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU(s) detected: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {gpu_mem:.2f} GB")
else:
    print("[WARN] CUDA tidak tersedia - PyTorch menggunakan CPU")
    print("To enable CUDA:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# 2. Check available models
print("\n[2] MODEL FILES CHECK")
print("-" * 80)
models_check = {
    "PyTorch (vits)": "checkpoints/depth_anything_v2_vits.pth",
    "ONNX (Full Precision)": "models/onnx/depth_anything_v2_vits.onnx",
    "ONNX (Quantized INT8)": "models/onnx/depth_anything_v2_vits_quantized_int8.onnx",
}

for name, path in models_check.items():
    exists = Path(path).exists()
    status = "[OK]" if exists else "[NO]"
    size = f"{Path(path).stat().st_size / 1024**2:.1f} MB" if exists else "N/A"
    print(f"{status} {name:30s} - {size:>10s}")

# 3. Check ONNX Runtime
print("\n[3] ONNX RUNTIME CHECK")
print("-" * 80)
try:
    import onnxruntime as ort
    print(f"[OK] ONNX Runtime version: {ort.__version__}")
    print(f"  Available providers: {ort.get_available_providers()}")
except ImportError:
    print("[NO] ONNX Runtime tidak terinstal")
    print("  Install: pip install onnxruntime")

# 4. Check OpenCV & camera
print("\n[4] OPENCV & CAMERA CHECK")
print("-" * 80)
print(f"[OK] OpenCV version: {cv2.__version__}")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print(f"[OK] Camera detected and working")
        cap.release()
    else:
        print(f"[NO] Camera not detected")
except:
    print(f"[NO] Camera error")

# 5. Performance expectations
print("\n[5] PERFORMANCE EXPECTATIONS (Inference latency)")
print("-" * 80)

perf_data = """
Model                           CPU Only      GPU (CUDA)    Notes
-----                           --------      ----------    -----
PyTorch (vits)                  ~1600ms       ~600-800ms    Raw model, accurate
ONNX Full Precision (vits)      ~800ms        ~400-500ms    Optimized, still accurate
ONNX Quantized INT8 (vits)      ~700ms        ~300-400ms    Fast, slight loss in detail

FPS Approximation (at 640x480 input):
- CPU with PyTorch:           0.4-0.6 FPS (1.6-2.5 sec/frame)
- CPU with ONNX Quantized:    1.4-2.0 FPS (0.5-0.7 sec/frame)
- GPU with ONNX Quantized:    2.5-3.5 FPS (0.3-0.4 sec/frame)

GPU acceleration: 3-5x speedup
ONNX optimization: 1.5-2x speedup
Quantization: 1-2x speedup
Combined (GPU + ONNX Quantized): 5-10x total speedup possible!
"""
print(perf_data)

# 6. Deployment recommendations
print("\n[6] DEPLOYMENT STRATEGY (RECOMMENDED)")
print("-" * 80)

if torch.cuda.is_available():
    current_status = "GPU AVAILABLE"
    print(f"[OK] {current_status}\n")
    
    print("TAHAP 1 - DEVELOPMENT (Accuracy Priority)")
    print("  python conversion/realtime_pytorch_accurate.py --skip 1")
    print("  - Use: Prototyping, algorithm tuning, evaluation")
    print("  - FPS: ~1.0-1.5 (with GPU)")
    print("  - Quality: Excellent\n")
    
    print("TAHAP 2 - PRODUCTION (Balanced)")
    print("  python conversion/realtime_onnx_accurate.py --skip 1")
    print("  - Use: Server deployment, real-time applications")
    print("  - FPS: ~1.5-2.0 (with GPU)")
    print("  - Quality: Very Good\n")
    
    print("TAHAP 3 - EDGE/MOBILE (Speed Priority)")
    print("  python conversion/realtime_onnx_quantized.py --skip 1")
    print("  - Use: Android/iOS, embedded devices, cloud edge")
    print("  - FPS: ~2.5-3.5 (with GPU)")
    print("  - Quality: Good (slight loss acceptable)")
else:
    current_status = "CPU ONLY - GPU NOT AVAILABLE"
    print(f"[NO] {current_status}\n")
    
    print("TAHAP 1 - RECOMMENDED (Best Balance)")
    print("  python conversion/realtime_onnx_accurate.py --skip 1")
    print("  - Use: Main deployment option on CPU")
    print("  - FPS: ~0.8-1.0")
    print("  - Quality: Very Good\n")
    
    print("TAHAP 2 - IF SPEED NEEDED (Fast)")
    print("  python conversion/realtime_onnx_quantized.py --skip 1")
    print("  - Use: When FPS too low, add frame skip")
    print("  - FPS: ~1.4-2.0")
    print("  - Quality: Good (slight loss)\n")
    
    print("TAHAP 3 - FOR MOBILE/EDGE")
    print("  Use ONNX Quantized model + increase skip parameter")
    print("  - python conversion/realtime_onnx_quantized.py --skip 2")
    print("  - Display smoothness maintained with frame skip")

# 7. Quantization trade-offs
print("\n[7] QUANTIZATION TRADE-OFFS (INT8 vs Full Precision)")
print("-" * 80)

tradeoff = """
Full Precision (FP32):
  Pros:
    - Maximum accuracy
    - Rich depth map details
    - Better for small objects/edges
  Cons:
    - Slower inference
    - Larger model size (94 MB)
    
Quantized INT8:
  Pros:
    - Much faster (1.5-2x speedup)
    - Smaller model (26 MB, 73% reduction!)
    - Better for deployment/mobile
  Cons:
    - Slight loss in accuracy
    - Depth gradients less smooth
    - May miss very small details
    
Recommendation:
  - Development/Research: Use Full Precision
  - Production servers: Use Full Precision or ONNX
  - Mobile/Edge devices: Use Quantized INT8
  - If accuracy critical: Skip quantization
  - If speed critical: Use quantized + GPU
"""
print(tradeoff)

# 8. Next steps
print("\n[8] NEXT STEPS")
print("-" * 80)

if torch.cuda.is_available():
    print("1. Your system has GPU - use it for 3-5x speedup!")
    print("2. Start with: python conversion/realtime_pytorch_accurate.py --skip 1")
    print("3. Benchmark: Compare FPS between PyTorch vs ONNX vs Quantized")
    print("4. For Android: Use ONNX Quantized format")
else:
    print("1. No GPU detected - consider GPU-accelerated PyTorch if available")
    print("2. Start with: python conversion/realtime_onnx_accurate.py --skip 1")
    print("3. If FPS low: Use --skip parameter to increase display smoothness")
    print("4. For Android: Use ONNX Quantized format")
    print("\nOptional: Install CUDA-enabled PyTorch")
    print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*80)
print("For model conversion to ncnn:")
print("  Convert ONNX to ncnn: onnx2ncnn model.onnx model.param model.bin")
print("  For quantization: ncnnoptimize model.param model.bin model_opt.param model_opt.bin 65536")
print("="*80 + "\n")
