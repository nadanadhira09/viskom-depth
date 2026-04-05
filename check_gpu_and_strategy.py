"""
check_gpu_and_strategy.py
=========================
Verifikasi GPU, CUDA, PyTorch, dan strategi deployment optimal
"""

import sys
import torch
import cv2
import numpy as np
from pathlib import Path

print("="*80)
print("HARDWARE ACCELERATION & DEPLOYMENT STRATEGY CHECK")
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
    print("[WARN] CUDA tidak tersedia - PyTorch akan menggunakan CPU")
    print("       Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

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

# 4. Check OpenCV
print("\n[4] OPENCV CHECK")
print("-" * 80)
print(f"[OK] OpenCV version: {cv2.__version__}")
print(f"  Camera support: {'Yes' if cv2.VideoCapture(0).isOpened() else 'No'}")

# 5. Deployment Strategy
print("\n[5] DEPLOYMENT STRATEGY REKOMENDASI")
print("-" * 80)

strategies = [
    {
        "name": "ACCURACY MODE (Development)",
        "model": "PyTorch (vits)",
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "fps": "0.8-1.2" if torch.cuda.is_available() else "0.4-0.6",
        "depth_quality": "Excellent",
        "use_case": "Prototyping, fine-tuning, evaluation",
        "command": "python conversion/realtime_pytorch_accurate.py --encoder vits --skip 1"
    },
    {
        "name": "BALANCED MODE (Production Server)",
        "model": "ONNX Full Precision",
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "fps": "1.2-1.8" if torch.cuda.is_available() else "0.8-1.0",
        "depth_quality": "Very Good",
        "use_case": "Cloud/server deployment, real-time applications",
        "command": "python conversion/realtime_onnx_accurate.py --skip 1"
    },
    {
        "name": "FAST MODE (Edge Device/Mobile)",
        "model": "ONNX Quantized INT8",
        "device": "CPU (optimized)",
        "fps": "1.4-2.0",
        "depth_quality": "Good (slight loss)",
        "use_case": "Mobile, embedded, edge computing (Android/iOS)",
        "command": "python conversion/realtime_onnx_quantized.py --skip 1"
    }
]

for i, strategy in enumerate(strategies, 1):
    print(f"\n{i}. {strategy['name']}")
    print(f"   Model:         {strategy['model']}")
    print(f"   Device:        {strategy['device']}")
    print(f"   FPS:           {strategy['fps']}")
    print(f"   Depth Quality: {strategy['depth_quality']}")
    print(f"   Use Case:      {strategy['use_case']}")
    print(f"   Command:       {strategy['command']}")

# 6. CUDA-optimized recommendation
print("\n[6] OPTIMISASI UNTUK SISTEM ANDA")
print("-" * 80)

if torch.cuda.is_available():
    print("[OK] GPU DETECTED - Recommended workflow:")
    print("\n  TAHAP 1 - DEVELOPMENT (Akurat):")
    print("    python conversion/realtime_pytorch_accurate.py --skip 1")
    print("    [FPS akan lebih baik di GPU - bisa 1-2 FPS dari 0.4-0.6 FPS di CPU]")
    print("\n  TAHAP 2 - OPTIMIZATION (Balance):")
    print("    python conversion/realtime_onnx_accurate.py --skip 1")
    print("    [ONNX sedikit lebih cepat dari PyTorch, depth tetap bagus]")
    print("\n  TAHAP 3 - DEPLOYMENT (Speed):")
    print("    python conversion/realtime_onnx_quantized.py --skip 1")
    print("    [Untuk edge devices - jauh lebih cepat]")
else:
    print("[NO] GPU NOT DETECTED - Recommended workflow (CPU only):")
    print("\n  TAHAP 1 - BALANCE (Rekomendasi):")
    print("    python conversion/realtime_onnx_accurate.py --skip 1")
    print("    [Lebih cepat dari PyTorch, depth masih akurat]")
    print("\n  TAHAP 2 - SPEED (Jika FPS masih kurang):")
    print("    python conversion/realtime_onnx_quantized.py --skip 2 atau --skip 3")
    print("    [Tambah frame skip untuk visual smoothness]")

# 7. Performance expectations
print("\n[7] PERKIRAAN PERFORMANCE")
print("-" * 80)

perf_table = f"""
┌─────────────────────────────┬──────────────┬──────────────┬─────────────┐
│ Model                       │ CPU Only     │ GPU (CUDA)   │ Quantized   │
├─────────────────────────────┼──────────────┼──────────────┼─────────────┤
│ PyTorch (vits)              │ 0.4-0.6 FPS  │ 1.0-1.5 FPS  │ N/A         │
│ ONNX Full Precision (vits)  │ 0.8-1.0 FPS  │ 1.2-1.8 FPS  │ N/A         │
│ ONNX Quantized INT8 (vits)  │ 1.4-2.0 FPS  │ 2.0-3.0 FPS  │ Yes (Best)  │
└─────────────────────────────┴──────────────┴──────────────┴─────────────┘

KESIMPULAN:
- GPU → 2-3x speedup dari CPU
- ONNX → 1.5-2x speedup dari PyTorch
- Quantized → 1.5-2x speedup dari full precision
- GPU + ONNX Quantized → potential 5-10x total speedup!
"""
print(perf_table)

print("\n[8] NEXT STEPS")
print("-" * 80)
if torch.cuda.is_available():
    print("1. ✓ GPU terdeteksi - mulai dengan realtime_pytorch_accurate.py")
    print("2. Bandingkan FPS antara PyTorch vs ONNX vs ONNX Quantized")
    print("3. Pilih strategi sesuai kebutuhan")
    print("4. Untuk deployment Android/Edge: gunakan ONNX Quantized")
else:
    print("1. Pertimbangkan install PyTorch dengan CUDA support:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("2. Jika GPU tersedia setelah install: restart Python dan run check ini lagi")
    print("3. Jika tidak ada GPU: gunakan ONNX Quantized + frame skip untuk smoothness")
    print("4. Untuk deployment: ONNX Quantized tetap pilihan terbaik")

print("\n" + "="*80)
