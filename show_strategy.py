#!/usr/bin/env python3
"""
show_strategy.py - Show deployment strategy
"""
import torch
from pathlib import Path

cuda = torch.cuda.is_available()
print("=" * 80)
print("DEPLOYMENT STRATEGY")
print("=" * 80)
print(f"\nGPU: {'YES' if cuda else 'NO'} | PyTorch: {torch.__version__}")
print("\nModels:")
for name, path in [("PyTorch", "checkpoints/depth_anything_v2_vits.pth"), ("ONNX Full", "models/onnx/depth_anything_v2_vits.onnx"), ("ONNX Int8", "models/onnx/depth_anything_v2_vits_quantized_int8.onnx")]:
    ok = "OK" if Path(path).exists() else "NO"
    print(f"  {name:12s} {ok}")

if cuda:
    print("\n[WITH GPU]")
    print("1. realtime_pytorch_accurate.py --skip 1")
    print("   FPS: 1.0-1.5 | Best quality")
    print("2. realtime_onnx_accurate.py --skip 1")
    print("   FPS: 1.5-2.0 | Good quality")
    print("3. realtime_onnx_quantized.py --skip 1")
    print("   FPS: 2.5-3.5 | Good quality, smallest model")
else:
    print("\n[CPU ONLY - NO GPU]")
    print("1. realtime_onnx_accurate.py --skip 1")
    print("   FPS: 0.8-1.0 | Best quality on CPU")
    print("2. realtime_onnx_quantized.py --skip 1")
    print("   FPS: 1.4-2.0 | Good quality, faster")
    print("\nGPU option:")
    print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\nQuantization (INT8):")
print("  Model size: 26 MB (vs 94 MB for full precision)")
print("  Speed gain: 1.5-2x faster")
print("  Quality: Minimal loss, still very good")
print("  Best for: Mobile, edge devices, Android deployment")

print("\n" + "=" * 80)
