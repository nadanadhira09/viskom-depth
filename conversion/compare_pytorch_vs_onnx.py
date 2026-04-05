"""
compare_pytorch_vs_onnx.py
==========================
Direct comparison: PyTorch vs ONNX Runtime inference time
"""

import numpy as np
import time
import torch
import sys
import cv2
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_DIR))
from depth_anything_v2.dpt import DepthAnythingV2

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def benchmark_pytorch(frame, size=518):
    """Benchmark PyTorch inference"""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }
    
    print("\n[INFO] Loading PyTorch model...")
    model = DepthAnythingV2(**model_configs['vits'])
    model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
    model = model.to('cpu').eval()
    print("[OK] PyTorch model loaded")
    
    # Warmup
    with torch.no_grad():
        _ = model.infer_image(frame, input_size=size)
    
    # Benchmark
    times = []
    num_iterations = 5
    print(f"\n[BENCHMARK] Running {num_iterations} iterations (size {size}x{size})...")
    
    for i in range(num_iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            _ = model.infer_image(frame, input_size=size)
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.2f}ms")
    
    avg_time = np.mean(times[1:])
    fps = 1.0 / avg_time
    
    return avg_time, fps


def benchmark_onnx(frame, size=518):
    """Benchmark ONNX inference"""
    print("\n[INFO] Loading ONNX model...")
    session = ort.InferenceSession('models/onnx/depth_anything_v2_vits.onnx',
                                  providers=['CPUExecutionProvider'])
    print("[OK] ONNX model loaded")
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Preprocess
    def preprocess(img):
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img
    
    # Warmup
    img_resized = cv2.resize(frame, (size, size))
    img_input = preprocess(img_resized)
    _ = session.run([output_name], {input_name: img_input})
    
    # Benchmark
    times = []
    num_iterations = 5
    print(f"\n[BENCHMARK] Running {num_iterations} iterations (size {size}x{size})...")
    
    for i in range(num_iterations):
        start = time.time()
        
        img_input = preprocess(img_resized)
        _ = session.run([output_name], {input_name: img_input})
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed*1000:.2f}ms")
    
    avg_time = np.mean(times[1:])
    fps = 1.0 / avg_time
    
    return avg_time, fps


def main():
    import cv2
    
    print("="*70)
    print("PyTorch vs ONNX Runtime - Performance Comparison")
    print("="*70)
    
    # Create test frame
    print("\n[INFO] Creating test frame (640x480)...")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Benchmark PyTorch
    print("\n" + "─"*70)
    print("PyTorch (CPU)")
    print("─"*70)
    pt_time, pt_fps = benchmark_pytorch(frame, size=518)
    
    # Benchmark ONNX
    if ONNX_AVAILABLE:
        print("\n" + "─"*70)
        print("ONNX Runtime (CPU)")
        print("─"*70)
        onnx_time, onnx_fps = benchmark_onnx(frame, size=518)
        
        # Comparison
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(f"\nPyTorch Average: {pt_time*1000:.2f}ms ({pt_fps:.2f} FPS)")
        print(f"ONNX Average:    {onnx_time*1000:.2f}ms ({onnx_fps:.2f} FPS)")
        
        speedup = pt_time / onnx_time
        print(f"\n🚀 SPEEDUP: {speedup:.1f}x faster with ONNX!")
        print(f"📈 FPS Improvement: {pt_fps:.2f} → {onnx_fps:.2f} FPS")
    else:
        print("\n[WARNING] ONNX not available")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
