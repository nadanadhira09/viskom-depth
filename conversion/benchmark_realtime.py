"""
benchmark_realtime.py
=====================
Profile dan benchmark untuk mengidentifikasi bottleneck FPS
"""

import cv2
import torch
import numpy as np
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_DIR))
from depth_anything_v2.dpt import DepthAnythingV2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def benchmark_depth_model(frame, model, device, sizes=[256, 384, 518]):
    """Benchmark depth inference di berbagai resolution"""
    print("\n" + "="*70)
    print("DEPTH MODEL BENCHMARKING")
    print("="*70)
    
    h, w = frame.shape[:2]
    print(f"Input Frame: {w}x{h}")
    
    model.eval()
    
    for size in sizes:
        # Warmup
        with torch.no_grad():
            _ = model.infer_image(frame, input_size=size)
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            with torch.no_grad():
                _ = model.infer_image(frame, input_size=size)
            times.append(time.time() - start)
        
        avg_time = np.mean(times[1:])  # Skip first
        fps = 1.0 / avg_time
        
        print(f"  Size {size:3d}x{size:3d}: {avg_time*1000:6.1f}ms | {fps:5.1f} FPS")


def benchmark_yolo_model(frame, model):
    """Benchmark YOLO inference"""
    print("\n" + "="*70)
    print("YOLO MODEL BENCHMARKING")
    print("="*70)
    
    h, w = frame.shape[:2]
    print(f"Input Frame: {w}x{h}")
    
    # Warmup
    _ = model(frame, verbose=False)
    
    # Benchmark
    times = []
    for _ in range(5):
        start = time.time()
        _ = model(frame, verbose=False)
        times.append(time.time() - start)
    
    avg_time = np.mean(times[1:])
    fps = 1.0 / avg_time
    
    print(f"  YOLO v12n: {avg_time*1000:6.1f}ms | {fps:5.1f} FPS")


def benchmark_pipeline(frame, depth_model, yolo_model, device):
    """Benchmark combined pipeline"""
    print("\n" + "="*70)
    print("PIPELINE BENCHMARKING (Depth + YOLO)")
    print("="*70)
    
    depth_model.eval()
    
    configs = [
        ("Depth Only (256)", lambda: depth_inference(frame, depth_model, 256)),
        ("Depth Only (384)", lambda: depth_inference(frame, depth_model, 384)),
        ("Depth Only (518)", lambda: depth_inference(frame, depth_model, 518)),
        ("YOLO Only", lambda: yolo_model(frame, verbose=False) if yolo_model else None),
    ]
    
    if yolo_model:
        configs.extend([
            ("YOLO + Depth(256)", lambda: pipeline(frame, depth_model, yolo_model, 256)),
            ("YOLO + Depth(384)", lambda: pipeline(frame, depth_model, yolo_model, 384)),
        ])
    
    for name, func in configs:
        # Warmup
        _ = func()
        
        # Benchmark
        times = []
        for _ in range(3):
            start = time.time()
            _ = func()
            times.append(time.time() - start)
        
        avg_time = np.mean(times[1:])
        fps = 1.0 / avg_time
        
        print(f"  {name:25s}: {avg_time*1000:6.1f}ms | {fps:5.1f} FPS")


def depth_inference(frame, model, size):
    """Simple depth inference"""
    with torch.no_grad():
        return model.infer_image(frame, input_size=size)


def pipeline(frame, depth_model, yolo_model, depth_size):
    """Full pipeline inference"""
    with torch.no_grad():
        depth = depth_model.infer_image(frame, input_size=depth_size)
    _ = yolo_model(frame, verbose=False)
    return depth


def main():
    print("="*70)
    print("REAL-TIME PERFORMANCE BENCHMARK")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")
    
    # Load models
    print("\n[INFO] Loading models...")
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }
    
    depth_model = DepthAnythingV2(**model_configs['vits'])
    depth_model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
    depth_model = depth_model.to(device).eval()
    print("[OK] Depth model loaded")
    
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            yolo_model = YOLO('yolov12n.pt')
            print("[OK] YOLO model loaded")
        except:
            print("[WARNING] YOLO not available")
    
    # Create test frame
    print("\n[INFO] Creating test frame (640x480)...")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run benchmarks
    benchmark_depth_model(frame, depth_model, device)
    
    if yolo_model:
        benchmark_yolo_model(frame, yolo_model)
    
    benchmark_pipeline(frame, depth_model, yolo_model, device)
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR OPTIMIZATION")
    print("="*70)
    print("""
1. For MAX SPEED (>5 FPS on CPU):
   - Use Depth Only (size 256)
   - Disable YOLO
   - Use frame skipping (--skip 2 or 3)
   
2. For BALANCED (3-5 FPS):
   - Depth + YOLO (size 384)
   - Frame skip 1
   
3. For QUALITY (1-2 FPS):
   - Depth Only (size 518)
   - Enable YOLO (if needed)
   - Frame skip 1
   
4. HARDWARE UPGRADE OPTIONS:
   - Use GPU (CUDA) → 5-20x speedup
   - Quantized models → 1.5-2x speedup
   - ONNX Runtime → 2-3x speedup
    """)
    
    print("="*70)


if __name__ == "__main__":
    main()
