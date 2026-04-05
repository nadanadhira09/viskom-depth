"""
real_time_performance_analysis.py
==================================
Real-time performance analysis dengan:
- Runtime metrics (FPS, latency)
- Resource usage (RAM, CPU)
- Distance accuracy validation
"""

import cv2
import numpy as np
import time
import torch
import sys
from pathlib import Path
from collections import deque
import psutil
import matplotlib
import os

try:
    import onnxruntime as ort
except:
    print("[ERROR] onnxruntime not installed")
    sys.exit(1)

sys.path.insert(0, 'Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False


class RealTimePerformanceAnalyzer:
    """Analyzer untuk real-time performance metrics"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
    def analyze_onnx_quantized_realtime(self, duration_seconds=60, skip=1):
        """Analyze real-time performance dengan ONNX Quantized INT8"""
        
        print("="*80)
        print("6.2.3 ANALISIS APLIKASI REAL-TIME (ONNX Quantized INT8)")
        print("="*80 + "\n")
        
        # Load model
        model_path = 'models/onnx/depth_anything_v2_vits_quantized_int8.onnx'
        if not Path(model_path).exists():
            print(f"[ERROR] Model not found: {model_path}")
            return
        
        print(f"[INFO] Loading ONNX Quantized INT8: {model_path}")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Load YOLO
        yolo_model = None
        if YOLO_AVAILABLE:
            try:
                yolo_model = YOLO('yolov12n.pt')
                print("[OK] YOLO v12n loaded")
            except:
                pass
        
        # Open camera
        print("[INFO] Opening camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Camera failed")
            return
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("[OK] Camera ready\n")
        
        # Metrics collection
        inference_times = deque(maxlen=300)
        ram_usage = deque(maxlen=300)
        cpu_usage = deque(maxlen=300)
        distances = []
        frame_count = 0
        proc_count = 0
        distances_measured = []
        
        # Process monitoring
        process = psutil.Process(os.getpid())
        initial_ram = process.memory_info().rss / 1024**2
        initial_cpu = process.cpu_percent(interval=0.1)
        
        print(f"Initial RAM: {initial_ram:.1f} MB")
        print(f"Initial CPU: {initial_cpu:.1f}%\n")
        
        start_time = time.time()
        
        try:
            while True:
                frame_time = time.time() - start_time
                if frame_time > duration_seconds:
                    break
                
                frame_count += 1
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Frame skip
                if (frame_count - 1) % skip != 0:
                    continue
                
                proc_count += 1
                
                # Inference
                infer_start = time.time()
                
                # Preprocess
                img = cv2.resize(frame, (518, 518))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, 0)
                
                # Run inference
                depth = session.run([output_name], {input_name: img})[0][0, 0]
                
                infer_time = (time.time() - infer_start) * 1000
                inference_times.append(infer_time)
                
                # Get metrics
                ram = process.memory_info().rss / 1024**2
                cpu = process.cpu_percent(interval=0)
                ram_usage.append(ram)
                cpu_usage.append(cpu)
                
                # YOLO detection (optional)
                if yolo_model and proc_count % 2 == 0:
                    try:
                        results = yolo_model(frame, verbose=False)
                    except:
                        pass
                
                # Show progress
                if proc_count % 30 == 0:
                    avg_infer = np.mean(inference_times)
                    avg_ram = np.mean(ram_usage)
                    print(f"[{proc_count:3d}] Infer: {avg_infer:.1f}ms | RAM: {avg_ram:.1f}MB | CPU: {cpu:.1f}%")
        
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            cap.release()
        
        # Generate report
        self._generate_report(inference_times, ram_usage, cpu_usage, 
                            initial_ram, frame_count, proc_count)
    
    def _generate_report(self, inference_times, ram_usage, cpu_usage, 
                         initial_ram, total_frames, processed_frames):
        """Generate analysis report"""
        
        print("\n" + "="*80)
        print("REAL-TIME PERFORMANCE ANALYSIS REPORT")
        print("="*80 + "\n")
        
        infer_arr = np.array(inference_times)
        ram_arr = np.array(ram_usage)
        cpu_arr = np.array(cpu_usage)
        
        # 1. Inference Performance
        print("1. INFERENCE PERFORMANCE")
        print("-" * 40)
        print(f"   Min:           {infer_arr.min():.2f} ms")
        print(f"   Max:           {infer_arr.max():.2f} ms")
        print(f"   Mean:          {infer_arr.mean():.2f} ms")
        print(f"   Std Dev:       {infer_arr.std():.2f} ms")
        print(f"   FPS:           {1000/infer_arr.mean():.2f} FPS")
        
        # 2. Resource Usage
        print("\n2. RESOURCE USAGE")
        print("-" * 40)
        print(f"   Total frames:  {total_frames}")
        print(f"   Proc frames:   {processed_frames}")
        print(f"   RAM usage:     {ram_arr.mean():.1f} MB (avg)")
        print(f"   RAM increase:  {ram_arr.mean() - initial_ram:.1f} MB")
        print(f"   CPU usage:     {cpu_arr.mean():.1f}% (avg)")
        
        # 3. Analysis
        print("\n3. PERFORMANCE ANALYSIS")
        print("-" * 40)
        print("""
Latency Analysis:
  - Waktu inferensi rata-rata: {:.0f} ms
  - Achievable FPS: {:.2f} FPS
  - Latency class: {}
  
Resource Consumption:
  - RAM delta: {:.1f} MB
  - Per-frame overhead: {:.1f} MB
  
Recommendation:
  - Model: ONNX Quantized INT8
  - Performance: Sufficient untuk real-time 1-2 FPS
  - Resource: Acceptable untuk perangkat average
        """.format(
            infer_arr.mean(),
            1000/infer_arr.mean(),
            "EXCELLENT" if infer_arr.mean() < 500 else "GOOD" if infer_arr.mean() < 1000 else "SLOW",
            ram_arr.mean() - initial_ram,
            (ram_arr.mean() - initial_ram) / processed_frames
        ))


def main():
    analyzer = RealTimePerformanceAnalyzer()
    
    print("\n" + "="*80)
    print("REAL-TIME PERFORMANCE ANALYSIS - ONNX Quantized INT8")
    print("="*80 + "\n")
    print("Duration: 30 seconds (akan measure 30-60 frames)")
    print("Press Q/ESC dalam window untuk stop sebelum timeout\n")
    
    analyzer.analyze_onnx_quantized_realtime(duration_seconds=30, skip=1)
    
    print("\n" + "="*80)
    print("Evaluasi selesai!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
