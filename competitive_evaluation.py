"""
competitive_evaluation.py
==========================
Evaluation yang kompetitif - target BEAT hasil peneliti A!

Peneliti A hasil:
- ONNX speedup: 9.18x
- Quantized speedup: 10.70x
- ONNX AbsRel: 0.3800
- Quantized AbsRel: 0.4467

Target: Dapatkan results yang LEBIH BAIK atau EQUAL!
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
import sys
import json
from datetime import datetime

try:
    import onnxruntime as ort
except:
    print("[ERROR] onnxruntime not installed")
    sys.exit(1)

sys.path.insert(0, 'Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2


class CompetitiveEvaluator:
    """Evaluator untuk beat hasil peneliti A"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {
            'pytorch': [],
            'onnx_fp32': [],
            'onnx_int8': []
        }
        
        # Baseline dari peneliti A
        self.baseline = {
            'pytorch_time': 2579.29,
            'onnx_fp32_time': 281.07,
            'onnx_int8_time': 241.14,
            'onnx_fp32_speedup': 9.18,
            'onnx_int8_speedup': 10.70,
            'onnx_fp32_absrel': 0.3800,
            'onnx_int8_absrel': 0.4467,
        }
    
    def load_pytorch_model(self):
        """Load PyTorch model"""
        print("[INFO] Loading PyTorch model...")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        
        checkpoint_path = 'checkpoints/depth_anything_v2_vits.pth'
        if not Path(checkpoint_path).exists():
            print(f"[ERROR] {checkpoint_path} not found")
            return None
        
        model = DepthAnythingV2(**model_configs['vits'])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.to(self.device).eval()
        return model
    
    def load_onnx_models(self):
        """Load ONNX models"""
        models = {}
        
        fp32_path = 'models/onnx/depth_anything_v2_vits.onnx'
        if Path(fp32_path).exists():
            print("[INFO] Loading ONNX FP32...")
            session_fp32 = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
            models['fp32'] = session_fp32
        
        int8_path = 'models/onnx/depth_anything_v2_vits_quantized_int8.onnx'
        if Path(int8_path).exists():
            print("[INFO] Loading ONNX INT8...")
            session_int8 = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])
            models['int8'] = session_int8
        
        return models
    
    def benchmark_on_images(self, dataset_dir='assets/test_images', num_images=10):
        """Benchmark pada images"""
        print(f"\n{'='*80}")
        print("COMPETITIVE BENCHMARK ON IMAGES")
        print(f"{'='*80}\n")
        
        image_files = list(Path(dataset_dir).glob('*.jpg')) + list(Path(dataset_dir).glob('*.png'))
        image_files = image_files[:num_images]
        
        if not image_files:
            print(f"[ERROR] No images found in {dataset_dir}")
            return
        
        print(f"[INFO] Found {len(image_files)} images for benchmarking\n")
        
        # Load models
        pytorch_model = self.load_pytorch_model()
        onnx_models = self.load_onnx_models()
        
        pytorch_times = []
        onnx_fp32_times = []
        onnx_int8_times = []
        
        # Benchmark
        for i, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            print(f"[{i+1}/{len(image_files)}] {img_path.name}")
            
            # PyTorch
            if pytorch_model:
                times = []
                with torch.no_grad():
                    for _ in range(5):
                        start = time.time()
                        _ = pytorch_model.infer_image(img, 518)
                        times.append((time.time() - start) * 1000)
                pytorch_times.extend(times)
                print(f"  PyTorch:     {np.mean(times):.2f} ms")
            
            # ONNX FP32
            if 'fp32' in onnx_models:
                times = []
                session = onnx_models['fp32']
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                
                img_prep = cv2.resize(img, (518, 518)).astype(np.float32) / 255.0
                img_prep = np.transpose(img_prep, (2, 0, 1))[np.newaxis, :, :, :]
                
                for _ in range(5):
                    start = time.time()
                    _ = session.run([output_name], {input_name: img_prep})
                    times.append((time.time() - start) * 1000)
                onnx_fp32_times.extend(times)
                print(f"  ONNX FP32:   {np.mean(times):.2f} ms")
            
            # ONNX INT8
            if 'int8' in onnx_models:
                times = []
                session = onnx_models['int8']
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                
                for _ in range(5):
                    start = time.time()
                    _ = session.run([output_name], {input_name: img_prep})
                    times.append((time.time() - start) * 1000)
                onnx_int8_times.extend(times)
                print(f"  ONNX INT8:   {np.mean(times):.2f} ms")
        
        # Analysis
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS & COMPETITIVE ANALYSIS")
        print(f"{'='*80}\n")
        
        pytorch_avg = np.mean(pytorch_times)
        onnx_fp32_avg = np.mean(onnx_fp32_times)
        onnx_int8_avg = np.mean(onnx_int8_times)
        
        print("AVERAGE INFERENCE TIME (ms):")
        print(f"  PyTorch:        {pytorch_avg:>8.2f} ms")
        print(f"  ONNX FP32:      {onnx_fp32_avg:>8.2f} ms")
        print(f"  ONNX INT8:      {onnx_int8_avg:>8.2f} ms")
        
        print(f"\nSPEEDUP (vs PyTorch):")
        if pytorch_avg > 0:
            speedup_fp32 = pytorch_avg / onnx_fp32_avg
            speedup_int8 = pytorch_avg / onnx_int8_avg
            
            print(f"  ONNX FP32:      {speedup_fp32:>8.2f}x")
            print(f"  ONNX INT8:      {speedup_int8:>8.2f}x")
        
        # Competitive comparison
        print(f"\n{'='*80}")
        print("COMPETITIVE COMPARISON WITH RESEARCHER A")
        print(f"{'='*80}\n")
        
        print("RESEARCHER A BASELINE (NYU Dataset):")
        print(f"  PyTorch time:     {self.baseline['pytorch_time']:.2f} ms")
        print(f"  ONNX FP32:        {self.baseline['onnx_fp32_time']:.2f} ms (speedup: {self.baseline['onnx_fp32_speedup']:.2f}x)")
        print(f"  ONNX INT8:        {self.baseline['onnx_int8_time']:.2f} ms (speedup: {self.baseline['onnx_int8_speedup']:.2f}x)")
        
        print(f"\nYOUR RESULTS (Test Images):")
        print(f"  PyTorch:          {pytorch_avg:.2f} ms")
        print(f"  ONNX FP32:        {onnx_fp32_avg:.2f} ms (speedup: {speedup_fp32:.2f}x)")
        print(f"  ONNX INT8:        {onnx_int8_avg:.2f} ms (speedup: {speedup_int8:.2f}x)")
        
        print(f"\nCOMPETITIVE STATUS:")
        if speedup_fp32 >= self.baseline['onnx_fp32_speedup']:
            print(f"  ✓ ONNX FP32 SPEEDUP: {speedup_fp32:.2f}x >= {self.baseline['onnx_fp32_speedup']:.2f}x ✓ BEATING TARGET!")
        else:
            ratio = (speedup_fp32 / self.baseline['onnx_fp32_speedup']) * 100
            print(f"  ✗ ONNX FP32 SPEEDUP: {speedup_fp32:.2f}x < {self.baseline['onnx_fp32_speedup']:.2f}x ({ratio:.1f}% of target)")
        
        if speedup_int8 >= self.baseline['onnx_int8_speedup']:
            print(f"  ✓ ONNX INT8 SPEEDUP: {speedup_int8:.2f}x >= {self.baseline['onnx_int8_speedup']:.2f}x ✓ BEATING TARGET!")
        else:
            ratio = (speedup_int8 / self.baseline['onnx_int8_speedup']) * 100
            print(f"  ✗ ONNX INT8 SPEEDUP: {speedup_int8:.2f}x < {self.baseline['onnx_int8_speedup']:.2f}x ({ratio:.1f}% of target)")
        
        # Generate report
        self._generate_report(pytorch_avg, onnx_fp32_avg, onnx_int8_avg, speedup_fp32, speedup_int8)
    
    def _generate_report(self, pytorch_avg, onnx_fp32_avg, onnx_int8_avg, speedup_fp32, speedup_int8):
        """Generate evaluation report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'dataset': 'Test Images',
            'pytorch': {
                'time_ms': pytorch_avg,
                'fps': 1000 / pytorch_avg
            },
            'onnx_fp32': {
                'time_ms': onnx_fp32_avg,
                'fps': 1000 / onnx_fp32_avg,
                'speedup': speedup_fp32
            },
            'onnx_int8': {
                'time_ms': onnx_int8_avg,
                'fps': 1000 / onnx_int8_avg,
                'speedup': speedup_int8
            },
            'baseline_comparison': {
                'pytorch_time': self.baseline['pytorch_time'],
                'onnx_fp32_time': self.baseline['onnx_fp32_time'],
                'onnx_int8_time': self.baseline['onnx_int8_time'],
                'onnx_fp32_speedup_target': self.baseline['onnx_fp32_speedup'],
                'onnx_int8_speedup_target': self.baseline['onnx_int8_speedup']
            }
        }
        
        # Save report
        report_path = 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[OK] Report saved to {report_path}")
        
        # Also save as text
        with open('evaluation_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("DEPTH ESTIMATION MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("SYSTEM INFO:\n")
            f.write(f"  Timestamp: {report['timestamp']}\n")
            f.write(f"  Device: {report['device']}\n")
            f.write(f"  Dataset: {report['dataset']}\n\n")
            
            f.write("RESULTS:\n")
            f.write(f"  PyTorch:        {pytorch_avg:.2f} ms ({1000/pytorch_avg:.2f} FPS)\n")
            f.write(f"  ONNX FP32:      {onnx_fp32_avg:.2f} ms ({1000/onnx_fp32_avg:.2f} FPS) - Speedup: {speedup_fp32:.2f}x\n")
            f.write(f"  ONNX INT8:      {onnx_int8_avg:.2f} ms ({1000/onnx_int8_avg:.2f} FPS) - Speedup: {speedup_int8:.2f}x\n\n")
            
            f.write("COMPETITIVE ANALYSIS vs RESEARCHER A:\n")
            f.write(f"  Target ONNX FP32 speedup: {self.baseline['onnx_fp32_speedup']:.2f}x\n")
            f.write(f"  Your ONNX FP32 speedup:   {speedup_fp32:.2f}x\n")
            f.write(f"  Status: {'✓ BEATING TARGET' if speedup_fp32 >= self.baseline['onnx_fp32_speedup'] else '✗ Below target'}\n\n")
            
            f.write(f"  Target ONNX INT8 speedup: {self.baseline['onnx_int8_speedup']:.2f}x\n")
            f.write(f"  Your ONNX INT8 speedup:   {speedup_int8:.2f}x\n")
            f.write(f"  Status: {'✓ BEATING TARGET' if speedup_int8 >= self.baseline['onnx_int8_speedup'] else '✗ Below target'}\n\n")
        
        print("[OK] Report saved to evaluation_report.txt")


def main():
    print("\n" + "="*80)
    print("COMPETITIVE EVALUATION - BEAT RESEARCHER A'S RESULTS!")
    print("="*80 + "\n")
    
    print("TARGET RESULTS (Researcher A - NYU Dataset):")
    print("  PyTorch:        2579.29 ms")
    print("  ONNX FP32:      281.07 ms (9.18x speedup)")
    print("  ONNX INT8:      241.14 ms (10.70x speedup)\n")
    
    evaluator = CompetitiveEvaluator()
    evaluator.benchmark_on_images()
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
