"""
CPU-ONLY OPTIMIZATION STRATEGY
================================
Strategi optimasi untuk sistem tanpa GPU:
- ONNX Runtime graph optimization (ORT_ENABLE_ALL)
- Threading configuration  
- Batch inference (theoretical projection)
- Hasil proyeksi untuk target Researcher A
"""

import os
import cv2
import numpy as np
import time
import onnxruntime as ort
from onnxruntime import SessionOptions, GraphOptimizationLevel
import torch
from datetime import datetime

print("=" * 80)
print("💻 CPU-ONLY OPTIMIZATION STRATEGY")
print("=" * 80)
print("Note: No GPU detected. Focusing on CPU-only optimizations.\n")

# ============================================================================
# CONFIGURATION (Using sizes that worked in previous tests)
# ============================================================================
IMG_SIZE = (252, 252)
TEST_IMAGES_DIR = "assets/test_images"
MODEL_ONNX_PATH = "models/onnx/depth_anything_v2_vits.onnx"
MODEL_ONNX_QUANT_PATH = "models/onnx/depth_anything_v2_vits_quantized_int8.onnx"

NUM_RUNS = 20
WARMUP_RUNS = 2

print("[INFO] Configuration:")
print(f"  • Image Size: {IMG_SIZE}")
print(f"  • Model FP32: {MODEL_ONNX_PATH}")
print(f"  • Model INT8: {MODEL_ONNX_QUANT_PATH}")
print(f"  • Benchmark Runs: {NUM_RUNS} (after {WARMUP_RUNS} warmup)")
print()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def load_test_image():
    """Load a single test image"""
    image_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f"❌ No images found in {TEST_IMAGES_DIR}")
        return None
    
    img_path = os.path.join(TEST_IMAGES_DIR, image_files[0])
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return img

# ============================================================================
# STEP 1: Baseline Benchmark
# ============================================================================
print("=" * 80)
print("[1/3] BASELINE BENCHMARK (Default ONNX Config)")
print("=" * 80)

test_img = load_test_image()
if test_img is None:
    print("❌ Failed to load test image")
    exit(1)

# Default session (no optimization)
session_default = ort.InferenceSession(MODEL_ONNX_PATH)
input_name = session_default.get_inputs()[0].name
output_name = session_default.get_outputs()[0].name

# Warmup
for _ in range(WARMUP_RUNS):
    session_default.run([output_name], {input_name: test_img})

# Benchmark FP32 baseline
times_fp32_base = []
for _ in range(NUM_RUNS):
    start = time.time()
    session_default.run([output_name], {input_name: test_img})
    end = time.time()
    times_fp32_base.append((end - start) * 1000)

avg_fp32_base = np.mean(times_fp32_base)
std_fp32_base = np.std(times_fp32_base)

print(f"\n✓ ONNX FP32 (Default Config):")
print(f"  • Average: {avg_fp32_base:.2f} ms")
print(f"  • Std Dev: {std_fp32_base:.2f} ms")
print(f"  • FPS: {1000/avg_fp32_base:.2f}")

# Benchmark INT8 baseline  
session_quant_base = ort.InferenceSession(MODEL_ONNX_QUANT_PATH)

for _ in range(WARMUP_RUNS):
    session_quant_base.run([output_name], {input_name: test_img})

times_int8_base = []
for _ in range(NUM_RUNS):
    start = time.time()
    session_quant_base.run([output_name], {input_name: test_img})
    end = time.time()
    times_int8_base.append((end - start) * 1000)

avg_int8_base = np.mean(times_int8_base)
std_int8_base = np.std(times_int8_base)

print(f"\n✓ ONNX INT8 (Default Config):")
print(f"  • Average: {avg_int8_base:.2f} ms")
print(f"  • Std Dev: {std_int8_base:.2f} ms")
print(f"  • FPS: {1000/avg_int8_base:.2f}")

# ============================================================================
# STEP 2: ONNX Graph Optimization (ORT_ENABLE_ALL)
# ============================================================================
print("\n" + "=" * 80)
print("[2/3] OPTIMIZED ONNX RUNTIME (ORT_ENABLE_ALL)")
print("=" * 80)

# Create optimized session
session_opts_opt = SessionOptions()
session_opts_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts_opt.inter_op_num_threads = 1
session_opts_opt.intra_op_num_threads = 4  # CPU cores

session_fp32_opt = ort.InferenceSession(MODEL_ONNX_PATH, session_opts_opt)

# Warmup
for _ in range(WARMUP_RUNS):
    session_fp32_opt.run([output_name], {input_name: test_img})

# Benchmark FP32 optimized
times_fp32_opt = []
for _ in range(NUM_RUNS):
    start = time.time()
    session_fp32_opt.run([output_name], {input_name: test_img})
    end = time.time()
    times_fp32_opt.append((end - start) * 1000)

avg_fp32_opt = np.mean(times_fp32_opt)
std_fp32_opt = np.std(times_fp32_opt)
improvement_fp32 = avg_fp32_base / avg_fp32_opt

print(f"\n✓ ONNX FP32 (ORT_ENABLE_ALL Optimized):")
print(f"  • Average: {avg_fp32_opt:.2f} ms")
print(f"  • Std Dev: {std_fp32_opt:.2f} ms")
print(f"  • FPS: {1000/avg_fp32_opt:.2f}")
print(f"  • Improvement: {improvement_fp32:.2f}x vs baseline")

# INT8 optimized
session_opts_int8 = SessionOptions()
session_opts_int8.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts_int8.inter_op_num_threads = 1
session_opts_int8.intra_op_num_threads = 4

session_int8_opt = ort.InferenceSession(MODEL_ONNX_QUANT_PATH, session_opts_int8)

for _ in range(WARMUP_RUNS):
    session_int8_opt.run([output_name], {input_name: test_img})

times_int8_opt = []
for _ in range(NUM_RUNS):
    start = time.time()
    session_int8_opt.run([output_name], {input_name: test_img})
    end = time.time()
    times_int8_opt.append((end - start) * 1000)

avg_int8_opt = np.mean(times_int8_opt)
std_int8_opt = np.std(times_int8_opt)
improvement_int8 = avg_int8_base / avg_int8_opt

print(f"\n✓ ONNX INT8 (ORT_ENABLE_ALL Optimized):")
print(f"  • Average: {avg_int8_opt:.2f} ms")
print(f"  • Std Dev: {std_int8_opt:.2f} ms")
print(f"  • FPS: {1000/avg_int8_opt:.2f}")
print(f"  • Improvement: {improvement_int8:.2f}x vs baseline")

# ============================================================================
# STEP 3: CPU-Only Optimization Projections
# ============================================================================
print("\n" + "=" * 80)
print("[3/3] CPU-ONLY OPTIMIZATION PROJECTIONS & STRATEGY")
print("=" * 80)

# Reference from competitive evaluation
baseline_pytorch = 1270.28  # ms (from competitive_evaluation.py)
target_pytorch = 2579.29  # Researcher A
target_fp32_speedup = 9.18
target_int8_speedup = 10.70

status_text = f"""
CURRENT STATUS (from competitive_evaluation.py):
  • PyTorch baseline: {baseline_pytorch:.2f} ms
  • ONNX FP32: {avg_fp32_base:.2f} ms (speedup baseline: {baseline_pytorch/avg_fp32_base:.2f}x)
  • ONNX INT8: {avg_int8_base:.2f} ms (speedup baseline: {baseline_pytorch/avg_int8_base:.2f}x)

AFTER GRAPH OPTIMIZATION (ORT_ENABLE_ALL):
  • ONNX FP32: {avg_fp32_opt:.2f} ms (speedup: {baseline_pytorch/avg_fp32_opt:.2f}x)
  • ONNX INT8: {avg_int8_opt:.2f} ms (speedup: {baseline_pytorch/avg_int8_opt:.2f}x)
  • Improvement: {improvement_fp32:.2f}x / {improvement_int8:.2f}x

THEORETICAL BATCH PROCESSING IMPACT (10-frame batch):
  • Estimated per-frame latency reduction: 30-40%
  • Projected FP32 speedup: {baseline_pytorch/(avg_fp32_opt*0.7):.2f}x
  • Projected INT8 speedup: {baseline_pytorch/(avg_int8_opt*0.7):.2f}x

TARGET BENCHMARKS:
  • FP32 Target: {target_fp32_speedup}x (gap: {max(0, target_fp32_speedup - baseline_pytorch/avg_fp32_opt):.2f}x needed)
  • INT8 Target: {target_int8_speedup}x (gap: {max(0, target_int8_speedup - baseline_pytorch/avg_int8_opt):.2f}x needed)

"""
print(status_text)

# ============================================================================
# STRATEGY RECOMMENDATION
# ============================================================================
print("=" * 80)
print("📊 OPTIMIZATION STRATEGY FOR CPU-ONLY")
print("=" * 80)

strategies = [
    {
        "name": "Strategy 1: ONNX Graph Optimization ONLY",
        "improvements": [
            ("Graph Optimization (ORT_ENABLE_ALL)", improvement_fp32),
        ],
        "fp32_result": baseline_pytorch / avg_fp32_opt,
        "int8_result": baseline_pytorch / avg_int8_opt,
    },
    {
        "name": "Strategy 2: Graph Opt + Batch Processing (10 frames)",
        "improvements": [
            ("Graph Optimization", improvement_fp32),
            ("Batch Processing", 1.35),  # Conservative estimate
        ],
        "fp32_result": (baseline_pytorch / avg_fp32_opt) * 1.35,
        "int8_result": (baseline_pytorch / avg_int8_opt) * 1.35,
    },
    {
        "name": "Strategy 3: Graph Opt + Batch + Thread Tuning",
        "improvements": [
            ("Graph Optimization", improvement_fp32),
            ("Batch Processing", 1.35),
            ("Advanced Threading", 1.10),
        ],
        "fp32_result": (baseline_pytorch / avg_fp32_opt) * 1.35 * 1.10,
        "int8_result": (baseline_pytorch / avg_int8_opt) * 1.35 * 1.10,
    },
]

for idx, strategy in enumerate(strategies, 1):
    print(f"\n{idx}. {strategy['name']}")
    components_str = ' × '.join([f"{k} ({v:.2f}x)" for k, v in strategy['improvements']])
    print(f"   Components: {components_str}")
    print(f"   Expected Results:")
    print(f"     • FP32 Speedup: {strategy['fp32_result']:.2f}x (target: {target_fp32_speedup}x)")
    print(f"     • INT8 Speedup: {strategy['int8_result']:.2f}x (target: {target_int8_speedup}x)")
    
    if strategy['int8_result'] >= target_int8_speedup:
        print(f"     ✅ BEATS TARGET!")
    else:
        gap = target_int8_speedup - strategy['int8_result']
        print(f"     ⚠️  Gap: {gap:.2f}x (need {gap/strategy['int8_result']*100:.0f}% more)")

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n" + "=" * 80)
print("🎯 RECOMMENDATION")
print("=" * 80)

rec_text = f"""
Given CPU-only setup (no GPU hardware):

ACHIEVABLE on CPU:
  • Graph Optimization (ORT_ENABLE_ALL): {improvement_fp32:.2f}x
  • Batch Processing: ~1.35x
  • Thread Tuning: ~1.10x

Total Expected: ~{improvement_fp32 * 1.35 * 1.10:.2f}x

Estimated Results after all CPU optimizations:
  • INT8 Speedup: ~{(baseline_pytorch / avg_int8_opt) * 1.35 * 1.10:.2f}x
  • vs Target: {target_int8_speedup}x

Gap Analysis:
  • Current INT8: {baseline_pytorch/avg_int8_base:.2f}x
  • Can achieve: ~{(baseline_pytorch / avg_int8_opt) * 1.35 * 1.10:.2f}x (CPU-only)
  • Still need: ~{max(0, target_int8_speedup / ((baseline_pytorch / avg_int8_opt) * 1.35 * 1.10)):.2f}x multiplier

To Beat Researcher A Target, you need:
  1. GPU Hardware (provides 3-5x boost)
  2. Or consider model optimization (distillation, pruning)
  3. Or accept CPU-only limitation and optimize for mobile deployment

RECOMMENDED PATH:
  → Deploy on Android with CPU optimizations
  → Achieve ~2x speedup with current optimizations
  → Provide excellent user experience on mid-range devices
  → Test on Infinix Hot 20S target device
"""
print(rec_text)

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
