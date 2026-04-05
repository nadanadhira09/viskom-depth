"""
BATCH INFERENCE OPTIMIZATION
=============================
Implementasi batch processing untuk meningkatkan throughput.
Membandingkan:
- Single-frame baseline
- Batch size 5
- Batch size 10
"""

import os
import torch
import cv2
import numpy as np
import time
from pathlib import Path
import onnxruntime as ort
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_SIZE = (252, 252)
TEST_IMAGES_DIR = "assets/test_images"
MODEL_ONNX_PATH = "models/onnx/depth_anything_v2_vits.onnx"
MODEL_ONNX_QUANT_PATH = "models/onnx/depth_anything_v2_vits_quantized_int8.onnx"

# Batch sizes to test
BATCH_SIZES = [1, 5, 10]
NUM_RUNS = 30
WARMUP_RUNS = 3

print("=" * 70)
print("⚡ BATCH INFERENCE OPTIMIZATION")
print("=" * 70)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def load_image(image_path, size):
    """Load and preprocess image"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return img

def load_all_images(batch_size, num_batches=5):
    """Load test images"""
    images = []
    image_files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])[:4]
    
    if not image_files:
        print(f"❌ No images found in {TEST_IMAGES_DIR}")
        return []
    
    # Repeat images to create enough batches
    for i in range(batch_size * num_batches):
        img_path = os.path.join(TEST_IMAGES_DIR, image_files[i % len(image_files)])
        img = load_image(img_path, IMG_SIZE)
        if img is not None:
            images.append(img)
    
    return images

def create_batches(images, batch_size):
    """Create batches from images"""
    batches = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        if len(batch) == batch_size:
            batches.append(np.concatenate(batch, axis=0))
    return batches

# ============================================================================
# BENCHMARK SINGLE-FRAME (BASELINE)
# ============================================================================
print("\n[1/3] Benchmarking single-frame inference (baseline)...")
print(f"      {'-' * 60}")

session = ort.InferenceSession(MODEL_ONNX_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

images = load_all_images(batch_size=1, num_batches=5)

# Warmup
for _ in range(WARMUP_RUNS):
    session.run([output_name], {input_name: images[0]})

# Benchmark
times_single = []
for _ in range(NUM_RUNS):
    start = time.time()
    session.run([output_name], {input_name: images[0]})
    end = time.time()
    times_single.append((end - start) * 1000)

avg_single = np.mean(times_single)
std_single = np.std(times_single)

print(f"  Single-frame (batch=1):")
print(f"  • Average: {avg_single:.2f} ms")
print(f"  • Std Dev: {std_single:.2f} ms")
print(f"  • Throughput: {1000/avg_single:.2f} frames/sec")

# ============================================================================
# BENCHMARK BATCH INFERENCE
# ============================================================================
print("\n[2/3] Benchmarking batch inference...")
print(f"      {'-' * 60}")

batch_results = {}

for batch_size in BATCH_SIZES:
    if batch_size == 1:
        continue  # Already benchmarked
    
    print(f"\n  Batch Size: {batch_size}")
    
    images = load_all_images(batch_size=batch_size, num_batches=5)
    batches = create_batches(images, batch_size)
    
    # Warmup
    for _ in range(WARMUP_RUNS):
        session.run([output_name], {input_name: batches[0]})
    
    # Benchmark (measure total batch time)
    times_batch = []
    for _ in range(NUM_RUNS):
        start = time.time()
        session.run([output_name], {input_name: batches[0]})
        end = time.time()
        times_batch.append((end - start) * 1000)
    
    avg_batch = np.mean(times_batch)
    std_batch = np.std(times_batch)
    
    # Per-frame time (batch time / batch size)
    per_frame_time = avg_batch / batch_size
    speedup = avg_single / per_frame_time
    throughput = 1000 / per_frame_time
    
    batch_results[batch_size] = {
        'total_time': avg_batch,
        'per_frame_time': per_frame_time,
        'speedup': speedup,
        'throughput': throughput,
        'std': std_batch
    }
    
    print(f"  • Total batch time: {avg_batch:.2f} ms")
    print(f"  • Per-frame time: {per_frame_time:.2f} ms")
    print(f"  • Speedup vs single: {speedup:.2f}x")
    print(f"  • Throughput: {throughput:.2f} frames/sec")

# ============================================================================
# BATCH OPTIMIZATION WITH INT8 QUANTIZATION
# ============================================================================
print("\n[3/3] Testing Batch + Quantization...")
print(f"      {'-' * 60}")

session_quant = ort.InferenceSession(MODEL_ONNX_QUANT_PATH)

batch_size_optimal = 10
images = load_all_images(batch_size=batch_size_optimal, num_batches=5)
batches = create_batches(images, batch_size_optimal)

# Warmup
for _ in range(WARMUP_RUNS):
    session_quant.run([output_name], {input_name: batches[0]})

# Benchmark INT8
times_int8 = []
for _ in range(NUM_RUNS):
    start = time.time()
    session_quant.run([output_name], {input_name: batches[0]})
    end = time.time()
    times_int8.append((end - start) * 1000)

avg_int8 = np.mean(times_int8)
per_frame_int8 = avg_int8 / batch_size_optimal
speedup_int8 = avg_single / per_frame_int8

print(f"\n  Batch {batch_size_optimal} + INT8 Quantization:")
print(f"  • Total batch time: {avg_int8:.2f} ms")
print(f"  • Per-frame time: {per_frame_int8:.2f} ms")
print(f"  • Speedup vs single: {speedup_int8:.2f}x")
print(f"  • Throughput: {1000/per_frame_int8:.2f} frames/sec")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("📊 BATCH OPTIMIZATION RESULTS")
print("=" * 70)

print(f"\nBaseline (Single-frame):")
print(f"  • Latency: {avg_single:.2f} ms/frame")
print(f"  • FPS: {1000/avg_single:.2f}")

print(f"\nBatch Sizes Performance:")
print(f"  {'-' * 60}")
for batch_size, results in sorted(batch_results.items()):
    print(f"  Batch {batch_size}: {results['speedup']:.2f}x speedup" +
          f" ({results['per_frame_time']:.2f} ms/frame, " +
          f"{results['throughput']:.2f} fps)")

print(f"\nOptimal Configuration (Batch 10 + INT8):")
print(f"  • Speedup: {speedup_int8:.2f}x")
print(f"  • Latency: {per_frame_int8:.2f} ms/frame")
print(f"  • FPS: {1000/per_frame_int8:.2f}")

# ============================================================================
# COMPETITIVE IMPACT
# ============================================================================
current_baseline = 1.42  # Current ONNX FP32 speedup
batch_multiplier = speedup_int8 / speedup_int8  # Relative improvement
new_speedup = current_baseline * batch_multiplier

print(f"\nCompetitive Impact:")
print(f"  • Current ONNX speedup: {current_baseline:.2f}x")
print(f"  • Batch optimization multiplier: {batch_multiplier:.2f}x")
print(f"  • New ONNX speedup: {new_speedup:.2f}x (vs target 10.70x)")
print(f"  • Gap to target: {(10.70/new_speedup - 1) * 100:.1f}%")

print("\n" + "=" * 70)
print("🎯 NEXT STEP: Apply ONNX Graph Optimization")
print("   Command: python onnx_graph_optimization.py")
print("=" * 70)
