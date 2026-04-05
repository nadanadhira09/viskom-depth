import os
import time
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

"""
Evaluasi Benchmarking Edge Device LOKAL (ONNX vs PTH vs NCNN)
Skrip ini mereplikasi evaluasi Google Colab secara lokal di mesin Anda untuk 
mendapatkan laporan waktu inferensi & akurasi.
"""

# ==============================================================
# CONFIGURATION
# ==============================================================
OUTPUT_DIR = "results/edge_evaluation"
TEST_IMG_DIR = "assets/test_images"  # Harap taro gambar pengujian KITTI/NYU disini

MODEL_PTH_PATH = "checkpoints/depth_anything_v2_vits.pth"
MODEL_ONNX_FP32 = "models/onnx/depth_anything_v2_vits.onnx"
# Model Quantized & NCNN. Anda bisa memasukkan file yg relevan nanti ketika ada
MODEL_ONNX_INT8 = "" 
MODEL_NCNN_PARAM = ""

IMG_SIZE = 518 # Harus sesuai dengan model export Anda (252 atau 518)
VISUALIZE_SAMPLES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)

# Import model dpt untuk PTH base (Asumsi folder repo Depth-Anything-V2 terhubung)
import sys
REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_DIR))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    has_pth = True
except ImportError:
    print("[WARNING] Tidak dapat memuat Python PyTorch Depth Anything V2. PTH base akan dilewati.")
    has_pth = False

# ==============================================================
# METRICS (AbsRel, RMSE)
# ==============================================================
def abs_rel(pred, gt):
    return np.mean(np.abs(pred - gt) / (gt + 1e-6))

def rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

# ==============================================================
# MAIN EVALUATION
# ==============================================================
def main():
    print("="*60)
    print(" EDGE INFERENCE EVALUATION (LOKAL) ")
    print("="*60)
    
    # 1. PERSIAPAN DATASET
    if not os.path.exists(TEST_IMG_DIR) or len(os.listdir(TEST_IMG_DIR)) == 0:
        print(f"[ERROR] Folder pengujian gambar kosong atau tidak ada di: {TEST_IMG_DIR}")
        print("Silahkan masukkan gambar (min 5 gambar) lalu jalankan lagi.")
        os.makedirs(TEST_IMG_DIR, exist_ok=True)
        return
        
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[INFO] Ditemukan {len(image_files)} gambar uji.")

    # 2. INISIALISASI SESI MODEL
    results_data = []
    
    # A. PyTorch (Baseline)
    model_pth = None
    if has_pth and os.path.exists(MODEL_PTH_PATH):
        print("[INFO] Memuat Model PyTorch (Baseline FP32)...")
        model_pth = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        model_pth.load_state_dict(torch.load(MODEL_PTH_PATH, map_location='cpu'))
        model_pth.eval()
    
    # B. ONNX FP32
    session_onnx = None
    if os.path.exists(MODEL_ONNX_FP32):
        print(f"[INFO] Memuat Model ONNX FP32... ({MODEL_ONNX_FP32})")
        # Menggunakan CPUExecutionProvider agar setara perbandingannya dengan HP Android CPU
        session_onnx = ort.InferenceSession(MODEL_ONNX_FP32, providers=['CPUExecutionProvider'])
        onnx_input_name = session_onnx.get_inputs()[0].name
        
    # C. Cek NCNN Python PNNX (Opsional jika user telah install paket ncnn)
    has_ncnn = False
    try:
        if os.path.exists(MODEL_NCNN_PARAM):
            import ncnn
            print("[INFO] Memuat Sesi NCNN VULKAN (Simulasi GPU HP)...")
            net_ncnn = ncnn.Net()
            net_ncnn.opt.use_vulkan_compute = True 
            net_ncnn.load_param(MODEL_NCNN_PARAM)
            net_ncnn.load_model(MODEL_NCNN_PARAM.replace(".param", ".bin"))
            has_ncnn = True
    except ImportError:
        print("Module 'ncnn' tidak diinstall. Skip evaluasi NCNN.")
        
    # 3. LOOP EVALUASI GAMBAR
    for idx, img_file in enumerate(tqdm(image_files)):
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        
        # Baca & Preprocess
        raw_img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        baseline_depth = None
        times = {"pth": 0.0, "onnx": 0.0, "onnx_int8": 0.0, "ncnn": 0.0}
        
        # Eksekusi PyTorch
        if model_pth:
            start = time.time()
            with torch.no_grad():
                baseline_depth = model_pth.infer_image(raw_img, IMG_SIZE)
            times["pth"] = (time.time() - start) * 1000 # to Ms
            
            # Normalisasi metrik referensi
            baseline_depth = baseline_depth / (baseline_depth.max() + 1e-6)
            
        # Eksekusi ONNX FP32
        if session_onnx:
            # Preprocess untuk ONNX: (1, 3, H, W) normalisasi /255.0
            resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            input_tensor = resized.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1)) # HWC to CHW
            input_tensor = np.expand_dims(input_tensor, axis=0) # [1, 3, H, W]
            
            start = time.time()
            outputs = session_onnx.run(None, {onnx_input_name: input_tensor})
            depth_onnx = outputs[0].squeeze()
            times["onnx"] = (time.time() - start) * 1000
            
            depth_onnx = depth_onnx / (depth_onnx.max() + 1e-6)
            
            if baseline_depth is not None:
                # Resize baseline untuk menyamakan resolusi evaluasi metrik
                baseline_rez = cv2.resize(baseline_depth, (depth_onnx.shape[1], depth_onnx.shape[0]))
                a_rel = abs_rel(depth_onnx, baseline_rez)
                rm = rmse(depth_onnx, baseline_rez)
                
                results_data.append({
                    "Image": img_file,
                    "ONNX_AbsRel": a_rel,
                    "ONNX_RMSE": rm,
                    "Time_PTH_ms": times["pth"],
                    "Time_ONNX_ms": times["onnx"]
                })
                
    # 4. KELUARKAN LAPORAN
    if len(results_data) > 0:
        df = pd.DataFrame(results_data)
        out_csv = f"{OUTPUT_DIR}/benchmark_results.csv"
        df.to_csv(out_csv, index=False)
        
        print("\n\n" + "="*50)
        print(" HASIL EVALUASI INFERENSI (RATA-RATA) ")
        print("="*50)
        print(f"Total Sampel : {len(df)} gambar")
        print(f"PyTorch Time : {df['Time_PTH_ms'].mean():.2f} ms")
        print(f"ONNX Time    : {df['Time_ONNX_ms'].mean():.2f} ms")
        print(f"Speedup ONNX : {(df['Time_PTH_ms'].mean() / df['Time_ONNX_ms'].mean()):.2f}x lebih cepat")
        print("-" * 50)
        print("Akurasi vs Baseline PyTorch (Target Mendekati 0.0):")
        print(f"AbsRel ONNX  : {df['ONNX_AbsRel'].mean():.4f}")
        print(f"RMSE ONNX    : {df['ONNX_RMSE'].mean():.4f}")
        print("="*50)
        print(f"Laporan lengkap tersimpan di: {out_csv}")
    else:
        print("[INFO] Evaluasi selesai, namun tidak melahirkan output tabel statistik.")
        
if __name__ == "__main__":
    main()