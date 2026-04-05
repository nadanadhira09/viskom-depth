"""
calibrate_depth.py
==================
Script kalibrasi untuk mendapatkan konstanta A dari metode tiga titik.

Penggunaan:
    python conversion/calibrate_depth.py --mode interactive
    python conversion/calibrate_depth.py --mode images --images img1.jpg img2.jpg img3.jpg
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import json
from datetime import datetime

# Tambahkan path ke Depth Anything V2
REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_DIR))

from depth_anything_v2.dpt import DepthAnythingV2


def extract_roi_depth(depth_map: np.ndarray, roi_size: int = 50) -> float:
    """
    Ekstrak nilai raw depth rata-rata dari Region of Interest (ROI) di tengah.
    
    Args:
        depth_map: Output depth map dari model (H x W)
        roi_size: Ukuran kotak ROI dalam piksel
    
    Returns:
        Nilai rata-rata raw depth di area tengah
    """
    h, w = depth_map.shape
    center_y, center_x = h // 2, w // 2
    
    # Tentukan batas ROI
    y1 = max(0, center_y - roi_size // 2)
    y2 = min(h, center_y + roi_size // 2)
    x1 = max(0, center_x - roi_size // 2)
    x2 = min(w, center_x + roi_size // 2)
    
    # Ekstrak ROI dan hitung rata-rata
    roi = depth_map[y1:y2, x1:x2]
    raw_depth = np.mean(roi)
    
    return raw_depth


def visualize_depth_with_roi(frame, depth, roi_size, raw_depth, distance, status_text):
    """Visualisasi frame dengan depth map dan ROI."""
    h, w = frame.shape[:2]
    center_y, center_x = h // 2, w // 2
    roi_half = roi_size // 2
    
    # Normalisasi depth untuk visualisasi
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    
    # Resize depth ke ukuran frame
    depth_colored = cv2.resize(depth_colored, (w, h))
    
    # Gambar ROI di frame asli
    cv2.rectangle(frame, 
                 (center_x - roi_half, center_y - roi_half),
                 (center_x + roi_half, center_y + roi_half),
                 (0, 255, 0), 2)
    
    # Gambar ROI di depth map
    cv2.rectangle(depth_colored, 
                 (center_x - roi_half, center_y - roi_half),
                 (center_x + roi_half, center_y + roi_half),
                 (0, 255, 0), 2)
    
    # Gabungkan side-by-side
    combined = np.hstack([frame, depth_colored])
    
    # Tambahkan info text
    info_y = 30
    cv2.putText(combined, status_text, (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(combined, f"Raw Depth: {raw_depth:.2f}", (10, info_y + 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if distance is not None:
        cv2.putText(combined, f"Distance: {distance:.1f}m", (10, info_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(combined, "SPASI=Capture | ESC=Quit", (10, combined.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return combined


def calibrate_interactive(model, device, distances=[1.0, 2.0, 3.0], 
                         input_size=518, roi_size=50):
    """
    Mode kalibrasi interaktif menggunakan webcam.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka webcam")
        return None, None
    
    # Set resolusi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    calibration_data = []
    current_distance_idx = 0
    
    print("\n" + "="*70)
    print("KALIBRASI DEPTH - MODE INTERAKTIF")
    print("="*70)
    print(f"Letakkan objek datar pada jarak {distances[current_distance_idx]:.1f}m")
    print("Tekan SPASI untuk capture, ESC untuk keluar")
    print("="*70 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prediksi depth
        depth = model.infer_image(frame, input_size)
        
        # Ekstrak raw depth di ROI
        raw_depth = extract_roi_depth(depth, roi_size)
        
        # Status
        status_text = f"Target: {distances[current_distance_idx]:.1f}m | Progress: {len(calibration_data)}/{len(distances)}"
        
        # Visualisasi
        vis = visualize_depth_with_roi(frame, depth, roi_size, raw_depth, 
                                       distances[current_distance_idx], status_text)
        
        cv2.imshow("Kalibrasi Depth", vis)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n[CANCELLED] Kalibrasi dibatalkan")
            break
        elif key == 32:  # SPASI
            # Capture data kalibrasi
            calibration_data.append({
                'distance': distances[current_distance_idx],
                'raw_depth': raw_depth
            })
            
            print(f"✓ Captured: {distances[current_distance_idx]:.1f}m → raw_depth={raw_depth:.2f}")
            
            current_distance_idx += 1
            if current_distance_idx >= len(distances):
                print("\n[OK] Kalibrasi selesai!")
                break
            else:
                print(f"\nSekarang letakkan objek pada jarak {distances[current_distance_idx]:.1f}m")
                print("Tekan SPASI untuk capture\n")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(calibration_data) < len(distances):
        print("[WARN] Kalibrasi tidak lengkap")
        return None, None
    
    # Hitung konstanta A untuk setiap titik
    A_values = []
    print("\n" + "-"*70)
    print("HASIL KALIBRASI:")
    print("-"*70)
    
    for i, data in enumerate(calibration_data, 1):
        A = data['distance'] * data['raw_depth']
        A_values.append(A)
        print(f"  Point {i}: {data['distance']:.1f}m → raw_depth={data['raw_depth']:.2f} → A={A:.2f}")
    
    # Rata-rata konstanta A
    A_final = np.mean(A_values)
    A_std = np.std(A_values)
    A_min = np.min(A_values)
    A_max = np.max(A_values)
    
    print("-"*70)
    print(f"KONSTANTA KALIBRASI:")
    print(f"  A_mean = {A_final:.2f}")
    print(f"  A_std  = {A_std:.2f}")
    print(f"  A_min  = {A_min:.2f}")
    print(f"  A_max  = {A_max:.2f}")
    print("-"*70)
    
    # Verifikasi akurasi
    print(f"\nVERIFIKASI AKURASI:")
    errors = []
    for data in calibration_data:
        predicted = A_final / data['raw_depth']
        error = abs(predicted - data['distance'])
        error_pct = (error / data['distance']) * 100
        errors.append(error)
        print(f"  {data['distance']:.1f}m: prediksi={predicted:.2f}m, "
              f"error={error:.3f}m ({error_pct:.1f}%)")
    
    mae = np.mean(errors)
    print(f"\nMean Absolute Error (MAE): {mae:.3f}m")
    
    return A_final, calibration_data


def calibrate_from_images(model, device, image_paths, distances, 
                          input_size=518, roi_size=50):
    """Mode kalibrasi dari gambar yang sudah ada."""
    if len(image_paths) != len(distances):
        print("[ERROR] Jumlah gambar harus sama dengan jumlah jarak")
        return None, None
    
    calibration_data = []
    
    print("\n" + "="*70)
    print("KALIBRASI DEPTH - MODE IMAGES")
    print("="*70)
    
    for img_path, dist in zip(image_paths, distances):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Tidak dapat membaca gambar: {img_path}")
            continue
        
        depth = model.infer_image(img, input_size)
        raw_depth = extract_roi_depth(depth, roi_size)
        
        calibration_data.append({
            'distance': dist,
            'raw_depth': raw_depth,
            'image': img_path
        })
        
        print(f"✓ {Path(img_path).name}: {dist:.1f}m → raw_depth={raw_depth:.2f}")
    
    # Hitung konstanta A
    A_values = [d['distance'] * d['raw_depth'] for d in calibration_data]
    A_final = np.mean(A_values)
    A_std = np.std(A_values)
    
    print(f"\n{'='*70}")
    print(f"KONSTANTA KALIBRASI: A = {A_final:.2f} ± {A_std:.2f}")
    print(f"{'='*70}\n")
    
    return A_final, calibration_data


def save_calibration_results(A, calibration_data, args, output_file):
    """Simpan hasil kalibrasi ke file."""
    results = {
        'calibration_constant': float(A),
        'model': args.encoder,
        'input_size': args.input_size,
        'roi_size': args.roi_size,
        'distances': args.distances,
        'timestamp': datetime.now().isoformat(),
        'data_points': [
            {
                'distance_m': d['distance'],
                'raw_depth': float(d['raw_depth']),
                'A': float(d['distance'] * d['raw_depth'])
            }
            for d in calibration_data
        ]
    }
    
    # Simpan JSON
    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Simpan TXT (untuk mudah copy-paste)
    with open(output_file, 'w') as f:
        f.write(f"# Hasil Kalibrasi Depth Anything V2\n")
        f.write(f"# Timestamp: {results['timestamp']}\n")
        f.write(f"# Model: {args.encoder}\n")
        f.write(f"# Input size: {args.input_size}\n")
        f.write(f"# ROI size: {args.roi_size}\n")
        f.write(f"# Jarak kalibrasi: {args.distances}\n\n")
        f.write(f"# ===== KONSTANTA UNTUK C++ CODE =====\n")
        f.write(f"const float A_CALIBRATION_CONSTANT = {A:.2f}f;\n")
        f.write(f"const int ROI_SIZE = {args.roi_size};\n\n")
        f.write(f"# ===== DATA POINTS =====\n")
        for dp in results['data_points']:
            f.write(f"# {dp['distance_m']:.1f}m: raw_depth={dp['raw_depth']:.2f}, A={dp['A']:.2f}\n")
    
    print(f"\n[OK] Hasil disimpan:")
    print(f"     JSON: {json_file}")
    print(f"     TXT:  {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Kalibrasi Depth untuk NCNN")
    parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--checkpoint', default='checkpoints/depth_anything_v2_vits.pth')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--roi-size', type=int, default=50, 
                       help='Ukuran ROI di tengah untuk ekstraksi depth')
    parser.add_argument('--distances', nargs='+', type=float, default=[1.0, 2.0, 3.0],
                       help='Jarak kalibrasi dalam meter (default: 1.0 2.0 3.0)')
    parser.add_argument('--mode', choices=['interactive', 'images'], default='interactive',
                       help='Mode kalibrasi: interactive (webcam) atau images')
    parser.add_argument('--images', nargs='+', 
                       help='Path gambar untuk mode images (urutannya sesuai --distances)')
    parser.add_argument('--output', default='conversion/calibration_result.txt',
                       help='File output untuk menyimpan hasil kalibrasi')
    
    args = parser.parse_args()
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device: {device}")
    
    # Load model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model = model.to(device).eval()
    
    print(f"[OK] Model loaded: {args.encoder}")
    
    # Kalibrasi
    if args.mode == 'interactive':
        A, calibration_data = calibrate_interactive(model, device, args.distances, 
                                                     args.input_size, args.roi_size)
    else:
        if not args.images:
            print("[ERROR] Mode 'images' memerlukan --images")
            return
        A, calibration_data = calibrate_from_images(model, device, args.images, args.distances,
                                                     args.input_size, args.roi_size)
    
    if A is not None and calibration_data is not None:
        # Simpan hasil
        output_path = Path(args.output)
        save_calibration_results(A, calibration_data, args, output_path)
        
        print(f"\n{'='*70}")
        print(f"GUNAKAN KONSTANTA INI DI KODE C++ NCNN:")
        print(f"const float A_CALIBRATION_CONSTANT = {A:.2f}f;")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
