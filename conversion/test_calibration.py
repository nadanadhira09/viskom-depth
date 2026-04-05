"""
test_calibration.py
===================
Script untuk testing akurasi hasil kalibrasi pada berbagai jarak.

Penggunaan:
    python conversion/test_calibration.py --calibration conversion/calibration_result.json
    python conversion/test_calibration.py --calibration conversion/calibration_result.json --test-distances 0.5 1.5 2.5 3.5
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Tambahkan path ke Depth Anything V2
REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_DIR))

from depth_anything_v2.dpt import DepthAnythingV2


def extract_roi_depth(depth_map: np.ndarray, roi_size: int = 50) -> float:
    """Ekstrak nilai raw depth rata-rata dari ROI di tengah."""
    h, w = depth_map.shape
    center_y, center_x = h // 2, w // 2
    
    y1 = max(0, center_y - roi_size // 2)
    y2 = min(h, center_y + roi_size // 2)
    x1 = max(0, center_x - roi_size // 2)
    x2 = min(w, center_x + roi_size // 2)
    
    roi = depth_map[y1:y2, x1:x2]
    raw_depth = np.mean(roi)
    
    return raw_depth


def predict_distance(raw_depth: float, A: float) -> float:
    """Prediksi jarak menggunakan konstanta kalibrasi."""
    if raw_depth < 1e-6:
        return float('inf')
    return A / raw_depth


def visualize_test(frame, depth, roi_size, raw_depth, predicted_dist, true_dist, A):
    """Visualisasi frame testing dengan info prediksi."""
    h, w = frame.shape[:2]
    center_y, center_x = h // 2, w // 2
    roi_half = roi_size // 2
    
    # Normalisasi depth untuk visualisasi
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    depth_colored = cv2.resize(depth_colored, (w, h))
    
    # Gambar ROI
    cv2.rectangle(frame, 
                 (center_x - roi_half, center_y - roi_half),
                 (center_x + roi_half, center_y + roi_half),
                 (0, 255, 0), 2)
    cv2.rectangle(depth_colored, 
                 (center_x - roi_half, center_y - roi_half),
                 (center_x + roi_half, center_y + roi_half),
                 (0, 255, 0), 2)
    
    # Gabungkan
    combined = np.hstack([frame, depth_colored])
    
    # Hitung error
    error = abs(predicted_dist - true_dist)
    error_pct = (error / true_dist) * 100
    
    # Tentukan warna berdasarkan error (hijau = akurat, merah = tidak akurat)
    if error_pct < 10:
        color = (0, 255, 0)  # Hijau
    elif error_pct < 20:
        color = (0, 255, 255)  # Kuning
    else:
        color = (0, 0, 255)  # Merah
    
    # Info text
    info_y = 30
    cv2.putText(combined, f"True Distance: {true_dist:.2f}m", (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, f"Predicted: {predicted_dist:.2f}m", (10, info_y + 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(combined, f"Error: {error:.3f}m ({error_pct:.1f}%)", (10, info_y + 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(combined, f"Raw Depth: {raw_depth:.2f}", (10, info_y + 105), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(combined, f"A = {A:.2f}", (10, info_y + 140), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    cv2.putText(combined, "SPASI=Next | ESC=Quit", (10, combined.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return combined, error, error_pct


def test_interactive(model, device, A, test_distances, input_size=518, roi_size=50):
    """Mode testing interaktif dengan webcam."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka webcam")
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    test_results = []
    current_distance_idx = 0
    
    print("\n" + "="*70)
    print("TESTING AKURASI KALIBRASI - MODE INTERAKTIF")
    print("="*70)
    print(f"Letakkan objek pada jarak {test_distances[current_distance_idx]:.2f}m")
    print("Tekan SPASI untuk test, ESC untuk keluar")
    print("="*70 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prediksi depth
        depth = model.infer_image(frame, input_size)
        raw_depth = extract_roi_depth(depth, roi_size)
        predicted_dist = predict_distance(raw_depth, A)
        true_dist = test_distances[current_distance_idx]
        
        # Visualisasi
        vis, error, error_pct = visualize_test(frame, depth, roi_size, raw_depth, 
                                               predicted_dist, true_dist, A)
        
        cv2.imshow("Test Kalibrasi", vis)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n[CANCELLED] Testing dibatalkan")
            break
        elif key == 32:  # SPASI
            # Record hasil test
            test_results.append({
                'true_distance': true_dist,
                'predicted_distance': predicted_dist,
                'raw_depth': raw_depth,
                'error_m': error,
                'error_pct': error_pct
            })
            
            print(f"✓ Test: {true_dist:.2f}m → pred={predicted_dist:.2f}m, error={error:.3f}m ({error_pct:.1f}%)")
            
            current_distance_idx += 1
            if current_distance_idx >= len(test_distances):
                print("\n[OK] Testing selesai!")
                break
            else:
                print(f"\nSekarang letakkan objek pada jarak {test_distances[current_distance_idx]:.2f}m")
                print("Tekan SPASI untuk test\n")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return test_results


def plot_results(test_results, calibration_data, A, output_dir):
    """Plot hasil testing vs kalibrasi."""
    if not test_results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Data untuk plotting
    cal_distances = [d['distance_m'] for d in calibration_data]
    cal_raw_depths = [d['raw_depth'] for d in calibration_data]
    
    test_true = [r['true_distance'] for r in test_results]
    test_pred = [r['predicted_distance'] for r in test_results]
    test_raw = [r['raw_depth'] for r in test_results]
    test_errors = [r['error_m'] for r in test_results]
    
    # Plot 1: Raw Depth vs Distance (Calibration + Test)
    axes[0, 0].scatter(cal_distances, cal_raw_depths, c='blue', s=100, label='Calibration Points', marker='o')
    axes[0, 0].scatter(test_true, test_raw, c='red', s=100, label='Test Points', marker='x')
    
    # Fit curve untuk visualisasi
    x_fit = np.linspace(0.5, max(max(cal_distances), max(test_true)) + 0.5, 100)
    y_fit = A / x_fit
    axes[0, 0].plot(x_fit, y_fit, 'g--', label=f'Fitted: depth = {A:.2f}/distance', linewidth=2)
    
    axes[0, 0].set_xlabel('True Distance (m)', fontsize=12)
    axes[0, 0].set_ylabel('Raw Depth', fontsize=12)
    axes[0, 0].set_title('Raw Depth vs Distance', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs True Distance
    axes[0, 1].scatter(test_true, test_pred, c='green', s=100, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(min(test_true), min(test_pred))
    max_val = max(max(test_true), max(test_pred))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[0, 1].set_xlabel('True Distance (m)', fontsize=12)
    axes[0, 1].set_ylabel('Predicted Distance (m)', fontsize=12)
    axes[0, 1].set_title('Predicted vs True Distance', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error per Test Point
    axes[1, 0].bar(range(len(test_errors)), test_errors, color=['green' if e < 0.1 else 'orange' if e < 0.2 else 'red' for e in test_errors])
    axes[1, 0].set_xlabel('Test Point Index', fontsize=12)
    axes[1, 0].set_ylabel('Absolute Error (m)', fontsize=12)
    axes[1, 0].set_title('Error per Test Point', fontsize=14, fontweight='bold')
    axes[1, 0].axhline(y=np.mean(test_errors), color='blue', linestyle='--', linewidth=2, label=f'Mean Error = {np.mean(test_errors):.3f}m')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error Percentage
    test_error_pct = [r['error_pct'] for r in test_results]
    axes[1, 1].bar(range(len(test_error_pct)), test_error_pct, color=['green' if e < 10 else 'orange' if e < 20 else 'red' for e in test_error_pct])
    axes[1, 1].set_xlabel('Test Point Index', fontsize=12)
    axes[1, 1].set_ylabel('Error (%)', fontsize=12)
    axes[1, 1].set_title('Error Percentage per Test Point', fontsize=14, fontweight='bold')
    axes[1, 1].axhline(y=np.mean(test_error_pct), color='blue', linestyle='--', linewidth=2, label=f'Mean Error = {np.mean(test_error_pct):.1f}%')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Simpan plot
    output_path = output_dir / f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Plot disimpan: {output_path}")
    
    plt.show()


def save_test_results(test_results, A, calibration_file, output_file):
    """Simpan hasil testing."""
    results = {
        'calibration_constant': float(A),
        'calibration_source': str(calibration_file),
        'timestamp': datetime.now().isoformat(),
        'test_points': test_results,
        'statistics': {
            'mean_error_m': float(np.mean([r['error_m'] for r in test_results])),
            'std_error_m': float(np.std([r['error_m'] for r in test_results])),
            'max_error_m': float(np.max([r['error_m'] for r in test_results])),
            'mean_error_pct': float(np.mean([r['error_pct'] for r in test_results])),
            'std_error_pct': float(np.std([r['error_pct'] for r in test_results])),
            'max_error_pct': float(np.max([r['error_pct'] for r in test_results])),
        }
    }
    
    # Simpan JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] Test results disimpan: {output_file}")
    
    # Print statistik
    print("\n" + "="*70)
    print("STATISTIK TESTING:")
    print("="*70)
    print(f"  Mean Absolute Error: {results['statistics']['mean_error_m']:.3f} ± {results['statistics']['std_error_m']:.3f} m")
    print(f"  Mean Error %:        {results['statistics']['mean_error_pct']:.1f} ± {results['statistics']['std_error_pct']:.1f} %")
    print(f"  Max Error:           {results['statistics']['max_error_m']:.3f} m ({results['statistics']['max_error_pct']:.1f}%)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Test akurasi hasil kalibrasi")
    parser.add_argument('--calibration', required=True, 
                       help='Path ke file kalibrasi JSON')
    parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--checkpoint', default='checkpoints/depth_anything_v2_vits.pth')
    parser.add_argument('--test-distances', nargs='+', type=float, default=[0.5, 1.5, 2.5],
                       help='Jarak untuk testing (default: 0.5 1.5 2.5)')
    parser.add_argument('--output', default='conversion/test_results.json',
                       help='File output untuk hasil testing')
    parser.add_argument('--plot', action='store_true', 
                       help='Tampilkan plot hasil testing')
    
    args = parser.parse_args()
    
    # Load kalibrasi
    calib_path = Path(args.calibration)
    if not calib_path.exists():
        print(f"[ERROR] File kalibrasi tidak ditemukan: {calib_path}")
        return
    
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    
    A = calib_data['calibration_constant']
    roi_size = calib_data.get('roi_size', 50)
    input_size = calib_data.get('input_size', 518)
    
    print(f"[INFO] Loaded calibration: A = {A:.2f}")
    
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
    
    # Testing
    test_results = test_interactive(model, device, A, args.test_distances, 
                                    input_size, roi_size)
    
    if test_results:
        # Simpan hasil
        output_path = Path(args.output)
        save_test_results(test_results, A, calib_path, output_path)
        
        # Plot jika diminta
        if args.plot:
            plot_results(test_results, calib_data['data_points'], A, output_path.parent)


if __name__ == "__main__":
    main()
