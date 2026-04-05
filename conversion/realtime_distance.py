"""
realtime_distance.py
====================
Visualisasi real-time prediksi jarak dengan smoothing dan safety hold.

Penggunaan:
    python conversion/realtime_distance.py --calibration conversion/calibration_result.json
    python conversion/realtime_distance.py --calibration conversion/calibration_result.json --smoothing 0.3 --safety-threshold 1.5
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import json
from collections import deque
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("[WARNING] Modul ultralytics (YOLO) tidak ditemukan. Deteksi objek tidak akan tersedia.")
    print("Install dengan: pip install ultralytics")
    YOLO = None

# Tambahkan path ke Depth Anything V2
REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_DIR))

from depth_anything_v2.dpt import DepthAnythingV2


class DistanceEstimator:
    """Class untuk estimasi jarak dengan smoothing dan safety hold."""
    
    def __init__(self, A, roi_size=50, smoothing_alpha=0.3, 
                 max_jump_threshold=1.5, min_distance=0.1, max_distance=5.0):
        self.A = A
        self.roi_size = roi_size
        self.smoothing_alpha = smoothing_alpha
        self.max_jump_threshold = max_jump_threshold
        self.min_distance = min_distance
        self.max_distance = max_distance
        
        # State variables
        self.last_smoothed_distance = -1.0
        self.distance_history = deque(maxlen=30)  # 30 frame history
        self.safety_hold_active = False
        self.safety_hold_counter = 0
        self.safety_hold_threshold = 5  # frames
        
    def extract_roi_depth(self, depth_map):
        """Ekstrak raw depth dari ROI."""
        h, w = depth_map.shape
        center_y, center_x = h // 2, w // 2
        
        y1 = max(0, center_y - self.roi_size // 2)
        y2 = min(h, center_y + self.roi_size // 2)
        x1 = max(0, center_x - self.roi_size // 2)
        x2 = min(w, center_x + self.roi_size // 2)
        
        roi = depth_map[y1:y2, x1:x2]
        return np.mean(roi)
    
    def predict_distance(self, depth_map):
        """
        Prediksi jarak dengan smoothing dan safety hold.
        Jika A=None, kita tampilkan 'Proximity Score' relatif saja.
        
        Returns:
            tuple: (smoothed_distance, current_estimate, safety_active)
        """
        # Ekstrak raw depth menggunakan median untuk kestabilan lebih baik dibanding mean
        h, w = depth_map.shape
        center_y, center_x = h // 2, w // 2
        y1, y2 = max(0, center_y - self.roi_size // 2), min(h, center_y + self.roi_size // 2)
        x1, x2 = max(0, center_x - self.roi_size // 2), min(w, center_x + self.roi_size // 2)
        roi = depth_map[y1:y2, x1:x2]
        raw_depth = np.median(roi)  # Perubahan krusial: Mean ke Median
        
        # Mode tanpa kalibrasi
        if self.A is None:
            # Gunakan skor pendekatan: misalnya kita scaling raw depth saja
            # Raw depth pada model ini umumnya merepresentasikan disparitas (makin tinggi makin dekat)
            current_estimate = np.clip(raw_depth / 100.0, 0, 100) # dummy max limit
        else:
            # Mode dengan kalibrasi: Konversi ke jarak metrik
            if raw_depth < 1e-6:
                current_estimate = self.max_distance
            else:
                current_estimate = self.A / raw_depth
            # Clamp ke range valid
            current_estimate = np.clip(current_estimate, self.min_distance, self.max_distance)
        
        # Inisialisasi jika frame pertama
        if self.last_smoothed_distance < 0:
            smoothed_distance = current_estimate
            self.last_smoothed_distance = smoothed_distance
            self.distance_history.append(smoothed_distance)
            return smoothed_distance, current_estimate, False
        
        # Deteksi sudden jump
        distance_change = abs(current_estimate - self.last_smoothed_distance)
        
        if distance_change > self.max_jump_threshold:
            # Sudden jump detected - activate safety hold
            self.safety_hold_active = True
            self.safety_hold_counter = self.safety_hold_threshold
            
            # Prioritaskan jarak terdekat (safety bias)
            current_estimate = min(current_estimate, self.last_smoothed_distance)
        
        # Smoothing menggunakan Exponential Moving Average
        smoothed_distance = (self.smoothing_alpha * current_estimate + 
                            (1.0 - self.smoothing_alpha) * self.last_smoothed_distance)
        
        # Update state
        self.last_smoothed_distance = smoothed_distance
        self.distance_history.append(smoothed_distance)
        
        # Decrement safety hold counter
        if self.safety_hold_counter > 0:
            self.safety_hold_counter -= 1
        if self.safety_hold_counter == 0:
            self.safety_hold_active = False
        
        return smoothed_distance, current_estimate, self.safety_hold_active
    
    def get_stability_metric(self):
        """Hitung metrik stabilitas dari history."""
        if len(self.distance_history) < 2:
            return 0.0
        
        recent_history = list(self.distance_history)[-10:]  # 10 frame terakhir
        std = np.std(recent_history)
        return std
    
    def reset(self):
        """Reset state estimator."""
        self.last_smoothed_distance = -1.0
        self.distance_history.clear()
        self.safety_hold_active = False
        self.safety_hold_counter = 0


def create_distance_bar(distance, max_dist=5.0, width=400, height=50, unit="m"):
    """Buat visualisasi bar jarak."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background
    cv2.rectangle(bar, (0, 0), (width, height), (50, 50, 50), -1)
    
    # Filled portion
    fill_width = int((min(distance, max_dist) / max_dist) * width)
    
    # Color gradient (Untuk Proximity, warnanya dibalik karena nilai besar artinya sangat dekat. Untuk metrik, nilai besar artinya jauh)
    if unit == "%":
        # Mode Proximity (Besar = Dekat)
        if distance > 70:
            color = (0, 0, 255)  # Merah - sangat dekat
        elif distance > 40:
            color = (0, 165, 255) # Orange
        elif distance > 20:
            color = (0, 255, 255) # Kuning
        else:
            color = (0, 255, 0) # Hijau - jauh
    else:
        # Mode Metrik Biasa (Besar = Jauh)
        if distance > 3.0:
            color = (0, 255, 0)  # Hijau - aman
        elif distance > 1.5:
            color = (0, 255, 255)  # Kuning - hati-hati
        elif distance > 0.5:
            color = (0, 165, 255)  # Orange - peringatan
        else:
            color = (0, 0, 255)  # Merah - bahaya
    
    cv2.rectangle(bar, (0, 0), (fill_width, height), color, -1)
    
    # Border
    cv2.rectangle(bar, (0, 0), (width, height), (255, 255, 255), 2)
    
    # Text
    text = f"{distance:.1f}{unit}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(bar, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(bar, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return bar


def create_plot_graph(history, width=400, height=150, max_dist=5.0):
    """Buat plot graph dari history jarak."""
    graph = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    if len(history) < 2:
        return graph
    
    # Border
    cv2.rectangle(graph, (0, 0), (width-1, height-1), (100, 100, 100), 2)
    
    # Grid lines
    for i in range(1, 5):
        y = int(height * i / 5)
        cv2.line(graph, (0, y), (width, y), (200, 200, 200), 1)
    
    # Plot line
    points = []
    for i, dist in enumerate(history):
        x = int((i / len(history)) * width)
        y = int(height - (min(dist, max_dist) / max_dist) * height)
        points.append((x, y))
    
    # Draw line
    for i in range(len(points) - 1):
        cv2.line(graph, points[i], points[i+1], (255, 0, 0), 2)
    
    # Y-axis labels
    for i in range(6):
        dist_label = i * max_dist / 5
        y_pos = height - int((i / 5) * height)
        cv2.putText(graph, f"{dist_label:.1f}", (5, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    return graph


def enhance_low_light(image):
    """
    Meningkatkan kecerahan dan kontras gambar di kondisi kurang cahaya (Low-Light)
    menggunakan metode CLAHE (Contrast Limited Adaptive Histogram Equalization)
    pada ruang warna LAB tanpa merusak warna aslinya.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplikasikan CLAHE pada L-channel (Lightness)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Gabungkan dan konversi kembali ke BGR
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def visualize_realtime(frame, depth, estimator, fps, show_depth=True, yolo_results=None):
    """Visualisasi real-time dengan semua info."""
    # Prediksi jarak
    smoothed_dist, current_est, safety_active = estimator.predict_distance(depth)
    stability = estimator.get_stability_metric()
    
    # Text helper untuk membedakan Meter vs Proximiy
    is_calibrated = estimator.A is not None
    unit_str = "m" if is_calibrated else "%"
    status_label = "Distance" if is_calibrated else "Proximity"
    
    # Normalisasi depth untuk visualisasi
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
                 
    # Gambar bounding boxes YOLO dan tampilkan jarak objek
    if yolo_results:
        # Resize depth agar sesuai dengan dimensi aslinya jika berbeda
        dh, dw = depth.shape[:2]
        sh, sw = frame.shape[:2]
        depth_resized = depth if (dh == sh and dw == sw) else cv2.resize(depth, (sw, sh))
        
        for result in yolo_results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]
                
                # Batasi kotak dalam dimensi gambar
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(sw - 1, x2), min(sh - 1, y2)
                
                bw, bh = x2 - x1, y2 - y1
                # Abaikan objek terlalu kecil
                if bw < 5 or bh < 5:
                    continue
                    
                # Hitung nilai depth di sekitar tengah kotak (20% tengah) menggunakan median
                cx, cy = x1 + bw // 2, y1 + bh // 2
                sample_hw, sample_hh = max(2, bw // 10), max(2, bh // 10)
                
                sy1, sy2 = max(0, cy - sample_hh), min(sh, cy + sample_hh)
                sx1, sx2 = max(0, cx - sample_hw), min(sw, cx + sample_hw)
                
                roi_depth = depth_resized[sy1:sy2, sx1:sx2]
                median_depth = np.median(roi_depth) if roi_depth.size > 0 else 0
                
                if estimator.A is None:
                    # Tampilkan Proximity Score (skala relatif terbalik sebagai kedekatan)
                    obj_dist = np.clip(median_depth / 100.0, 0, 100) # nilai skala
                    text = f"{label} {conf:.2f} | C{obj_dist:.1f}%" # C for Close (Kedekatan)
                    
                    # Warna berdasarkan proximity (Mendekati 100% = merah, di bawah 40% = biru)
                    if obj_dist > 60:
                        box_color = (0, 0, 255) # Merah (Dekat)
                    elif obj_dist > 40:
                        box_color = (0, 165, 255) # Oranye/Kuning
                    else:
                        box_color = (255, 0, 0) # Biru (Jauh)
                else:
                    if median_depth < 1e-6:
                        obj_dist = estimator.max_distance
                    else:
                        obj_dist = np.clip(estimator.A / median_depth, estimator.min_distance, estimator.max_distance)
                    text = f"{label} {conf:.2f} | {obj_dist:.1f}m"
                    
                    # Warna berdasarkan meter (Di bawah 1 meter = merah, di atas 2 meter = Biru)
                    if obj_dist < 1.0:
                        box_color = (0, 0, 255) # Merah (Dekat)
                    elif obj_dist < 2.0:
                        box_color = (0, 165, 255) # Oranye/Kuning
                    else:
                        box_color = (255, 0, 0) # Biru (Jauh)
                
                # Gambar kotak pada frame asli
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, text, (x1, max(15, y1 - 5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # Gambar kotak pada depth image
                if show_depth:
                    cv2.rectangle(depth_colored, (x1, y1), (x2, y2), box_color, 1)

    # Gabungkan frame dan depth jika diminta
    if show_depth:
        main_view = np.hstack([frame, depth_colored])
    else:
        main_view = frame
    
    # Hanya kembalikan layar utama tanpa panel informasi bawah
    return main_view


def main():
    parser = argparse.ArgumentParser(description="Real-time distance estimation dengan smoothing")
    parser.add_argument('--calibration', default=None, 
                       help='Path ke file kalibrasi JSON (Opsional)')
    parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--checkpoint', default='checkpoints/depth_anything_v2_vits.pth')
    parser.add_argument('--smoothing', type=float, default=0.3,
                       help='Alpha untuk smoothing (0-1, default: 0.3)')
    parser.add_argument('--safety-threshold', type=float, default=1.5,
                       help='Threshold untuk sudden jump detection (meter, default: 1.5)')
    parser.add_argument('--no-depth', action='store_true',
                       help='Jangan tampilkan depth map (hanya frame asli)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--yolo-model', type=str, default='yolov12n.pt',
                       help='Pilih model YOLO untuk deteksi objek (default: yolov12n.pt)')
    parser.add_argument('--enhance', action='store_true',
                       help='Aktifkan peningkatan kontras cerdas untuk lingkungan kurang cahaya (Low Light)')
    
    args = parser.parse_args()
    
    # Load kalibrasi jika tersedia
    if args.calibration:
        calib_path = Path(args.calibration)
        if not calib_path.exists():
            print(f"[ERROR] File kalibrasi tidak ditemukan: {calib_path}")
            return
        
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        A = calib_data['calibration_constant']
        roi_size = calib_data.get('roi_size', 50)
        input_size = calib_data.get('input_size', 518)
        print(f"[INFO] Mode Terkalibrasi. Calibration constant: A = {A:.2f}")
    else:
        A = None
        roi_size = 50
        input_size = 518
        print("[INFO] Mode Tanpa Kalibrasi. Output berupa 'Nilai Kendekatan' (Proximity Score).")
        
    print(f"[INFO] Smoothing alpha: {args.smoothing}")
    print(f"[INFO] Safety threshold: {args.safety_threshold}m")
    
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
    
    # Load YOLO model
    if YOLO is not None:
        try:
            print(f"[INFO] Loading YOLO model: {args.yolo_model}")
            yolo_model = YOLO(args.yolo_model)
            print("[OK] YOLO model loaded.")
        except Exception as e:
            print(f"[ERROR] Gagal memuat YOLO: {e}")
            yolo_model = None
    else:
        yolo_model = None

    # Initialize distance estimator
    estimator = DistanceEstimator(
        A=A,
        roi_size=roi_size,
        smoothing_alpha=args.smoothing,
        max_jump_threshold=args.safety_threshold
    )
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka kamera {args.camera}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*70)
    print("REAL-TIME DISTANCE ESTIMATION")
    print("="*70)
    print("Tekan 'r' untuk reset | 'q' atau ESC untuk keluar")
    print("="*70 + "\n")
    
    # FPS calculation
    fps_history = deque(maxlen=30)
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Peningkatan Low-Light jika diaktifkan (Membantu MDE dan YOLO mencari kontur dalam skenario gelap)
        if args.enhance:
            frame = enhance_low_light(frame)
        
        # Prediksi depth
        depth = model.infer_image(frame, input_size)
        
        # Deteksi Objek dengan YOLO (jika tersedia)
        yolo_results = None
        if yolo_model is not None:
            # Gunakan resolusi asli atau disesuaikan dengan kebutuhan
            yolo_results = yolo_model(frame, stream=False, verbose=False)

        # Calculate FPS
        float_fps_denom = time.time() - start_time
        fps = 1.0 / float_fps_denom if float_fps_denom > 0 else 0
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        
        # Visualisasi
        vis = visualize_realtime(frame.copy(), depth, estimator, avg_fps, 
                                yolo_results=yolo_results,
                                show_depth=not args.no_depth)
        
        cv2.imshow("Real-time Distance Estimation", vis)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or key == ord('q'):  # ESC or Q
            break
        elif key == ord('r'):  # Reset
            estimator.reset()
            print("[INFO] Estimator reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n[OK] Selesai")


if __name__ == "__main__":
    main()
