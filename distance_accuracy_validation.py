"""
distance_accuracy_validation.py
================================
Validasi akurasi estimasi jarak real-time pada berbagai distance
Menghasilkan laporan seperti Tabel 6.7 pada contoh pengguna
"""

import cv2
import numpy as np
import matplotlib
import sys
from pathlib import Path

try:
    import onnxruntime as ort
except:
    print("[ERROR] onnxruntime not installed")
    sys.exit(1)

sys.path.insert(0, 'Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2


class DistanceAccuracyValidator:
    """Validator untuk akurasi estimasi jarak"""
    
    def __init__(self, model_type='onnx_int8'):
        self.model_type = model_type
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        self.load_model()
    
    def load_model(self):
        """Load model berdasarkan type"""
        if self.model_type == 'onnx_int8':
            model_path = 'models/onnx/depth_anything_v2_vits_quantized_int8.onnx'
            print(f"[INFO] Loading ONNX Quantized INT8: {model_path}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        elif self.model_type == 'onnx_fp32':
            model_path = 'models/onnx/depth_anything_v2_vits.onnx'
            print(f"[INFO] Loading ONNX FP32: {model_path}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        print("[OK] Model loaded\n")
    
    def infer_depth(self, frame):
        """Run depth inference"""
        img = cv2.resize(frame, (518, 518))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        
        depth = self.session.run([self.output_name], {self.input_name: img})[0][0, 0]
        return depth
    
    def measure_distance_at_point(self, frame, center_x=None, center_y=None, window_size=50):
        """Measure distance at specific point (center region average)"""
        depth = self.infer_depth(frame)
        
        h, w = frame.shape[:2]
        if center_x is None:
            center_x = w // 2
        if center_y is None:
            center_y = h // 2
        
        # Get region around center
        x1 = max(0, center_x - window_size // 2)
        x2 = min(w, center_x + window_size // 2)
        y1 = max(0, center_y - window_size // 2)
        y2 = min(h, center_y + window_size // 2)
        
        region_depth = depth[y1:y2, x1:x2]
        avg_depth = np.mean(region_depth)
        
        # Normalize to actual distance (rough calibration)
        # Depth value 0-1 mapped to distance 0-10 meters
        estimated_distance = avg_depth * 10.0
        
        return estimated_distance, depth
    
    def validate_distances(self):
        """Interactive distance validation"""
        print("="*80)
        print("6.2.3 ANALISIS AKURASI ESTIMASI JARAK REAL-TIME")
        print("="*80 + "\n")
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Camera failed")
            return
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("""
INSTRUCTIONS:
=============
1. Posisikan objek di depan kamera pada jarak yang diinginkan
2. Pastikan objek berada di CENTER CIRCLE (tengah layar)
3. Tekan SPACE untuk capture/measure distance
4. Input actual distance yang diukur dengan ruler/meter
5. Ulangi untuk berbagai jarak (dekat, menengah, jauh)
6. Tekan Q/ESC untuk selesai

Siap? Mulai!\n
        """)
        
        measurements = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Get depth estimate at center
                h, w = frame.shape[:2]
                estimated_dist, depth = self.measure_distance_at_point(frame)
                
                # Normalize and colorize depth
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
                depth_colored = (self.cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                
                # Draw center circle
                center_x, center_y = w // 2, h // 2
                cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                
                # Display info
                cv2.putText(frame, f"Est Distance: {estimated_dist:.2f}m", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE=Measure | Q=Quit", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
                
                # Combine
                display = np.hstack([frame, depth_colored])
                cv2.imshow("Distance Validation - Left: RGB | Right: Depth", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    # Measure
                    print(f"\n[MEASUREMENT {len(measurements)+1}]")
                    estimated = estimated_dist
                    print(f"Estimated distance: {estimated:.2f} m")
                    
                    actual_input = input("Enter actual distance measured (meters): ")
                    try:
                        actual = float(actual_input)
                        error = abs(estimated - actual)
                        measurements.append({
                            'actual': actual,
                            'estimated': estimated,
                            'error': error
                        })
                        print(f"Error: {error:.2f} m")
                        print(f"Recorded!\n")
                    except:
                        print("Invalid input, skipped\n")
                
                elif key == ord('q') or key == 27:
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Generate report
        if measurements:
            self._generate_distance_report(measurements)
    
    def _generate_distance_report(self, measurements):
        """Generate distance validation report (seperti Tabel 6.7)"""
        
        print("\n" + "="*80)
        print("TABEL 6.7 - VALIDASI JARAK ESTIMASI DEPTH REAL-TIME")
        print("="*80 + "\n")
        
        print("Hasil Pengujian Validasi Jarak:")
        print("-" * 80)
        print(f"{'No':>3} | {'Jarak Aktu':<12} | {'Jarak Est':<12} | {'Error':<10} | {'Analisis':<15}")
        print("-" * 80)
        
        for i, m in enumerate(measurements, 1):
            actual = m['actual']
            estimated = m['estimated']
            error = m['error']
            
            # Classify distance
            if actual < 0.5:
                class_type = "Ekstrem Dekat"
            elif actual < 1.0:
                class_type = "Sangat Dekat"
            elif actual < 2.0:
                class_type = "Dekat"
            elif actual < 3.0:
                class_type = "Menengah"
            elif actual < 5.0:
                class_type = "Jauh"
            else:
                class_type = "Sangat Jauh"
            
            print(f"{i:>3} | {actual:>10.2f}m | {estimated:>10.2f}m | {error:>8.2f}m | {class_type:<15}")
        
        print("-" * 80)
        
        # Analysis
        mean_error = np.mean([m['error'] for m in measurements])
        max_error = max([m['error'] for m in measurements])
        
        print(f"\nSummary:")
        print(f"  Jumlah sampel: {len(measurements)}")
        print(f"  Rata-rata error: {mean_error:.3f} m")
        print(f"  Maksimum error: {max_error:.3f} m")
        
        # Detailed analysis
        print("\n" + "="*80)
        print("ANALISIS DETAIL AKURASI")
        print("="*80 + "\n")
        
        # Group by distance
        close_dist = [m for m in measurements if m['actual'] < 1.5]
        mid_dist = [m for m in measurements if 1.5 <= m['actual'] < 3.0]
        far_dist = [m for m in measurements if m['actual'] >= 3.0]
        
        if close_dist:
            print("1. JARAK DEKAT (< 1.5m):")
            errors = [m['error'] for m in close_dist]
            print(f"   Rata-rata error: {np.mean(errors):.3f} m")
            print(f"   Maks error: {np.max(errors):.3f} m")
            print(f"   Status: {'EXCELLENT' if np.mean(errors) < 0.1 else 'GOOD' if np.mean(errors) < 0.2 else 'ACCEPTABLE'}")
            print(f"   Analisis: Sistem menunjukkan performa {} pada jarak dekat".format(
                "EXCELLENT" if np.mean(errors) < 0.1 else "GOOD" if np.mean(errors) < 0.2 else "ACCEPTABLE"))
            print()
        
        if mid_dist:
            print("2. JARAK MENENGAH (1.5m - 3.0m):")
            errors = [m['error'] for m in mid_dist]
            print(f"   Rata-rata error: {np.mean(errors):.3f} m")
            print(f"   Maks error: {np.max(errors):.3f} m")
            print(f"   Status: {'EXCELLENT' if np.mean(errors) < 0.1 else 'GOOD' if np.mean(errors) < 0.2 else 'ACCEPTABLE'}")
            print(f"   Analisis: Sistem menunjukkan performa {} pada jarak menengah".format(
                "EXCELLENT" if np.mean(errors) < 0.1 else "GOOD" if np.mean(errors) < 0.2 else "ACCEPTABLE"))
            print()
        
        if far_dist:
            print("3. JARAK JAUH (>= 3.0m):")
            errors = [m['error'] for m in far_dist]
            print(f"   Rata-rata error: {np.mean(errors):.3f} m")
            print(f"   Maks error: {np.max(errors):.3f} m")
            print(f"   Status: {'EXCELLENT' if np.mean(errors) < 0.1 else 'GOOD' if np.mean(errors) < 0.2 else 'ACCEPTABLE'}")
            print(f"   Analisis: Sistem menunjukkan performa {} pada jarak jauh".format(
                "EXCELLENT" if np.mean(errors) < 0.1 else "GOOD" if np.mean(errors) < 0.2 else "ACCEPTABLE"))
            print()
        
        print("="*80)


def main():
    print("\n" + "="*80)
    print("DISTANCE ACCURACY VALIDATION")
    print("="*80 + "\n")
    
    validator = DistanceAccuracyValidator(model_type='onnx_int8')
    validator.validate_distances()


if __name__ == "__main__":
    main()
