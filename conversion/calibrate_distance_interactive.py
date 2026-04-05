"""
calibrate_distance_interactive.py
==================================
Kalibrasi jarak untuk mendapatkan akurasi yang tepat
User menunjukkan benda dengan jarak yang diketahui
"""

import cv2
import torch
import numpy as np
import sys
from pathlib import Path

# Add Depth Anything V2 path
REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_DIR))

from depth_anything_v2.dpt import DepthAnythingV2


class DistanceCalibrator:
    def __init__(self, device='cpu'):
        self.device = device
        self.depth_model = None
        self.cap = None
        
        # Calibration variables
        self.calibration_points = []  # [(depth_value, known_distance), ...]
        self.depth_scale = 10.0
        
        self.current_frame = None
        self.current_depth = None
        self.roi_start = None
        self.roi_end = None
        self.drawing = False
    
    def load_model(self):
        """Load Depth Anything V2"""
        print("[INFO] Loading Depth Anything V2 (vits)...")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model.load_state_dict(
            torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu')
        )
        self.depth_model = self.depth_model.to(self.device).eval()
        print("[OK] Model loaded")
    
    def init_camera(self):
        """Initialize camera"""
        print("[INFO] Opening camera...")
        
        for backend_id in [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]:
            self.cap = cv2.VideoCapture(0, backend_id)
            if self.cap.isOpened():
                break
        
        if not self.cap.isOpened():
            print("[ERROR] Camera failed")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("[OK] Camera ready")
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback untuk drawing ROI"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_end = (x, y)
            self.drawing = False
    
    def get_depth_in_roi(self):
        """Get average depth dalam ROI"""
        if self.roi_start is None or self.roi_end is None:
            return None
        
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        
        # Ensure coordinates are valid
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        # Scale ke depth map size
        h, w = self.current_depth.shape
        sx = w / 640
        sy = h / 480
        
        dx1 = int(x1 * sx)
        dy1 = int(y1 * sy)
        dx2 = int(x2 * sx)
        dy2 = int(y2 * sy)
        
        # Extract depth region
        depth_region = self.current_depth[dy1:dy2, dx1:dx2]
        
        if depth_region.size == 0:
            return None
        
        # Return median value
        return np.median(depth_region)
    
    def run(self):
        """Main calibration loop"""
        self.load_model()
        
        if not self.init_camera():
            return
        
        # Setup mouse callback
        cv2.namedWindow("Depth Calibration")
        cv2.setMouseCallback("Depth Calibration", self.mouse_callback)
        
        print("\n" + "="*70)
        print("INTERACTIVE DISTANCE CALIBRATION")
        print("="*70)
        print("\nInstructions:")
        print("1. Arahkan kamera ke benda dengan jarak yang DIKETAHUI")
        print("2. Drag untuk select ROI (Region of Interest) di benda tersebut")
        print("3. Ketik jarak sebenarnya dalam METER (contoh: 2.5)")
        print("4. Ulangi untuk 3-5 titik berbeda")
        print("5. Tekan SPACE untuk calculate calibration")
        print("6. Tekan Q untuk quit")
        print("="*70 + "\n")
        
        calibration_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.current_frame = frame.copy()
                
                # Compute depth
                with torch.no_grad():
                    depth = self.depth_model.infer_image(frame, input_size=384)
                
                # Normalize
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max > depth_min:
                    self.current_depth = (depth - depth_min) / (depth_max - depth_min)
                else:
                    self.current_depth = np.zeros_like(depth)
                
                # Inverse untuk konvensi (jauh = rendah)
                self.current_depth = 1.0 - self.current_depth
                
                # Colorize depth
                display = self.current_frame.copy()
                depth_colored = cv2.applyColorMap(
                    (self.current_depth * 255).astype(np.uint8),
                    cv2.COLORMAP_MAGMA
                )
                
                # Draw calibration points
                for i, (depth_val, known_dist) in enumerate(self.calibration_points):
                    color = (0, 255, 0)
                    text = f"Cal {i+1}: {known_dist}m (d={depth_val:.3f})"
                    cv2.putText(display, text, (10, 30 + i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw ROI if selecting
                if self.roi_start and self.roi_end:
                    x1, y1 = self.roi_start
                    x2, y2 = self.roi_end
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
                    # Show depth value in ROI
                    depth_in_roi = self.get_depth_in_roi()
                    if depth_in_roi is not None:
                        cv2.putText(display, f"Depth: {depth_in_roi:.3f}", (x1, max(20, y1-10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Combine
                h, w = display.shape[:2]
                depth_resized = cv2.resize(depth_colored, (w, h))
                combined = np.hstack([display, depth_resized])
                
                # Info
                info_y = combined.shape[0] - 60
                cv2.putText(combined, "CALIBRATION - Drag ROI on object, enter distance, press ENTER", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
                cv2.putText(combined, f"Points collected: {len(self.calibration_points)} | Press SPACE to calculate scale", 
                           (10, info_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
                
                cv2.imshow("Depth Calibration", combined)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    print("\n[INFO] Quit")
                    break
                
                elif key == ord('\r') or key == ord('\n'):  # ENTER
                    depth_value = self.get_depth_in_roi()
                    if depth_value is not None:
                        # Input jarak
                        distance_str = input(f"\nMasukkan jarak sebenarnya (meter): ")
                        try:
                            distance = float(distance_str)
                            self.calibration_points.append((depth_value, distance))
                            calibration_count += 1
                            print(f"[OK] Point {calibration_count} recorded: depth={depth_value:.3f}, distance={distance}m")
                        except ValueError:
                            print("[ERROR] Input tidak valid")
                    else:
                        print("[ERROR] Silakan select ROI terlebih dahulu")
                    
                    self.roi_start = None
                    self.roi_end = None
                
                elif key == ord(' '):  # SPACE
                    if len(self.calibration_points) >= 2:
                        self.calculate_scale()
                        print("\n[OK] Calibration complete!")
                        print(f"Calculated depth_scale: {self.depth_scale:.2f}")
                        break
                    else:
                        print("[ERROR] Minimal 2 titik kalibrasi dibutuhkan")
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def calculate_scale(self):
        """Calculate optimal depth scale menggunakan linear regression"""
        if len(self.calibration_points) < 2:
            print("[ERROR] Tidak cukup data")
            return
        
        # Extract depth values dan distances
        depths = np.array([d for d, dis in self.calibration_points])
        distances = np.array([dis for d, dis in self.calibration_points])
        
        print("\n[INFO] Calibration Points:")
        for i, (d, dis) in enumerate(self.calibration_points):
            print(f"  {i+1}. depth={d:.4f}, distance={dis}m")
        
        # Linear regression: distance = A / (depth + B)
        # Simple approach: fit distance = scale / depth
        # Scale = mean(distance * depth)
        
        scales = distances / (depths + 0.001)
        self.depth_scale = np.mean(scales)
        
        print(f"\n[OK] Calculated scales: {scales}")
        print(f"[OK] Final depth_scale: {self.depth_scale:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    calibrator = DistanceCalibrator(device=args.device)
    calibrator.run()
    
    # Save scale
    if calibrator.depth_scale > 1.0:
        print(f"\n[RESULT] Optimal depth_scale: {calibrator.depth_scale:.4f}")
        print(f"[TIP] Gunakan parameter: --scale {calibrator.depth_scale:.2f}")
        print(f"[TIP] Contoh: python realtime_yolo_depth_distance.py --scale {calibrator.depth_scale:.2f}")


if __name__ == "__main__":
    main()
