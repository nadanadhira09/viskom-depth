"""
realtime_yolo_depth_distance.py
================================
Real-time YOLO + Depth dengan menampilkan JARAK dalam METER
Setiap bounding box menunjukkan jarak dari kamera
"""

import cv2
import torch
import numpy as np
import sys
import time
from pathlib import Path
from collections import deque
import threading

# Add Depth Anything V2 path
REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_DIR))

from depth_anything_v2.dpt import DepthAnythingV2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARNING] ultralytics not installed")
    YOLO_AVAILABLE = False


class RealtimeProcessorWithDistance:
    def __init__(self, use_yolo=True, frame_skip=1, inference_size=384, device='cpu'):
        """
        Args:
            use_yolo: Enable YOLO detection
            frame_skip: Process every Nth frame
            inference_size: Depth inference resolution
            device: 'cpu' or 'cuda'
        """
        self.use_yolo = use_yolo
        self.frame_skip = frame_skip
        self.inference_size = inference_size
        self.device = device
        
        self.depth_model = None
        self.yolo_model = None
        self.cap = None
        
        # Calibration params untuk konversi depth ke meter
        # Disesuaikan berdasarkan kalibrasi kamera
        self.depth_scale = 10.0  # Faktor skala (meter)
        self.max_distance = 8.0  # Jarak maksimal yang ditampilkan
        
        print(f"[CONFIG] Frame Skip: {frame_skip} | Inference Size: {inference_size} | YOLO: {use_yolo}")
        print(f"[CONFIG] Depth Scale: {self.depth_scale}m | Max Distance: {self.max_distance}m")
    
    def load_models(self):
        """Load Depth + YOLO models"""
        print("\n[INFO] Loading Depth Anything V2 (vits)...")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model.load_state_dict(
            torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu')
        )
        self.depth_model = self.depth_model.to(self.device).eval()
        print("[OK] Depth model ready")
        
        # Load YOLO
        self.yolo_model = None
        if self.use_yolo and YOLO_AVAILABLE:
            try:
                print("[INFO] Loading YOLO v12n...")
                self.yolo_model = YOLO('yolov12n.pt')
                print("[OK] YOLO model ready")
            except Exception as e:
                print(f"[WARNING] YOLO failed: {e}")
                self.use_yolo = False
    
    def init_camera(self):
        """Initialize camera"""
        print("\n[INFO] Opening camera...")
        
        for backend_id in [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]:
            self.cap = cv2.VideoCapture(0, backend_id)
            
            if self.cap.isOpened():
                break
        
        if not self.cap.isOpened():
            print("[ERROR] Could not open camera")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("[OK] Camera ready")
        return True
    
    def get_distance_from_bbox(self, depth_map, x1, y1, x2, y2):
        """
        Ambil jarak dari depth map di area bounding box
        Gunakan nilai maximum dari area untuk estimasi (bagian terdekat dari objek)
        
        Args:
            depth_map: Depth map (normalized 0-1)
                      1.0 = foreground/dekat
                      0.0 = background/jauh
            x1, y1, x2, y2: Bounding box coordinates
        
        Returns:
            distance_meter: Jarak dalam meter
        """
        h, w = depth_map.shape
        
        # Scale bbox ke dimensi depth map
        scale_x = w / 640
        scale_y = h / 480
        
        bx1 = max(0, int(x1 * scale_x))
        by1 = max(0, int(y1 * scale_y))
        bx2 = min(w, int(x2 * scale_x))
        by2 = min(h, int(y2 * scale_y))
        
        # Extract depth region
        if bx1 >= bx2 or by1 >= by2:
            return None
        
        depth_region = depth_map[by1:by2, bx1:bx2]
        
        if depth_region.size == 0:
            return None
        
        # Ambil nilai MAXIMUM (bagian terdekat dalam ROI)
        depth_value = np.max(depth_region)
        
        # Formula: distance = scale * (1 - depth_value)
        # Jika depth_value = 1.0 (dekat) → distance = 0
        # Jika depth_value = 0.0 (jauh) → distance = scale
        
        distance = self.depth_scale * (1.0 - depth_value)
        
        # Clip ke range yang masuk akal
        distance = np.clip(distance, 0.2, self.max_distance)
        
        return distance
    
    def process_frame(self, frame):
        """Process single frame: Depth + YOLO dengan distance"""
        with torch.no_grad():
            depth = self.depth_model.infer_image(frame, input_size=self.inference_size)
        
        # Normalize depth
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth)
        
        # JANGAN inverse! depth_normalized sekarang:
        # - mendekati 1.0 = foreground/dekat
        # - mendekati 0.0 = background/jauh
        
        depth_colored = cv2.applyColorMap(
            (depth_normalized * 255).astype(np.uint8),
            cv2.COLORMAP_MAGMA
        )
        
        # YOLO detection
        yolo_results = None
        if self.yolo_model is not None:
            try:
                yolo_results = self.yolo_model(frame, verbose=False)
            except:
                pass
        
        # Draw detections dengan distance
        display_frame = frame.copy()
        if yolo_results and len(yolo_results) > 0:
            result = yolo_results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = result.names[cls]
                    
                    # Ambil jarak dari depth map
                    distance = self.get_distance_from_bbox(depth_normalized, x1, y1, x2, y2)
                    
                    # Gambar bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Label dengan jarak dalam meter
                    if distance is not None:
                        label_text = f"{label} {distance:.2f}m"
                        color = (0, 255, 0)  # Green
                        
                        # Color coding berdasarkan jarak
                        if distance < 1.0:
                            color = (0, 0, 255)  # Red: sangat dekat
                        elif distance < 2.0:
                            color = (0, 165, 255)  # Orange: dekat
                        elif distance < 4.0:
                            color = (0, 255, 255)  # Yellow: sedang
                        else:
                            color = (0, 255, 0)  # Green: jauh
                    else:
                        label_text = f"{label} N/A"
                        color = (255, 255, 255)
                    
                    # Draw text background
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(display_frame, 
                                 (x1, max(20, y1-text_size[1]-5)),
                                 (x1+text_size[0], max(20, y1-5)),
                                 color, -1)
                    
                    # Draw text
                    cv2.putText(display_frame, label_text, (x1, max(20, y1-5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return display_frame, depth_colored, depth_normalized
    
    def run(self):
        """Main loop"""
        self.load_models()
        
        if not self.init_camera():
            return
        
        print("\n" + "="*70)
        print("REAL-TIME YOLO + DEPTH DISTANCE ESTIMATION")
        print("="*70)
        print("Commands:")
        print("  Q/ESC = Quit")
        print("  1 = Toggle YOLO")
        print("  +/- = Adjust frame skip")
        print("  S/D = Adjust inference size")
        print("  [/] = Adjust depth scale")
        print("="*70 + "\n")
        
        fps_history = deque(maxlen=30)
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                start_time = time.time()
                frame_count += 1
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Frame skipping
                if (frame_count - 1) % self.frame_skip != 0:
                    continue
                
                processed_count += 1
                display_frame, depth_colored, depth_map = self.process_frame(frame)
                
                # Combine views
                h, w = display_frame.shape[:2]
                depth_resized = cv2.resize(depth_colored, (w, h))
                combined = np.hstack([display_frame, depth_resized])
                
                # FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_history.append(fps)
                avg_fps = np.mean(fps_history)
                
                # Add overlay info
                cv2.putText(combined, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(combined, f"Frame: {frame_count} | Proc: {processed_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(combined, f"Time: {elapsed*1000:.1f}ms", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                mode_text = f"Skip:{self.frame_skip} | Size:{self.inference_size} | Scale:{self.depth_scale}m"
                cv2.putText(combined, mode_text, (10, combined.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
                
                # Distance legend
                cv2.putText(combined, "Red:<1m | Orange:1-2m | Yellow:2-4m | Green:>4m", 
                           (10, combined.shape[0]-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Display
                cv2.imshow("YOLO + Distance (L:RGB | R:Depth)", combined)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[INFO] User quit")
                    break
                elif key == ord('1'):
                    self.use_yolo = not self.use_yolo
                    print(f"[INFO] YOLO: {'ON' if self.use_yolo else 'OFF'}")
                elif key == ord('+') or key == ord('='):
                    self.frame_skip = min(10, self.frame_skip + 1)
                    print(f"[INFO] Frame Skip: {self.frame_skip}")
                elif key == ord('-'):
                    self.frame_skip = max(1, self.frame_skip - 1)
                    print(f"[INFO] Frame Skip: {self.frame_skip}")
                elif key == ord('s'):
                    self.inference_size = max(256, self.inference_size - 128)
                    print(f"[INFO] Inference Size: {self.inference_size}")
                elif key == ord('d'):
                    self.inference_size = min(768, self.inference_size + 128)
                    print(f"[INFO] Inference Size: {self.inference_size}")
                elif key == ord('['):
                    self.depth_scale = max(1.0, self.depth_scale - 0.5)
                    print(f"[INFO] Depth Scale: {self.depth_scale}m")
                elif key == ord(']'):
                    self.depth_scale = min(20.0, self.depth_scale + 0.5)
                    print(f"[INFO] Depth Scale: {self.depth_scale}m")
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n[OK] Done")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', action='store_true', default=True,
                       help='Enable YOLO detection (default: ON)')
    parser.add_argument('--no-yolo', dest='yolo', action='store_false',
                       help='Disable YOLO detection')
    parser.add_argument('--skip', type=int, default=2,
                       help='Process every Nth frame (1=all, 2=every 2nd, etc)')
    parser.add_argument('--size', type=int, default=384,
                       help='Inference resolution (256-768)')
    parser.add_argument('--scale', type=float, default=10.0,
                       help='Depth scale in meters')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    processor = RealtimeProcessorWithDistance(
        use_yolo=args.yolo,
        frame_skip=args.skip,
        inference_size=args.size,
        device=args.device
    )
    processor.depth_scale = args.scale
    processor.run()


if __name__ == "__main__":
    main()
