"""
realtime_yolo_depth_optimized.py
================================
Optimized untuk FPS tinggi dengan multiple options:
- Frame skipping
- Resolution control
- YOLO on/off toggle
- Threading optimization
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


class RealtimeProcessor:
    def __init__(self, use_yolo=True, frame_skip=1, inference_size=384, device='cpu'):
        """
        Args:
            use_yolo: Enable YOLO detection
            frame_skip: Process every Nth frame (1=all, 2=every other)
            inference_size: Depth inference resolution (smaller = faster)
            device: 'cpu' or 'cuda'
        """
        self.use_yolo = use_yolo
        self.frame_skip = frame_skip
        self.inference_size = inference_size
        self.device = device
        
        self.depth_model = None
        self.yolo_model = None
        self.cap = None
        
        self.frame_buffer = None
        self.lock = threading.Lock()
        
        print(f"[CONFIG] Frame Skip: {frame_skip} | Inference Size: {inference_size} | YOLO: {use_yolo}")
    
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
        """Initialize camera with best backend"""
        print("[INFO] Opening camera...")
        
        for backend_id in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            self.cap = cv2.VideoCapture(0, backend_id)
            if self.cap.isOpened():
                print(f"[OK] Camera opened (backend: {backend_id})")
                break
        
        if not self.cap.isOpened():
            print("[ERROR] Could not open camera")
            return False
        
        # Set low latency mode
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Single buffer for low latency
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def process_frame(self, frame):
        """Process single frame: Depth + optional YOLO"""
        with torch.no_grad():
            depth = self.depth_model.infer_image(frame, input_size=self.inference_size)
        
        # Normalize depth
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth)
        
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
        
        # Draw detections on frame
        display_frame = frame.copy()
        if yolo_results and len(yolo_results) > 0:
            result = yolo_results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = result.names[cls]
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, max(20, y1-5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return display_frame, depth_colored
    
    def run(self):
        """Main loop"""
        self.load_models()
        
        if not self.init_camera():
            return
        
        print("\n" + "="*70)
        print("REAL-TIME OPTIMIZED INFERENCE")
        print("="*70)
        print("Commands: Q=Quit | 1=Toggle YOLO | +/-=Frame Skip | S/D=Size")
        print(f"Current: Frame Skip={self.frame_skip} | Size={self.inference_size} | YOLO={'ON' if self.use_yolo else 'OFF'}")
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
                
                # Frame skipping logic
                if (frame_count - 1) % self.frame_skip != 0:
                    display_fps = np.mean(fps_history) if fps_history else 0
                    self._show_skipped_frame(display_fps, frame_count, processed_count)
                    continue
                
                # Process frame
                processed_count += 1
                display_frame, depth_colored = self.process_frame(frame)
                
                # Combine views
                h, w = display_frame.shape[:2]
                depth_resized = cv2.resize(depth_colored, (w, h))
                combined = np.hstack([display_frame, depth_resized])
                
                # FPS calculation
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_history.append(fps)
                avg_fps = np.mean(fps_history)
                
                # Add info overlay
                self._add_overlay(combined, avg_fps, frame_count, processed_count, elapsed)
                
                # Display
                cv2.imshow("Real-Time Optimized (L:RGB | R:Depth)", combined)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[INFO] User quit")
                    break
                elif key == ord('1'):  # Toggle YOLO
                    self.use_yolo = not self.use_yolo
                    print(f"[INFO] YOLO: {'ON' if self.use_yolo else 'OFF'}")
                elif key == ord('+') or key == ord('='):  # Increase frame skip
                    self.frame_skip = min(10, self.frame_skip + 1)
                    print(f"[INFO] Frame Skip: {self.frame_skip}")
                elif key == ord('-'):  # Decrease frame skip
                    self.frame_skip = max(1, self.frame_skip - 1)
                    print(f"[INFO] Frame Skip: {self.frame_skip}")
                elif key == ord('s'):  # Smaller size
                    self.inference_size = max(256, self.inference_size - 128)
                    print(f"[INFO] Inference Size: {self.inference_size}")
                elif key == ord('d'):  # Larger size
                    self.inference_size = min(768, self.inference_size + 128)
                    print(f"[INFO] Inference Size: {self.inference_size}")
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
        except Exception as e:
            print(f"\n[ERROR] {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n[OK] Done")
    
    def _show_skipped_frame(self, fps, frame_count, processed_count):
        """Show FPS for skipped frame"""
        pass
    
    def _add_overlay(self, img, fps, frame_count, processed_count, elapsed):
        """Add performance metrics overlay"""
        h, w = img.shape[:2]
        
        # FPS
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Frame info
        cv2.putText(img, f"Total: {frame_count} | Proc: {processed_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Time per frame
        cv2.putText(img, f"Time: {elapsed*1000:.1f}ms", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Mode
        mode = f"YOLO: {'ON' if self.use_yolo else 'OFF'} | Skip: {self.frame_skip} | Size: {self.inference_size}"
        cv2.putText(img, mode, (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', action='store_true', default=True,
                       help='Enable YOLO detection (default: ON)')
    parser.add_argument('--no-yolo', dest='yolo', action='store_false',
                       help='Disable YOLO detection')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth frame (1=all, 2=every 2nd, etc)')
    parser.add_argument('--size', type=int, default=384,
                       help='Inference resolution (lower=faster, 256-768)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    processor = RealtimeProcessor(
        use_yolo=args.yolo,
        frame_skip=args.skip,
        inference_size=args.size,
        device=args.device
    )
    processor.run()


if __name__ == "__main__":
    main()
