"""
realtime_yolo_depth.py
======================
Simple real-time YOLO object detection + Depth Anything V2 inference
Dengan mode monitoring yang lebih robust.
"""

import cv2
import torch
import numpy as np
import sys
from pathlib import Path
import time
from collections import deque

# Add Depth Anything V2 path
REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_DIR))

from depth_anything_v2.dpt import DepthAnythingV2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARNING] ultralytics not installed, continuing without YOLO")
    YOLO_AVAILABLE = False


def main():
    print("="*70)
    print("INITIALIZING REAL-TIME YOLO + DEPTH INFERENCE")
    print("="*70)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[INFO] Device: {device}")
    
    # Load Depth model
    print("[INFO] Loading Depth Anything V2 (vits)...")
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }
    
    depth_model = DepthAnythingV2(**model_configs['vits'])
    depth_model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
    depth_model = depth_model.to(device).eval()
    print("[OK] Depth model loaded")
    
    # Load YOLO model
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            print("[INFO] Loading YOLO v12n...")
            yolo_model = YOLO('yolov12n.pt')
            print("[OK] YOLO model loaded")
        except Exception as e:
            print(f"[WARNING] YOLO load failed: {e}")
    
    # Initialize camera with timeout
    print("\n[INFO] Opening camera (index 0)...")
    
    # Try camera with various backends
    for backend_id in [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]:
        print(f"  Trying backend: {backend_id}...", end=" ", flush=True)
        cap = cv2.VideoCapture(0, backend_id)
        
        if cap.isOpened():
            print("OK")
            break
        print("FAILED")
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("[OK] Camera ready")
    
    print("\n" + "="*70)
    print("REAL-TIME YOLO + DEPTH ESTIMATION")
    print("="*70)
    print("Commands: Q=Quit | R=Reset | S=Save Frame")
    print("="*70 + "\n")
    
    # FPS tracking
    fps_history = deque(maxlen=30)
    frame_count = 0
    
    try:
        while True:
            start_time = time.time()
            frame_count += 1
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            # Infer depth
            with torch.no_grad():
                depth = depth_model.infer_image(frame, input_size=518)
            
            # Normalize depth for visualization
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:
                depth_normalized = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(depth)
            
            depth_colored = cv2.applyColorMap(
                (depth_normalized * 255).astype(np.uint8),
                cv2.COLORMAP_MAGMA
            )
            
            # Run YOLO detection
            yolo_results = None
            if yolo_model is not None:
                try:
                    yolo_results = yolo_model(frame, verbose=False)
                except Exception as e:
                    print(f"[WARNING] YOLO inference failed: {e}")
            
            # Draw detections
            display_frame = frame.copy()
            if yolo_results and len(yolo_results) > 0:
                result = yolo_results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = result.names[cls]
                        
                        text = f"{label} {conf:.2f}"
                        
                        # Bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, text, (x1, max(20, y1-5)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Combine views
            h, w = display_frame.shape[:2]
            depth_colored_resized = cv2.resize(depth_colored, (w, h))
            combined = np.hstack([display_frame, depth_colored_resized])
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            
            # Add FPS info
            cv2.putText(combined, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, f"Frame: {frame_count}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Real-Time YOLO + Depth (Left: RGB | Right: Depth)", combined)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("\n[INFO] User quit")
                break
            elif key == ord('r'):  # Reset
                print("[INFO] Reset (frame counter)")
                frame_count = 0
            elif key == ord('s'):  # Save
                filename = f"frame_{frame_count:04d}.png"
                cv2.imwrite(filename, combined)
                print(f"[OK] Saved: {filename}")
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[OK] Done")


if __name__ == "__main__":
    main()
