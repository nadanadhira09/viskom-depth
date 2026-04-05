"""
realtime_pytorch_accurate.py
=============================
Menggunakan PyTorch model langsung (PALING AKURAT!)
Dengan encoder yang lebih besar untuk depth lebih detail
"""

import cv2
import numpy as np
import sys
import time
import torch
import matplotlib
from pathlib import Path
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Import model
sys.path.insert(0, 'Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2


class PyTorchDepth:
    def __init__(self, encoder='vitb', input_size=518):
        """Load PyTorch model (paling akurat!)"""
        print(f"[INFO] Loading PyTorch Depth Anything V2 ({encoder})...")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {self.device}")
        
        # Model config
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        if encoder not in model_configs:
            print(f"[ERROR] Unknown encoder: {encoder}")
            encoder = 'vitb'
        
        # Check checkpoint exists
        checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
        if not Path(checkpoint_path).exists():
            print(f"[WARN] Checkpoint not found: {checkpoint_path}")
            print(f"[INFO] Available: depth_anything_v2_vits.pth")
            encoder = 'vits'
            checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
        
        # Create and load model
        self.model = DepthAnythingV2(**model_configs[encoder])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device).eval()
        
        self.input_size = input_size
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        print(f"[OK] Model loaded with {encoder} encoder")
    
    def infer(self, frame):
        """Run inference"""
        with torch.no_grad():
            depth = self.model.infer_image(frame, self.input_size)
        return depth


def run_realtime(encoder='vitb', skip=1):
    """Main real-time loop"""
    
    print("="*80)
    print(f"REAL-TIME PyTorch Depth Anything V2 - {encoder.upper()} Encoder (PALING AKURAT!)")
    print("="*80)
    
    # Load models
    print(f"\n[INFO] Loading models...")
    depth_model = PyTorchDepth(encoder=encoder)
    
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            print("[INFO] Loading YOLO v12n...")
            yolo_model = YOLO('yolov12n.pt')
            print("[OK] YOLO ready")
        except Exception as e:
            print(f"[WARN] YOLO load failed: {e}")
    
    # Open camera
    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Camera failed")
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[OK] Camera ready")
    
    print("\n" + "="*80)
    print("Commands: Q=Quit | 1=YOLO Toggle | 2=Save Frame")
    print("="*80 + "\n")
    
    fps_history = deque(maxlen=30)
    frame_count = 0
    proc_count = 0
    use_yolo = False
    
    try:
        while True:
            start = time.time()
            frame_count += 1
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame skip
            if (frame_count - 1) % skip != 0:
                continue
            
            proc_count += 1
            
            # Depth inference (PyTorch - paling akurat!)
            depth = depth_model.infer(frame)
            
            # Normalize menggunakan percentile untuk hasil yang lebih baik
            depth_min = np.percentile(depth, 2)
            depth_max = np.percentile(depth, 98)
            
            if depth_max > depth_min:
                depth_norm = np.clip((depth - depth_min) / (depth_max - depth_min), 0, 1)
            else:
                depth_norm = np.zeros_like(depth)
            
            # Apply colormap Spectral_r
            depth_colored = (depth_model.cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            # YOLO detection
            display_frame = frame.copy()
            if use_yolo and yolo_model:
                try:
                    results = yolo_model(frame, verbose=False)
                    if results and len(results) > 0:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            label = results[0].names[cls]
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except:
                    pass
            
            # Combine views
            h, w = display_frame.shape[:2]
            depth_r = cv2.resize(depth_colored, (w, h))
            combined = np.hstack([display_frame, depth_r])
            
            # Calculate FPS
            elapsed = time.time() - start
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            
            # Add overlay
            cv2.putText(combined, f"FPS: {avg_fps:.2f} | {encoder.upper()} | PyTorch", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(combined, f"Frame: {frame_count:5d} | Proc: {proc_count:5d}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(combined, f"Infer: {elapsed*1000:.1f}ms | Mode: {'YOLO+' if use_yolo else ''}Depth", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            
            device_info = f"GPU" if torch.cuda.is_available() else "CPU"
            cv2.putText(combined, f"{device_info} | Akurat", (w-200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 1)
            
            cv2.imshow(f"PyTorch Depth Anything V2 ({encoder}) - Left: RGB | Right: Depth", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[INFO] User quit")
                break
            elif key == ord('1'):
                use_yolo = not use_yolo
                print(f"[INFO] YOLO: {'ON' if use_yolo else 'OFF'}")
            elif key == ord('2'):
                # Save depth map
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"depth_{timestamp}.png", depth_colored)
                cv2.imwrite(f"combined_{timestamp}.png", combined)
                print(f"[OK] Saved depth_{timestamp}.png and combined_{timestamp}.png")
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='vitb', 
                       choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='Model encoder (default: vitb for balance; vitl for best quality)')
    parser.add_argument('--skip', type=int, default=1, help='Frame skip (1=no skip)')
    args = parser.parse_args()
    
    print(f"\n[INFO] Encoder tips:")
    print(f"  - vits: Fastest (64 features)")
    print(f"  - vitb: Balance speed/accuracy (128 features) <- RECOMMENDED")
    print(f"  - vitl: Best quality, slower (256 features)")
    print(f"  - vitg: Highest quality, slowest (384 features)")
    print(f"\n[INFO] Using: {args.encoder}\n")
    
    run_realtime(encoder=args.encoder, skip=args.skip)
