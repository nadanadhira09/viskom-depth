"""
realtime_onnx_accurate.py
=========================
Menggunakan ONNX FULL PRECISION model (lebih akurat, depth lebih detail!)
"""

import cv2
import numpy as np
import sys
import time
import matplotlib
from pathlib import Path
from collections import deque

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("[ERROR] onnxruntime not installed")
    ONNX_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ONNXAccurateDepth:
    def __init__(self, model_path='models/onnx/depth_anything_v2_vits.onnx'):
        """Load ONNX model FULL PRECISION (lebih akurat!)"""
        print(f"[INFO] Loading ONNX: {model_path}")
        
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load colormap sama seperti official GitHub repo
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        print(f"[OK] Model loaded (full precision)")
    
    def preprocess(self, frame, size=512):
        """Preprocess untuk ONNX"""
        img = cv2.resize(frame, (size, size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img
    
    def infer(self, frame, size=518):
        """Run inference"""
        input_data = self.preprocess(frame, size)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        depth = outputs[0][0, 0]
        return depth


def run_realtime(skip=1):
    """Main real-time loop"""
    
    print("="*70)
    print("REAL-TIME ONNX FULL PRECISION (Accurate Depth!)")
    print("="*70)
    
    # Check model exists
    model_path = Path('models/onnx/depth_anything_v2_vits.onnx')
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    # Load models
    print(f"\n[INFO] Loading models...")
    depth_model = ONNXAccurateDepth(str(model_path))
    
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            print("[INFO] Loading YOLO v12n...")
            yolo_model = YOLO('yolov12n.pt')
            print("[OK] YOLO ready")
        except:
            pass
    
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
    
    print("\n" + "="*70)
    print("Commands: Q=Quit | 1=YOLO Toggle")
    print("="*70 + "\n")
    
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
            
            # Depth inference (ONNX Full Precision - lebih akurat!)
            depth = depth_model.infer(frame, size=518)
            
            # Normalize depth dengan percentile untuk contrast lebih baik
            depth_min = np.percentile(depth, 2)  # Remove outliers
            depth_max = np.percentile(depth, 98)
            
            if depth_max > depth_min:
                depth_norm = np.clip((depth - depth_min) / (depth_max - depth_min), 0, 1)
            else:
                depth_norm = np.zeros_like(depth)
            
            # Apply colormap Spectral_r (official GitHub reference)
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
            cv2.putText(combined, f"FPS: {avg_fps:.1f} (ONNX Full Precision)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(combined, f"Frame: {frame_count:4d} | Proc: {proc_count:4d}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(combined, f"Time: {elapsed*1000:.1f}ms | Mode: {'YOLO+' if use_yolo else ''}Depth | Accurate", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            
            cv2.imshow("ONNX Full Precision (Accurate!) - Left: RGB | Right: Depth", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[INFO] User quit")
                break
            elif key == ord('1'):
                use_yolo = not use_yolo
                print(f"[INFO] YOLO: {'ON' if use_yolo else 'OFF'}")
    
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
    parser.add_argument('--skip', type=int, default=1, help='Frame skip')
    args = parser.parse_args()
    
    if not ONNX_AVAILABLE:
        print("[ERROR] onnxruntime required!")
        print("Install: pip install onnxruntime")
        sys.exit(1)
    
    run_realtime(skip=args.skip)
