"""
realtime_onnx_yolo_depth.py
===========================
ONNX Runtime untuk inference 2-3x lebih cepat dari PyTorch
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from collections import deque

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("[WARNING] onnxruntime not installed. Install: pip install onnxruntime")
    ONNX_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ONNXDepthInference:
    def __init__(self, model_path):
        """Load ONNX model untuk depth inference"""
        print(f"[INFO] Loading ONNX model: {model_path}")
        
        # Use CPU execution
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"[OK] ONNX model loaded")
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_name}")
    
    def preprocess(self, frame, size=512):
        """Preprocess frame untuk ONNX input"""
        img = cv2.resize(frame, (size, size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)  # Add batch dimension
        return img
    
    def infer(self, frame, size=512):
        """Run inference"""
        input_data = self.preprocess(frame, size)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        depth = outputs[0][0, 0]  # Extract single depth map
        return depth


class RealtimeONNXProcessor:
    def __init__(self, use_yolo=True, frame_skip=1, inference_size=384):
        self.use_yolo = use_yolo
        self.frame_skip = frame_skip
        self.inference_size = inference_size
        
        self.depth_model = None
        self.yolo_model = None
        self.cap = None
        
        print(f"[CONFIG] Frame Skip: {frame_skip} | Size: {inference_size} | YOLO: {use_yolo}")
    
    def load_models(self):
        """Load ONNX Depth + YOLO"""
        # Check ONNX model exists
        onnx_path = Path('models/onnx/depth_anything_v2_vits.onnx')
        if not onnx_path.exists():
            print(f"[ERROR] ONNX model not found: {onnx_path}")
            print("[INFO] Using PyTorch fallback...")
            return False
        
        if ONNX_AVAILABLE:
            try:
                self.depth_model = ONNXDepthInference(str(onnx_path))
            except Exception as e:
                print(f"[WARNING] ONNX load failed: {e}")
                return False
        else:
            print("[WARNING] onnxruntime not available")
            return False
        
        # Load YOLO
        if self.use_yolo and YOLO_AVAILABLE:
            try:
                print("[INFO] Loading YOLO v12n...")
                self.yolo_model = YOLO('yolov12n.pt')
                print("[OK] YOLO loaded")
            except:
                print("[WARNING] YOLO load failed")
                self.use_yolo = False
        
        return True
    
    def init_camera(self):
        """Initialize camera"""
        print("[INFO] Opening camera...")
        
        for backend_id in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            self.cap = cv2.VideoCapture(0, backend_id)
            if self.cap.isOpened():
                break
        
        if not self.cap.isOpened():
            print("[ERROR] Camera failed")
            return False
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("[OK] Camera ready")
        return True
    
    def process_frame(self, frame):
        """Process frame with ONNX"""
        # ONNX depth inference (much faster!)
        depth = self.depth_model.infer(frame, self.inference_size)
        
        # Normalize
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth)
        
        depth_colored = cv2.applyColorMap(
            (depth_norm * 255).astype(np.uint8),
            cv2.COLORMAP_MAGMA
        )
        
        # YOLO
        display_frame = frame.copy()
        if self.yolo_model:
            try:
                results = self.yolo_model(frame, verbose=False)
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
        
        return display_frame, depth_colored
    
    def run(self):
        """Main loop"""
        if not self.load_models():
            print("[ERROR] Model loading failed")
            return
        
        if not self.init_camera():
            print("[ERROR] Camera init failed")
            return
        
        print("\n" + "="*70)
        print("REAL-TIME ONNX OPTIMIZED (2-3x Faster!)")
        print("="*70)
        print("Commands: Q=Quit | 1=YOLO | +/-=Skip | S/D=Size")
        print("="*70 + "\n")
        
        fps_history = deque(maxlen=30)
        frame_count = 0
        proc_count = 0
        
        try:
            while True:
                start = time.time()
                frame_count += 1
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if (frame_count - 1) % self.frame_skip != 0:
                    continue
                
                proc_count += 1
                display_frame, depth_colored = self.process_frame(frame)
                
                h, w = display_frame.shape[:2]
                depth_r = cv2.resize(depth_colored, (w, h))
                combined = np.hstack([display_frame, depth_r])
                
                elapsed = time.time() - start
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_history.append(fps)
                avg_fps = np.mean(fps_history)
                
                cv2.putText(combined, f"FPS: {avg_fps:.1f} (ONNX)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(combined, f"Total: {frame_count} | Proc: {proc_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(combined, f"Time: {elapsed*1000:.1f}ms", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("ONNX Optimized (L:RGB | R:Depth)", combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('1'):
                    self.use_yolo = not self.use_yolo
                    print(f"YOLO: {'ON' if self.use_yolo else 'OFF'}")
                
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[OK] Done")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', action='store_true', default=False)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--size', type=int, default=384)
    
    args = parser.parse_args()
    
    if not ONNX_AVAILABLE:
        print("[ERROR] onnxruntime not installed!")
        print("[INFO] Install: pip install onnxruntime")
        return
    
    processor = RealtimeONNXProcessor(
        use_yolo=args.yolo,
        frame_skip=args.skip,
        inference_size=args.size
    )
    processor.run()


if __name__ == "__main__":
    main()
