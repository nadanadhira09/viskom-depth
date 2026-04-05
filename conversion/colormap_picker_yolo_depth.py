"""
realtime_yolo_depth_colormap_picker.py
=======================================
YOLO + Depth dengan pilihan colormap untuk melihat mana yang cocok
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from collections import deque

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


COLORMAPS_CV = {
    'MAGMA': cv2.COLORMAP_MAGMA,
    'INFERNO': cv2.COLORMAP_INFERNO,
    'PLASMA': cv2.COLORMAP_PLASMA,
    'VIRIDIS': cv2.COLORMAP_VIRIDIS,
    'JET': cv2.COLORMAP_JET,
    'HOT': cv2.COLORMAP_HOT,
    'COOL': cv2.COLORMAP_COOL,
    'SPRING': cv2.COLORMAP_SPRING,
    'SUMMER': cv2.COLORMAP_SUMMER,
    'AUTUMN': cv2.COLORMAP_AUTUMN,
    'WINTER': cv2.COLORMAP_WINTER,
}


class YOLODepthColorMapPicker:
    def __init__(self):
        print("="*70)
        print("YOLO v12 + DEPTH COLORMAP PICKER")
        print("="*70)
        
        self.depth_model = None
        self.yolo_model = None
        self.cap = None
        self.current_colormap_idx = 0
        self.colormaps_list = list(COLORMAPS_CV.keys())
        self.current_colormap_name = self.colormaps_list[0]
        
        print(f"\nAvailable Colormaps: {', '.join(self.colormaps_list)}")
        print(f"Current: {self.current_colormap_name}")
        print("Controls:")
        print("  N = Next colormap")
        print("  P = Previous colormap")
        print("  Q/ESC = Quit")
        print("  G = Grayscale")
        print("  1 = Toggle YOLO")
    
    def load_models(self):
        """Load ONNX Depth + YOLO"""
        print("\n[INFO] Loading models...")
        
        # ONNX Depth (fast)
        if ort:
            try:
                print("[INFO] Loading ONNX Depth model...")
                session = ort.InferenceSession(
                    'models/onnx/depth_anything_v2_vits_quantized_int8.onnx',
                    providers=['CPUExecutionProvider']
                )
                self.depth_session = session
                print("[OK] ONNX Depth model loaded")
            except Exception as e:
                print(f"[WARNING] ONNX failed: {e}")
                self.depth_session = None
        
        # YOLO
        if YOLO_AVAILABLE:
            try:
                print("[INFO] Loading YOLO v12n...")
                self.yolo_model = YOLO('yolov12n.pt')
                print("[OK] YOLO loaded")
            except Exception as e:
                print(f"[WARNING] YOLO failed: {e}")
                self.yolo_model = None
    
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
    
    def infer_depth_onnx(self, frame):
        """ONNX depth inference"""
        img = cv2.resize(frame, (518, 518))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        
        input_name = self.depth_session.get_inputs()[0].name
        output_name = self.depth_session.get_outputs()[0].name
        outputs = self.depth_session.run([output_name], {input_name: img})
        depth = outputs[0][0, 0]
        
        return depth
    
    def process_frame(self, frame, use_yolo=True, use_grayscale=False):
        """Process frame"""
        # Depth inference
        depth = self.infer_depth_onnx(frame)
        
        # Normalize
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth).astype(np.uint8)
        
        # Colormap
        if use_grayscale:
            depth_colored = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
        else:
            colormap_id = COLORMAPS_CV[self.current_colormap_name]
            depth_colored = cv2.applyColorMap(depth_norm, colormap_id)
        
        # YOLO detection
        display_frame = frame.copy()
        if use_yolo and self.yolo_model:
            try:
                results = self.yolo_model(frame, verbose=False)
                if results and len(results) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = results[0].names[cls]
                        
                        # Green boxes untuk YOLO
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, max(20, y1-5)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except:
                pass
        
        return display_frame, depth_colored
    
    def run(self):
        """Main loop"""
        self.load_models()
        
        if not self.init_camera():
            return
        
        print("\n" + "="*70)
        print("REAL-TIME COLORMAP PICKER")
        print("="*70 + "\n")
        
        fps_history = deque(maxlen=30)
        frame_count = 0
        use_yolo = True
        use_grayscale = False
        
        try:
            while True:
                start = time.time()
                frame_count += 1
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process
                display_frame, depth_colored = self.process_frame(
                    frame, 
                    use_yolo=use_yolo,
                    use_grayscale=use_grayscale
                )
                
                # Combine views
                h, w = display_frame.shape[:2]
                depth_r = cv2.resize(depth_colored, (w, h))
                combined = np.hstack([display_frame, depth_r])
                
                # FPS
                elapsed = time.time() - start
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_history.append(fps)
                avg_fps = np.mean(fps_history)
                
                # Add overlay
                cv2.putText(combined, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Colormap info
                colormap_text = ("GRAYSCALE" if use_grayscale else 
                                self.current_colormap_name)
                cv2.putText(combined, f"Colormap: {colormap_text}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Status
                status_text = f"YOLO: {'ON' if use_yolo else 'OFF'} (1) | N/P: Colormap | Q: Quit"
                cv2.putText(combined, status_text, (10, h-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                cv2.imshow("YOLO + Depth Colormap Picker (L:RGB+YOLO | R:Depth)", combined)
                
                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[INFO] User quit")
                    break
                elif key == ord('n'):  # Next colormap
                    self.current_colormap_idx = (self.current_colormap_idx + 1) % len(self.colormaps_list)
                    self.current_colormap_name = self.colormaps_list[self.current_colormap_idx]
                    use_grayscale = False
                    print(f"[INFO] Colormap: {self.current_colormap_name}")
                elif key == ord('p'):  # Previous colormap
                    self.current_colormap_idx = (self.current_colormap_idx - 1) % len(self.colormaps_list)
                    self.current_colormap_name = self.colormaps_list[self.current_colormap_idx]
                    use_grayscale = False
                    print(f"[INFO] Colormap: {self.current_colormap_name}")
                elif key == ord('g'):  # Grayscale
                    use_grayscale = not use_grayscale
                    print(f"[INFO] Grayscale: {'ON' if use_grayscale else 'OFF'}")
                elif key == ord('1'):  # Toggle YOLO
                    use_yolo = not use_yolo
                    print(f"[INFO] YOLO: {'ON' if use_yolo else 'OFF'}")
        
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[OK] Done")


def main():
    picker = YOLODepthColorMapPicker()
    picker.run()


if __name__ == "__main__":
    main()
