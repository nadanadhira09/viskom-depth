"""
auto_calibrate_distance.py
==========================
Otomatis analisis depth range dan recommend scale factor
"""

import cv2
import torch
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add Depth Anything V2 path
REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
sys.path.insert(0, str(REPO_DIR))

from depth_anything_v2.dpt import DepthAnythingV2


class AutoCalibrator:
    def __init__(self, device='cpu'):
        self.device = device
        self.depth_model = None
        self.cap = None
    
    def load_model(self):
        """Load Depth Anything V2"""
        print("[INFO] Loading Depth Anything V2...")
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
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("[ERROR] Camera failed")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("[OK] Camera ready")
        return True
    
    def gather_depth_samples(self, num_frames=100):
        """Gather depth samples dari berbagai posisi"""
        print(f"\n[INFO] Gathering {num_frames} depth samples...")
        print("[TIP] Arahkan kamera ke berbagai jarak untuk sampling")
        
        all_depths = []
        centers = []  # Depth di center frame (kursi depan)
        
        for frame_idx in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            with torch.no_grad():
                depth = self.depth_model.infer_image(frame, input_size=384)
            
            # Normalize
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max > depth_min:
                depth_norm = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = np.zeros_like(depth)
            
            # Inverse
            depth_norm = 1.0 - depth_norm
            
            all_depths.extend(depth_norm.flatten())
            
            # Center region (closest object presumably)
            h, w = depth_norm.shape
            center_region = depth_norm[h//4:3*h//4, w//4:3*w//4]
            centers.append(np.median(center_region))
            
            # Progress
            if (frame_idx + 1) % 10 == 0:
                print(f"  {frame_idx + 1}/{num_frames} frames")
            
            cv2.imshow("Sampling...", cv2.resize(
                cv2.applyColorMap((depth_norm*255).astype(np.uint8), cv2.COLORMAP_MAGMA),
                (400, 300)
            ))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        return np.array(all_depths), np.array(centers)
    
    def analyze_and_recommend(self):
        """Analyze depth distribution dan recommend scale"""
        self.load_model()
        
        if not self.init_camera():
            return
        
        all_depths, centers = self.gather_depth_samples(150)
        
        print("\n" + "="*70)
        print("DEPTH ANALYSIS RESULTS")
        print("="*70)
        
        print(f"\nAll depths statistics:")
        print(f"  Min:    {all_depths.min():.4f}")
        print(f"  Max:    {all_depths.max():.4f}")
        print(f"  Mean:   {all_depths.mean():.4f}")
        print(f"  Median: {np.median(all_depths):.4f}")
        print(f"  Std:    {all_depths.std():.4f}")
        
        print(f"\nCenter regions (closest objects) statistics:")
        print(f"  Min:    {centers.min():.4f}")
        print(f"  Max:    {centers.max():.4f}")
        print(f"  Mean:   {centers.mean():.4f}")
        print(f"  Median: {np.median(centers):.4f}")
        
        # Recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS FOR DEPTH SCALE")
        print("="*70)
        
        close_depth = np.percentile(centers, 10)  # Closest 10%
        far_depth = np.percentile(all_depths, 90)  # Farthest 10%
        
        # Assume close objects are ~0.5m, far objects are ~5m
        scale_from_close = 0.5 / (close_depth + 0.01)
        scale_from_far = 5.0 / (far_depth + 0.01)
        
        recommended_scale = (scale_from_close + scale_from_far) / 2
        
        print(f"\nAssuming:")
        print(f"  - Closest objects: ~0.5m (depth={close_depth:.3f})")
        print(f"  - Farthest objects: ~5.0m (depth={far_depth:.3f})")
        
        print(f"\nCalculated scales:")
        print(f"  From close: {scale_from_close:.2f}")
        print(f"  From far:   {scale_from_far:.2f}")
        print(f"  Recommended: {recommended_scale:.2f}")
        
        print("\n" + "="*70)
        print("SUGGESTED COMMANDS TO RUN")
        print("="*70)
        print(f"\nWith current scale ({recommended_scale:.2f}):")
        print(f"  python conversion/realtime_yolo_depth_distance.py --scale {recommended_scale:.2f}")
        
        # Try different scales
        print(f"\nOr try alternative scales:")
        for scale in [5.0, 8.0, 10.0, 12.0, 15.0]:
            print(f"  python conversion/realtime_yolo_depth_distance.py --scale {scale}")
        
        self.cap.release()


def main():
    calibrator = AutoCalibrator(device='cpu')
    calibrator.analyze_and_recommend()


if __name__ == "__main__":
    main()
