"""
start_calibration.py
====================
Helper script untuk memudahkan proses kalibrasi.
Akan otomatis mendeteksi kamera dan menjalankan kalibrasi.

Cara pakai:
    python conversion/start_calibration.py
"""

import sys
import os
import subprocess
from pathlib import Path

def print_header(text):
    """Print header dengan border."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_opencv():
    """Cek apakah OpenCV terinstall."""
    try:
        import cv2
        print(f"✓ OpenCV terdeteksi: v{cv2.__version__}")
        return True
    except ImportError:
        print("✗ OpenCV tidak terinstall!")
        print("  Jalankan: pip install opencv-python")
        return False

def check_torch():
    """Cek apakah PyTorch terinstall."""
    try:
        import torch
        print(f"✓ PyTorch terdeteksi: v{torch.__version__}")
        return True
    except ImportError:
        print("✗ PyTorch tidak terinstall!")
        return False

def detect_camera():
    """Deteksi kamera yang tersedia."""
    try:
        import cv2
        print("\n[INFO] Mendeteksi kamera...")
        
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append((i, width, height))
                print(f"  ✓ Camera {i}: TERSEDIA ({width}x{height})")
                cap.release()
        
        if not available_cameras:
            print("  ✗ Tidak ada kamera yang terdeteksi!")
            print("\n[SOLUSI]")
            print("  1. Pastikan webcam USB sudah tercolok")
            print("  2. Tutup aplikasi yang menggunakan webcam (Zoom, Teams, dll)")
            print("  3. Cek permission kamera di Windows Settings")
            return None
        
        return available_cameras[0][0]  # Return index kamera pertama
    
    except Exception as e:
        print(f"  ✗ Error saat deteksi kamera: {e}")
        return None

def check_checkpoint():
    """Cek apakah model checkpoint tersedia."""
    checkpoint = Path("checkpoints/depth_anything_v2_vits.pth")
    if checkpoint.exists():
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        print(f"✓ Model checkpoint: {checkpoint} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"✗ Model checkpoint tidak ditemukan: {checkpoint}")
        print("  Jalankan: python download_assets.py")
        return False

def run_calibration(camera_index):
    """Jalankan script kalibrasi."""
    print_header("MEMULAI KALIBRASI")
    
    print("\n[INSTRUKSI]")
    print("  1. Siapkan objek datar (papan/dinding/buku besar)")
    print("  2. Siapkan meteran untuk mengukur jarak")
    print("  3. Jendela preview akan muncul dengan kotak hijau ROI")
    print("  4. Letakkan objek pada jarak yang diminta")
    print("  5. Tekan SPASI untuk capture")
    print("  6. Tekan ESC untuk keluar/batal")
    
    print("\n[JARAK KALIBRASI]")
    print("  • 1.0 meter (ukur dengan meteran)")
    print("  • 2.0 meter")
    print("  • 3.0 meter")
    
    input("\nTekan ENTER untuk melanjutkan...")
    
    # Build command
    cmd = [
        sys.executable,
        "conversion/calibrate_depth.py",
        "--mode", "interactive",
        "--encoder", "vits",
        "--checkpoint", "checkpoints/depth_anything_v2_vits.pth",
        "--distances", "1.0", "2.0", "3.0"
    ]
    
    if camera_index != 0:
        cmd.extend(["--camera", str(camera_index)])
    
    print(f"\n[CMD] {' '.join(cmd)}\n")
    
    # Run kalibrasi
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\n[INFO] Kalibrasi dibatalkan oleh user")
        return False
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return False

def main():
    print_header("SETUP KALIBRASI DEPTH ANYTHING V2")
    
    # Check dependencies
    print("\n[STEP 1] Mengecek dependencies...")
    if not check_opencv() or not check_torch():
        print("\n[ERROR] Dependencies tidak lengkap!")
        sys.exit(1)
    
    # Check checkpoint
    print("\n[STEP 2] Mengecek model checkpoint...")
    if not check_checkpoint():
        print("\n[ERROR] Model checkpoint tidak ditemukan!")
        sys.exit(1)
    
    # Detect camera
    print("\n[STEP 3] Mendeteksi kamera...")
    camera_index = detect_camera()
    
    if camera_index is None:
        print("\n[ERROR] Tidak dapat melanjutkan tanpa kamera!")
        print("\n[ALTERNATIF] Gunakan mode images jika punya foto:")
        print("  python conversion/calibrate_depth.py --mode images \\")
        print("    --images foto_1m.jpg foto_2m.jpg foto_3m.jpg \\")
        print("    --distances 1.0 2.0 3.0")
        sys.exit(1)
    
    print(f"\n✓ Akan menggunakan Camera {camera_index}")
    
    # Run kalibrasi
    success = run_calibration(camera_index)
    
    if success:
        print_header("KALIBRASI SELESAI")
        print("\n[OUTPUT FILES]")
        print("  • conversion/calibration_result.txt")
        print("  • conversion/calibration_result.json")
        print("\n[LANGKAH SELANJUTNYA]")
        print("  1. Lihat konstanta A di file .txt")
        print("  2. Update conversion/depth_config.h dengan konstanta A")
        print("  3. Test akurasi: python conversion/test_calibration.py --calibration conversion/calibration_result.json --plot")
        print("  4. Real-time viz: python conversion/realtime_distance.py --calibration conversion/calibration_result.json")
    else:
        print_header("KALIBRASI GAGAL ATAU DIBATALKAN")
        print("\nJika ada error, coba:")
        print("  • Pastikan webcam tidak digunakan aplikasi lain")
        print("  • Restart terminal dan coba lagi")
        print("  • Gunakan mode images sebagai alternatif")

if __name__ == "__main__":
    main()
