import cv2

print("Mendeteksi kamera yang tersedia...\n")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera {i}: TERSEDIA ({int(width)}x{int(height)})")
        cap.release()
    else:
        print(f"Camera {i}: Tidak ditemukan")

print("\nGunakan index kamera yang TERSEDIA untuk kalibrasi.")
