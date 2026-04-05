# Workflow Kalibrasi dan Testing Depth Estimation

Dokumentasi lengkap untuk workflow kalibrasi depth estimation dengan metode tiga titik, testing akurasi, dan deployment ke NCNN.

## 📋 Daftar Isi

1. [Overview](#overview)
2. [Prasyarat](#prasyarat)
3. [Langkah 1: Kalibrasi](#langkah-1-kalibrasi)
4. [Langkah 2: Testing Akurasi](#langkah-2-testing-akurasi)
5. [Langkah 3: Visualisasi Real-time](#langkah-3-visualisasi-real-time)
6. [Langkah 4: Deployment ke NCNN](#langkah-4-deployment-ke-ncnn)
7. [Parameter Tuning](#parameter-tuning)

---

## Overview

Workflow ini mengimplementasikan **metode kalibrasi tiga titik** untuk mengkonversi relative depth menjadi metric depth (jarak absolut dalam meter) dengan fitur:

- ✅ **Inverse Depth Calibration**: Kalibrasi menggunakan 3+ titik referensi
- ✅ **Exponential Moving Average (EMA)**: Smoothing 30% data baru + 70% historis
- ✅ **Safety Hold Logic**: Deteksi sudden jump & bias ke jarak terdekat
- ✅ **Real-time Visualization**: Monitoring jarak dengan grafik stabilitas

**Rumus Dasar:**

```
Jarak (meter) = A / raw_depth
```

Dimana `A` adalah konstanta kalibrasi yang didapat dari proses kalibrasi.

---

## Prasyarat

### Software

```bash
# Pastikan environment aktif
.venv\Scripts\Activate.ps1

# Install dependencies (sudah ada di requirements.txt)
pip install torch opencv-python numpy matplotlib
```

### Hardware

- Webcam (untuk mode interactive)
- Objek datar (papan/dinding) untuk target kalibrasi
- Meteran untuk mengukur jarak akurat

### Model

```bash
# Pastikan checkpoint sudah terdownload
# File: checkpoints/depth_anything_v2_vits.pth
```

---

## Langkah 1: Kalibrasi

### 1.1 Mode Interactive (Webcam)

**Recommended untuk akurasi tinggi**

```bash
python conversion/calibrate_depth.py \
    --encoder vits \
    --checkpoint checkpoints/depth_anything_v2_vits.pth \
    --distances 1.0 2.0 3.0 \
    --roi-size 50 \
    --mode interactive \
    --output conversion/calibration_result.txt
```

**Langkah-langkah:**

1. Program akan membuka webcam
2. Letakkan objek datar pada jarak **1.0 meter** (ukur dengan meteran)
3. Arahkan kamera sehingga objek berada di tengah (dalam kotak hijau ROI)
4. Tekan **SPASI** untuk capture data
5. Ulangi untuk jarak **2.0 meter** dan **3.0 meter**
6. Program akan menghitung konstanta `A` dan menyimpan hasil

**Output:**

```
conversion/calibration_result.txt  # Konstanta untuk C++
conversion/calibration_result.json # Data lengkap (JSON)
```

### 1.2 Mode Images (Dari Gambar)

**Jika sudah punya foto objek dengan jarak terukur**

```bash
python conversion/calibrate_depth.py \
    --mode images \
    --images assets/cal_1m.jpg assets/cal_2m.jpg assets/cal_3m.jpg \
    --distances 1.0 2.0 3.0 \
    --output conversion/calibration_result.txt
```

### 1.3 Interpretasi Hasil

Contoh output:

```
HASIL KALIBRASI:
----------------------------------------------------------------------
  Point 1: 1.0m → raw_depth=1850.50 → A=1850.50
  Point 2: 2.0m → raw_depth=925.25  → A=1850.50
  Point 3: 3.0m → raw_depth=616.83  → A=1850.50
----------------------------------------------------------------------
KONSTANTA KALIBRASI:
  A_mean = 1850.50
  A_std  = 0.00
  A_min  = 1850.50
  A_max  = 1850.50
----------------------------------------------------------------------

VERIFIKASI AKURASI:
  1.0m: prediksi=1.00m, error=0.000m (0.0%)
  2.0m: prediksi=2.00m, error=0.000m (0.0%)
  3.0m: prediksi=3.00m, error=0.000m (0.0%)

Mean Absolute Error (MAE): 0.000m
```

**Kriteria Baik:**

- `A_std` < 50 (konsisten antar titik)
- MAE < 0.05m (akurat dalam 5cm)

---

## Langkah 2: Testing Akurasi

Setelah kalibrasi, test akurasi pada jarak yang **berbeda** dengan jarak kalibrasi.

### 2.1 Testing Interactive

```bash
python conversion/test_calibration.py \
    --calibration conversion/calibration_result.json \
    --encoder vits \
    --checkpoint checkpoints/depth_anything_v2_vits.pth \
    --test-distances 0.5 1.5 2.5 3.5 \
    --output conversion/test_results.json \
    --plot
```

**Langkah-langkah:**

1. Program membuka webcam
2. Letakkan objek pada jarak **0.5m** (ukur dengan meteran)
3. Tekan **SPASI** untuk test
4. Ulangi untuk jarak 1.5m, 2.5m, 3.5m
5. Program menampilkan statistik error dan plot grafik

### 2.2 Interpretasi Hasil Test

Contoh output:

```
STATISTIK TESTING:
======================================================================
  Mean Absolute Error: 0.087 ± 0.045 m
  Mean Error %:        8.2 ± 4.1 %
  Max Error:           0.145 m (12.3%)
======================================================================
```

**Kriteria Layak Deploy:**

- Mean Error < 15% (untuk navigasi kursi roda)
- Max Error < 0.3m (tidak berbahaya)

**Plot yang dihasilkan:**

- Raw Depth vs Distance (verifikasi hubungan invers)
- Predicted vs True Distance (akurasi prediksi)
- Error per Test Point (visualisasi outlier)
- Error Percentage (konsistensi akurasi)

---

## Langkah 3: Visualisasi Real-time

Monitoring jarak secara real-time dengan smoothing dan safety hold.

### 3.1 Jalankan Visualisasi

```bash
python conversion/realtime_distance.py \
    --calibration conversion/calibration_result.json \
    --encoder vits \
    --checkpoint checkpoints/depth_anything_v2_vits.pth \
    --smoothing 0.3 \
    --safety-threshold 1.5
```

**Fitur yang ditampilkan:**

- Frame asli + Depth map (side-by-side)
- Distance bar dengan color coding:
  - 🟢 Hijau (> 3m): Aman
  - 🟡 Kuning (1.5-3m): Hati-hati
  - 🟠 Orange (0.5-1.5m): Peringatan
  - 🔴 Merah (< 0.5m): Bahaya
- Graf history jarak (30 frame terakhir)
- Info smoothed vs current estimate
- Warning "SAFETY HOLD ACTIVE" saat sudden jump terdeteksi
- FPS dan stability metric

### 3.2 Parameter Real-time

| Parameter            | Default | Keterangan                                |
| -------------------- | ------- | ----------------------------------------- |
| `--smoothing`        | 0.3     | Alpha EMA (0.0 = smooth, 1.0 = responsif) |
| `--safety-threshold` | 1.5     | Threshold sudden jump (meter)             |
| `--no-depth`         | false   | Jangan tampilkan depth map                |
| `--camera`           | 0       | Index kamera                              |

**Keyboard shortcuts:**

- `r`: Reset estimator (clear history)
- `q` atau `ESC`: Quit

---

## Langkah 4: Deployment ke NCNN

### 4.1 Update Konstanta Kalibrasi

Edit file `conversion/depth_config.h`:

```cpp
// Ganti dengan hasil kalibrasi Anda
const float A_CALIBRATION_CONSTANT = 1850.5f;  // <-- Update ini
const int ROI_SIZE = 50;
```

### 4.2 Integrasi ke Android (C++)

**Include header files:**

```cpp
#include "depth_config.h"
#include "depth_estimator.hpp"
```

**Inisialisasi estimator:**

```cpp
// Di onCreate atau saat load model
DepthEstimator estimator(A_CALIBRATION_CONSTANT);
```

**Prediksi jarak:**

```cpp
// Setelah inference NCNN
ncnn::Mat depth_output;  // Output dari model NCNN (H x W, float)

// Estimasi jarak dengan smoothing
float distance = estimator.estimate(depth_output);

// Cek safety hold
if (estimator.isSafetyHoldActive()) {
    // Warning: Sudden jump terdeteksi
    showWarning("PERINGATAN: Perubahan jarak tiba-tiba!");
}

// Cek stabilitas
float stability = estimator.getStabilityMetric();
if (stability > 0.1f) {
    // Jarak tidak stabil (kamera bergerak atau objek bergerak)
}
```

**Contoh decision logic:**

```cpp
float distance = estimator.estimate(depth_output);

if (distance < 0.5f) {
    // BAHAYA: Rintangan sangat dekat
    triggerEmergencyStop();
    playSound("STOP! Rintangan 0.5 meter");
} else if (distance < 1.5f) {
    // PERINGATAN: Rintangan dekat
    slowDownSpeed();
    playSound("Hati-hati, rintangan 1.5 meter");
} else if (distance < 3.0f) {
    // NORMAL: Rintangan terdeteksi
    continueNormalSpeed();
} else {
    // AMAN: Tidak ada rintangan
    continueNormalSpeed();
}
```

### 4.3 Optimasi untuk Real-time

**1. Reduce input size untuk FPS lebih tinggi:**

```bash
# Saat konversi NCNN, gunakan input size lebih kecil
python conversion/convert_ncnn.py \
    --input-size 256  # Lebih cepat dari 518
```

**2. Adjust smoothing untuk responsiveness:**

```cpp
// Alpha lebih tinggi = lebih responsif (tapi lebih jitter)
DepthEstimator estimator(A, 50, 0.5f);  // alpha=0.5

// Alpha lebih rendah = lebih smooth (tapi delay lebih besar)
DepthEstimator estimator(A, 50, 0.2f);  // alpha=0.2
```

**3. Multi-threading:**

```cpp
// Jalankan depth estimation di thread terpisah
std::thread depth_thread([&]() {
    while (running) {
        ncnn::Mat depth = model.infer(frame);
        float dist = estimator.estimate(depth);
        updateUI(dist);
    }
});
```

---

## Parameter Tuning

### Smoothing Alpha (`smoothing_alpha`)

| Alpha | Behavior                      | Use Case                         |
| ----- | ----------------------------- | -------------------------------- |
| 0.1   | Sangat smooth, delay tinggi   | Kursi roda indoor (objek statis) |
| 0.3   | **Balanced (recommended)**    | General purpose                  |
| 0.5   | Responsif, moderate jitter    | Outdoor, objek bergerak          |
| 0.8   | Sangat responsif, high jitter | Racing, fast motion              |

### Safety Threshold (`max_jump_threshold`)

| Threshold | Sensitivity                | Use Case          |
| --------- | -------------------------- | ----------------- |
| 0.5m      | Sangat sensitif            | Lingkungan sempit |
| 1.0m      | Moderate                   | Indoor normal     |
| **1.5m**  | **Balanced (recommended)** | General purpose   |
| 2.0m      | Low sensitivity            | Outdoor luas      |

### ROI Size (`roi_size`)

| Size     | Area                        | Use Case                |
| -------- | --------------------------- | ----------------------- |
| 30px     | Sangat fokus (titik center) | Deteksi pinpoint        |
| **50px** | **Balanced (recommended)**  | General purpose         |
| 100px    | Area lebar                  | Deteksi rintangan besar |

**Formula untuk menyesuaikan ROI:**

```cpp
// ROI size sebagai persentase dari frame width
int adaptive_roi = frame_width * 0.1;  // 10% dari lebar frame
```

---

## Troubleshooting

### Error: "Kalibrasi tidak konsisten (A_std tinggi)"

**Penyebab:**

- Pencahayaan berubah antar capture
- Kamera bergerak saat capture
- Objek tidak rata/sejajar kamera

**Solusi:**

- Gunakan tripod untuk stabilitas
- Pastikan pencahayaan konstan
- Gunakan objek datar (papan/dinding)
- Ulangi kalibrasi

### Error: "Mean Error tinggi (> 20%)"

**Penyebab:**

- Jarak kalibrasi terlalu sempit (hanya 1-3m)
- Model tidak cocok dengan kondisi lighting
- Kamera berbeda dengan saat kalibrasi

**Solusi:**

- Tambah titik kalibrasi: `--distances 0.5 1.0 1.5 2.0 2.5 3.0 3.5`
- Kalibrasi ulang di kondisi pencahayaan target
- Kalibrasi per-kamera jika multi-camera

### Jarak fluktuatif (jitter tinggi)

**Solusi:**

- Turunkan alpha: `--smoothing 0.2`
- Naikkan safety threshold: `--safety-threshold 2.0`
- Perbesar ROI: `--roi-size 100`

### Delay terlalu besar (lambat bereaksi)

**Solusi:**

- Naikkan alpha: `--smoothing 0.5`
- Kurangi safety hold frames di C++: `SAFETY_HOLD_FRAMES = 2`

---

## File Reference

| File                      | Fungsi                          |
| ------------------------- | ------------------------------- |
| `calibrate_depth.py`      | Kalibrasi 3 titik → konstanta A |
| `test_calibration.py`     | Testing akurasi kalibrasi       |
| `realtime_distance.py`    | Visualisasi real-time           |
| `depth_config.h`          | Header C++ (konstanta)          |
| `depth_estimator.hpp`     | Class C++ (implementasi)        |
| `calibration_result.txt`  | Output kalibrasi (TXT)          |
| `calibration_result.json` | Output kalibrasi (JSON)         |
| `test_results.json`       | Output testing akurasi          |

---

## Best Practices

### ✅ DO:

- Kalibrasi setiap kali ganti kamera
- Test pada berbagai kondisi pencahayaan
- Tune parameter berdasarkan use case
- Monitor stability metric secara real-time
- Simpan hasil kalibrasi per-device

### ❌ DON'T:

- Gunakan konstanta A dari paper/orang lain
- Skip testing phase
- Deploy tanpa verifikasi akurasi
- Hardcode parameter tanpa dokumentasi
- Lupa update konstanta di C++ header

---

## Contoh Complete Pipeline

```bash
# 1. Kalibrasi
python conversion/calibrate_depth.py --mode interactive

# 2. Testing
python conversion/test_calibration.py \
    --calibration conversion/calibration_result.json \
    --test-distances 0.5 1.5 2.5 3.5 \
    --plot

# 3. Visualisasi real-time
python conversion/realtime_distance.py \
    --calibration conversion/calibration_result.json

# 4. Update C++ header
# Edit conversion/depth_config.h dengan konstanta dari step 1

# 5. Deploy ke NCNN/Android
# Copy depth_config.h dan depth_estimator.hpp ke project Android
```

---

**Dokumentasi dibuat:** 10 Maret 2026  
**Versi:** 1.0  
**Author:** GitHub Copilot Assistant
