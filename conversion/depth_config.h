/**
 * depth_config.h
 * ==============
 * Header file untuk konfigurasi depth estimation di NCNN (Android/C++).
 * 
 * Konstanta kalibrasi diperoleh dari calibrate_depth.py
 * Update nilai A_CALIBRATION_CONSTANT dengan hasil kalibrasi Anda.
 */

#ifndef DEPTH_CONFIG_H
#define DEPTH_CONFIG_H

// ============================================================================
// KONSTANTA KALIBRASI
// ============================================================================
// IMPORTANT: Ganti nilai ini dengan hasil dari calibrate_depth.py
// Contoh: Jika kalibrasi menghasilkan A = 1850.5, maka:
const float A_CALIBRATION_CONSTANT = 1850.5f;


// ============================================================================
// PARAMETER ROI (Region of Interest)
// ============================================================================
// Ukuran kotak ROI di tengah layar untuk ekstraksi depth
const int ROI_SIZE = 50;  // pixels


// ============================================================================
// PARAMETER SMOOTHING
// ============================================================================
// Alpha untuk Exponential Moving Average (EMA)
// - 0.0 = 100% data lama (sangat smooth tapi lambat bereaksi)
// - 1.0 = 100% data baru (sangat responsif tapi jitter tinggi)
// - 0.3 = 30% data baru, 70% data lama (recommended)
const float SMOOTHING_ALPHA = 0.3f;


// ============================================================================
// PARAMETER SAFETY
// ============================================================================
// Threshold untuk deteksi sudden jump (meter)
// Jika perubahan jarak > threshold, aktifkan safety hold
const float MAX_JUMP_THRESHOLD = 1.5f;

// Durasi safety hold (dalam frame)
const int SAFETY_HOLD_FRAMES = 5;

// Range jarak valid
const float MIN_VALID_DISTANCE = 0.1f;   // meter
const float MAX_VALID_DISTANCE = 5.0f;   // meter


// ============================================================================
// PARAMETER MODEL
// ============================================================================
// Input size model (harus sama dengan saat konversi NCNN)
const int MODEL_INPUT_WIDTH = 518;
const int MODEL_INPUT_HEIGHT = 518;

// Normalisasi input (ImageNet standard)
const float MEAN_VALUES[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
const float NORM_VALUES[3] = {1.0f / (0.229f * 255.0f), 1.0f / (0.224f * 255.0f), 1.0f / (0.225f * 255.0f)};


#endif // DEPTH_CONFIG_H
