/**
 * depth_estimator.hpp
 * ===================
 * Class C++ untuk estimasi jarak dengan smoothing dan safety hold.
 * Untuk digunakan dengan NCNN di Android.
 * 
 * Contoh penggunaan:
 *   DepthEstimator estimator(A_CALIBRATION_CONSTANT);
 *   float distance = estimator.estimate(depth_map);
 */

#ifndef DEPTH_ESTIMATOR_HPP
#define DEPTH_ESTIMATOR_HPP

#include <ncnn/mat.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "depth_config.h"

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "DepthEstimator"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(...) printf(__VA_ARGS__)
#define LOGW(...) printf(__VA_ARGS__)
#endif


class DepthEstimator {
public:
    /**
     * Constructor
     * 
     * @param A Konstanta kalibrasi dari calibrate_depth.py
     * @param roi_size Ukuran ROI dalam piksel
     * @param smoothing_alpha Alpha untuk EMA (0-1)
     * @param max_jump_threshold Threshold untuk sudden jump detection (meter)
     */
    DepthEstimator(
        float A = A_CALIBRATION_CONSTANT,
        int roi_size = ROI_SIZE,
        float smoothing_alpha = SMOOTHING_ALPHA,
        float max_jump_threshold = MAX_JUMP_THRESHOLD
    ) : A_(A),
        roi_size_(roi_size),
        smoothing_alpha_(smoothing_alpha),
        max_jump_threshold_(max_jump_threshold),
        last_smoothed_distance_(-1.0f),
        safety_hold_active_(false),
        safety_hold_counter_(0) {
        
        distance_history_.reserve(30);
        LOGD("DepthEstimator initialized: A=%.2f, roi_size=%d, alpha=%.2f", 
             A_, roi_size_, smoothing_alpha_);
    }
    
    /**
     * Ekstrak nilai raw depth dari ROI di tengah depth map
     * 
     * @param depth_map Output dari model NCNN (H x W, tipe float)
     * @return Nilai rata-rata raw depth di ROI
     */
    float extractRoiDepth(const ncnn::Mat& depth_map) {
        int h = depth_map.h;
        int w = depth_map.w;
        
        // Tentukan batas ROI di tengah
        int center_y = h / 2;
        int center_x = w / 2;
        int roi_half = roi_size_ / 2;
        
        int y1 = std::max(0, center_y - roi_half);
        int y2 = std::min(h, center_y + roi_half);
        int x1 = std::max(0, center_x - roi_half);
        int x2 = std::min(w, center_x + roi_half);
        
        // Hitung rata-rata depth di ROI
        float sum = 0.0f;
        int count = 0;
        
        for (int y = y1; y < y2; y++) {
            const float* ptr = depth_map.row(y);
            for (int x = x1; x < x2; x++) {
                sum += ptr[x];
                count++;
            }
        }
        
        float raw_depth = (count > 0) ? (sum / count) : 0.0f;
        return raw_depth;
    }
    
    /**
     * Konversi raw depth ke jarak metrik menggunakan rumus invers
     * 
     * @param raw_depth Nilai raw depth dari model
     * @return Jarak dalam meter
     */
    float rawDepthToDistance(float raw_depth) {
        if (raw_depth < 1e-6f) {
            return MAX_VALID_DISTANCE;
        }
        
        float distance = A_ / raw_depth;
        
        // Clamp ke range valid
        distance = std::max(MIN_VALID_DISTANCE, std::min(MAX_VALID_DISTANCE, distance));
        
        return distance;
    }
    
    /**
     * Estimasi jarak dengan smoothing dan safety hold
     * 
     * @param depth_map Output depth map dari model NCNN
     * @return Jarak yang sudah di-smooth (meter)
     */
    float estimate(const ncnn::Mat& depth_map) {
        // 1. Ekstrak raw depth dari ROI
        float raw_depth = extractRoiDepth(depth_map);
        
        // 2. Konversi ke jarak metrik
        float current_estimate = rawDepthToDistance(raw_depth);
        
        // 3. Inisialisasi jika frame pertama
        if (last_smoothed_distance_ < 0.0f) {
            last_smoothed_distance_ = current_estimate;
            distance_history_.push_back(current_estimate);
            return current_estimate;
        }
        
        // 4. Deteksi sudden jump
        float distance_change = std::abs(current_estimate - last_smoothed_distance_);
        
        if (distance_change > max_jump_threshold_) {
            // Sudden jump detected - activate safety hold
            safety_hold_active_ = true;
            safety_hold_counter_ = SAFETY_HOLD_FRAMES;
            
            // Prioritaskan jarak terdekat (safety bias)
            current_estimate = std::min(current_estimate, last_smoothed_distance_);
            
            LOGW("Sudden jump detected: %.2fm -> %.2fm (clamped to safety)", 
                 last_smoothed_distance_, current_estimate);
        }
        
        // 5. Smoothing menggunakan Exponential Moving Average (EMA)
        float smoothed_distance = smoothing_alpha_ * current_estimate + 
                                 (1.0f - smoothing_alpha_) * last_smoothed_distance_;
        
        // 6. Update state
        last_smoothed_distance_ = smoothed_distance;
        distance_history_.push_back(smoothed_distance);
        
        // Keep only recent history (30 frames)
        if (distance_history_.size() > 30) {
            distance_history_.erase(distance_history_.begin());
        }
        
        // 7. Decrement safety hold counter
        if (safety_hold_counter_ > 0) {
            safety_hold_counter_--;
            if (safety_hold_counter_ == 0) {
                safety_hold_active_ = false;
            }
        }
        
        return smoothed_distance;
    }
    
    /**
     * Mendapatkan status safety hold
     * 
     * @return true jika safety hold aktif
     */
    bool isSafetyHoldActive() const {
        return safety_hold_active_;
    }
    
    /**
     * Mendapatkan metrik stabilitas (standard deviation dari history)
     * 
     * @return Nilai standard deviation dari 10 frame terakhir
     */
    float getStabilityMetric() const {
        if (distance_history_.size() < 2) {
            return 0.0f;
        }
        
        // Ambil 10 frame terakhir
        size_t start_idx = (distance_history_.size() > 10) ? 
                          (distance_history_.size() - 10) : 0;
        
        // Hitung mean
        float sum = 0.0f;
        size_t count = 0;
        for (size_t i = start_idx; i < distance_history_.size(); i++) {
            sum += distance_history_[i];
            count++;
        }
        float mean = sum / count;
        
        // Hitung standard deviation
        float sq_sum = 0.0f;
        for (size_t i = start_idx; i < distance_history_.size(); i++) {
            float diff = distance_history_[i] - mean;
            sq_sum += diff * diff;
        }
        
        return std::sqrt(sq_sum / count);
    }
    
    /**
     * Reset state estimator
     */
    void reset() {
        last_smoothed_distance_ = -1.0f;
        distance_history_.clear();
        safety_hold_active_ = false;
        safety_hold_counter_ = 0;
        LOGD("DepthEstimator reset");
    }
    
    /**
     * Get current smoothed distance tanpa update state
     */
    float getCurrentDistance() const {
        return last_smoothed_distance_;
    }

private:
    // Konstanta
    float A_;
    int roi_size_;
    float smoothing_alpha_;
    float max_jump_threshold_;
    
    // State variables
    float last_smoothed_distance_;
    std::vector<float> distance_history_;
    bool safety_hold_active_;
    int safety_hold_counter_;
};


#endif // DEPTH_ESTIMATOR_HPP
