package com.example.yolodepthestimator.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.yolodepthestimator.data.inference.DetectionResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel untuk CameraScreen
 * 
 * Responsibilities:
 * - Manage detection results state
 * - Track FPS & latency metrics
 * - Handle model lifecycle
 */
data class CameraUiState(
    val detections: List<DetectionResult> = emptyList(),
    val fps: Float = 0f,
    val latencyMs: Long = 0L,
    val isProcessing: Boolean = false,
    val errorMessage: String? = null
)

class CameraViewModel : ViewModel() {
    
    private val _uiState = MutableStateFlow(CameraUiState())
    val uiState: StateFlow<CameraUiState> = _uiState
    
    private var lastFpsUpdate = System.currentTimeMillis()
    private var frameCount = 0
    
    fun updateDetections(detections: List<DetectionResult>) {
        viewModelScope.launch {
            _uiState.emit(
                _uiState.value.copy(
                    detections = detections,
                    isProcessing = false
                )
            )
        }
    }
    
    fun updateFps(fps: Float) {
        viewModelScope.launch {
            _uiState.emit(
                _uiState.value.copy(fps = fps)
            )
        }
    }
    
    fun updateLatency(latencyMs: Long) {
        viewModelScope.launch {
            _uiState.emit(
                _uiState.value.copy(latencyMs = latencyMs)
            )
        }
    }
    
    fun setError(message: String) {
        viewModelScope.launch {
            _uiState.emit(
                _uiState.value.copy(errorMessage = message)
            )
        }
    }
    
    fun clearError() {
        viewModelScope.launch {
            _uiState.emit(
                _uiState.value.copy(errorMessage = null)
            )
        }
    }
}
