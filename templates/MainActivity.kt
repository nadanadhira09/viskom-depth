package com.example.yolodepthestimator

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.yolodepthestimator.data.inference.OnnxRuntimeManager
import com.example.yolodepthestimator.ui.screens.CameraScreen
import com.example.yolodepthestimator.ui.theme.YoloDepthEstimatorTheme
import com.example.yolodepthestimator.viewmodel.CameraViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize ONNX Runtime (one-time)
        OnnxRuntimeManager.initializeEnvironment()
        
        setContent {
            YoloDepthEstimatorTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val viewModel = viewModel<CameraViewModel>()
                    
                    // Load models on first composition
                    LaunchedEffect(Unit) {
                        OnnxRuntimeManager.loadYoloModel(this@MainActivity)
                        OnnxRuntimeManager.loadDepthModel(this@MainActivity)
                    }
                    
                    CameraScreen(
                        onDetectionsUpdated = { detections ->
                            viewModel.updateDetections(detections)
                        },
                        onFramesPerSecond = { fps ->
                            viewModel.updateFps(fps)
                        }
                    )
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Cleanup ONNX Runtime
        OnnxRuntimeManager.cleanup()
    }
}
