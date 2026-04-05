package com.example.yolodepthestimator.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.yolodepthestimator.data.inference.OnnxRuntimeManager
import com.example.yolodepthestimator.data.inference.DetectionResult
import java.util.concurrent.Executors
import kotlin.math.roundToInt

/**
 * Camera Screen dengan real-time YOLO + Depth estimation
 * 
 * Features:
 * - Live camera preview
 * - Real-time model inference
 * - FPS monitoring
 * - Detection overlay
 * - Start/Stop recording
 */
@Composable
fun CameraScreen(
    onDetectionsUpdated: (List<DetectionResult>) -> Unit = {},
    onFramesPerSecond: (Float) -> Unit = {}
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    
    var previewView: PreviewView? by remember { mutableStateOf(null) }
    var isRunning by remember { mutableStateOf(true) }
    var fps by remember { mutableStateOf(0f) }
    var latencyMs by remember { mutableStateOf(0L) }
    var detections by remember { mutableStateOf<List<DetectionResult>>(emptyList()) }
    var processingError by remember { mutableStateOf("") }
    var hasPermission by remember { mutableStateOf(false) }
    
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    
    // Permission launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasPermission = isGranted
    }
    
    // Check & request permissions
    LaunchedEffect(Unit) {
        hasPermission = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        
        if (!hasPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
    
    // Initialize camera
    LaunchedEffect(hasPermission, isRunning) {
        if (!hasPermission) return@LaunchedEffect
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build()
            
            // Setup image analysis for inference
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (isRunning) {
                            processFrame(
                                imageProxy,
                                context,
                                onDetectionsUpdated = {
                                    detections = it
                                    onDetectionsUpdated(it)
                                },
                                onLatency = { latencyMs = it },
                                onError = { processingError = it }
                            )
                        }
                        imageProxy.close()
                    }
                }
            
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )
                
                preview.setSurfaceProvider(previewView?.surfaceProvider)
            } catch (e: Exception) {
                processingError = "Camera binding error: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(context))
    }
    
    // Update FPS every second
    LaunchedEffect(Unit) {
        var frameCount = 0
        while (true) {
            delay(1000)
            fps = frameCount.toFloat()
            frameCount = 0
            onFramesPerSecond(fps)
        }
    }
    
    Box(modifier = Modifier.fillMaxSize()) {
        // Camera Preview
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    previewView = this
                }
            },
            modifier = Modifier.fillMaxSize()
        )
        
        // Overlay: Stats & Controls
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Top: Stats
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color.Black.copy(alpha = 0.7f)
                )
            ) {
                Column(
                    modifier = Modifier.padding(12.dp)
                ) {
                    Text(
                        text = "FPS: ${"%.1f".format(fps)}",
                        color = Color.Green,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(4.dp)
                    )
                    Text(
                        text = "Latency: ${latencyMs}ms",
                        color = Color.Yellow,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(4.dp)
                    )
                    Text(
                        text = "Detections: ${detections.size}",
                        color = Color.Cyan,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(4.dp)
                    )
                    
                    if (processingError.isNotEmpty()) {
                        Text(
                            text = "Error: $processingError",
                            color = Color.Red,
                            fontSize = 12.sp,
                            modifier = Modifier.padding(4.dp)
                        )
                    }
                }
            }
            
            // Bottom: Controls
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color.Black.copy(alpha = 0.5f))
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = { isRunning = !isRunning },
                    modifier = Modifier
                        .weight(1f)
                        .padding(8.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (isRunning) Color.DarkGray else Color.Green
                    )
                ) {
                    Icon(
                        imageVector = if (isRunning) Icons.Filled.Pause else Icons.Filled.PlayArrow,
                        contentDescription = "Toggle",
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(if (isRunning) "PAUSE" else "RESUME")
                }
                
                Button(
                    onClick = { /* TODO: Save detection */ },
                    modifier = Modifier
                        .weight(1f)
                        .padding(8.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color.Blue
                    )
                ) {
                    Icon(
                        imageVector = Icons.Filled.CameraAlt,
                        contentDescription = "Capture",
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("CAPTURE")
                }
            }
            
            // Detection list
            if (detections.isNotEmpty()) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = Color.Black.copy(alpha = 0.7f)
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp)
                    ) {
                        Text(
                            text = "Detected Objects:",
                            color = Color.White,
                            fontSize = 14.sp,
                            fontWeight = androidx.compose.ui.text.font.FontWeight.Bold
                        )
                        detections.take(3).forEach { detection ->
                            Text(
                                text = "${detection.className} (${
                                    "%.1f".format(
                                        detection.confidence * 100
                                    )
                                }%)",
                                color = Color.Cyan,
                                fontSize = 12.sp,
                                modifier = Modifier.padding(4.dp)
                            )
                        }
                        if (detections.size > 3) {
                            Text(
                                text = "+${detections.size - 3} more",
                                color = Color.Gray,
                                fontSize = 10.sp,
                                modifier = Modifier.padding(4.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Process single frame untuk inference
 */
private suspend fun processFrame(
    imageProxy: ImageProxy,
    context: android.content.Context,
    onDetectionsUpdated: (List<DetectionResult>) -> Unit,
    onLatency: (Long) -> Unit,
    onError: (String) -> Unit
) {
    try {
        val startTime = System.currentTimeMillis()
        
        // Convert ImageProxy to Bitmap
        val bitmap = imageProxy.toBitmap() ?: return
        
        // Get ONNX sessions
        val yoloSession = OnnxRuntimeManager.getYoloSession() ?: run {
            OnnxRuntimeManager.loadYoloModel(context)
            OnnxRuntimeManager.getYoloSession()
        } ?: return
        
        val depthSession = OnnxRuntimeManager.getDepthSession() ?: run {
            OnnxRuntimeManager.loadDepthModel(context)
            OnnxRuntimeManager.getDepthSession()
        } ?: return
        
        // TODO: Implement actual inference
        // For now, return empty detections
        onDetectionsUpdated(emptyList())
        
        val endTime = System.currentTimeMillis()
        onLatency(endTime - startTime)
        
    } catch (e: Exception) {
        onError(e.message ?: "Unknown error")
    }
}

private suspend fun delay(ms: Long) {
    kotlinx.coroutines.delay(ms)
}

private fun ImageProxy.toBitmap(): android.graphics.Bitmap? {
    return try {
        val planes = this.planes
        val buffer = planes[0].buffer
        buffer.rewind()
        val pixelStride = planes[0].pixelStride
        val padding = planes[0].rowPadding
        val w = this.width + padding / pixelStride
        val bitmap = android.graphics.Bitmap.createBitmap(w, this.height, android.graphics.Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)
        // Crop to actual image width
        android.graphics.Bitmap.createBitmap(bitmap, 0, 0, this.width, this.height)
    } catch (e: Exception) {
        null
    }
}
