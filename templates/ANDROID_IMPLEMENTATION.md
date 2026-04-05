# Android Implementation Template

Berikut adalah struktur file-file yang harus dibuat di Android Studio:

## File Structure

```
YoloDepthEstimator/app/src/main/
├── java/com/example/yolodepthestimator/
│   ├── MainActivity.kt
│   ├── ui/
│   │   ├── screens/
│   │   │   ├── CameraScreen.kt
│   │   │   └── ResultScreen.kt
│   │   └── components/
│   │       ├── BoundingBoxOverlay.kt
│   │       └── DepthVisualization.kt
│   ├── data/
│   │   ├── inference/
│   │   │   ├── OnnxRuntimeManager.kt
│   │   │   ├── ImagePreprocessor.kt
│   │   │   └── InferenceHelpers.kt
│   │   └── models/
│   │       ├── DetectionResult.kt
│   │       └── AnalysisState.kt
│   └── viewmodel/
│       └── CameraViewModel.kt
├── assets/
│   ├── yolov12n.onnx
│   └── depth_anything_v2_vits_quantized_int8.onnx
└── AndroidManifest.xml
```

## Key Implementation Points

### 1. Initialization in MainActivity

```kotlin
override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    // Initialize ONNX Runtime
    OnnxRuntimeManager.initializeEnvironment()
    OnnxRuntimeManager.loadYoloModel(this)
    OnnxRuntimeManager.loadDepthModel(this)
}

override fun onDestroy() {
    super.onDestroy()
    OnnxRuntimeManager.cleanup()
}
```

### 2. Camera Frame Processing

```kotlin
private fun processFrame(frame: ImageProxy) {
    val bitmap = frame.toBitmap() ?: return

    viewModelScope.launch(Dispatchers.Default) {
        val yoloSession = OnnxRuntimeManager.getYoloSession() ?: return@launch
        val depthSession = OnnxRuntimeManager.getDepthSession() ?: return@launch

        val yoloInference = YoloInference(yoloSession, env)
        val depthInference = DepthInference(depthSession, env)

        val detections = yoloInference.detect(bitmap, bitmap.width, bitmap.height)
        val depthMap = depthInference.estimateDepth(bitmap)

        // Update UI with results
        updateResults(detections, depthMap)
    }
}
```

### 3. Performance Considerations

- **Model Loading**: Load once in singleton, reuse sessions
- **Threading**: Run inference di Dispatchers.Default (background thread)
- **Memory**: Use quantized INT8 models untuk efficiency
- **Batch Processing**: Process camera frames dengan controlled rate

### 4. GPU Support (Optional)

```kotlin
val sessionOptions = OrtSession.SessionOptions().apply {
    addNnapi()           // Enable NNAPI untuk acceleration
    // addCoreMLFlags()   // iOS-style, not for Android
    // addDnnl()          // x86 only
    setIntraOpNumThreads(4)
}
```

## Build & Test Checklist

- [ ] Gradle dependencies semua terpasang
- [ ] Model files ada di assets/
- [ ] Permissions di AndroidManifest
- [ ] Runtime permissions granted (Camera)
- [ ] Emulator atau device API 26+
- [ ] Build variant: release (untuk performance)

## Next Steps

1. Create Android Studio project dengan template ini
2. Copy ONNX model files ke assets/
3. Implement CameraScreen UI
4. Test inference dengan sample frames
5. Optimize performance berdasarkan device capabilities
