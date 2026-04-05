# Android Implementation Documentation Index

> Panduan lengkap implementasi ONNX + YOLO + Depth Estimation di Android Studio

**Last Updated:** March 30, 2026  
**Project:** YoloDepthEstimator  
**Language:** Kotlin + Jetpack Compose  
**Min SDK:** Android 8.0 (API 26)

---

## 📚 Documentation Map

### 🚀 Getting Started (Start Here!)

1. **[QUICK_START.md](QUICK_START.md)** ⭐ **START HERE**
   - 5-minute quick setup
   - Essential steps only
   - Expected behavior checklist

2. **[COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)**
   - Full 7-step detailed implementation
   - Gradle configuration
   - File structure
   - Common issues & solutions

3. **[TESTING_DEBUGGING_GUIDE.md](TESTING_DEBUGGING_GUIDE.md)**
   - Unit testing strategies
   - Integration testing
   - Manual testing checklist
   - Debugging techniques
   - Performance profiling
   - Device compatibility matrix

---

## 📁 Code Files (Ready to Use)

### Core Infrastructure

- **[OnnxRuntimeManager.kt](OnnxRuntimeManager.kt)**
  - Singleton pattern for ONNX session management
  - Model loading & cleanup
  - Memory efficient

- **[ImagePreprocessor.kt](ImagePreprocessor.kt)**
  - Image normalization (CHW format)
  - Resize & rotation utilities
  - Mean/STD normalization for both YOLO & Depth models

- **[InferenceHelpers.kt](InferenceHelpers.kt)**
  - `YoloInference` class for detection
  - `DepthInference` class for depth estimation
  - Helper methods for output parsing

### UI Components

- **[CameraScreen.kt](CameraScreen.kt)** ⭐ MAIN UI
  - Live camera preview
  - Real-time inference
  - FPS & latency display
  - Detection overlay
  - Start/pause controls
  - Error handling

- **[MainActivity.kt](MainActivity.kt)**
  - Activity entry point
  - ONNX Runtime initialization
  - Lifecycle management
  - Cleanup on destroy

### State Management

- **[CameraViewModel.kt](CameraViewModel.kt)**
  - Jetpack ViewModel pattern
  - State flow for UI updates
  - Detection results tracking
  - FPS/latency metrics

---

## 🔧 Configuration Files

### Gradle Setup

```kotlin
// Key dependencies in build.gradle.kts
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.3")
implementation("androidx.camera:camera-camera2:1.4.0-alpha03")
implementation("androidx.camera:camera-lifecycle:1.4.0-alpha03")
implementation("androidx.camera:camera-view:1.4.0-alpha03")
implementation("androidx.compose.material3:material3:1.2.0")
```

### Manifest Permissions

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-feature android:name="android.hardware.camera" android:required="true" />
```

---

## 📊 Feature Checklist

| Feature             | Status      | File                  |
| ------------------- | ----------- | --------------------- |
| Camera preview      | ✅ Ready    | CameraScreen.kt       |
| YOLO detection      | ✅ Ready    | InferenceHelpers.kt   |
| Depth estimation    | ✅ Ready    | InferenceHelpers.kt   |
| Real-time FPS       | ✅ Ready    | CameraScreen.kt       |
| Latency tracking    | ✅ Ready    | CameraScreen.kt       |
| Model management    | ✅ Ready    | OnnxRuntimeManager.kt |
| Image preprocessing | ✅ Ready    | ImagePreprocessor.kt  |
| Error handling      | ✅ Ready    | CameraScreen.kt       |
| Permission handling | ✅ Ready    | CameraScreen.kt       |
| Theme support       | ⏳ Template | Theme.kt              |
| Result display      | ⏳ TODO     | ResultScreen.kt       |

---

## 🎯 Implementation Roadmap

### Phase 1: Setup (1 hour)

- [ ] Create Android Studio project
- [ ] Update gradle dependencies
- [ ] Copy ONNX models to assets/
- [ ] Copy template files
- [ ] Run first build

### Phase 2: Development (2-3 hours)

- [ ] Update MainActivity
- [ ] Implement CameraScreen
- [ ] Test on device
- [ ] Fix any issues

### Phase 3: Testing (1-2 hours)

- [ ] Manual testing checklist
- [ ] Performance profiling
- [ ] Device compatibility testing
- [ ] Debugging & optimization

### Phase 4: Release (30 mins)

- [ ] ProGuard configuration
- [ ] Sign APK/AAB
- [ ] Create release build

---

## 🧠 Architecture Overview

```plaintext
MainActivity (Activity Lifecycle)
    ↓
OnnxRuntimeManager (Singleton)
    ↓ loads models
┌─────────────────────────────────┐
│   ONNX Runtime Sessions         │
│  ✅ YOLO Detection Model        │
│  ✅ Depth Estimation Model      │
└─────────────────────────────────┘
    ↑
    │ inference
CameraScreen (Composable)
    ↓
ImagePreprocessor
    ↓
InferenceHelpers
    ↓
CameraViewModel (State Management)
```

---

## 📱 Model Files

| Model                                      | Size  | Location | Recommended    |
| ------------------------------------------ | ----- | -------- | -------------- |
| yolov12n.onnx                              | 26 MB | assets/  | For reference  |
| depth_anything_v2_vits.onnx                | 95 MB | assets/  | ❌ Too large   |
| depth_anything_v2_vits_quantized_int8.onnx | 24 MB | assets/  | ✅ Recommended |

---

## 🚀 Quick Reference

### Build Project

```bash
./gradlew build              # Build debug APK
./gradlew buildRelease       # Build release APK
```

### Run on Device

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.example.yolodepthestimator/.MainActivity
```

### View Logs

```bash
adb logcat | grep -i "yolo\|depth\|onnx"
adb logcat -c              # Clear logs
```

### Device Requirements

- Min SDK: API 26 (Android 8.0)
- Recommended RAM: 4GB+
- Recommended CPU: Snapdragon 778+/MediaTek Dimensity+
- GPU: NNAPI supported (optional but recommended)

---

## 🆘 Troubleshooting Quick Links

1. **Build Issues** → See COMPLETE_SETUP_GUIDE.md → "Common Build Issues"
2. **Runtime Crashes** → See TESTING_DEBUGGING_GUIDE.md → "Troubleshooting Common Issues"
3. **Performance Problems** → See TESTING_DEBUGGING_GUIDE.md → "Performance Optimization"
4. **Permission Errors** → Check AndroidManifest.xml had permissions

---

## 📊 Performance Expectations

| Metric                   | Target           | Typical          |
| ------------------------ | ---------------- | ---------------- |
| App Load Time            | < 3s             | 2-3s             |
| Model Load Time          | < 3s             | 2-3s             |
| Per Frame (YOLO + Depth) | < 1000ms         | 500-800ms        |
| FPS                      | 0.5-1            | 0.5-1            |
| Memory Usage             | < 400MB          | 250-350MB        |
| Battery Impact           | ~ 30% drain/hour | Varies by device |

---

## ✅ Pre-Flight Checklist

Before committing code:

- [ ] Project builds without errors
- [ ] Min SDK is 26
- [ ] Permissions in manifest
- [ ] Models in assets/
- [ ] No UnresolvedReference warnings
- [ ] Gradle synced successfully

Before deploying to device:

- [ ] APK generated successfully
- [ ] Device has API 26+
- [ ] USB debugging enabled
- [ ] Permissions granted

Before release:

- [ ] Tested on min 2 devices
- [ ] No crashes in 5-min test
- [ ] FPS stable
- [ ] Memory doesn't leak
- [ ] ProGuard rules configured

---

## 🔗 External References

- ONNX Runtime Android: https://onnxruntime.ai/docs/install/
- Jetpack Compose: https://developer.android.com/compose
- CameraX: https://developer.android.com/training/camerax
- Android Developers: https://developer.android.com/

---

## 📝 Notes

### Version Information

- ONNX Runtime: 1.16.3
- Gradle: 8.0+
- Android Studio: Arctic Fox 2021.3.1+
- Kotlin: 1.8.0+
- Compose: 1.5.0+

### Supported Models

- ✅ YOLOv12n (26 MB)
- ✅ Depth Anything V2 ViTS Quantized INT8 (24 MB)

### Future Enhancements

- [ ] Result display screen
- [ ] Video recording
- [ ] Image gallery integration
- [ ] Export detections to JSON
- [ ] Multi-threading optimization
- [ ] GPU computation support

---

## 📞 Support

For issues:

1. Check this documentation index
2. Read the detailed guide relevant to your issue
3. Check logcat output
4. Profile with Android Profiler
5. Test on different device

---

**Last Updated:** March 30, 2026  
**Status:** ✅ Ready for Implementation  
**Questions?** Start with [QUICK_START.md](QUICK_START.md)
