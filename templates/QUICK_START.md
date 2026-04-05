# ⚡ Android Implementation - Quick Start

> Panduan singkat untuk segera mulai implementasi

---

## 📦 Files Ready to Use

Semua file sudah tersedia di `templates/` folder:

```
templates/
├── CameraScreen.kt              ← Main UI
├── MainActivity.kt              ← Entry point
├── CameraViewModel.kt           ← State management
├── OnnxRuntimeManager.kt        ← Model manager
├── ImagePreprocessor.kt         ← Image normalization
├── InferenceHelpers.kt          ← YOLO + Depth helpers
├── COMPLETE_SETUP_GUIDE.md      ← Full 7-step setup
├── TESTING_DEBUGGING_GUIDE.md   ← Testing strategies
└── ANDROID_IMPLEMENTATION.md    ← Structure reference
```

---

## 🚀 5-Minute Quick Setup

### Step 1: Create Android Project (1 min)

```bash
# Android Studio:
File → New → New Android Project
- Empty Activity (Compose)
- Name: YoloDepthEstimator
- Min SDK: 26
```

### Step 2: Update Gradle (1 min)

Copy this ke `app/build.gradle.kts`:

```kotlin
dependencies {
    // Core essentials
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // Compose UI
    implementation("androidx.activity:activity-compose:1.8.1")
    implementation("androidx.compose.ui:ui:1.6.4")
    implementation("androidx.compose.material3:material3:1.2.0")

    // Camera (critical for live feed)
    implementation("androidx.camera:camera-camera2:1.4.0-alpha03")
    implementation("androidx.camera:camera-lifecycle:1.4.0-alpha03")
    implementation("androidx.camera:camera-view:1.4.0-alpha03")

    // ⭐ ONNX Runtime - this is the main dependency
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.3")
}
```

Tekan: **Sync Now**

### Step 3: Copy Model Files (1 min)

```bash
# Terminal/PowerShell
cd your-android-project/app/src/main
mkdir -p assets
cp path/to/viskom/models/onnx/*.onnx assets/
```

Atau via Android Studio:

- Right-click `app/src/main` → New → Folder → Assets
- Drag models dari explorer ke folder

### Step 4: Copy Template Files (2 min)

```bash
# Copy from viskom/templates/ ke app/src/main/java/com/example/yolodepthestimator/

# 1. Create folder structure:
data/inference/          ← OnnxRuntimeManager, ImagePreprocessor, InferenceHelpers
ui/screens/             ← CameraScreen
viewmodel/              ← CameraViewModel

# 2. Copy:
OnnxRuntimeManager.kt → data/inference/
ImagePreprocessor.kt → data/inference/
InferenceHelpers.kt → data/inference/
CameraScreen.kt → ui/screens/
CameraViewModel.kt → viewmodel/
MainActivity.kt → (root)
```

### Step 5: Create Theme File (quick)

Create `ui/theme/Theme.kt`:

```kotlin
package com.example.yolodepthestimator.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable

@Composable
fun YoloDepthEstimatorTheme(content: @Composable () -> Unit) {
    MaterialTheme(content = content)
}
```

### Step 6: Update AndroidManifest.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.INTERNET" />

    <uses-feature android:name="android.hardware.camera" android:required="true" />

    <application ...>
        <activity android:name=".MainActivity" android:exported="true" />
    </application>

</manifest>
```

---

## ✅ Build & Run

1. **Build:**
   - Ctrl+B or Build → Build Bundles/APKs → Build APK

2. **Run:**
   - Connect device (API 26+) OR open emulator
   - Ctrl+R or Run → Run 'app'

3. **First Run:**
   - App should launch
   - Camera preview appears
   - Grant permission when prompted
   - FPS counter shows ~0.5 FPS

---

## 🎯 What's Next?

### If App Launches Successfully ✅

→ Go to: **TESTING_DEBUGGING_GUIDE.md**

- Test on different devices
- Profile performance
- Optimize inference

### If App Crashes ❌

→ Check:

1. ONNX dependency added? (sync gradle!)
2. Models in assets/? (check file sizes)
3. Min SDK = 26?
4. Permissions in manifest?

---

## 📊 Expected Behavior

### First Launch

- Camera preview ~100ms to appear
- Model loading ~2-3 seconds
- Detection overlay should appear

### During Operation

- FPS: 0.5-1 frame/second
- Latency: 500-1000ms per frame
- Memory: 250-350 MB
- CPU: 30-50% usage

### Device Compatibility

| Device Type   | Min SDK | Works?        |
| ------------- | ------- | ------------- |
| Phone (2020+) | 26      | ✅ Yes        |
| Phone (2018)  | 26      | ⚠️ Slow CPU   |
| Tablet        | 26      | ✅ Yes        |
| Emulator      | 26      | ✅ Yes (slow) |

---

## 🔗 File Reference Map

Need help? Find your file here:

| Task                | File                         |
| ------------------- | ---------------------------- |
| Setup steps 1-7     | `COMPLETE_SETUP_GUIDE.md`    |
| Testing strategies  | `TESTING_DEBUGGING_GUIDE.md` |
| Full implementation | `ANDROID_IMPLEMENTATION.md`  |
| Camera UI code      | `CameraScreen.kt`            |
| Model inference     | `InferenceHelpers.kt`        |
| Image prep          | `ImagePreprocessor.kt`       |

---

## 💡 Pro Tips

1. **Use Quantized Models**
   - Prefer: `depth_anything_v2_vits_quantized_int8.onnx` (24MB)
   - Avoid: `depth_anything_v2_vits.onnx` (95MB) if slow

2. **Enable NNAPI**

   ```kotlin
   sessionOptions.apply {
       addNnapi()  // Use mobile accelerators
       setIntraOpNumThreads(4)
   }
   ```

3. **Test on Real Device**
   - Emulator is good for development
   - Real device needed for performance testing
   - Recommend: Pixel 5+ or Samsung S21+

4. **Monitor Logcat**
   ```bash
   adb logcat | grep -i "yolo\|depth\|onnx"
   ```

---

## 🆘 Common Issues

| Problem                       | Fix                          |
| ----------------------------- | ---------------------------- |
| Build fails: "ONNX not found" | Sync gradle, check version   |
| App crashes on launch         | Check min SDK, ONNX version  |
| Camera permission denied      | Grant in app settings        |
| Models not loading            | Verify path in assets/       |
| Very slow inference           | Enable NNAPI, use INT8 model |

---

## ⏭️ Next Steps

1. ✅ Complete this 5-minute setup
2. 📝 Read `COMPLETE_SETUP_GUIDE.md` for detailed explanation
3. 🧪 Use `TESTING_DEBUGGING_GUIDE.md` for validation
4. 🚀 Deploy to device

---

**Ready? Let's go! 🚀**

Questions? Check the detailed guides in templates/
