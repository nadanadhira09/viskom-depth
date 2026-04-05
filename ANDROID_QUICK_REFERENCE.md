# 🚀 ANDROID DEPLOYMENT - QUICK REFERENCE CARD

## ⏱️ 5-MINUTE SUMMARY

### What to Do

1. **Create project** in Android Studio
2. **Copy 6 Kotlin files** from `templates/`
3. **Copy 2 model files** to `assets/`
4. **Update gradle** dependencies
5. **Run on device!**

---

## 📋 GRADLE DEPENDENCIES (Copy-Paste Ready)

```gradle
dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("androidx.activity:activity-compose:1.8.1")
    implementation("androidx.compose.ui:ui:1.6.4")
    implementation("androidx.compose.ui:ui-graphics:1.6.4")
    implementation("androidx.compose.ui:ui-tooling-preview:1.6.4")
    implementation("androidx.compose.material3:material3:1.2.0")
    implementation("androidx.camera:camera-camera2:1.4.0-alpha03")
    implementation("androidx.camera:camera-lifecycle:1.4.0-alpha03")
    implementation("androidx.camera:camera-view:1.4.0-alpha03")
    implementation("androidx.camera:camera-core:1.4.0-alpha03")
    implementation("com.google.accompanist:accompanist-permissions:0.35.0-alpha")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.3")
}
```

---

## 📁 FILE COPY MAPPING

| Template File           | Destination       | Package                                         |
| ----------------------- | ----------------- | ----------------------------------------------- |
| `CameraScreen.kt`       | `ui/screens/`     | `com.example.yolodepthestimator.ui.screens`     |
| `MainActivity.kt`       | Root              | `com.example.yolodepthestimator`                |
| `CameraViewModel.kt`    | `viewmodel/`      | `com.example.yolodepthestimator.viewmodel`      |
| `OnnxRuntimeManager.kt` | `data/inference/` | `com.example.yolodepthestimator.data.inference` |
| `ImagePreprocessor.kt`  | `data/inference/` | `com.example.yolodepthestimator.data.inference` |
| `InferenceHelpers.kt`   | `data/inference/` | `com.example.yolodepthestimator.data.inference` |

---

## 🎯 MODEL FILES TO ASSETS

```
Source:                                    Destination:
models/onnx/depth_anything_v2_vits_quantized_int8.onnx  →  app/src/main/assets/
models/onnx/yolov12n.onnx                               →  app/src/main/assets/
```

---

## ✅ BUILD & RUN

```powershell
# In Android Studio:
1. Build → Make Project
2. Run → Run 'app'
3. Select Device: Infinix Hot 20S
4. Wait 2-3 minutes for first install

# Or from terminal:
./gradlew build
./gradlew installDebug
adb shell am start -n com.example.yolodepthestimator/.MainActivity
```

---

## 🧪 QUICK TEST

After app launches:

- [ ] Camera preview visible
- [ ] Depth map showing
- [ ] FPS counter > 1
- [ ] Distance updating
- [ ] TTS announces values
- [ ] Vibration working

**If any fails**: Check `templates/TESTING_DEBUGGING_GUIDE.md`

---

## 🎓 FILES REFERENCE

| File                              | Purpose                      |
| --------------------------------- | ---------------------------- |
| `ANDROID_IMPLEMENTATION_FINAL.md` | THIS → Complete step-by-step |
| `QUICK_START.md`                  | 5-minute ultra-quick version |
| `COMPLETE_SETUP_GUIDE.md`         | Detailed 7-step guide        |
| `TESTING_DEBUGGING_GUIDE.md`      | Troubleshooting help         |
| `ANDROID_IMPLEMENTATION.md`       | Architecture reference       |

---

## 🚨 COMMON ISSUES & FIXES

### App crashes on startup

```
→ Check: assets/ folder has models
→ Check: package names match
→ Check: permissions in manifest
```

### Camera not working

```
→ Grant camera permission (Settings)
→ Check: android:name="android.permission.CAMERA"
→ Check: CameraX initialization in code
```

### Very slow (< 1 FPS)

```
→ Normal for CPU-only! (Expected 2-3 FPS)
→ INT8 already optimized
→ Check logcat for ONNX errors
```

### Build fails

```
./gradlew clean
./gradlew sync --info
File → Invalidate Caches → Restart
```

---

## 📊 EXPECTED RESULTS

- **FPS**: 2-3 (acceptable for accessibility)
- **Latency**: 330-400 ms per frame
- **Accuracy**: ±15% distance error
- **Memory**: ~250-400 MB
- **Battery**: Efficient (INT8)

---

## 🎯 SUCCESS =

✅ Code compiles
✅ Deploys to device  
✅ Camera works
✅ Depth shows
✅ Can demonstrate

**READY FOR THESIS!**

---

**Start Time**: Now  
**Estimated Time**: 2-3 hours  
**Next Step**: Open Android Studio → Create Project!
