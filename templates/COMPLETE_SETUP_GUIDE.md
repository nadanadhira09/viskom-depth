# Android Implementation Complete Guide

## 📋 Full Step-by-Step Implementation

### PART 1: Project Setup

#### Step 1: Create New Android Project

```
1. Open Android Studio
2. File → New → New Android Project
3. Select "Empty Activity (Compose)" template
4. Project details:
   - Name: YoloDepthEstimator
   - Package: com.example.yolodepthestimator
   - Save location: Choose a folder
   - Language: Kotlin
   - Minimum SDK: API 26 (Android 8.0)
```

#### Step 2: Configure Gradle

Edit `app/build.gradle.kts`:

```kotlin
plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.example.yolodepthestimator"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.yolodepthestimator"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.11"
    }
}

dependencies {
    // Core
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // Compose
    implementation("androidx.activity:activity-compose:1.8.1")
    implementation("androidx.compose.ui:ui:1.6.4")
    implementation("androidx.compose.material3:material3:1.2.0")
    implementation("androidx.compose.material:material-icons-extended:1.6.4")
    debugImplementation("androidx.compose.ui:ui-tooling:1.6.4")

    // Camera
    implementation("androidx.camera:camera-camera2:1.4.0-alpha03")
    implementation("androidx.camera:camera-lifecycle:1.4.0-alpha03")
    implementation("androidx.camera:camera-view:1.4.0-alpha03")

    // ONNX Runtime ⭐ CRITICAL
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.3")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
}
```

#### Step 3: Create Folder Structure

Create these directories under `app/src/main/`:

```
java/com/example/yolodepthestimator/
├── ui/
│   ├── screens/
│   ├── components/
│   └── theme/
├── data/
│   ├── inference/
│   └── models/
├── viewmodel/
└── MainActivity.kt
```

#### Step 4: Copy Assets

```bash
# Create assets folder if not exists
mkdir -p app/src/main/assets

# Copy ONNX models from Python project
cp path/to/viskom/models/onnx/yolov12n.onnx app/src/main/assets/
cp path/to/viskom/models/onnx/depth_anything_v2_vits_quantized_int8.onnx app/src/main/assets/
```

---

### PART 2: Copy Template Files

Copy these files dari `viskom/templates/` ke Android project:

#### 1. Inference Infrastructure

```bash
# OnnxRuntimeManager.kt → app/src/main/java/com/example/yolodepthestimator/data/inference/
# ImagePreprocessor.kt → app/src/main/java/com/example/yolodepthestimator/data/inference/
# InferenceHelpers.kt → app/src/main/java/com/example/yolodepthestimator/data/inference/
```

#### 2. UI Layer

```bash
# CameraScreen.kt → app/src/main/java/com/example/yolodepthestimator/ui/screens/
# MainActivity.kt → app/src/main/java/com/example/yolodepthestimator/
```

#### 3. ViewModel

```bash
# CameraViewModel.kt → app/src/main/java/com/example/yolodepthestimator/viewmodel/
```

---

### PART 3: Create Theme File (if needed)

Create `app/src/main/java/com/example/yolodepthestimator/ui/theme/Theme.kt`:

```kotlin
package com.example.yolodepthestimator.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable

@Composable
fun YoloDepthEstimatorTheme(content: @Composable () -> Unit) {
    MaterialTheme(content = content)
}
```

---

### PART 4: Update AndroidManifest.xml

Edit `app/src/main/AndroidManifest.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.INTERNET" />

    <uses-feature
        android:name="android.hardware.camera"
        android:required="true" />
    <uses-feature
        android:name="android.hardware.camera.autofocus"
        android:required="false" />

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:supportsRtl="true">

        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

    </application>

</manifest>
```

---

### PART 5: Create Run Configuration

1. **Edit Run Configuration:**
   - Click: Run → Edit Configurations
   - Select: Android App
   - Deployment: APK from Gradle

2. **Build APK:**

   ```
   Build → Build Bundle(s)/APK(s) → Build APK(s)
   ```

3. **Run on Device:**
   - Connect Android device (API 26+)
   - Enable USB debugging
   - Ctrl+R atau Run → Run 'app'

---

## ✅ Build & Deployment Checklist

- [ ] Gradle dependencies all added
- [ ] Min SDK set to API 26
- [ ] ONNX models copied to assets/
- [ ] All Kotlin files imported correctly
- [ ] AndroidManifest.xml updated
- [ ] Permissions configured
- [ ] Theme file created
- [ ] Build successful (no errors)
- [ ] APK generated

---

## 🧪 First Test Checklist

When app launches:

- [ ] Camera preview appears
- [ ] Stats overlay visible (FPS, Latency)
- [ ] No immediate crashes
- [ ] PAUSE/RESUME button works
- [ ] Permissions requested & granted
- [ ] Models loaded successfully

---

## 🐛 Common Build Issues & Solutions

| Issue                              | Solution                                       |
| ---------------------------------- | ---------------------------------------------- |
| `Unresolved reference OnnxRuntime` | Sync gradle again, check ONNX dependency       |
| `Could not find method addNnapi()` | Update ONNX Runtime to 1.16.3+                 |
| `Cannot create Camera provider`    | Check permissions in AndroidManifest           |
| `Assets file not found`            | Verify file in app/src/main/assets/            |
| `Out of Memory`                    | Reduce input image resolution, use INT8 models |
| `No compatible devices`            | Check min SDK, use emulator API 26+            |

---

## 📱 Recommended Test Devices

| Device        | Specs           | Recommendation          |
| ------------- | --------------- | ----------------------- |
| Emulator      | Pixel 5 API 30+ | ✅ Good for development |
| Samsung S21+  | Snapdragon 888+ | ✅✅ Best performance   |
| OnePlus 9+    | Snapdragon 870+ | ✅ Good performance     |
| Redmi Note 11 | Snapdragon 680  | ⚠️ CPU-only, slow       |

---

## Next: Testing & Debugging

See `TESTING_DEBUGGING_GUIDE.md`
