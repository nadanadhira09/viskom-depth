# Setup ONNX Runtime untuk Android Studio

## Project YOLO + Depth Estimation

---

## 📱 **Persyaratan Sistem**

| Item               | Spesifikasi                         |
| ------------------ | ----------------------------------- |
| **Android Studio** | Arctic Fox 2021.3.1 atau lebih baru |
| **Min SDK**        | Android 8.0 (API 26)                |
| **Target SDK**     | Android 14+ (API 34)                |
| **JDK**            | 11+                                 |
| **Gradle**         | 8.0+                                |
| **Language**       | Kotlin + Jetpack Compose            |

---

## 🔧 **Step 1: Setup Project Baru di Android Studio**

### 1.1 Buat project baru

```bash
File → New → New Android Project
- Template: Empty Activity (Compose)
- Name: YoloDepthEstimator
- Package: com.example.yolodepthestimator
- Min SDK: 26 (Android 8.0)
- Language: Kotlin
```

### 1.2 Folder Structure Target

```
YoloDepthEstimator/
├── app/
│   ├── src/main/
│   │   ├── assets/
│   │   │   ├── yolov12n.onnx         (copy dari project Python)
│   │   │   └── depth_anything_v2_vits_quantized_int8.onnx
│   │   ├── java/com/example/yolodepthestimator/
│   │   │   ├── ui/
│   │   │   │   ├── screens/
│   │   │   │   │   ├── CameraScreen.kt
│   │   │   │   │   └── ResultScreen.kt
│   │   │   │   └── components/
│   │   │   ├── data/
│   │   │   │   ├── models/
│   │   │   │   │   ├── YoloModel.kt
│   │   │   │   │   └── DepthModel.kt
│   │   │   │   └── inference/
│   │   │   │       ├── OnnxRuntime.kt
│   │   │   │       ├── Preprocessor.kt
│   │   │   │       └── Postprocessor.kt
│   │   │   └── MainActivity.kt
│   ├── build.gradle.kts
│   └── proguard-rules.pro
└── settings.gradle.kts
```

---

## 📦 **Step 2: Konfigurasi Gradle Dependencies**

### 2.1 Edit `app/build.gradle.kts`

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

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
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
    // Core Android
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // Jetpack Compose
    implementation("androidx.activity:activity-compose:1.8.1")
    implementation("androidx.compose.ui:ui:1.6.4")
    implementation("androidx.compose.material3:material3:1.2.0")
    implementation("androidx.compose.ui:ui-tooling-preview:1.6.4")
    debugImplementation("androidx.compose.ui:ui-tooling:1.6.4")

    // Camera
    implementation("androidx.camera:camera-camera2:1.4.0-alpha03")
    implementation("androidx.camera:camera-lifecycle:1.4.0-alpha03")
    implementation("androidx.camera:camera-view:1.4.0-alpha03")
    implementation("androidx.camera:camera-extensions:1.4.0-alpha03")

    // ONNX Runtime
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.3")

    // Image Processing (Optional tapi recommended)
    implementation("org.opencv:opencv-android:4.8.0")

    // MLKit untuk utility (optional)
    implementation("com.google.android.gms:play-services-mlkit-common:16.6.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
```

### 2.2 Edit `settings.gradle.kts`

```kotlin
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "YoloDepthEstimator"
include(":app")
```

---

## 🧠 **Step 3: Core Model Classes**

### 3.1 File: `data/models/YoloModel.kt`

```kotlin
package com.example.yolodepthestimator.data.models

import android.graphics.Bitmap
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream

data class BoundingBox(
    val classLabel: String,
    val confidence: Float,
    val x1: Float, val y1: Float, val x2: Float, val y2: Float
)

class YoloModel(
    private val env: OrtEnvironment,
    modelBytes: ByteArray
) {
    private val session: OrtSession
    private val inputName: String

    init {
        val sessionOptions = OrtSession.SessionOptions().apply {
            addNnapi() // Enable NNAPI for Android neural accelerator
            setIntraOpNumThreads(4)
        }
        session = env.createSession(modelBytes, sessionOptions)
        inputName = session.inputNames.first()
    }

    suspend fun detect(bitmap: Bitmap): List<BoundingBox> = withContext(Dispatchers.Default) {
        // Preprocessing
        val inputSize = 640
        val resizedBmp = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val floatBuffer = preprocessImage(resizedBmp)

        // Inference
        val inputShape = longArrayOf(1L, 3L, 640L, 640L)
        val ortTensor = ai.onnxruntime.OnnxTensor.createTensor(env, floatBuffer, inputShape)
        val results = session.run(mapOf(inputName to ortTensor))

        // Postprocessing
        parseYoloOutput(results, bitmap.width, bitmap.height)
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val floatArray = FloatArray(bitmap.width * bitmap.height * 3)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        pixels.forEachIndexed { idx, pixel ->
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            floatArray[idx] = (r - mean[0]) / std[0]
            floatArray[bitmap.width * bitmap.height + idx] = (g - mean[1]) / std[1]
            floatArray[2 * bitmap.width * bitmap.height + idx] = (b - mean[2]) / std[2]
        }

        return floatArray
    }

    private fun parseYoloOutput(results: Map<String, Any>, origW: Int, origH: Int): List<BoundingBox> {
        // Implementasi parsing output YOLO (detaily tergantung model architecture)
        // Placeholder untuk tahap ini
        return emptyList()
    }

    fun close() {
        session.close()
    }
}
```

### 3.2 File: `data/models/DepthModel.kt`

```kotlin
package com.example.yolodepthestimator.data.models

import android.graphics.Bitmap
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class DepthModel(
    private val env: OrtEnvironment,
    modelBytes: ByteArray
) {
    private val session: OrtSession
    private val inputName: String

    init {
        val sessionOptions = OrtSession.SessionOptions().apply {
            addNnapi()
            setIntraOpNumThreads(4)
        }
        session = env.createSession(modelBytes, sessionOptions)
        inputName = session.inputNames.first()
    }

    suspend fun estimateDepth(bitmap: Bitmap): FloatArray = withContext(Dispatchers.Default) {
        val inputSize = 252
        val resizedBmp = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val floatBuffer = preprocessImage(resizedBmp)

        // Inference
        val inputShape = longArrayOf(1L, 3L, 252L, 252L)
        val ortTensor = ai.onnxruntime.OnnxTensor.createTensor(env, floatBuffer, inputShape)
        val results = session.run(mapOf(inputName to ortTensor))

        // Extract output
        (results[0] as Array<*>).flatMap { it as List<*> }
            .map { (it as Number).toFloat() }
            .toFloatArray()
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val floatArray = FloatArray(bitmap.width * bitmap.height * 3)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        pixels.forEachIndexed { idx, pixel ->
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            floatArray[idx] = (r - mean[0]) / std[0]
            floatArray[bitmap.width * bitmap.height + idx] = (g - mean[1]) / std[1]
            floatArray[2 * bitmap.width * bitmap.height + idx] = (b - mean[2]) / std[2]
        }

        return floatArray
    }

    fun close() {
        session.close()
    }
}
```

---

## 🎬 **Step 4: Camera & Composable UI**

### 4.1 File: `ui/screens/CameraScreen.kt`

```kotlin
package com.example.yolodepthestimator.ui.screens

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.LifecycleOwner
import com.example.yolodepthestimator.data.models.BoundingBox
import java.util.concurrent.Executors

@Composable
fun CameraScreen(
    onFrameAnalyzed: (List<BoundingBox>) -> Unit
) {
    var previewView: PreviewView? by remember { mutableStateOf(null) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    LaunchedEffect(Unit) {
        // Initialize camera
    }

    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        // Camera preview
        AndroidView(
            factory = { context ->
                PreviewView(context).apply {
                    previewView = this
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        )

        // Control buttons
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(onClick = { /* Capture & Analyze */ }) {
                Text("Analyze")
            }
            Button(onClick = { /* Toggle model */ }) {
                Text("Switch Model")
            }
        }
    }
}
```

---

## 📋 **Step 5: AndroidManifest.xml Permissions**

### 5.1 Edit `AndroidManifest.xml`

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">

    <!-- Permissions -->
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.INTERNET" />

    <!-- Features -->
    <uses-feature android:name="android.hardware.camera" />
    <uses-feature android:name="android.hardware.camera.autofocus" />

    <application
        android:allowBackup="true"
        android:dataExtractionRules="@xml/data_extraction_rules"
        android:fullBackupContent="@xml/backup_rules"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:supportsRtl="true"
        android:theme="@style/Theme.YoloDepthEstimator"
        android:usesCleartextTraffic="false">

        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:theme="@style/Theme.YoloDepthEstimator">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>

</manifest>
```

---

## 🚀 **Step 6: Siapkan Model Files**

### 6.1 Copy model files ke Android

```bash
# Di project Python Anda
cp models/onnx/yolov12n.onnx /path/to/Android/app/src/main/assets/
cp models/onnx/depth_anything_v2_vits_quantized_int8.onnx /path/to/Android/app/src/main/assets/
```

### 6.2 Verifikasi file ada di assets/

```
app/src/main/assets/
├── yolov12n.onnx
└── depth_anything_v2_vits_quantized_int8.onnx
```

---

## ⚙️ **Step 7: ProGuard Rules (Opsional tapi Recommended)**

### 7.1 Edit `app/proguard-rules.pro`

```proguard
# ONNX Runtime
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }

# Keep native methods
-keepclassmembers class * {
    native <methods>;
}

# Jetpack Compose
-keep @androidx.compose.runtime.Stable class * { *; }
```

---

## 📊 **Checklist Build & Deploy**

- [ ] Android SDK 26+ terinstal
- [ ] Gradle 8.0+ configured
- [ ] ONNX Runtime dependency added
- [ ] Model files di `assets/`
- [ ] Permissions di AndroidManifest.xml
- [ ] ProGuard rules configured
- [ ] Build variant: Release (untuk optimization)
- [ ] Test device terhubung atau emulator running

### Build & Run:

```bash
./gradlew build
./gradlew installRelease

# atau
Shift + F10 (di Android Studio)
```

---

## 🔍 **Performance Tuning**

| Optimization     | Code                                                |
| ---------------- | --------------------------------------------------- |
| **Enable NNAPI** | `sessionOptions.addNnapi()`                         |
| **GPU Compute**  | `sessionOptions.addCoreMLFlags()` / `addDirectML()` |
| **Thread Pool**  | `setIntraOpNumThreads(4)`                           |
| **Quantization** | Use INT8 model files                                |
| **Memory**       | Load model once di `ViewModel`                      |

---

## 📱 **Target Devices untuk Testing**

- ✅ Midrange (Snapdragon 778G+) - Recommended
- ✅ Flagship (Snapdragon 8 Gen 3) - Best performance
- ⚠️ Budget (Snapdragon 680) - CPU-only, slower
- ❌ API < 26 - Not supported

---

## 🆘 **Common Issues & Solutions**

| Issue                  | Solusi                                          |
| ---------------------- | ----------------------------------------------- |
| `UnsatisfiedLinkError` | Update ONNX Runtime version                     |
| `OutOfMemoryError`     | Reduce input resolution, use model quantization |
| `Camera not starting`  | Check permissions, grant runtime permissions    |
| `Slow inference`       | Enable NNAPI, use GPU, reduce input size        |

---

## 📚 **Referensi Lengkap**

- ONNX Runtime Android: https://onnxruntime.ai/docs/install/
- Jetpack Compose: https://developer.android.com/compose
- Camera X: https://developer.android.com/training/camerax
- Android Developers: https://developer.android.com/

---

**Next Steps:**

1. ✅ Siapkan AS project
2. ✅ Configure gradle
3. ✅ Buat model classes
4. ✅ Setup camera UI
5. ✅ Deploy & test

**Siap untuk implementasi lebih detail?** 🚀
