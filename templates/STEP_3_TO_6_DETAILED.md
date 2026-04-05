# 🎯 Step-by-Step Detail Guide: Step 3-6 (dengan Visual)

> Panduan SANGAT detail untuk copy files dan deploy (dengan instruksi klik-demi-klik)

---

## 📋 Prerequisites Checklist

Pastikan sudah selesai ini dulu:

- [ ] Android Studio sudah buka
- [ ] Project "YoloDepthEstimator" sudah dibuat
- [ ] Gradle dependencies sudah diupdate di `app/build.gradle.kts`
- [ ] Klik "Sync Now" selesai (tidak ada error)

**Jika belum:** Baca QUICK_START.md Step 1-2 dulu, baru lanjut ke sini.

---

# STEP 3: Copy Model Files ke Assets

## 3.1 Create Assets Folder (Cara 1: Via Android Studio - RECOMMENDED)

```
📍 Di Android Studio:

1. Buka "Project" panel (di sebelah kiri)
   - Jika tidak terlihat: View → Tool Windows → Project

2. Expand folder:
   app → src → main

   Anda akan melihat:
   ├── java
   ├── res
   └── (tidak ada "assets" - kita akan bikin)

3. Right-click pada folder "main"
   → Pilih: New → Folder → Assets Folder

4. Dialog akan muncul, klik OK

Sekarang folder structure menjadi:
   ├── java
   ├── res
   └── assets  ✅ BARU DIBUAT
```

## 3.2 Copy Model Files

```
📍 Buka file explorer (Windows):

1. Buka folder: D:\S2\Semester 2\Visi Komputer\viskom\models\onnx

   Anda akan lihat files:
   ├── depth_anything_v2_vits.onnx (95 MB)
   ├── depth_anything_v2_vits_quantized_int8.onnx (24 MB)  ⭐ AMBIL INI
   └── yolov12n.onnx (26 MB)

2. Seleksi file:
   depth_anything_v2_vits_quantized_int8.onnx

   (PILIH YANG INT8, jangan yang 95MB)

3. Ctrl+C (copy)

4. Buka Android Studio project:
   app/src/main/assets/

5. Ctrl+V (paste) di folder assets

Hasil:
   assets/
   └── depth_anything_v2_vits_quantized_int8.onnx ✅
```

## 3.3 Verify Assets Folder

```
✅ Cek di Android Studio:

   app → src → main → assets

   Harus ada file:
   - depth_anything_v2_vits_quantized_int8.onnx (24 MB)

   Ukuran file penting! Jika salah model, cek ulang.
```

---

# STEP 4: Copy Template Kotlin Files

## 4.1 Create Folder Structure di Android Project

```
📍 Di Android Studio, create folders dengan urutan ini:

Mulai dari: app/src/main/java/com/example/yolodepthestimator/

✅ STRUCTURE YANG HARUS DIBUAT:
├── data/
│   └── inference/     ← CREATE THIS FOLDER
├── ui/
│   ├── screens/       ← CREATE THIS FOLDER
│   └── theme/         ← CREATE THIS FOLDER
└── viewmodel/         ← CREATE THIS FOLDER

HOW TO CREATE:

a) Right-click pada: com/example/yolodepthestimator/
   → New → Package
   → Type: "data" → OK

   Ulangi untuk: inference, ui, screens, theme, viewmodel

   ATAU:

b) Right-click pada: com/example/yolodepthestimator/
   → New → Directory
   → Create directory untuk setiap folder
```

## 4.2 Copy Files (Detailed)

```
📍 FOLDER STRUCTURE FINAL:

YoloDepthEstimator/app/src/main/java/com/example/yolodepthestimator/
│
├── 📄 MainActivity.kt ............................ (COPY KE SINI)
│
├── 📁 data/
│   └── 📁 inference/
│       ├── 📄 OnnxRuntimeManager.kt ............ (COPY KE SINI)
│       ├── 📄 ImagePreprocessor.kt ............ (COPY KE SINI)
│       └── 📄 InferenceHelpers.kt ............. (COPY KE SINI)
│
├── 📁 ui/
│   ├── 📁 screens/
│   │   └── 📄 CameraScreen.kt ................. (COPY KE SINI)
│   └── 📁 theme/
│       └── 📄 Theme.kt ........................ (BUAT FILE INI SENDIRI)
│
└── 📁 viewmodel/
    └── 📄 CameraViewModel.kt .................. (COPY KE SINI)


STEP-BY-STEP COPY:

1️⃣ COPY: OnnxRuntimeManager.kt
   ├─ From: D:\...\viskom\templates\OnnxRuntimeManager.kt
   ├─ To:   Android_Project/app/src/main/java/com/example/yolodepthestimator/data/inference/
   │
2️⃣ COPY: ImagePreprocessor.kt
   ├─ From: D:\...\viskom\templates\ImagePreprocessor.kt
   ├─ To:   Android_Project/app/src/main/java/com/example/yolodepthestimator/data/inference/
   │
3️⃣ COPY: InferenceHelpers.kt
   ├─ From: D:\...\viskom\templates\InferenceHelpers.kt
   ├─ To:   Android_Project/app/src/main/java/com/example/yolodepthestimator/data/inference/
   │
4️⃣ COPY: CameraScreen.kt
   ├─ From: D:\...\viskom\templates\CameraScreen.kt
   ├─ To:   Android_Project/app/src/main/java/com/example/yolodepthestimator/ui/screens/
   │
5️⃣ COPY: CameraViewModel.kt
   ├─ From: D:\...\viskom\templates\CameraViewModel.kt
   ├─ To:   Android_Project/app/src/main/java/com/example/yolodepthestimator/viewmodel/
   │
6️⃣ COPY: MainActivity.kt ⚠️ REPLACE YANG ADA
   ├─ From: D:\...\viskom\templates\MainActivity.kt
   ├─ To:   Android_Project/app/src/main/java/com/example/yolodepthestimator/
   └─ ACTION: Replace existing file
```

## 4.3 Create Theme.kt File (Buat sendiri)

```
📍 Di Android Studio:

1. Right-click: app/src/main/java/.../ui/theme/
   → New → Kotlin File/Class
   → Name: "Theme"
   → OK

2. Copy-paste code ini ke Theme.kt:

```

---

# STEP 5: Create/Update Key Files

## 5.1 Update MainActivity.kt

**Status:** Sudah di-copy di step 4 → OK, next!

## 5.2 Create Theme.kt

```kotlin
// 📁 File: app/src/main/java/com/example/yolodepthestimator/ui/theme/Theme.kt

package com.example.yolodepthestimator.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable

@Composable
fun YoloDepthEstimatorTheme(content: @Composable () -> Unit) {
    MaterialTheme(content = content)
}
```

## 5.3 Update AndroidManifest.xml

```
📍 Di Android Studio:

1. Buka: app/src/main/AndroidManifest.xml

2. Replace ENTIRE content dengan:
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:tools="http://schemas.android.com/tools">

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
        android:dataExtractionRules="@xml/data_extraction_rules"
        android:fullBackupContent="@xml/backup_rules"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:supportsRtl="true"
        android:theme="@style/Theme.YoloDepthEstimator">

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

# STEP 6: Build & Deploy

## 6.1 Resolve Gradle Sync Issues (Jika ada error)

```
📍 Jika ada error merah saat copy files:

1. Top menu: Build → Clean Project
   (Tunggu selesai)

2. File → Sync project with Gradle Files
   (atau Ctrl+Shift+S)

3. Tunggu Gradle sync selesai
   Harus terlihat: "Build completed successfully"
```

## 6.2 Build APK di Android Studio

```
📍 BUILD PROCESS:

1. Menu: Build → Build Bundles/APKs → Build APK

   ATAU shortcut:
   Ctrl+B

2. Bottom panel akan muncul:
   "BUILD: Starting Gradle build..."

   TUNGGU SAMPAI SELESAI!
   Ini akan memakan 2-3 menit.

3. Setelah selesai, pesan akan muncul:
   "BUILD: Build completed successfully"

   Green ✅ = OK, next step!
   Red ❌ = Ada error, scroll up cek pesan error

JIKA ADA ERROR:
├─ Check: ONNX dependency ada di build.gradle.kts?
├─ Check: Min SDK = 26?
├─ Check: Sync gradle sudah berhasil?
└─ Check: File paths benar?
```

## 6.3 Setup Device atau Emulator

```
PILIH SALAH SATU:

OPTION A: Physical Device
├─ 1. Connect phone ke laptop via USB
├─ 2. Enable USB debugging:
│     Settings → Developer Options → USB Debugging ON
├─ 3. Klik OK di notifikasi "Allow USB debugging"
└─ 4. Siap untuk deploy!

OPTION B: Android Emulator
├─ 1. Buka: Tools → Device Manager
├─ 2. Jika sudah ada device, click "Play" untuk start
├─ 3. Jika belum ada, create new:
│     - Click: Create Device
│     - Select: Phone (Pixel 5)
│     - Select System Image: Android 13 (API 33+)
│     - Click Finish
│     - Click Play untuk start
├─ 4. Tunggu emulator boot (~1-2 menit)
└─ 5. Siap untuk deploy!
```

## 6.4 Deploy (Run) Aplikasi

```
📍 DEPLOY STEP:

1. Menu: Run → Run 'app'

   ATAU shortcut:
   Ctrl+R

2. Dialog akan muncul:
   "Select Deployment Target"

   ├─ Jika physical device terpilih → OK click
   ├─ Jika emulator terpilih → OK click
   └─ Jika tidak ada pilihan:
      - Check device/emulator connected properly
      - Refresh: Tools → Device Manager

3. Click OK, Android Studio akan:
   ├─ Compile code
   ├─ Generate APK
   ├─ Install ke device/emulator
   └─ Launch aplikasi

4. Tunggu ~2-3 menit sampai selesai

5. Di device/emulator:
   ├─ Aplikasi akan launch
   ├─ Camera permission popup muncul
   ├─ Klik "Allow"
   ├─ Camera preview harus muncul
   └─ FPS counter terlihat

   ✅ SUCCESS!
```

---

# 🎉 FINAL CHECKLIST

Sebelum run, pastikan semua ini ✅:

```
✅ STEP 1-2: Project & Gradle
  ├─ [ ] Project "YoloDepthEstimator" created
  ├─ [ ] Min SDK = 26
  ├─ [ ] build.gradle.kts updated
  └─ [ ] Sync successful (green ✅)

✅ STEP 3: Assets
  ├─ [ ] assets/ folder ada
  ├─ [ ] depth_anything_v2_vits_quantized_int8.onnx ada
  └─ [ ] File size = 24 MB (bukan 95 MB!)

✅ STEP 4: Kotlin Files
  ├─ [ ] data/inference/ folder ada
  │  ├─ OnnxRuntimeManager.kt ✓
  │  ├─ ImagePreprocessor.kt ✓
  │  └─ InferenceHelpers.kt ✓
  ├─ [ ] ui/screens/ folder ada
  │  └─ CameraScreen.kt ✓
  ├─ [ ] ui/theme/ folder ada
  │  └─ Theme.kt ✓
  ├─ [ ] viewmodel/ folder ada
  │  └─ CameraViewModel.kt ✓
  └─ [ ] MainActivity.kt replaced ✓

✅ STEP 5: Config Files
  ├─ [ ] AndroidManifest.xml updated
  └─ [ ] Permissions ada (CAMERA, INTERNET)

✅ STEP 6: Build & Deploy
  ├─ [ ] Ctrl+B Build successful (green ✅)
  ├─ [ ] Device/Emulator connected
  ├─ [ ] Ctrl+R Deploy successful
  └─ [ ] App launches dengan camera preview
```

---

# 🆘 Troubleshooting: Jika Ada Error

| Error                                                   | Solusi                                                  |
| ------------------------------------------------------- | ------------------------------------------------------- |
| "Package com.example.yolodepthestimator does not exist" | Folder structure salah, check package path              |
| "Unresolved reference: OnnxTensor"                      | ONNX dependency belum add di gradle, sync ulang         |
| "Build failed"                                          | Clear cache: Build → Clean Project, rebuild             |
| "App crashes on launch"                                 | Check MainActivity.kt copied benar, check manifest      |
| "Camera black screen"                                   | Check permission granted, camera not in use by app lain |
| "Models not found"                                      | Check file ada di assets/, path benar                   |

---

# 🚀 Next After Success

Jika build & deploy berhasil:

1. ✅ Camera preview muncul
2. ✅ FPS counter terlihat (0.5-1 FPS)
3. ✅ App stable tanpa crash

**Maka:**
→ Buka: **TESTING_DEBUGGING_GUIDE.md** untuk:

- Performance optimization
- Testing pada devices lain
- Debugging tips

---

## 📱 Expected Result

```
┌─────────────────────────────────────┐
│   YoloDepthEstimator App            │
├─────────────────────────────────────┤
│                                     │
│   ┌─────────────────────────────┐   │
│   │   CAMERA PREVIEW            │   │
│   │   (Live feed dari camera)   │   │
│   └─────────────────────────────┘   │
│                                     │
│   FPS: 0.8                          │
│   Latency: 680ms                    │
│   Detections: 0                     │
│                                     │
│   [PAUSE] [CAPTURE]                 │
│                                     │
└─────────────────────────────────────┘
```

**Beres! Aplikasi Anda sudah jalan! 🎉**

---

## 💬 Masih Bingung?

Tanya saja:

1. "Bingung di mana tepatnya?"
2. "Ada error apa?"
3. "Screenshot error message"

Saya bantu fix! 💪
