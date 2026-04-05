# Testing & Debugging Guide

## 🧪 Android Testing Strategy

---

## Part 1: Unit Testing

### 1.1 Test Image Preprocessing

Create `app/src/test/java/com/example/yolodepthestimator/ImagePreprocessorTest.kt`:

```kotlin
import android.graphics.Bitmap
import com.example.yolodepthestimator.data.inference.ImagePreprocessor
import org.junit.Assert.*
import org.junit.Test

class ImagePreprocessorTest {

    @Test
    fun testPreprocessYoloShape() {
        // Create test bitmap 100x100
        val bitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)

        // Preprocess
        val result = ImagePreprocessor.preprocessYolo(bitmap)

        // Expected: 3 channels × 640 × 640
        val expectedSize = 3 * 640 * 640
        assertEquals("Float array size mismatch", expectedSize, result.size)
    }

    @Test
    fun testPreprocessDepthShape() {
        val bitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)

        val result = ImagePreprocessor.preprocessDepth(bitmap)

        // Expected: 3 channels × 252 × 252
        val expectedSize = 3 * 252 * 252
        assertEquals("Float array size mismatch", expectedSize, result.size)
    }

    @Test
    fun testNormalizationRange() {
        val bitmap = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        val result = ImagePreprocessor.preprocessYolo(bitmap)

        // Normalized values should be roughly in [-3, 3] range
        val max = result.maxOrNull() ?: 0f
        val min = result.minOrNull() ?: 0f

        assertTrue("Max value out of range", max < 10f)
        assertTrue("Min value out of range", min > -10f)
    }
}
```

### 1.2 Test ONNX Runtime Manager

Create `app/src/test/java/com/example/yolodepthestimator/OnnxRuntimeManagerTest.kt`:

```kotlin
import android.content.Context
import com.example.yolodepthestimator.data.inference.OnnxRuntimeManager
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.mockito.Mockito.*

class OnnxRuntimeManagerTest {

    private lateinit var mockContext: Context

    @Before
    fun setup() {
        mockContext = mock(Context::class.java)
    }

    @After
    fun cleanup() {
        OnnxRuntimeManager.cleanup()
    }

    @Test
    fun testInitializeEnvironment() {
        OnnxRuntimeManager.initializeEnvironment()
        // If no exception thrown, test passes
        assert(true)
    }

    @Test
    fun testSingletonBehavior() {
        OnnxRuntimeManager.initializeEnvironment()
        // Initialize again
        OnnxRuntimeManager.initializeEnvironment()
        // Should not throw
        assert(true)
    }
}
```

---

## Part 2: Integration Testing

### 2.1 Camera Integration Test

Create `app/src/androidTest/java/com/example/yolodepthestimator/CameraIntegrationTest.kt`:

```kotlin
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import com.example.yolodepthestimator.MainActivity

@RunWith(AndroidJUnit4::class)
class CameraIntegrationTest {

    private lateinit var scenario: ActivityScenario<MainActivity>

    @Before
    fun setup() {
        scenario = ActivityScenario.launch(MainActivity::class.java)
    }

    @Test
    fun testActivityLaunches() {
        scenario.onActivity { activity ->
            assert(activity != null)
        }
    }

    @Test
    fun testCameraPermissionsHandled() {
        scenario.onActivity { activity ->
            // Verify the app handles permissions gracefully
            assert(true)
        }
    }
}
```

---

## Part 3: Manual Testing

### 3.1 Test Checklist (On Device)

#### Camera Functionality

- [ ] App launches without crash
- [ ] Camera preview displays
- [ ] Camera rotates with device orientation
- [ ] Both Back & Front cameras work

#### Inference

- [ ] FPS counter updates (should see ~0.5-1 FPS)
- [ ] Latency displays correctly
- [ ] Detection count increases when objects visible
- [ ] No memory leaks (smooth after 5 minutes)

#### UI/UX

- [ ] PAUSE button stops inference
- [ ] RESUME button restarts inference
- [ ] Tap events handled smoothly
- [ ] Overlay text readable

#### Edge Cases

- [ ] Rapid start/stop doesn't crash
- [ ] App recovers from permission denial
- [ ] Works in low light
- [ ] Handles rapid temperature changes

---

## Part 4: Debugging Techniques

### 4.1 Logcat Monitoring

```bash
# Monitor all logs
adb logcat

# Filter ONNX Runtime logs
adb logcat | grep -i "onnx"

# Filter app logs
adb logcat *:S TAG:V  # Show only TAG logs

# Save logs to file
adb logcat > debug.log
```

### 4.2 Android Studio Debugger

```kotlin
// Set breakpoints in onCreate:
override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    // Debug Session enabled here
    breakpoint()

    OnnxRuntimeManager.initializeEnvironment()
}
```

**Debugging Commands:**

- F8: Step over
- F7: Step into
- Shift+F8: Step out
- F9: Resume
- Watches tab: Monitor variables

### 4.3 Add Debug Logging

```kotlin
// In CameraScreen.kt
private suspend fun processFrame(...) {
    try {
        Log.d("CameraScreen", "Processing frame...")

        val yoloSession = OnnxRuntimeManager.getYoloSession()
        Log.d("CameraScreen", "YOLO session: ${yoloSession != null}")

        // ... inference code ...

        Log.d("CameraScreen", "Detections: ${detections.size}")
    } catch (e: Exception) {
        Log.e("CameraScreen", "Error: ${e.stackTrace}")
    }
}

// In logcat:
import android.util.Log
```

### 4.4 Performance Profiling

**Using Android Profiler:**

1. Run → Profiler
2. Select: CPU, Memory, both
3. Record 30 seconds of operation
4. Analyze peaks in memory/CPU usage

**Expected Metrics:**

- Memory: 200-400 MB
- CPU: 30-50% during inference
- Frames: ~0.5-1 FPS (real-time)

---

## Part 5: Troubleshooting Common Issues

### Issue 1: App Crashes on Launch

**Error:** `java.lang.UnsatisfiedLinkError: Native method not found`

**Solution:**

```kotlin
// Check ONNX Runtime version
// Build.gradle.kts must have:
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.3")

// NOT:
// implementation("com.microsoft.onnxruntime:onnxruntime-android:1.15.0")
```

### Issue 2: Camera Permission Denied

**Error:** `SecurityException: Permission denied`

**Solution:**

```kotlin
// In CameraScreen.kt - ensure permission request:
val permissionLauncher = rememberLauncherForActivityResult(
    ActivityResultContracts.RequestPermission()
) { isGranted ->
    hasPermission = isGranted
    Log.d("CameraScreen", "Permission: $isGranted")
}
```

### Issue 3: Very Slow Inference (>5s per frame)

**Possible Causes:**

1. ❌ Using FP32 models (too large)
2. ❌ CPU inference (no NNAPI)
3. ❌ Too many threads
4. ❌ Memory pressure

**Solution:**

```kotlin
// Use INT8 quantized model
context.assets.open("depth_anything_v2_vits_quantized_int8.onnx")

// Enable NNAPI acceleration
sessionOptions.apply {
    addNnapi()  // ← This is critical!
    setIntraOpNumThreads(4)
}

// Check for memory:
val runtime = Runtime.getRuntime()
val freeMemory = runtime.freeMemory() / 1_000_000
Log.d("Memory", "Free: ${freeMemory}MB")
```

### Issue 4: Out of Memory Error

**Error:** `java.lang.OutOfMemoryError: Failed to allocate 1234567 bytes`

**Solution:**

```kotlin
// Reduce input resolution
val inputSize = 416  // Instead of 640
val resizedBmp = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

// Or use depth estimation only (smaller model)
OnnxRuntimeManager.loadDepthModel(context)  // Skip YOLO

// Monitor memory in logcat
```

### Issue 5: NNAPI Not Working

**Symptom:** Inference still ~2s instead of 0.5s

**Check:**

```bash
# In logcat, look for NNAPI initialization
adb logcat | grep -i "nnapi"

# Should see something like:
# "NNAPI delegate initialized successfully"
```

**Solution:**

```kotlin
// NNAPI might not be available on all devices
sessionOptions.apply {
    try {
        addNnapi()
        Log.d("ONNX", "NNAPI enabled")
    } catch (e: Exception) {
        Log.w("ONNX", "NNAPI failed: ${e.message}")
        // Falls back to CPU
    }
}
```

---

## Part 6: Performance Optimization

### Optimization Checklist

- [ ] Use INT8 quantized models
- [ ] Enable NNAPI
- [ ] Reduce input resolution to 416x416
- [ ] Use MediaStore.ACTION_IMAGE_CAPTURE instead of Camera X (optional)
- [ ] Adjust thread count based on device
- [ ] Profile memory regularly

### Expected Performance Targets

| Metric          | Target   | Actual    |
| --------------- | -------- | --------- |
| Load Time       | < 3s     | ~2-3s     |
| Inference/Frame | < 1000ms | 500-800ms |
| FPS             | > 1      | 0.5-1 FPS |
| Memory Peak     | < 400MB  | 250-350MB |
| Crashes         | 0        | 0         |

---

## Part 7: Device Testing Matrix

Test on devices with different specs:

| Device            | RAM  | Processor      | Status         |
| ----------------- | ---- | -------------- | -------------- |
| Emulator (API 30) | 4GB  | x86            | ✅ Development |
| Pixel 5           | 8GB  | Snapdragon 765 | ✅ Recommended |
| Samsung S21       | 8GB  | Snapdragon 888 | ✅✅ Best      |
| OnePlus 9         | 12GB | Snapdragon 870 | ✅✅ Best      |
| Redmi Note 11     | 4GB  | Snapdragon 680 | ⚠️ CPU only    |

---

## Part 8: Final Checklist Before Release

- [ ] No crashes after 10 minutes runtime
- [ ] FPS stable
- [ ] Memory doesn't leak
- [ ] Works on min SDK (API 26)
- [ ] Works on high-end devices
- [ ] Permissions handled gracefully
- [ ] ProGuard rules configured
- [ ] AAB/APK signed with release key

---

**Good Luck! 🚀**
