# Panduan Implementasi YOLO & Depth Anything V2 di Android (Kotlin)

Dokumen ini berisi rancangan pembaruan untuk proyek Android Anda (menggunakan Jetpack Compose dan ONNX Runtime) untuk **menggabungkan model Object Detection (YOLOv12) dan Monocular Depth Estimation (Depth Anything V2)**.

## 1. Pembaruan Data Class & Inisialisasi

```kotlin
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

// State untuk menampung dua model ONNX sekaligus
data class DualModelState(
    val env: OrtEnvironment?,
    val yoloSession: OrtSession?,
    val depthSession: OrtSession?,
    val isLoading: Boolean = true,
    val error: String? = null
)

// Pemuatan model di dalam fungsi Composable (Background Thread)
// val modelState by produceState(...) {
//     val env = OrtEnvironment.getEnvironment()
//     val yoloBytes = context.assets.open("yolov12n_mobile_quantized.onnx").readBytes()
//     val depthBytes = context.assets.open("depth_anything_v2_mobile_quantized.onnx").readBytes()
//
//     // Konfigurasi Delegasi Eksekusi & Optimasi (Kuantisasi + NPU/CPU)
//     val sessionOptions = OrtSession.SessionOptions().apply {
//         // Opsi 1: Mengaktifkan NNAPI (Neural Networks API) untuk memakai NPU/DSP di Android
//         // Ini sangat optimal untuk model kuantisasi INT8/FP16
//         addNnapi()
//
//         // Opsi 2 (Alternatif): Jika perangkat tidak pakai NPU, gunakan XNNPACK untuk optimasi CPU
//         // addXnnpack(mapOf("num_threads" to "4"))
//
//         setIntraOpNumThreads(4) // Atur jumlah thread CPU
//     }
//
//     val yoloSession = env.createSession(yoloBytes, sessionOptions)
//     val depthSession = env.createSession(depthBytes, sessionOptions)
//     value = DualModelState(env, yoloSession, depthSession, false, null)
// }
```

## 2. Kelas Data untuk Hasil Analisis

```kotlin
data class BoundingBox(
    val classLabel: String,
    val confidence: Float,
    val x1: Int, val y1: Int, val x2: Int, val y2: Int
)

data class DetectedObject(
    val boundingBox: BoundingBox,
    val distanceInMeters: Float
)

data class AnalysisResult(
    val objects: List<DetectedObject>,
    val fullDepthMap: Bitmap?,
    val inferenceTimeMs: Long
)
```

## 3. Logika Analyze (Penggabungan Tensor YOLO + Depth)

Berikut adalah fungsi `analyze` di dalam kelas `ImageAnalysis.Analyzer`.

```kotlin
override fun analyze(image: ImageProxy) {
    val startTime = System.currentTimeMillis()

    // 1. Ekstrak Bitmap Asli & Rotasi
    val originalBmp = image.toBitmap() ?: run { image.close(); return }
    val rotation = image.imageInfo.rotationDegrees.toFloat()
    val rotatedBmp = rotateBitmap(originalBmp, rotation)

    val origW = rotatedBmp.width
    val origH = rotatedBmp.height

    // ====================================================================
    // A. INFERENSI YOLO (OBJECT DETECTION)
    // ====================================================================
    val yoloSize = 640
    val yoloBmp = scaleBitmap(rotatedBmp, yoloSize, yoloSize)
    val yoloTensor = preprocessYoloToOnnxTensor(env, yoloBmp) // FloatBuffer

    val yoloResults = yoloSession.run(mapOf(yoloSession.inputNames.first() to yoloTensor))

    // Parse output tensor YOLO menjadi list kotak (Bounding Box)
    // Ingat: Koordinat BBox dikembalikan ke ukuran frame asli (origW x origH)
    val boundingBoxes = parseYoloOutput(yoloResults, origW, origH)

    // Jika tidak ada objek yang terdeteksi, kita asumsikan jarak kosong (Optimalisasi Daya)
    // Atau Anda bisa tetap lanjut jika butuh background depth map.

    // ====================================================================
    // B. INFERENSI DEPTH ANYTHING (MONOCULAR DEPTH)
    // ====================================================================
    val depthSize = 252
    val depthBmp = scaleBitmap(rotatedBmp, depthSize, depthSize)
    val depthTensor = preprocessDepthToOnnxTensor(env, depthBmp)

    val depthResults = depthSession.run(mapOf(depthSession.inputNames.first() to depthTensor))
    val depthMapFloatArray = parseDepthToFloatArray(depthResults, depthSize, depthSize)

    // ====================================================================
    // C. PENGGABUNGAN YOLO & DEPTH (FUSION/ALIGNMENT)
    // ====================================================================
    val finalDetectedObjects = mutableListOf<DetectedObject>()

    for (box in boundingBoxes) {
        // 1. Petakan koordinat box (Frame Asli) ke ukuran Depth Map (252x252)
        val mappedX1 = (box.x1.toFloat() / origW * depthSize).toInt().coerceIn(0, depthSize - 1)
        val mappedY1 = (box.y1.toFloat() / origH * depthSize).toInt().coerceIn(0, depthSize - 1)
        val mappedX2 = (box.x2.toFloat() / origW * depthSize).toInt().coerceIn(0, depthSize - 1)
        val mappedY2 = (box.y2.toFloat() / origH * depthSize).toInt().coerceIn(0, depthSize - 1)

        // 2. Ambil sampel koordinat ROI (Misal: ambil 20% area tepat di tengah kotak objek)
        val bw = mappedX2 - mappedX1
        val bh = mappedY2 - mappedY1
        val safeMarginW = (bw * 0.4).toInt() // potong 40% kiri kanan
        val safeMarginH = (bh * 0.4).toInt() // potong 40% atas bawah

        val startX = mappedX1 + safeMarginW
        val endX = mappedX2 - safeMarginW
        val startY = mappedY1 + safeMarginH
        val endY = mappedY2 - safeMarginH

        // 3. Kumpulkan nilai Raw Depth di dalam jangkauan tengah tersebut
        val roiDepths = mutableListOf<Float>()
        for (y in startY..endY) {
            for (x in startX..endX) {
                // Perhitungan indeks 1D dari array 2D
                val index = y * depthSize + x
                roiDepths.add(depthMapFloatArray[index])
            }
        }

        // 4. Hitung Nilai Median (Lebih kuat menahan noise background tembus pandang daripada rata-rata)
        if (roiDepths.isNotEmpty()) {
            roiDepths.sort()
            val medianRawDepth = roiDepths[roiDepths.size / 2]

            // Konversi dari tebakan disparitas relatif ke Meter (Gunakan konstanta A kalibrasi Android Anda)
            // Rumus: Jarak_M = A / medianRawDepth
            val A_CALIBRATION = 15.5f // Sesuaikan dengan hasil kalibrasi Anda!
            var distanceMeter = A_CALIBRATION / (medianRawDepth + 1e-6f)
            distanceMeter = distanceMeter.coerceIn(0.1f, 5.0f) // Clamp jarak wajar 0.1m - 5m

            finalDetectedObjects.add(DetectedObject(box, distanceMeter))
        }
    }

    // Logika UI & Safety Hold Smoothing bisa diaplikasikan ke finalDetectedObjects
    val processTime = System.currentTimeMillis() - startTime

    // Kirim Hasil ke UI/Feedback
    onResult(AnalysisResult(finalDetectedObjects, null, processTime))

    image.close()
    yoloTensor.close()
    depthTensor.close()
}
```

## 4. Prioritas Feedback Audio & Bergetar

Iterasi atas `finalDetectedObjects` untuk mencari **jarak paling kecil** (rintangan terdekat di depan netra pengguna), dan gunakan objek tersebut untuk memicu alarm bahaya.

```kotlin
val closestObject = currentResult.objects.minByOrNull { it.distanceInMeters }
if (closestObject != null) {
    if (closestObject.distanceInMeters < 1.0f) {
        // Bunyikan Alarm! "Awas, ada ${closestObject.boundingBox.classLabel} di jarak..."
    }
}
```
