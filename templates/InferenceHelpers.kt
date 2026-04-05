package com.example.yolodepthestimator.data.inference

import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

/**
 * Data class untuk hasil YOLO detection
 */
data class DetectionResult(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
)

/**
 * Helper untuk inference dengan YOLO model
 * 
 * Output format YOLO (typical):
 * Shape: (1, 84, 8400) atau (1, 25200, 84)
 * - 80 class probabilities
 * - 4 bbox coordinates (cx, cy, w, h)
 */
class YoloInference(
    private val session: OrtSession,
    private val env: OrtEnvironment
) {
    private val inputName = session.inputNames.first()  // ✅ FIXED: use first() instead of [0]
    private val classNames = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )
    
    suspend fun detect(
        bitmap: Bitmap,
        originalWidth: Int,
        originalHeight: Int,
        confThreshold: Float = 0.25f
    ): List<DetectionResult> = withContext(Dispatchers.Default) {
        // 1. Preprocess
        val floatArray = ImagePreprocessor.preprocessYolo(bitmap)
        
        // 2. Create tensor (✅ FIXED: wrap FloatArray in FloatBuffer)
        val inputShape = longArrayOf(1L, 3L, 640L, 640L)
        val floatBuffer = FloatBuffer.wrap(floatArray)
        val inputTensor = OnnxTensor.createTensor(env, floatBuffer, inputShape)
        
        // 3. Run inference
        val outputs = session.run(mapOf(inputName to inputTensor))
        
        // 4. Parse output (sesuaikan dengan model export Anda)
        parseOutput(outputs, originalWidth, originalHeight, confThreshold)
    }
    
    private fun parseOutput(
        outputs: Map<String, Any>,
        origW: Int,
        origH: Int,
        confThreshold: Float
    ): List<DetectionResult> {
        // Placeholder untuk parsing YOLO output
        // Format tergantung pada export settings Anda
        // Umum: output shape (1, 25200, 85) untuk YOLOv8/v12
        
        return emptyList()  // TODO: Implement based pada model output
    }
}

/**
 * Helper untuk inference dengan Depth model
 * 
 * Output: Float array shape (1, H, W) representing depth values
 */
class DepthInference(
    private val session: OrtSession,
    private val env: OrtEnvironment
) {
    private val inputName = session.inputNames.first()  // ✅ FIXED: use first() instead of [0]
    private val inputSize = 518  // Depth Anything V2 vit-s standard input
    
    suspend fun estimateDepth(bitmap: Bitmap): FloatArray = withContext(Dispatchers.Default) {
        // 1. Preprocess
        val floatArray = ImagePreprocessor.preprocessDepth(bitmap)
        
        // 2. Create tensor (✅ FIXED: wrap FloatArray in FloatBuffer)
        val inputShape = longArrayOf(1L, 3L, inputSize.toLong(), inputSize.toLong())
        val floatBuffer = FloatBuffer.wrap(floatArray)
        val inputTensor = OnnxTensor.createTensor(env, floatBuffer, inputShape)
        
        // 3. Run inference
        val outputs = session.run(mapOf(inputName to inputTensor))
        
        // 4. Extract & return result
        extractDepthMap(outputs)
    }
    
    private fun extractDepthMap(outputs: Map<String, Any>): FloatArray {
        // Placeholder untuk extracting depth output
        // Output shape biasanya: (1, 1, H, W) atau (1, H, W)
        
        val outputTensor = outputs.values.first()
        
        // TODO: Extract based pada actual output format
        return FloatArray(inputSize * inputSize)
    }
    
    /**
     * Get distance dari depth map untuk region of interest (ROI)
     * 
     * @param depthMap Raw depth values
     * @param x1 Top-left x
     * @param y1 Top-left y
     * @param x2 Bottom-right x
     * @param y2 Bottom-right y
     * @param calibrationConstant A dari kalibasi Anda (default: 15.5)
     * @return Distance dalam meter
     */
    fun getDistanceFromROI(
        depthMap: FloatArray,
        x1: Int,
        y1: Int,
        x2: Int,
        y2: Int,
        calibrationConstant: Float = 15.5f
    ): Float {
        // Ambil central region (40% dari tengah box) untuk robustness
        val bw = x2 - x1
        val bh = y2 - y1
        val marginW = (bw * 0.4).toInt()
        val marginH = (bh * 0.4).toInt()
        
        val roiX1 = x1 + marginW
        val roiY1 = y1 + marginH
        val roiX2 = x2 - marginW
        val roiY2 = y2 - marginH
        
        // Collect depth values
        val depthValues = mutableListOf<Float>()
        for (y in roiY1..roiY2) {
            for (x in roiX1..roiX2) {
                val idx = y * inputSize + x
                if (idx in depthMap.indices) {
                    depthValues.add(depthMap[idx])
                }
            }
        }
        
        if (depthValues.isEmpty()) return -1f
        
        // Use median (more robust than mean)
        depthValues.sort()
        val medianDepth = depthValues[depthValues.size / 2]
        
        // Convert to distance: meters = A / depthValue
        var distance = calibrationConstant / (medianDepth + 1e-6f)
        distance = distance.coerceIn(0.1f, 5.0f)  // Clamp ke range reasonable
        
        return distance
    }
}