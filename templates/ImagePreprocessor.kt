package com.example.yolodepthestimator.data.inference

import android.graphics.Bitmap
import android.graphics.Matrix
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import org.opencv.core.Mat
import org.opencv.core.CvType
import org.opencv.imgproc.Imgproc

/**
 * Preprocessing utilities untuk preparing gambar untuk ONNX models
 */
object ImagePreprocessor {
    
    private val YOLO_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val YOLO_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    
    private val DEPTH_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val DEPTH_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    
    /**
     * Preprocess untuk YOLO model (640x640 input)
     */
    fun preprocessYolo(bitmap: Bitmap): FloatArray {
        val resizedBmp = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        return bitmapToNormalizedFloatArray(resizedBmp, YOLO_MEAN, YOLO_STD)
    }
    
    /**
     * Preprocess untuk Depth model (252x252 input)
     */
    fun preprocessDepth(bitmap: Bitmap): FloatArray {
        val resizedBmp = Bitmap.createScaledBitmap(bitmap, 252, 252, true)
        return bitmapToNormalizedFloatArray(resizedBmp, DEPTH_MEAN, DEPTH_STD)
    }
    
    /**
     * Convert Bitmap → Normalized Float Array (HWC → CHW format untuk ONNX)
     */
    private fun bitmapToNormalizedFloatArray(
        bitmap: Bitmap,
        mean: FloatArray,
        std: FloatArray
    ): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        
        // Output array: CHW format (3, H, W)
        val floatArray = FloatArray(3 * height * width)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            
            // Normalize dengan mean & std
            floatArray[i] = (r - mean[0]) / std[0]                          // R channel
            floatArray[height * width + i] = (g - mean[1]) / std[1]         // G channel
            floatArray[2 * height * width + i] = (b - mean[2]) / std[2]     // B channel
        }
        
        return floatArray
    }
    
    /**
     * Rotate bitmap dengan degree tertentu
     */
    fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        if (degrees == 0f) return bitmap
        
        val matrix = Matrix().apply {
            postRotate(degrees)
        }
        
        return Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height,
            matrix, true
        )
    }
}
