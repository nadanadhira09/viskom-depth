package com.example.yolodepthestimator.data.inference

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Singleton untuk mengelola ONNX Runtime environment
 * Bertujuan: efficient memory usage, reuse session
 */
object OnnxRuntimeManager {
    private var ortEnvironment: OrtEnvironment? = null
    private var yoloSession: OrtSession? = null
    private var depthSession: OrtSession? = null
    
    @Synchronized
    fun initializeEnvironment() {
        if (ortEnvironment == null) {
            ortEnvironment = OrtEnvironment.getEnvironment()
        }
    }
    
    @Synchronized
    fun loadYoloModel(context: Context): OrtSession {
        if (yoloSession != null) return yoloSession!!
        
        initializeEnvironment()
        val modelBytes = loadModelFromAssets(context, "yolov12n.onnx")
        
        val sessionOptions = OrtSession.SessionOptions().apply {
            addNnapi()  // Enable NNAPI (Mobile Neural Accelerator)
            setIntraOpNumThreads(4)
            setInterOpNumThreads(2)
            setGraphOptimizationLevel(OrtSession.GraphOptimizationLevel.ORT_ENABLE_ALL)
        }
        
        yoloSession = ortEnvironment!!.createSession(modelBytes, sessionOptions)
        return yoloSession!!
    }
    
    @Synchronized
    fun loadDepthModel(context: Context): OrtSession {
        if (depthSession != null) return depthSession!!
        
        initializeEnvironment()
        val modelBytes = loadModelFromAssets(context, "depth_anything_v2_vits_quantized_int8.onnx")
        
        val sessionOptions = OrtSession.SessionOptions().apply {
            addNnapi()
            setIntraOpNumThreads(4)
            setInterOpNumThreads(2)
            setGraphOptimizationLevel(OrtSession.GraphOptimizationLevel.ORT_ENABLE_ALL)
        }
        
        depthSession = ortEnvironment!!.createSession(modelBytes, sessionOptions)
        return depthSession!!
    }
    
    private fun loadModelFromAssets(context: Context, modelName: String): ByteArray {
        return context.assets.open(modelName).use { inputStream ->
            inputStream.readBytes()
        }
    }
    
    fun getYoloSession(): OrtSession? = yoloSession
    fun getDepthSession(): OrtSession? = depthSession
    
    fun cleanup() {
        yoloSession?.close()
        depthSession?.close()
        ortEnvironment?.close()
        yoloSession = null
        depthSession = null
        ortEnvironment = null
    }
}
