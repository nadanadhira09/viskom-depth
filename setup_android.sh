#!/bin/bash
# Script untuk setup Android project dari template
# Jalankan di folder Android Project root

set -e

ANDROID_PROJECT_DIR="$1"
PYTHON_PROJECT_DIR="$2"

if [ -z "$ANDROID_PROJECT_DIR" ] || [ -z "$PYTHON_PROJECT_DIR" ]; then
    echo "Usage: ./setup_android.sh <android_project_path> <python_project_path>"
    echo "Example: ./setup_android.sh ~/AndroidStudioProjects/YoloDepth ~/python-projects/viskom"
    exit 1
fi

echo "=== Setting up Android Project ==="
echo "Android Project: $ANDROID_PROJECT_DIR"
echo "Python Project: $PYTHON_PROJECT_DIR"

# 1. Copy ONNX models
echo "📋 Copying ONNX models..."
mkdir -p "$ANDROID_PROJECT_DIR/app/src/main/assets"
cp "$PYTHON_PROJECT_DIR/models/onnx/yolov12n.onnx" "$ANDROID_PROJECT_DIR/app/src/main/assets/" 2>/dev/null || echo "⚠️ yolov12n.onnx not found"
cp "$PYTHON_PROJECT_DIR/models/onnx/depth_anything_v2_vits_quantized_int8.onnx" "$ANDROID_PROJECT_DIR/app/src/main/assets/" 2>/dev/null || echo "✅ depth_anything_v2_vits_quantized_int8.onnx copied"

# 2. Copy template Kotlin files
echo "📋 Copying Kotlin template files..."
KOTLIN_SRC="$ANDROID_PROJECT_DIR/app/src/main/java/com/example/yolodepthestimator"

mkdir -p "$KOTLIN_SRC/data/inference"
mkdir -p "$KOTLIN_SRC/ui/screens"
mkdir -p "$KOTLIN_SRC/ui/components"
mkdir -p "$KOTLIN_SRC/viewmodel"

# Copy from Python project templates
cp "$PYTHON_PROJECT_DIR/templates/OnnxRuntimeManager.kt" "$KOTLIN_SRC/data/inference/" 2>/dev/null || echo "⚠️ OnnxRuntimeManager.kt not found"
cp "$PYTHON_PROJECT_DIR/templates/ImagePreprocessor.kt" "$KOTLIN_SRC/data/inference/" 2>/dev/null || echo "✅ ImagePreprocessor.kt copied"
cp "$PYTHON_PROJECT_DIR/templates/InferenceHelpers.kt" "$KOTLIN_SRC/data/inference/" 2>/dev/null || echo "✅ InferenceHelpers.kt copied"

# 3. Verify files
echo ""
echo "=== Verification ==="
if [ -f "$ANDROID_PROJECT_DIR/app/src/main/assets/yolov12n.onnx" ]; then
    SIZE=$(ls -lh "$ANDROID_PROJECT_DIR/app/src/main/assets/yolov12n.onnx" | awk '{print $5}')
    echo "✅ yolov12n.onnx ($SIZE)"
else
    echo "⚠️ yolov12n.onnx not found"
fi

if [ -f "$ANDROID_PROJECT_DIR/app/src/main/assets/depth_anything_v2_vits_quantized_int8.onnx" ]; then
    SIZE=$(ls -lh "$ANDROID_PROJECT_DIR/app/src/main/assets/depth_anything_v2_vits_quantized_int8.onnx" | awk '{print $5}')
    echo "✅ depth_anything_v2_vits_quantized_int8.onnx ($SIZE)"
else
    echo "⚠️ depth_anything_v2_vits_quantized_int8.onnx not found"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Open Android Studio"
echo "2. File > Open > $ANDROID_PROJECT_DIR"
echo "3. Add ONNX Runtime dependency in app/build.gradle.kts"
echo "4. Implement CameraScreen UI"
echo "5. Build & test on device (API 26+)"
