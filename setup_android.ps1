@echo off
REM Setup script untuk Android project (Windows PowerShell version)
REM Usage: .\setup_android.ps1 -AndroidPath "path\to\android\project" -PythonPath "path\to\python\project"

param(
    [Parameter(Mandatory = $true)]
    [string]$AndroidPath,
    
    [Parameter(Mandatory = $true)]
    [string]$PythonPath
)

Write-Host "=== Setting up Android Project ===" -ForegroundColor Cyan
Write-Host "Android Project: $AndroidPath"
Write-Host "Python Project: $PythonPath"
Write-Host ""

# 1. Create assets directory
Write-Host "📋 Creating assets directory..." -ForegroundColor Yellow
$assetsDir = Join-Path $AndroidPath "app\src\main\assets"
New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null

# 2. Copy ONNX models
Write-Host "📋 Copying ONNX models..." -ForegroundColor Yellow
$model1 = Join-Path $PythonPath "models\onnx\yolov12n.onnx"
$model2 = Join-Path $PythonPath "models\onnx\depth_anything_v2_vits_quantized_int8.onnx"

if (Test-Path $model1) {
    Copy-Item -Path $model1 -Destination $assetsDir -Force
    Write-Host "✅ yolov12n.onnx copied" -ForegroundColor Green
}
else {
    Write-Host "⚠️  yolov12n.onnx not found" -ForegroundColor Yellow
}

if (Test-Path $model2) {
    Copy-Item -Path $model2 -Destination $assetsDir -Force
    Write-Host "✅ depth_anything_v2_vits_quantized_int8.onnx copied" -ForegroundColor Green
}
else {
    Write-Host "⚠️  depth_anything_v2_vits_quantized_int8.onnx not found" -ForegroundColor Yellow
}

# 3. Copy Kotlin template files
Write-Host "📋 Copying Kotlin template files..." -ForegroundColor Yellow
$kotlinDir = Join-Path $AndroidPath "app\src\main\java\com\example\yolodepthestimator"
$templatesDir = Join-Path $PythonPath "templates"

$inferenceDir = Join-Path $kotlinDir "data\inference"
New-Item -ItemType Directory -Force -Path $inferenceDir | Out-Null

$files = @(
    "OnnxRuntimeManager.kt",
    "ImagePreprocessor.kt",
    "InferenceHelpers.kt"
)

foreach ($file in $files) {
    $srcFile = Join-Path $templatesDir $file
    if (Test-Path $srcFile) {
        Copy-Item -Path $srcFile -Destination $inferenceDir -Force
        Write-Host "✅ $file copied" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  $file not found" -ForegroundColor Yellow
    }
}

# 4. Verification
Write-Host ""
Write-Host "=== Verification ===" -ForegroundColor Cyan
$yoloFile = Join-Path $assetsDir "yolov12n.onnx"
$depthFile = Join-Path $assetsDir "depth_anything_v2_vits_quantized_int8.onnx"

if (Test-Path $yoloFile) {
    $size = (Get-Item $yoloFile).Length / 1MB
    Write-Host "✅ yolov12n.onnx ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
}

if (Test-Path $depthFile) {
    $size = (Get-Item $depthFile).Length / 1MB
    Write-Host "✅ depth_anything_v2_vits_quantized_int8.onnx ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Open Android Studio"
Write-Host "2. File > Open > $AndroidPath"
Write-Host "3. Update build.gradle.kts with ONNX Runtime dependency"
Write-Host "4. Implement CameraScreen UI"
Write-Host "5. Build & test on device (API 26+)"
