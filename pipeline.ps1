# Pipeline Konversi & Evaluasi Depth Anything V2
# Jalankan step-by-step di PowerShell dari root folder c:\viskom

# ──────────────────────────────────────────────────────────────
# STEP 0: Install dependencies (jalankan sekali)
# ──────────────────────────────────────────────────────────────
pip install -r requirements.txt

# ──────────────────────────────────────────────────────────────
# STEP 1: Clone repo asli + Download model + DA-2K
# ──────────────────────────────────────────────────────────────
python download_assets.py --all --da2k-dir ./data/DA-2K

# ──────────────────────────────────────────────────────────────
# STEP 2: Export model PyTorch -> ONNX (untuk evaluasi di PC)
# ──────────────────────────────────────────────────────────────
python conversion/export_onnx.py `
    --checkpoint ./checkpoints/depth_anything_v2_vits.pth `
    --output     ./models/onnx/depth_anything_v2_vits_518.onnx `
    --encoder    vits `
    --input-size 518

# Ekspor juga versi input 256x256 (lebih ringan untuk mobile)
python conversion/export_onnx.py `
    --checkpoint ./checkpoints/depth_anything_v2_vits.pth `
    --output     ./models/onnx/depth_anything_v2_vits_256.onnx `
    --encoder    vits `
    --input-size 256

# ──────────────────────────────────────────────────────────────
# STEP 3: Konversi PyTorch -> NCNN menggunakan pnnx
# ncnnoptimize tersedia di:
# C:\ncnn\ncnn-20260113-windows-vs2022\x64\bin\ncnnoptimize.exe
# ──────────────────────────────────────────────────────────────
$NCNN_TOOLS = "C:\ncnn\ncnn-20260113-windows-vs2022\x64\bin"

# Konversi ukuran 518
python conversion/convert_ncnn.py `
    --checkpoint  ./checkpoints/depth_anything_v2_vits.pth `
    --output-dir  ./models/ncnn `
    --encoder     vits `
    --input-size  518 `
    --ncnn-tools  $NCNN_TOOLS

# Konversi ukuran 256
python conversion/convert_ncnn.py `
    --checkpoint  ./checkpoints/depth_anything_v2_vits.pth `
    --output-dir  ./models/ncnn `
    --encoder     vits `
    --input-size  256 `
    --ncnn-tools  $NCNN_TOOLS

# ──────────────────────────────────────────────────────────────
# STEP 4: Verifikasi konsistensi output ONNX vs NCNN
# ──────────────────────────────────────────────────────────────
python conversion/verify_outputs.py `
    --image       ./assets/test_images/sample.jpg `
    --onnx        ./models/onnx/depth_anything_v2_vits_518.onnx `
    --ncnn-param  ./models/ncnn/depth_anything_v2_vits_518.param `
    --ncnn-bin    ./models/ncnn/depth_anything_v2_vits_518.bin `
    --input-size  518 `
    --runs        20 `
    --save-dir    ./assets/verification

# ──────────────────────────────────────────────────────────────
# STEP 5: Evaluasi akurasi pada DA-2K benchmark
# ──────────────────────────────────────────────────────────────
python evaluation/eval_da2k.py `
    --da2k-dir   ./data/DA-2K `
    --onnx       ./models/onnx/depth_anything_v2_vits_518.onnx `
    --ncnn-param ./models/ncnn/depth_anything_v2_vits_518.param `
    --ncnn-bin   ./models/ncnn/depth_anything_v2_vits_518.bin `
    --input-size 518 `
    --output-csv ./results/da2k_results_518.csv

# Evaluasi ukuran 256
python evaluation/eval_da2k.py `
    --da2k-dir   ./data/DA-2K `
    --onnx       ./models/onnx/depth_anything_v2_vits_256.onnx `
    --ncnn-param ./models/ncnn/depth_anything_v2_vits_256.param `
    --ncnn-bin   ./models/ncnn/depth_anything_v2_vits_256.bin `
    --input-size 256 `
    --output-csv ./results/da2k_results_256.csv
