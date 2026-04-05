"""
verify_outputs.py
=================
Bandingkan output depth map antara model PyTorch (referensi),
ONNX Runtime, dan NCNN untuk memverifikasi konsistensi konversi.

Metrik yang dihitung:
  - MAE   : Mean Absolute Error antara dua depth map
  - RMSE  : Root Mean Square Error
  - SSIM  : Structural Similarity Index (via scikit-image)
  - MaxAE : Maximum Absolute Error

Cara penggunaan:
    python verify_outputs.py --image ../assets/test_images/sample.jpg \
                             --onnx ../models/onnx/depth_anything_v2_vits.onnx \
                             --ncnn-param ../models/ncnn/depth_anything_v2_vits.param \
                             --ncnn-bin   ../models/ncnn/depth_anything_v2_vits.bin \
                             --input-size 518 \
                             --save-dir   ../assets/verification
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────────────
# Pre/Post-processing helpers
# ──────────────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Resize + normalisasi gambar untuk inferensi.
    Returns:
        blob  : (1, 3, size, size) float32 dalam range [0, 1], normalized
        orig_size : (H, W) ukuran asli gambar
    """
    orig_size = (image_bgr.shape[0], image_bgr.shape[1])
    img = cv2.resize(image_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    blob = img.transpose(2, 0, 1)[np.newaxis, ...]   # (1, 3, H, W)
    return blob.astype(np.float32), orig_size


def postprocess(depth_raw: np.ndarray, orig_size: tuple[int, int]) -> np.ndarray:
    """
    Normalisasi depth map ke [0, 1] dan resize ke ukuran asli.
    """
    d = depth_raw.squeeze()  # hapus dimensi batch/channel
    d_min, d_max = d.min(), d.max()
    d = (d - d_min) / (d_max - d_min + 1e-8)
    d = cv2.resize(d, (orig_size[1], orig_size[0]))
    return d.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Inferensi per framework
# ──────────────────────────────────────────────────────────────────────
def infer_onnx(onnx_path: str, blob: np.ndarray, warmup: int = 3, runs: int = 10) -> tuple[np.ndarray, float]:
    """Inferensi menggunakan ONNX Runtime. Return (output, avg_ms)."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_opts=sess_opts,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: blob})

    # Benchmark
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: blob})
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = float(np.mean(latencies))
    return outputs[0], avg_ms


def infer_ncnn(param_path: str, bin_path: str, blob: np.ndarray,
               warmup: int = 3, runs: int = 10) -> tuple[np.ndarray, float]:
    """
    Inferensi menggunakan NCNN via ncnn Python bindings.
    Pastikan ncnn Python wheel sudah terinstal:
        pip install ncnn
    """
    try:
        import ncnn
    except ImportError:
        print("[ERROR] Package 'ncnn' tidak terinstal.")
        print("        Install dengan: pip install ncnn")
        sys.exit(1)

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.load_param(param_path)
    net.load_model(bin_path)

    _, C, H, W = blob.shape
    mat_in = ncnn.Mat(W, H, C, blob[0].copy())

    def run_once():
        ex = net.create_extractor()
        ex.input("image", mat_in)
        _, mat_out = ex.extract("depth")
        return np.array(mat_out)

    # Warmup
    for _ in range(warmup):
        run_once()

    # Benchmark
    latencies = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = run_once()
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = float(np.mean(latencies))
    return result, avg_ms


# ──────────────────────────────────────────────────────────────────────
# Metrik perbandingan
# ──────────────────────────────────────────────────────────────────────
def compute_metrics(map_a: np.ndarray, map_b: np.ndarray) -> dict:
    """
    Hitung metrik perbandingan antara dua depth map (a = referensi).
    Keduanya harus dinormalisasi ke [0, 1] dengan ukuran sama.
    """
    from skimage.metrics import structural_similarity as ssim

    diff = np.abs(map_a - map_b)
    mae  = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    ssim_val = float(ssim(map_a, map_b, data_range=1.0))
    max_ae = float(diff.max())

    return {
        "MAE":   mae,
        "RMSE":  rmse,
        "SSIM":  ssim_val,
        "MaxAE": max_ae,
    }


# ──────────────────────────────────────────────────────────────────────
# Visualisasi
# ──────────────────────────────────────────────────────────────────────
def save_comparison(
    image_bgr: np.ndarray,
    depth_onnx: np.ndarray,
    depth_ncnn: np.ndarray | None,
    diff_map: np.ndarray | None,
    metrics_onnx: dict,
    metrics_ncnn: dict | None,
    save_path: Path,
) -> None:
    """Simpan visualisasi perbandingan side-by-side."""
    n_cols = 4 if depth_ncnn is not None else 2

    fig = plt.figure(figsize=(5 * n_cols, 5))
    gs = gridspec.GridSpec(1, n_cols)

    def colorize(depth: np.ndarray) -> np.ndarray:
        cm = plt.cm.get_cmap("inferno")
        return (cm(depth)[:, :, :3] * 255).astype(np.uint8)

    # Gambar asli
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("Input Image", fontsize=10)
    ax.axis("off")

    # Depth ONNX
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(colorize(depth_onnx))
    m = metrics_onnx
    title = "ONNX Runtime\n" + f"MAE={m['MAE']:.5f} SSIM={m['SSIM']:.5f}"
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    if depth_ncnn is not None and n_cols >= 3:
        # Depth NCNN
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(colorize(depth_ncnn))
        m = metrics_ncnn
        title = "NCNN\n" + f"MAE={m['MAE']:.5f} SSIM={m['SSIM']:.5f}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        # Difference map
        if diff_map is not None:
            ax = fig.add_subplot(gs[0, 3])
            im = ax.imshow(diff_map, cmap="hot", vmin=0, vmax=0.1)
            ax.set_title(f"Difference |ONNX-NCNN|\nMax={diff_map.max():.4f}", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Visualisasi disimpan: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Verifikasi dan bandingkan output ONNX vs NCNN"
    )
    parser.add_argument("--image",      "-i",  required=True, help="Path gambar input")
    parser.add_argument("--onnx",              required=True, help="Path file .onnx")
    parser.add_argument("--ncnn-param",        default=None,  help="Path file .param NCNN")
    parser.add_argument("--ncnn-bin",          default=None,  help="Path file .bin NCNN")
    parser.add_argument("--input-size", "-s",  type=int, default=518,
                        help="Ukuran input model (default: 518)")
    parser.add_argument("--warmup",            type=int, default=3,
                        help="Jumlah warmup runs (default: 3)")
    parser.add_argument("--runs",              type=int, default=10,
                        help="Jumlah benchmark runs (default: 10)")
    parser.add_argument("--save-dir",          default="../assets/verification",
                        help="Direktori penyimpanan hasil visualisasi")
    args = parser.parse_args()

    base = Path(__file__).parent
    save_dir = Path(args.save_dir) if Path(args.save_dir).is_absolute() \
        else base / args.save_dir

    # ── Baca gambar ──────────────────────────────────────────────────
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Gambar tidak ditemukan: {img_path}")
        sys.exit(1)

    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"[ERROR] Gagal membaca gambar: {img_path}")
        sys.exit(1)

    blob, orig_size = preprocess(image_bgr, args.input_size)
    print(f"[INFO] Gambar: {img_path.name} | Ukuran asli: {orig_size} | Blob: {blob.shape}")

    # ── ONNX Inferensi ───────────────────────────────────────────────
    print(f"\n[INFO] Menjalankan ONNX Runtime ({args.runs} runs) ...")
    onnx_raw, onnx_ms = infer_onnx(args.onnx, blob, args.warmup, args.runs)
    depth_onnx = postprocess(onnx_raw, orig_size)
    metrics_onnx_vs_self = {"MAE": 0.0, "RMSE": 0.0, "SSIM": 1.0, "MaxAE": 0.0}

    print(f"  Latensi rata-rata ONNX: {onnx_ms:.2f} ms")

    # ── NCNN Inferensi (opsional) ────────────────────────────────────
    depth_ncnn   = None
    metrics_ncnn = None
    diff_map     = None
    ncnn_ms      = None

    if args.ncnn_param and args.ncnn_bin:
        print(f"\n[INFO] Menjalankan NCNN ({args.runs} runs) ...")
        ncnn_raw, ncnn_ms = infer_ncnn(
            args.ncnn_param, args.ncnn_bin, blob, args.warmup, args.runs
        )
        depth_ncnn = postprocess(ncnn_raw, orig_size)
        metrics_ncnn = compute_metrics(depth_onnx, depth_ncnn)
        diff_map = np.abs(depth_onnx - depth_ncnn)
        print(f"  Latensi rata-rata NCNN: {ncnn_ms:.2f} ms")
    else:
        print("[INFO] Path NCNN tidak disediakan, hanya ONNX yang dijalankan")

    # ── Cetak ringkasan ──────────────────────────────────────────────
    print("\n" + "═" * 48)
    print("  HASIL PERBANDINGAN ONNX vs NCNN")
    print("═" * 48)
    print(f"  {'Metrik':<12} {'ONNX (ref)':<18} {'NCNN':<18}")
    print("─" * 48)
    print(f"  {'Latensi':<12} {onnx_ms:>8.2f} ms        ", end="")
    if ncnn_ms:
        speedup = onnx_ms / ncnn_ms
        print(f"{ncnn_ms:>8.2f} ms  ({speedup:.2f}x)")
    else:
        print("N/A")

    if metrics_ncnn:
        for key in ["MAE", "RMSE", "SSIM", "MaxAE"]:
            print(f"  {key:<12} {'ref':<18} {metrics_ncnn[key]:.6f}")

    print("═" * 48)
    print("  Keterangan: MAE/RMSE/MaxAE mendekati 0 = output identik")
    print("              SSIM mendekati 1.0 = struktur depth sangat mirip")

    # ── Simpan visualisasi ───────────────────────────────────────────
    out_stem = img_path.stem
    save_comparison(
        image_bgr=image_bgr,
        depth_onnx=depth_onnx,
        depth_ncnn=depth_ncnn,
        diff_map=diff_map,
        metrics_onnx=metrics_onnx_vs_self,
        metrics_ncnn=metrics_ncnn,
        save_path=save_dir / f"{out_stem}_comparison.png",
    )

    # Simpan depth map sebagai gambar grayscale
    cv2.imwrite(str(save_dir / f"{out_stem}_depth_onnx.png"),
                (depth_onnx * 255).astype(np.uint8))
    if depth_ncnn is not None:
        cv2.imwrite(str(save_dir / f"{out_stem}_depth_ncnn.png"),
                    (depth_ncnn * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
