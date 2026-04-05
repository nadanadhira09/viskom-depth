"""
eval_da2k.py
============
Evaluasi akurasi relative depth dari model ONNX dan NCNN
menggunakan benchmark DA-2K (resmi dari paper Depth Anything V2).

DA-2K berisi:
  - 1.035 gambar dari 8 scene type
  - 2.069 pasang anotasi pair-wise relative depth

Download DA-2K:
    huggingface-cli download depth-anything/DA-2K --repo-type dataset \
        --local-dir ./data/DA-2K

Cara penggunaan:
    python eval_da2k.py --da2k-dir ./data/DA-2K \
                        --onnx ../models/onnx/depth_anything_v2_vits.onnx \
                        --ncnn-param ../models/ncnn/depth_anything_v2_vits.param \
                        --ncnn-bin   ../models/ncnn/depth_anything_v2_vits.bin \
                        --input-size 518 \
                        --output-csv ./results/da2k_results.csv
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Path lokal ke modul metrics
sys.path.insert(0, str(Path(__file__).parent))
from metrics import relative_depth_accuracy, framework_consistency, aggregate_metrics

# Scene tipe DA-2K
DA2K_SCENE_TYPES = [
    "indoor", "outdoor", "non_real", "transparent_reflective",
    "adverse_style", "aerial", "underwater", "object",
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
# Pre/Post-processing
# ──────────────────────────────────────────────────────────────────────
def preprocess(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """Return blob (1, 3, size, size) float32 normalized."""
    img = cv2.resize(image_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def postprocess_and_resize(depth_raw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Normalisasi + resize ke ukuran asli gambar."""
    d = depth_raw.squeeze()
    d_min, d_max = d.min(), d.max()
    d = (d - d_min) / (d_max - d_min + 1e-8)
    return cv2.resize(d, (target_w, target_h)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Inisialisasi session inferensi
# ──────────────────────────────────────────────────────────────────────
def init_onnx_session(onnx_path: str):
    """Inisialisasi ONNX Runtime session."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx_path, sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session, input_name


def init_ncnn_net(param_path: str, bin_path: str):
    """Inisialisasi NCNN Net."""
    try:
        import ncnn
    except ImportError:
        print("[ERROR] Package 'ncnn' tidak terinstal: pip install ncnn")
        sys.exit(1)
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.load_param(param_path)
    net.load_model(bin_path)
    return net


# ──────────────────────────────────────────────────────────────────────
# Fungsi inferensi satu gambar
# ──────────────────────────────────────────────────────────────────────
def infer_onnx_single(session, input_name: str, blob: np.ndarray) -> np.ndarray:
    return session.run(None, {input_name: blob})[0]


def infer_ncnn_single(net, blob: np.ndarray) -> np.ndarray:
    import ncnn
    _, C, H, W = blob.shape
    mat_in = ncnn.Mat(W, H, C, blob[0].copy())
    ex = net.create_extractor()
    ex.input("image", mat_in)
    _, mat_out = ex.extract("depth")
    return np.array(mat_out)


# ──────────────────────────────────────────────────────────────────────
# Evaluasi utama
# ──────────────────────────────────────────────────────────────────────
def evaluate(
    da2k_dir: Path,
    onnx_path: str | None,
    ncnn_param: str | None,
    ncnn_bin: str | None,
    input_size: int,
    scene_types: list[str] | None = None,
    max_images: int | None = None,
):
    """
    Jalankan evaluasi pada DA-2K dan kembalikan DataFrame hasil.
    """
    # Muat anotasi
    ann_file = da2k_dir / "annotations.json"
    if not ann_file.exists():
        print(f"[ERROR] File anotasi tidak ditemukan: {ann_file}")
        sys.exit(1)

    with open(ann_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    print(f"[INFO] Total gambar dengan anotasi: {len(annotations)}")

    # Filter scene type jika ditentukan
    scene_types = scene_types or DA2K_SCENE_TYPES

    # Inisialisasi session
    onnx_session = onnx_input_name = None
    ncnn_net = None

    if onnx_path:
        print(f"[INFO] Inisialisasi ONNX Runtime dari: {onnx_path}")
        onnx_session, onnx_input_name = init_onnx_session(onnx_path)

    if ncnn_param and ncnn_bin:
        print(f"[INFO] Inisialisasi NCNN dari: {ncnn_param}")
        ncnn_net = init_ncnn_net(ncnn_param, ncnn_bin)

    # Proses setiap gambar
    records = []
    image_paths = list(annotations.keys())
    if max_images:
        image_paths = image_paths[:max_images]

    print(f"\n[INFO] Evaluasi {len(image_paths)} gambar ...\n")

    for img_rel_path in tqdm(image_paths, desc="Evaluasi DA-2K"):
        img_full_path = da2k_dir / img_rel_path
        if not img_full_path.exists():
            continue

        # Tentukan scene type dari path
        scene = next((s for s in DA2K_SCENE_TYPES if s in img_rel_path), "unknown")
        if scene not in scene_types:
            continue

        anns = annotations[img_rel_path]

        # Baca gambar
        img_bgr = cv2.imread(str(img_full_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        blob = preprocess(img_bgr, input_size)

        record = {
            "image": img_rel_path,
            "scene": scene,
            "n_pairs": len(anns),
        }

        depth_onnx = depth_ncnn = None

        # ── ONNX ──────────────────────────────────────────────────────
        if onnx_session:
            t0 = time.perf_counter()
            raw_onnx = infer_onnx_single(onnx_session, onnx_input_name, blob)
            t1 = time.perf_counter()
            depth_onnx = postprocess_and_resize(raw_onnx, H, W)
            record["onnx_latency_ms"]  = (t1 - t0) * 1000
            record["onnx_accuracy"]    = relative_depth_accuracy(depth_onnx, anns)

        # ── NCNN ──────────────────────────────────────────────────────
        if ncnn_net:
            t0 = time.perf_counter()
            raw_ncnn = infer_ncnn_single(ncnn_net, blob)
            t1 = time.perf_counter()
            depth_ncnn = postprocess_and_resize(raw_ncnn, H, W)
            record["ncnn_latency_ms"]  = (t1 - t0) * 1000
            record["ncnn_accuracy"]    = relative_depth_accuracy(depth_ncnn, anns)

        # ── Konsistensi output ONNX vs NCNN ───────────────────────────
        if depth_onnx is not None and depth_ncnn is not None:
            consistency = framework_consistency(depth_onnx, depth_ncnn)
            for k, v in consistency.items():
                record[f"consistency_{k}"] = v

        records.append(record)

    df = pd.DataFrame(records)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Cetak tabel ringkasan hasil evaluasi."""
    print("\n" + "═" * 70)
    print("  RINGKASAN EVALUASI DA-2K")
    print("═" * 70)

    # Per scene type
    has_onnx = "onnx_accuracy" in df.columns
    has_ncnn = "ncnn_accuracy" in df.columns

    if has_onnx or has_ncnn:
        print(f"\n  {'Scene Type':<28} {'N':<6}", end="")
        if has_onnx:
            print(f"  {'ONNX Acc.':<14}", end="")
        if has_ncnn:
            print(f"  {'NCNN Acc.':<14}", end="")
        if "onnx_latency_ms" in df.columns and "ncnn_latency_ms" in df.columns:
            print(f"  {'ONNX ms':<10} {'NCNN ms':<10}", end="")
        print()
        print("─" * 70)

        for scene in DA2K_SCENE_TYPES + ["ALL"]:
            if scene == "ALL":
                sub = df
                label = "  ALL SCENES"
            else:
                sub = df[df["scene"] == scene]
                label = f"  {scene}"

            if sub.empty:
                continue

            n = len(sub)
            print(f"  {label:<28} {n:<6}", end="")

            if has_onnx:
                acc = sub["onnx_accuracy"].mean()
                print(f"  {acc*100:>8.2f} %     ", end="")
            if has_ncnn:
                acc = sub["ncnn_accuracy"].mean()
                print(f"  {acc*100:>8.2f} %     ", end="")
            if "onnx_latency_ms" in df.columns and "ncnn_latency_ms" in df.columns:
                onnx_ms = sub["onnx_latency_ms"].mean()
                ncnn_ms = sub["ncnn_latency_ms"].mean()
                print(f"  {onnx_ms:>8.1f}   {ncnn_ms:>8.1f}", end="")
            print()

    # Konsistensi
    cons_cols = [c for c in df.columns if c.startswith("consistency_")]
    if cons_cols:
        print("\n  ── Konsistensi Output ONNX vs NCNN ──")
        for col in cons_cols:
            label = col.replace("consistency_", "")
            print(f"  {label:<15} : {df[col].mean():.6f} ± {df[col].std():.6f}")

    print("\n" + "═" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluasi ONNX & NCNN pada DA-2K")
    parser.add_argument("--da2k-dir",    required=True, help="Direktori dataset DA-2K")
    parser.add_argument("--onnx",        default=None,  help="Path file .onnx (opsional)")
    parser.add_argument("--ncnn-param",  default=None,  help="Path file .param NCNN (opsional)")
    parser.add_argument("--ncnn-bin",    default=None,  help="Path file .bin NCNN (opsional)")
    parser.add_argument("--input-size",  type=int, default=518)
    parser.add_argument("--scene-types", nargs="+", default=None,
                        choices=DA2K_SCENE_TYPES,
                        help="Filter scene type (default: semua)")
    parser.add_argument("--max-images",  type=int, default=None,
                        help="Batasi jumlah gambar (untuk tes cepat)")
    parser.add_argument("--output-csv",  default="./results/da2k_results.csv",
                        help="Path simpan hasil CSV")
    args = parser.parse_args()

    base = Path(__file__).parent
    da2k_dir = Path(args.da2k_dir) if Path(args.da2k_dir).is_absolute() \
        else base / args.da2k_dir
    output_csv = Path(args.output_csv) if Path(args.output_csv).is_absolute() \
        else base / args.output_csv

    if not da2k_dir.exists():
        print(f"[ERROR] Direktori DA-2K tidak ditemukan: {da2k_dir}")
        print("        Download dengan:")
        print("        huggingface-cli download depth-anything/DA-2K --repo-type dataset \\")
        print(f"            --local-dir {da2k_dir}")
        sys.exit(1)

    if args.onnx is None and args.ncnn_param is None:
        print("[ERROR] Setidaknya satu model harus disediakan (--onnx atau --ncnn-param/--ncnn-bin)")
        sys.exit(1)

    df = evaluate(
        da2k_dir=da2k_dir,
        onnx_path=args.onnx,
        ncnn_param=args.ncnn_param,
        ncnn_bin=args.ncnn_bin,
        input_size=args.input_size,
        scene_types=args.scene_types,
        max_images=args.max_images,
    )

    print_summary(df)

    # Simpan CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Hasil disimpan ke: {output_csv}")


if __name__ == "__main__":
    main()
