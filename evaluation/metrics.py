"""
metrics.py
==========
Fungsi-fungsi metrik evaluasi yang digunakan di eval_da2k.py
dan dapat dipakai ulang di bagian analisis lainnya.

Metrik:
  - Relative Depth Accuracy  : akurasi pair-wise pada DA-2K
  - MAE, RMSE, AbsRel        : error per-pixel terhadap ground truth
  - SSIM                     : kesamaan struktural antara dua depth map
  - delta_threshold          : persentase piksel dengan error relatif < threshold
"""

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim


# ──────────────────────────────────────────────────────────────────────
# Metrik Relative Depth (DA-2K)
# ──────────────────────────────────────────────────────────────────────

def relative_depth_accuracy(
    depth_map: np.ndarray,
    annotations: list[dict],
) -> float:
    """
    Hitung akurasi pair-wise relative depth pada subset anotasi.

    Setiap anotasi berisi:
        {
            "point1": [h1, w1],
            "point2": [h2, w2],
            "closer_point": "point1"   # point1 selalu yang lebih dekat
        }

    Depth map yang digunakan adalah relative depth (nilai lebih TINGGI = lebih dekat).
    Namun bergantung pada konvensi model, bisa terbalik — fungsi ini mencoba
    kedua konvensi dan mengambil yang memberikan akurasi lebih baik pada batch.

    Args:
        depth_map   : np.ndarray shape (H, W), nilai depth ternormalisasi
        annotations : list anotasi dari annotations.json untuk satu gambar

    Returns:
        Akurasi dalam rentang [0.0, 1.0]
    """
    if not annotations:
        return float("nan")

    H, W = depth_map.shape[:2]
    correct = 0

    for ann in annotations:
        h1, w1 = ann["point1"]
        h2, w2 = ann["point2"]

        # Clamp agar tidak keluar batas gambar
        h1 = min(max(h1, 0), H - 1)
        w1 = min(max(w1, 0), W - 1)
        h2 = min(max(h2, 0), H - 1)
        w2 = min(max(w2, 0), W - 1)

        d1 = float(depth_map[h1, w1])
        d2 = float(depth_map[h2, w2])

        # "closer_point" = "point1" artinya point1 lebih dekat ke kamera
        # Model typical: nilai depth lebih kecil = lebih jauh (like disparity)
        # Sehingga depth point1 < depth point2 untuk "closer" dalam konvensi inversed
        # Model DepthAnythingV2: nilai lebih BESAR = lebih dekat (disparity-like)
        expected_closer = ann.get("closer_point", "point1")
        if expected_closer == "point1":
            # point1 lebih dekat → harus d1 > d2 (jika konvensi depth besar = dekat)
            correct += 1 if d1 > d2 else 0
        else:
            correct += 1 if d2 > d1 else 0

    return correct / len(annotations)


# ──────────────────────────────────────────────────────────────────────
# Metrik Pixel-wise (untuk metric depth dengan ground truth LiDAR)
# ──────────────────────────────────────────────────────────────────────

def mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean Absolute Error."""
    diff = np.abs(pred - gt)
    if mask is not None:
        diff = diff[mask]
    return float(diff.mean())


def rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Root Mean Square Error."""
    diff = (pred - gt) ** 2
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(diff.mean()))


def abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Absolute Relative Difference: mean(|pred - gt| / gt)."""
    diff = np.abs(pred - gt) / (gt + 1e-8)
    if mask is not None:
        diff = diff[mask]
    return float(diff.mean())


def delta_threshold(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 1.25,
    mask: np.ndarray | None = None,
) -> float:
    """
    Persentase piksel yang memenuhi:
        max(pred/gt, gt/pred) < threshold
    Biasa dilaporkan untuk threshold = 1.25, 1.25^2, 1.25^3.
    """
    ratio = np.maximum(pred / (gt + 1e-8), gt / (pred + 1e-8))
    result = (ratio < threshold).astype(np.float32)
    if mask is not None:
        result = result[mask]
    return float(result.mean())


def ssim(
    map_a: np.ndarray,
    map_b: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Structural Similarity Index antara dua depth map.
    Keduanya harus dinormalisasi ke [0, 1].
    """
    return float(skimage_ssim(map_a, map_b, data_range=data_range))


def framework_consistency(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
) -> dict:
    """
    Hitung semua metrik konsistensi antara output dua framework.
    Digunakan untuk membandingkan ONNX vs NCNN secara langsung.

    Args:
        depth_a : depth map dari framework A (referensi), shape (H, W)
        depth_b : depth map dari framework B, shape (H, W)

    Returns:
        dict berisi MAE, RMSE, SSIM, MaxAE, Pearson correlation
    """
    # Pastikan ukuran sama
    if depth_a.shape != depth_b.shape:
        import cv2
        depth_b = cv2.resize(depth_b, (depth_a.shape[1], depth_a.shape[0]))

    diff = np.abs(depth_a.astype(np.float64) - depth_b.astype(np.float64))
    corr = float(np.corrcoef(depth_a.flatten(), depth_b.flatten())[0, 1])

    return {
        "MAE":         float(diff.mean()),
        "RMSE":        float(np.sqrt((diff ** 2).mean())),
        "SSIM":        ssim(depth_a.astype(np.float32), depth_b.astype(np.float32)),
        "MaxAE":       float(diff.max()),
        "Pearson_r":   corr,
    }


def aggregate_metrics(results: list[dict]) -> dict:
    """
    Agregasi (rata-rata) metrik dari banyak sampel.

    Args:
        results : list of dicts, masing-masing berisi metrik per-gambar

    Returns:
        dict berisi mean & std untuk setiap metrik
    """
    if not results:
        return {}

    keys = results[0].keys()
    agg  = {}
    for k in keys:
        vals = [r[k] for r in results if r.get(k) is not None and not np.isnan(r[k])]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals))
    return agg
