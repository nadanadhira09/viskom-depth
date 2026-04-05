"""
export_onnx.py
==============
Export Depth Anything V2 (ViT-S) checkpoint ke format ONNX.

Cara penggunaan:
    python export_onnx.py --checkpoint ../checkpoints/depth_anything_v2_vits.pth \
                          --output ../models/onnx/depth_anything_v2_vits.onnx \
                          --input-size 518

Catatan:
    - Unduh checkpoint dari:
      https://huggingface.co/depth-anything/Depth-Anything-V2-Small
    - Clone repositori asli ke folder 'Depth-Anything-V2' di root workspace:
      git clone https://github.com/DepthAnything/Depth-Anything-V2
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Tambahkan path ke repositori asli Depth Anything V2
REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
if REPO_DIR.exists():
    sys.path.insert(0, str(REPO_DIR))
else:
    print(f"[WARN] Repositori tidak ditemukan di {REPO_DIR}")
    print("       Pastikan sudah clone: git clone https://github.com/DepthAnything/Depth-Anything-V2")


# ──────────────────────────────────────────────
# Konfigurasi model per encoder
# ──────────────────────────────────────────────
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class DepthAnythingV2WithPrePost(nn.Module):
    """
    Wrapper yang menggabungkan pre-processing dan post-processing
    ke dalam graph ONNX agar aplikasi Android tidak perlu logika tambahan.
    Input  : tensor float32, shape (1, 3, H, W), range [0, 255]
    Output : tensor float32, shape (1, 1, H, W), relative depth (dinormalisasi ke [0, 1])
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", self.MEAN)
        self.register_buffer("std",  self.STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 3, H, W) float32 dalam rentang [0, 255]
        x = x / 255.0
        x = (x - self.mean) / self.std
        depth = self.model(x)        # output: (1, H, W) atau (H, W)

        # Pastikan output 4D: (1, 1, H, W)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)

        # Normalisasi ke [0, 1] untuk konsistensi lintas framework
        d_min = depth.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1)
        d_max = depth.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
        depth = (depth - d_min) / (d_max - d_min + 1e-8)

        return depth


def load_model(encoder: str, checkpoint_path: str) -> nn.Module:
    """Muat model DepthAnythingV2 dari checkpoint."""
    from depth_anything_v2.dpt import DepthAnythingV2

    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[OK] Model '{encoder}' berhasil dimuat dari: {checkpoint_path}")
    return model


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_size: int = 518,
    opset: int = 17,
    dynamic: bool = False,
    use_wrapper: bool = True,
    simplify: bool = True,
) -> None:
    """Export model ke format ONNX."""
    if use_wrapper:
        export_model = DepthAnythingV2WithPrePost(model)
        print("[INFO] Menggunakan wrapper pre/post-processing")
    else:
        export_model = model

    dummy = torch.zeros(1, 3, input_size, input_size)
    if use_wrapper:
        dummy = dummy * 255.0  # simulasi input raw [0, 255]

    # Dynamic axes untuk mendukung berbagai ukuran input
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 2: "height", 3: "width"},
        }
        print("[INFO] Dynamic shapes diaktifkan")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Mengekspor ke ONNX (opset={opset}) ...")
    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["depth"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
    print(f"[OK] ONNX tersimpan di: {output_path}")

    # Verifikasi
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[OK] Verifikasi model ONNX: valid")

    # Simplifikasi dengan onnxslim (opsional)
    if simplify:
        try:
            import onnxslim
            slimmed = onnxslim.slim(output_path)
            onnx.save(slimmed, output_path)
            print("[OK] Model disederhanakan dengan onnxslim")
        except ImportError:
            print("[WARN] onnxslim tidak tersedia, lewati langkah simplifikasi")
        except Exception as e:
            print(f"[WARN] onnxslim gagal: {e}")


def print_model_info(output_path: str) -> None:
    """Cetak informasi model ONNX."""
    import onnx
    model = onnx.load(output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    inputs  = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
               for i in model.graph.input]
    outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
               for o in model.graph.output]

    print("\n─── Informasi Model ONNX ───")
    print(f"  File   : {output_path}")
    print(f"  Ukuran : {size_mb:.2f} MB")
    print(f"  Input  : {inputs}")
    print(f"  Output : {outputs}")
    print("────────────────────────────")


def main():
    parser = argparse.ArgumentParser(
        description="Export Depth Anything V2 ke ONNX"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="../checkpoints/depth_anything_v2_vits.pth",
        help="Path ke file .pth checkpoint",
    )
    parser.add_argument(
        "--output", "-o",
        default="../models/onnx/depth_anything_v2_vits.onnx",
        help="Path output file .onnx",
    )
    parser.add_argument(
        "--encoder", "-e",
        default="vits",
        choices=list(MODEL_CONFIGS.keys()),
        help="Encoder yang digunakan (default: vits)",
    )
    parser.add_argument(
        "--input-size", "-s",
        type=int,
        default=518,
        help="Ukuran input (tinggi & lebar sama). Default: 518",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Versi opset ONNX (default: 17)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Aktifkan dynamic shapes (batch, H, W)",
    )
    parser.add_argument(
        "--no-wrapper",
        action="store_true",
        help="Jangan sertakan pre/post-processing dalam model ONNX",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Lewati simplifikasi dengan onnxslim",
    )
    args = parser.parse_args()

    # Ubah path relatif ke absolut berdasarkan workspace root
    base_dir = Path(__file__).parent.parent  # workspace root, bukan conversion/

    checkpoint = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() \
        else base_dir / args.checkpoint
    output = Path(args.output) if Path(args.output).is_absolute() \
        else base_dir / args.output

    if not checkpoint.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {checkpoint}")
        print(
            "        Unduh dari: "
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
            "depth_anything_v2_vits.pth"
        )
        sys.exit(1)

    # Load → Export
    model = load_model(args.encoder, str(checkpoint))
    export_onnx(
        model=model,
        output_path=str(output),
        input_size=args.input_size,
        opset=args.opset,
        dynamic=args.dynamic,
        use_wrapper=not args.no_wrapper,
        simplify=not args.no_simplify,
    )
    print_model_info(str(output))


if __name__ == "__main__":
    main()
