"""
convert_ncnn.py
===============
Konversi model PyTorch ke format NCNN (.param + .bin) menggunakan pnnx.

Pipeline konversi:
    .pth  ──[pnnx]──►  .ncnn.param + .ncnn.bin  ──[ncnnoptimize]──►  optimized

Prasyarat:
    pip install pnnx

    ncnnoptimize tersedia di:
    C:\ncnn\ncnn-20260113-windows-vs2022\x64\bin\ncnnoptimize.exe

Cara penggunaan:
    python convert_ncnn.py \
        --checkpoint ../checkpoints/depth_anything_v2_vits.pth \
        --output-dir ../models/ncnn \
        --input-size 518
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

# Tambahkan path ke repositori asli Depth Anything V2
REPO_DIR = Path(__file__).parent.parent / "Depth-Anything-V2"
if REPO_DIR.exists():
    sys.path.insert(0, str(REPO_DIR))

# ──────────────────────────────────────────────────────────────────────
# Lokasi binary NCNN (terdeteksi otomatis)
# ──────────────────────────────────────────────────────────────────────
DEFAULT_NCNN_TOOLS = Path("C:/ncnn/ncnn-20260113-windows-vs2022/x64/bin")


def find_binary(name: str, search_dirs: List[Path]) -> Optional[Path]:
    """Cari binary di beberapa direktori dan PATH sistem."""
    system_path = shutil.which(name)
    if system_path:
        return Path(system_path)
    for d in search_dirs:
        for suffix in ["", ".exe"]:
            candidate = d / (name + suffix)
            if candidate.exists():
                return candidate
    return None


def run_command(cmd: list[str], step_name: str) -> None:
    """Jalankan perintah shell dan tampilkan hasilnya."""
    print(f"\n[STEP] {step_name}")
    print(f"       CMD: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] {step_name} gagal (exit code {result.returncode})")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)
    print(f"[OK] {step_name} selesai")


# ──────────────────────────────────────────────────────────────────────
# Konfigurasi model
# ──────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def load_pytorch_model(encoder: str, checkpoint_path: str):
    """Muat model DepthAnythingV2 dari checkpoint PyTorch."""
    from depth_anything_v2.dpt import DepthAnythingV2
    import torch

    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    print(f"[OK] Model '{encoder}' dimuat dari: {checkpoint_path}")
    return model


def convert_pytorch_to_ncnn(
    checkpoint_path: Path,
    output_dir: Path,
    ncnn_tools_dir: Path,
    stem: str,
    encoder: str = "vits",
    input_size: int = 518,
    fp16: bool = True,
    optimize: bool = True,
) -> Tuple[Path, Path]:
    """
    Konversi checkpoint PyTorch ke NCNN menggunakan pnnx.

    pnnx.export signature:
        (model, ptpath, inputs, ..., ncnnparam=, ncnnbin=, fp16=True)

    Returns:
        Tuple (param_path, bin_path)
    """
    try:
        import pnnx
    except ImportError:
        print("[ERROR] Package 'pnnx' tidak terinstal.")
        print("        Install dengan: pip install pnnx")
        sys.exit(1)

    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    # Muat model
    model = load_pytorch_model(encoder, str(checkpoint_path))

    # Dummy input
    dummy = torch.zeros(1, 3, input_size, input_size)

    # Path output yang ditentukan
    pt_path    = output_dir / f"{stem}.pt"       # file TorchScript sementara
    param_path = output_dir / f"{stem}.param"
    bin_path   = output_dir / f"{stem}.bin"

    print(f"\n[STEP] pnnx (PyTorch → NCNN)")
    print(f"       Input size : {input_size}x{input_size}")
    print(f"       Output     : {param_path.name} + {bin_path.name}")
    print(f"       FP16       : {fp16}")

    pnnx.export(
        model,
        str(pt_path),
        dummy,
        ncnnparam=str(param_path),
        ncnnbin=str(bin_path),
        fp16=fp16,
    )

    # Hapus file sementara yang dibuat pnnx
    for tmp_ext in [".pt", "_pnnx.py", ".pnnx.param", ".pnnx.bin",
                    ".pnnx.onnx", "_ncnn.py"]:
        tmp = output_dir / (stem + tmp_ext)
        tmp.unlink(missing_ok=True)
    # Hapus juga file .pt dari pnnx
    pt_path.unlink(missing_ok=True)

    if not param_path.exists():
        print("[ERROR] pnnx tidak menghasilkan file .param")
        sys.exit(1)

    print(f"[OK] pnnx selesai → {param_path.name} + {bin_path.name}")

    # ncnnoptimize (opsional)
    if optimize:
        _run_ncnnoptimize(param_path, bin_path, ncnn_tools_dir, fp16)

    return param_path, bin_path


def _run_ncnnoptimize(
    param_path: Path,
    bin_path: Path,
    ncnn_tools_dir: Path,
    fp16: bool = True,
) -> None:
    """Jalankan ncnnoptimize jika tersedia."""
    search_dirs = [
        ncnn_tools_dir,
        Path("C:/ncnn/ncnn-20260113-windows-vs2022/x64/bin"),
        Path("C:/ncnn/ncnn-20260113-windows-vs2022/arm64/bin"),
    ]
    ncnnopt = find_binary("ncnnoptimize", search_dirs)
    if not ncnnopt:
        print("[WARN] 'ncnnoptimize' tidak ditemukan, langkah optimasi dilewati")
        return

    opt_param = param_path.with_name(param_path.stem + "_opt.param")
    opt_bin   = bin_path.with_name(bin_path.stem + "_opt.bin")
    flag_fp16 = "65536" if fp16 else "0"

    run_command(
        [str(ncnnopt), str(param_path), str(bin_path),
         str(opt_param), str(opt_bin), flag_fp16],
        step_name=f"ncnnoptimize (fp16={fp16})"
    )

    opt_param.replace(param_path)
    opt_bin.replace(bin_path)
    print("[INFO] File dioptimasi dan diganti")


def print_ncnn_info(param_path: Path, bin_path: Path) -> None:
    """Cetak informasi file NCNN."""
    param_size = param_path.stat().st_size / 1024
    bin_size   = bin_path.stat().st_size / (1024 * 1024)

    print("\n─── Informasi Model NCNN ───")
    print(f"  .param : {param_path}  ({param_size:.1f} KB)")
    print(f"  .bin   : {bin_path}  ({bin_size:.2f} MB)")
    print(f"  Total  : {(param_size / 1024 + bin_size):.2f} MB")
    print("────────────────────────────")

    try:
        lines = param_path.read_text(encoding="utf-8").splitlines()
        print("\n  Preview .param (5 baris pertama):")
        for ln in lines[:5]:
            print(f"    {ln}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Konversi PyTorch checkpoint ke NCNN menggunakan pnnx"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="../checkpoints/depth_anything_v2_vits.pth",
        help="Path ke file .pth checkpoint PyTorch",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="../models/ncnn",
        help="Direktori output untuk .param dan .bin",
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
        help="Ukuran input model (default: 518)",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Nama file output (tanpa ekstensi). Default: depth_anything_v2_vits",
    )
    parser.add_argument(
        "--ncnn-tools",
        default=str(DEFAULT_NCNN_TOOLS),
        help=f"Direktori binary ncnnoptimize (default: {DEFAULT_NCNN_TOOLS})",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Gunakan FP32 saat optimasi (default: FP16)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Lewati langkah ncnnoptimize",
    )
    args = parser.parse_args()

    base_dir   = Path(__file__).parent
    checkpoint = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() \
        else base_dir / args.checkpoint
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() \
        else base_dir / args.output_dir
    ncnn_tools = Path(args.ncnn_tools)
    stem       = args.stem or f"depth_anything_v2_{args.encoder}_{args.input_size}"

    if not checkpoint.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {checkpoint}")
        print("        Jalankan download_assets.py --model terlebih dahulu.")
        sys.exit(1)

    print(f"[INFO] Checkpoint : {checkpoint}")
    print(f"[INFO] Output dir : {output_dir}")
    print(f"[INFO] Encoder    : {args.encoder}")
    print(f"[INFO] Input size : {args.input_size}")
    print(f"[INFO] Stem       : {stem}")
    print(f"[INFO] NCNN tools : {ncnn_tools}")

    param_path, bin_path = convert_pytorch_to_ncnn(
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        ncnn_tools_dir=ncnn_tools,
        stem=stem,
        encoder=args.encoder,
        input_size=args.input_size,
        fp16=not args.no_fp16,
        optimize=not args.no_optimize,
    )

    print_ncnn_info(param_path, bin_path)


if __name__ == "__main__":
    main()
