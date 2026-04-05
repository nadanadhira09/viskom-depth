"""
download_assets.py
==================
Script untuk mengunduh:
  1. Checkpoint DepthAnythingV2-Small (ViT-S)
  2. Dataset DA-2K (benchmark evaluasi)

Cara penggunaan:
    python download_assets.py --all
    python download_assets.py --model
    python download_assets.py --dataset
    python download_assets.py --dataset --da2k-dir ./data/DA-2K
"""

import argparse
import sys
import urllib.request
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# URL dan path
# ──────────────────────────────────────────────────────────────────────
CHECKPOINT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small"
    "/resolve/main/depth_anything_v2_vits.pth"
)
CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "depth_anything_v2_vits.pth"


def show_progress(block_num, block_size, total_size):
    """Callback untuk menampilkan progress unduhan."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb_done  = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  [{pct:5.1f}%]  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)


def download_model_checkpoint(force: bool = False) -> None:
    """Unduh checkpoint ViT-S dari HuggingFace."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT_FILE.exists() and not force:
        size_mb = CHECKPOINT_FILE.stat().st_size / (1024 * 1024)
        print(f"[OK] Checkpoint sudah ada ({size_mb:.1f} MB): {CHECKPOINT_FILE}")
        return

    print(f"[INFO] Mengunduh checkpoint DepthAnythingV2-Small ...")
    print(f"       Dari : {CHECKPOINT_URL}")
    print(f"       Ke   : {CHECKPOINT_FILE}")

    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_FILE, show_progress)
        print()  # newline setelah progress
        size_mb = CHECKPOINT_FILE.stat().st_size / (1024 * 1024)
        print(f"[OK] Download selesai ({size_mb:.1f} MB): {CHECKPOINT_FILE}")
    except Exception as e:
        print(f"\n[ERROR] Download gagal: {e}")
        print("        Coba download manual dari:")
        print(f"        {CHECKPOINT_URL}")
        sys.exit(1)


def download_da2k(output_dir: Path, force: bool = False) -> None:
    """
    Unduh dataset DA-2K menggunakan huggingface_hub.
    """
    ann_file = output_dir / "annotations.json"
    if ann_file.exists() and not force:
        print(f"[OK] DA-2K sudah ada di: {output_dir}")
        return

    print(f"[INFO] Mengunduh dataset DA-2K ke: {output_dir}")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="depth-anything/DA-2K",
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        print(f"[OK] DA-2K berhasil diunduh ke: {output_dir}")
    except ImportError:
        print("[ERROR] huggingface_hub tidak terinstal.")
        print("        Install dengan: pip install huggingface-hub")
        print()
        print("        Atau gunakan CLI:")
        print(f"        huggingface-cli download depth-anything/DA-2K --repo-type dataset --local-dir {output_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Download DA-2K gagal: {e}")
        print()
        print("        Alternatif:")
        print("        1. Install huggingface-cli:  pip install huggingface-hub")
        print(f"        2. Jalankan: huggingface-cli download depth-anything/DA-2K --repo-type dataset --local-dir {output_dir}")
        sys.exit(1)


def clone_repo() -> None:
    """Clone repositori asli DepthAnythingV2 untuk modul model."""
    import subprocess
    repo_dir = Path(__file__).parent / "Depth-Anything-V2"

    if repo_dir.exists():
        print(f"[OK] Repositori sudah ada di: {repo_dir}")
        return

    print("[INFO] Clone repositori DepthAnythingV2 ...")
    result = subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/DepthAnything/Depth-Anything-V2",
         str(repo_dir)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[OK] Clone selesai: {repo_dir}")
    else:
        print("[WARN] git clone gagal:")
        print(result.stderr)
        print("       Clone manual: git clone https://github.com/DepthAnything/Depth-Anything-V2")


def main():
    parser = argparse.ArgumentParser(description="Download semua aset yang diperlukan")
    parser.add_argument("--all",     action="store_true", help="Download semua (model + dataset + repo)")
    parser.add_argument("--model",   action="store_true", help="Download checkpoint ViT-S")
    parser.add_argument("--dataset", action="store_true", help="Download DA-2K")
    parser.add_argument("--repo",    action="store_true", help="Clone repositori asli DAv2")
    parser.add_argument("--da2k-dir", default="./data/DA-2K",
                        help="Direktori tujuan dataset DA-2K (default: ./data/DA-2K)")
    parser.add_argument("--force",   action="store_true", help="Unduh ulang meski sudah ada")
    args = parser.parse_args()

    base = Path(__file__).parent
    da2k_dir = Path(args.da2k_dir) if Path(args.da2k_dir).is_absolute() \
        else base / args.da2k_dir

    if not any([args.all, args.model, args.dataset, args.repo]):
        parser.print_help()
        return

    if args.all or args.repo:
        clone_repo()

    if args.all or args.model:
        download_model_checkpoint(force=args.force)

    if args.all or args.dataset:
        download_da2k(da2k_dir, force=args.force)

    print("\n[DONE] Semua aset siap.")


if __name__ == "__main__":
    main()
