<<<<<<< Updated upstream
from __future__ import annotations
import os
import shutil
from pathlib import Path
import zipfile

def download_kaggle_dataset():
    """
    Download and extract the hotel booking demand dataset from Kaggle.
    Requires kaggle.json API credentials.
    """
    import kaggle

    kaggle_json_src = Path("/Users/thomasfey-grytnes/Documents/Artificial Intelligence - Studying/kaggle.json")
    kaggle_json_dest = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    shutil.copy(kaggle_json_src, kaggle_json_dest)
    os.chmod(kaggle_json_dest, 0o600)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "hotel-booking-demand.zip"

    # Download dataset
    kaggle.api.dataset_download_files(
        "jessemostipak/hotel-booking-demand",
        path=str(output_dir),
        unzip=False
    )

    # Unzip
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Downloaded and extracted dataset to {output_dir}")


if __name__ == "__main__":
    download_kaggle_dataset()
=======
"""Ingestion utilities for Hotel Cancellation Risk (HCR).

This module adds a small CLI to download and extract the "jessemostipak/hotel-booking-demand"
dataset from Kaggle using the Kaggle API. It expects you to have a valid `kaggle.json` file.

Usage (from repo root):
  python -m hcr.ingest --kaggle-json /path/to/kaggle.json --out data/raw

If --kaggle-json is not provided, it will look for
  ~/.kaggle/kaggle.json

The script will create the output directory (if needed), download the dataset zip, and extract it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path


def ensure_kaggle_credentials(kaggle_json: Path | None) -> Path:
    """Ensure kaggle.json exists and is accessible. Returns the path used.

    If kaggle_json is provided it will be copied to ~/.kaggle/kaggle.json (with 600 perms).
    Otherwise the function expects ~/.kaggle/kaggle.json to exist.
    """
    home_kaggle = Path.home() / ".kaggle"
    home_kaggle.mkdir(exist_ok=True)
    dest = home_kaggle / "kaggle.json"

    if kaggle_json:
        kaggle_json = Path(kaggle_json).expanduser()
        if not kaggle_json.exists():
            raise FileNotFoundError(f"Provided kaggle.json not found: {kaggle_json}")
        shutil.copy(kaggle_json, dest)
        os.chmod(dest, 0o600)
        return dest

    # if the usual ~/.kaggle/kaggle.json exists, use it
    if dest.exists():
        return dest

    # Some users keep kaggle.json next to the repository (one level up). Try that.
    parent_kaggle = Path.cwd().parent / "kaggle.json"
    if parent_kaggle.exists():
        shutil.copy(parent_kaggle, dest)
        os.chmod(dest, 0o600)
        print(f"Found kaggle.json at {parent_kaggle}; copied to {dest}")
        return dest

    # Also try a kaggle.json in the current working directory as a last resort
    local_kaggle = Path.cwd() / "kaggle.json"
    if local_kaggle.exists():
        shutil.copy(local_kaggle, dest)
        os.chmod(dest, 0o600)
        print(f"Found kaggle.json at {local_kaggle}; copied to {dest}")
        return dest

    raise FileNotFoundError(
        "kaggle.json not found. Provide --kaggle-json, place it at ~/.kaggle/kaggle.json, or put kaggle.json one level above the repo (../kaggle.json)."
    )


def download_and_extract(output_dir: Path) -> None:
    """Download dataset from Kaggle and extract into output_dir.

    Raises RuntimeError if download fails.
    """
    try:
        import kaggle
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("kaggle package is required. Install with: pip install kaggle") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_name = "hotel-booking-demand.zip"
    zip_path = output_dir / zip_name

    # Kaggle API: downloads a zip file named dataset.zip by default in the target dir
    print("Downloading dataset from Kaggle: jessemostipak/hotel-booking-demand ...")
    kaggle.api.dataset_download_files("jessemostipak/hotel-booking-demand", path=str(output_dir), unzip=False)

    if not zip_path.exists():
        # the API sometimes names the zip differently; try to find a .zip in the output dir
        zips = list(output_dir.glob("*.zip"))
        if not zips:
            raise RuntimeError(f"Download completed but no zip found in {output_dir}")
        zip_path = zips[0]

    print(f"Extracting {zip_path} to {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    print("Extraction complete.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hcr.ingest", description="Download hotel booking dataset from Kaggle")
    ap.add_argument("--kaggle-json", type=str, default=None, help="Path to kaggle.json (optional)")
    ap.add_argument("--out", type=str, default="data/raw", help="Output directory for raw data")
    args = ap.parse_args(argv)

    try:
        ensure_kaggle_credentials(Path(args.kaggle_json) if args.kaggle_json else None)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2

    try:
        download_and_extract(Path(args.out))
    except Exception as e:
        print("Error while downloading/extracting dataset:", e, file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

>>>>>>> Stashed changes
