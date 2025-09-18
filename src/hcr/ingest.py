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
