from __future__ import annotations

from pathlib import Path
import zipfile
from typing import Optional


def download_from_kaggle(output_dir: Path | str = "data/raw",
                         dataset: str = "jessemostipak/hotel-booking-demand",
                         unzip: bool = True) -> Path:
    """Download a Kaggle dataset (zip) and extract it into output_dir.

    Returns the path to the downloaded zip (or the first matching zip if the
    Kaggle API names the file differently).

    Raises RuntimeError if the kaggle package is not available or download fails.
    """
    try:
        import kaggle  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("kaggle package is required (pip install kaggle)") from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Ask kaggle to download the dataset as a zip into the output dir
    kaggle.api.dataset_download_files(dataset, path=str(out), unzip=False)

    # Try to find a zip file in the output dir (API sometimes names it unpredictably)
    zips = list(out.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"Download completed but no .zip file found in {out}")
    zip_path = zips[0]

    if unzip:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out)

    return zip_path


__all__ = ["download_from_kaggle"]

