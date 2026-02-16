import shutil
import kagglehub
from pathlib import Path

# (paste the helper header here)

def main():
    # Download latest version
    dataset_path = kagglehub.dataset_download(
        "ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset"
    )
    print("Dataset downloaded to:", dataset_path)

    # Find the raw csv inside the downloaded folder
    dataset_path = Path(dataset_path)
    src = dataset_path / "earthquake_data_tsunami.csv"   # (this is what your code reads)
    if not src.exists():
        raise FileNotFoundError(f"Could not find {src} inside Kaggle download.")

    dst = RAW_DIR / "earthquake_data_tsunami.csv"
    shutil.copy(src, dst)
    print("Raw data copied to:", dst)

if __name__ == "__main__":
    main()

