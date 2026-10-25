"""Project paths shared by every pipeline stage.

All paths are derived from the repository root, never from the terminal's
current working directory.  The data directories are intentionally read-only
inputs here; pipeline stages decide whether an output path may be created.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "Raw_datasets"
SOURCE_DATASET_DIR = PROJECT_ROOT / "Processed_datasets" / "Processed_Network_dataset"
DATA_PREPROCESSING_DIR = PROJECT_ROOT / "dataPreprocessing"
DATA_OUTPUT_DIR = DATA_PREPROCESSING_DIR / "outputs"
DATA_REPORT_DIR = DATA_PREPROCESSING_DIR / "reports"
FEATURES_DIR = DATA_OUTPUT_DIR / "features"
EXPERIMENTS_DIR = DATA_OUTPUT_DIR / "experiments"
NETWORK_1M_DATA_DIR = EXPERIMENTS_DIR / "network_1m"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "modelTraining" / "outputs"
NETWORK_1M_MODEL_OUTPUT_DIR = MODEL_OUTPUT_DIR / "experiments" / "network_1m"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "reports"
BACKUP_PARENT = PROJECT_ROOT.parent / "pipeline_backups"


def project_relative(path: Path) -> str:
    """Return a stable repository-relative path for logs and reports."""

    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
