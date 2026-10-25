"""Portable paths and explicit defaults for all three models."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataPreprocessing" / "outputs" / "features"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

DEFAULTS = {
    "rf": {
        "n_estimators": 100, "max_depth": 20, "min_samples_split": 4,
        "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": None,
        "n_jobs": -1, "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
        "reg_lambda": 1.0, "tree_method": "hist", "device": "cpu",
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 400, "num_leaves": 31, "max_depth": -1,
        "learning_rate": 0.05, "min_child_samples": 40, "subsample": 1.0,
        "colsample_bytree": 0.8, "reg_lambda": 1.0, "class_weight": None,
        "n_jobs": -1, "random_state": 42, "device_type": "cpu",
    },
}


def model_config(name, device="cpu", overrides=None, smoke=False):
    """Return one copy; never change defaults or data by model."""
    config = DEFAULTS[name].copy()
    if device != "cpu":
        if name == "rf":
            raise ValueError("Random Forest only supports CPU in this program")
        config["device" if name == "xgboost" else "device_type"] = "cuda" if name == "xgboost" else "gpu"
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"Unknown {name} parameters: {sorted(unknown)}")
        config.update(overrides)
    if isinstance(config.get("class_weight"), dict):
        config["class_weight"] = {int(key): value for key, value in config["class_weight"].items()}
    if smoke:
        config["n_estimators"] = min(int(config["n_estimators"]), 15)
        if name == "rf":
            config["n_jobs"] = min(2, config["n_jobs"]) if config["n_jobs"] > 0 else 2
    return config
