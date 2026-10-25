"""XGBoost sklearn interface with hist trees and validation-only early stop."""

from pathlib import Path


def create(config, early_stopping_rounds=None):
    import xgboost as xgb
    if str(config.get("tree_method")) != "hist":
        raise ValueError("This program supports tree_method='hist' for XGBoost")
    params = {**config, "objective": "binary:logistic", "eval_metric": "logloss"}
    if early_stopping_rounds:
        params["early_stopping_rounds"] = int(early_stopping_rounds)
    return xgb.XGBClassifier(**params)


def fit(model, x_train, y_train, validation=None):
    if validation is None:
        model.fit(x_train, y_train, verbose=False)
    else:
        model.fit(x_train, y_train, eval_set=[validation], verbose=False)
    return model


def save(model, directory):
    path = Path(directory) / "model.json"
    model.save_model(path)
    return path


def load(path):
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model
