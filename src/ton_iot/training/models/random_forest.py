"""Random Forest; fit is in-memory and has no partial-fit path."""

from pathlib import Path


def create(config, early_stopping_rounds=None):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**config)


def fit(model, x_train, y_train, validation=None):
    model.fit(x_train, y_train)
    return model


def save(model, directory):
    import joblib
    path = Path(directory) / "model.joblib"
    joblib.dump(model, path, compress=3)
    return path


def load(path):
    import joblib
    return joblib.load(path)
