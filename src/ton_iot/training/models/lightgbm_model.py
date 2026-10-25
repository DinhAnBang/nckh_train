"""LightGBM sklearn interface with validation-only early stop."""

from pathlib import Path


def create(config, early_stopping_rounds=None):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**config, objective="binary", verbosity=-1)


def fit(model, x_train, y_train, validation=None, early_stopping_rounds=None):
    if validation is None:
        model.fit(x_train, y_train)
    else:
        from inspect import signature
        import lightgbm as lgb
        callbacks = [lgb.early_stopping(int(early_stopping_rounds), verbose=False)] if early_stopping_rounds else None
        if "eval_X" in signature(model.fit).parameters:
            model.fit(x_train, y_train, eval_X=validation[0], eval_y=validation[1],
                      eval_metric="binary_logloss", callbacks=callbacks)
        else:
            model.fit(x_train, y_train, eval_set=[validation],
                      eval_metric="binary_logloss", callbacks=callbacks)
    return model


def save(model, directory):
    import joblib
    path = Path(directory) / "model.joblib"
    joblib.dump(model, path, compress=3)
    return path


def load(path):
    import joblib
    return joblib.load(path)
