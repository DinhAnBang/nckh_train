"""CLI for safe dry-runs, technical smoke tests and explicit full training."""

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from . import data_loader, evaluate
from .experiments import TARGET_ROWS
from ton_iot.config.paths import NETWORK_1M_DATA_DIR, NETWORK_1M_MODEL_OUTPUT_DIR
from ton_iot.config.training import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, model_config
from .models import lightgbm_model, random_forest, xgboost_model
from .utils import (available_resources, gpu_status, library_versions, rss_gib,
                    ResourceMonitor, run_id, sha256_file, write_json)


MODELS = {"rf": random_forest, "xgboost": xgboost_model, "lightgbm": lightgbm_model}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], required=True)
    parser.add_argument("--experiment", choices=["network_1m"], help="Use the reproducible 1-million-row experiment sample")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Audit all inputs and resources; never train")
    mode.add_argument("--smoke-test", action="store_true", help="Small, class-balanced technical test")
    mode.add_argument("--train-mode", choices=["full"], help="Explicitly allow full training")
    mode.add_argument("--gpu-check", action="store_true", help="Fit a tiny XGBoost GPU test and exit without saving a model")
    parser.add_argument("--data-dir", type=Path, help="Feature directory; defaults depend on --experiment")
    parser.add_argument("--output-dir", type=Path, help="Result directory; defaults depend on --experiment")
    parser.add_argument("--config", type=Path, help="Optional JSON with model parameter overrides")
    parser.add_argument("--train-limit", type=int, help="Technical sample size; never official")
    parser.add_argument("--validation-limit", type=int, default=400, help="Smoke/technical validation sample size")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, help="Limit model CPU threads; useful on small Colab runtimes")
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--max-ram-gib", type=float, help="Optional RAM cap below physical availability")
    parser.add_argument("--monitor-interval", type=float, default=1.0, help="Resource sampling interval in seconds")
    args = parser.parse_args()
    if args.train_limit is not None and args.train_limit < 4:
        parser.error("--train-limit must be at least 4")
    if args.validation_limit < 4:
        parser.error("--validation-limit must be at least 4")
    if args.early_stopping_rounds < 0:
        parser.error("--early-stopping-rounds must be nonnegative")
    if args.max_ram_gib is not None and args.max_ram_gib <= 0:
        parser.error("--max-ram-gib must be positive")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.monitor_interval <= 0:
        parser.error("--monitor-interval must be positive")
    if args.experiment == "network_1m" and args.model == "all":
        parser.error("network_1m must run one model per process; choose rf, xgboost or lightgbm")
    if args.gpu_check and (args.model != "xgboost" or args.device != "gpu"):
        parser.error("--gpu-check requires --model xgboost --device gpu")
    return args


def configured_model(name, args, overrides, smoke=False):
    """Apply only explicit experiment resource limits to a model config."""

    config = model_config(name, args.device, overrides, smoke=smoke)
    if args.cpu_threads is not None:
        config["n_jobs"] = args.cpu_threads
    elif args.experiment == "network_1m":
        config["n_jobs"] = min(2, os.cpu_count() or 1)
    return config


def load_overrides(path):
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(name not in MODELS or not isinstance(parameters, dict)
                                          for name, parameters in value.items()):
        raise ValueError("Config JSON must map model names to parameter objects")
    return value


def set_logger(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("modelTraining")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_fingerprint(record_ids):
    return hashlib.sha256("\n".join(record_ids).encode("utf-8")).hexdigest()


def baseline_for_batches(layout, split):
    labels, attack_types = [], []
    for y, kinds in data_loader.label_type_batches(layout, split):
        labels.append(y)
        attack_types.extend(kinds)
    return evaluate.always_attack(np.concatenate(labels), attack_types)


def gpu_check(args, layout, overrides):
    """Verify an actual XGBoost CUDA fit on a tiny real sample."""

    x_train, y_train, _, _ = data_loader.balanced_sample(layout, "train", 256)
    x_train = x_train.toarray()
    config = configured_model("xgboost", args, overrides, smoke=True)
    config["n_estimators"] = 2
    model = xgboost_model.create(config, early_stopping_rounds=None)
    monitor = ResourceMonitor(args.monitor_interval)
    monitor.start()
    try:
        xgboost_model.fit(model, x_train, y_train)
    finally:
        resources = monitor.stop()
    print(f"GPU CHECK PASS; backend={config.get('device')}; resources={resources}")
    return 0


def fit_one(name, args, layout, config, run, logger, estimate, resources):
    wrapper = MODELS[name]
    technical = args.smoke_test or args.train_limit is not None
    model_dir = args.output_dir / "models" / name / run
    metrics_dir = args.output_dir / "metrics" / run / name
    train_rows_expected = layout["rows"]["train"]
    if technical:
        size = args.train_limit or 2000
        x_train, y_train, train_ids, _ = data_loader.balanced_sample(layout, "train", size)
        x_valid, y_valid, valid_ids, valid_types = data_loader.balanced_sample(
            layout, "validation", args.validation_limit)
        sample_method = "first N rows per class in sorted source parts; class-balanced technical sample"
    else:
        available = resources["ram_available_gib"]
        if args.max_ram_gib is not None:
            available = min(available, args.max_ram_gib)
        minimum = data_loader.require_full_memory(layout, estimate, available, name)
        logger.info("RAM guard for %s: estimated at least %.2f GiB; available %.2f GiB", name, minimum, available)
        x_train, y_train = data_loader.load_full(layout, "train")
        train_ids = valid_ids = None
        valid_types = None
        x_valid = y_valid = None
        if name != "rf" and args.early_stopping_rounds:
            x_valid, y_valid = data_loader.load_full(layout, "validation")
        sample_method = "all train rows; no sampling"
    if not np.isin(y_train, [0, 1]).all() or len(np.unique(y_train)) != 2:
        raise ValueError("Training data must contain both binary classes")
    if name == "xgboost":
        # XGBoost treats omitted sparse entries as missing; transformed zeros are real values.
        x_train = x_train.toarray()
        if x_valid is not None:
            x_valid = x_valid.toarray()
    model_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir.mkdir(parents=True, exist_ok=False)
    early_rounds = min(args.early_stopping_rounds, 5) if args.smoke_test else args.early_stopping_rounds
    if name == "rf":
        early_rounds = 0
    model = wrapper.create(config, early_stopping_rounds=early_rounds)
    effective_parameters = model.get_params(deep=False)
    logger.info("Fitting %s: rows=%s, features=%s, technical=%s", name, len(y_train), len(layout["columns"]), technical)
    started = perf_counter()
    validation = (x_valid, y_valid) if x_valid is not None else None
    monitor = ResourceMonitor(args.monitor_interval)
    monitor.start()
    try:
        if name == "lightgbm":
            wrapper.fit(model, x_train, y_train, validation, early_stopping_rounds=early_rounds)
        else:
            wrapper.fit(model, x_train, y_train, validation)
    finally:
        resource_stats = monitor.stop()
    fit_seconds = perf_counter() - started
    memory_after_fit = rss_gib()
    model_path = wrapper.save(model, model_dir)
    reloaded = wrapper.load(model_path)
    if technical:
        reference = evaluate.scores(model, x_valid[:10])
        saved = evaluate.scores(reloaded, x_valid[:10])
        if not np.allclose(reference, saved, rtol=1e-7, atol=1e-9):
            raise ValueError(f"Saved/reloaded {name} predictions changed")
    del model, x_train, y_train
    gc.collect()
    results = []
    baseline_result = None
    if technical:
        started = perf_counter()
        probability = evaluate.scores(reloaded, x_valid)
        prediction_seconds = perf_counter() - started
        metrics, by_type = evaluate.summarize(y_valid, probability, valid_types)
        metrics["prediction_seconds"] = round(prediction_seconds, 3)
        results.append(("validation", metrics, by_type))
        if args.experiment:
            baseline_result = evaluate.always_attack(y_valid, valid_types)
        del x_valid, y_valid
    else:
        if x_valid is not None:
            del x_valid, y_valid
            gc.collect()
        for split in ("validation", "test"):
            logger.info("Evaluating %s on %s", name, split)
            batches = data_loader.eval_batches(layout, split, dense=name == "xgboost")
            metrics, by_type = evaluate.evaluate_batches(reloaded, batches)
            results.append((split, metrics, by_type))
    status = "network_1m_experiment" if args.experiment else ("technical" if technical else "official")
    summary = {
        "run_id": run, "model": name, "status": status,
        "experiment": args.experiment,
        "data_dir": str(layout["root"]), "rules_sha256": sha256_file(layout["rules_path"]),
        "feature_count": len(layout["columns"]), "feature_columns": layout["columns"],
        "input_format": "dense float64" if name == "xgboost" else "CSR float64",
        "source_rows": dict(layout["rows"]), "train_rows_used": size if technical else train_rows_expected,
        "sample_method": sample_method, "train_sample_sha256": sample_fingerprint(train_ids) if technical else None,
        "validation_sample_sha256": sample_fingerprint(valid_ids) if technical else None,
        "config": config, "effective_parameters": effective_parameters,
        "random_state": config["random_state"],
        "early_stopping_rounds": early_rounds, "threshold": 0.5,
        "fit_seconds": round(fit_seconds, 3), "rss_after_fit_gib": memory_after_fit,
        "peak_rss_gib": resource_stats["peak_rss_gib"],
        "peak_gpu_used_gib": resource_stats["peak_gpu_used_gib"],
        "library_versions": library_versions(),
        "validation_only_attack_type": "password",
        "sampling_report_sha256": sha256_file(args.data_dir / "sampling_report.json") if args.experiment else None,
        "evaluation_scope": (
            "network_1m fixed sample; validation/test are experiment results"
            if args.experiment else ("technical sample, not research result" if technical else "full fixed splits")
        ),
    }
    write_json(model_dir / "run_manifest.json", summary)
    metric_rows = []
    type_rows = []
    for split, metrics, by_type in results:
        metric_rows.append({"run_id": run, "model": name, "split": split, "status": summary["status"], **metrics})
        type_rows.extend({"run_id": run, "model": name, "split": split, **row} for row in by_type)
    if args.experiment:
        baseline_results = []
        if technical:
            baseline_results.append(("validation", *baseline_result))
        else:
            for split in ("validation", "test"):
                baseline_results.append((split, *baseline_for_batches(layout, split)))
        for split, metrics, by_type in baseline_results:
            metric_rows.append({
                "run_id": run, "model": "baseline_always_attack", "split": split,
                "status": "baseline", **metrics,
            })
            type_rows.extend({"run_id": run, "model": "baseline_always_attack", "split": split, **row} for row in by_type)
    write_csv(metrics_dir / "metrics.csv", metric_rows)
    write_csv(metrics_dir / "attack_type_results.csv", type_rows)
    logger.info("Saved %s (%s) to %s", name, summary["status"], model_dir)
    del reloaded
    gc.collect()
    return metric_rows


def main():
    args = arguments()
    selected = list(MODELS) if args.model == "all" else [args.model]
    overrides = load_overrides(args.config)
    if args.experiment == "network_1m":
        args.data_dir = (args.data_dir or NETWORK_1M_DATA_DIR).expanduser().resolve()
        args.output_dir = (args.output_dir or NETWORK_1M_MODEL_OUTPUT_DIR).expanduser().resolve()
        if not (args.data_dir / "sampling_report.json").is_file():
            raise FileNotFoundError(f"network_1m sampling_report.json missing: {args.data_dir}")
        expected_rows = TARGET_ROWS
    else:
        args.data_dir = (args.data_dir or DEFAULT_DATA_DIR).expanduser().resolve()
        args.output_dir = (args.output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
        expected_rows = None
    if args.device == "gpu":
        status = gpu_status()
        if status.startswith("nvidia-smi unavailable"):
            raise RuntimeError(status + ". No GPU training attempted.")
    layout = data_loader.inspect_layout(args.data_dir, expected_rows=expected_rows)
    needs_full_audit = args.dry_run or args.gpu_check or (args.train_mode == "full" and args.train_limit is None)
    labels, density = data_loader.audit_data(layout) if needs_full_audit else (None, None)
    if density is None:
        # A single bounded part gives a rough resource estimate, not a fit guarantee.
        first = layout["parts"]["train"][0]
        import pyarrow.parquet as pq
        table = pq.read_table(first)
        density = data_loader.to_csr(table, layout["columns"]).nnz / table.num_rows
    estimate = data_loader.memory_estimate(layout, density)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resources = available_resources(args.output_dir)
    versions = library_versions()
    print("========== NETWORK TRAINING SETUP ==========")
    print(f"Features: {len(layout['columns'])}; rows: {dict(layout['rows'])}")
    print(f"Parquet X: {estimate['x_parquet_gib']} GiB; estimated dense train float64: {estimate['train_dense_float64_gib']} GiB; estimated CSR train: {estimate['train_csr_est_gib']} GiB")
    print("Input representation: RF/LightGBM CSR float64; XGBoost dense float64 to preserve zero values")
    print(f"Resources: {resources}; GPU: {gpu_status() if args.device == 'gpu' else 'not requested'}")
    for name in selected:
        config = configured_model(name, args, overrides.get(name), smoke=args.smoke_test)
        print(f"{name}: {config}")
        library = {"rf": "scikit-learn", "xgboost": "xgboost", "lightgbm": "lightgbm"}[name]
        if versions[library] == "not installed":
            raise ImportError(f"{library} is missing; install requirements.txt")
    if args.dry_run:
        print(f"All X/y/metadata parts audited; label counts: {dict(labels)}")
        print("DRY-RUN PASS; no model fitted")
        return 0
    if args.gpu_check:
        return gpu_check(args, layout, overrides.get("xgboost"))
    run = run_id()
    logger = set_logger(args.output_dir / "logs" / f"{run}.log")
    logger.info("Run %s; mode=%s; dataset=%s", run, "smoke" if args.smoke_test else "full", layout["root"])
    all_results = []
    for name in selected:
        config = configured_model(name, args, overrides.get(name), smoke=args.smoke_test)
        all_results.extend(fit_one(name, args, layout, config, run, logger, estimate, resources))
    if len(selected) > 1:
        write_csv(args.output_dir / "metrics" / run / "model_comparison.csv", all_results)
    print(f"RUN COMPLETE: {run}; {'TECHNICAL SMOKE TEST' if args.smoke_test or args.train_limit else 'OFFICIAL FULL TRAINING'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MemoryError, ImportError, ValueError, RuntimeError) as error:
        logging.getLogger("modelTraining").exception("Training stopped: %s", error)
        raise SystemExit(1) from error
