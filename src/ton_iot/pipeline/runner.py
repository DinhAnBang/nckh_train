"""Run the existing Network pipeline once, stopping at the first failed gate."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pyarrow.parquet as pq

from ton_iot.config.paths import (
    BACKUP_PARENT,
    DATA_OUTPUT_DIR,
    DATA_PREPROCESSING_DIR,
    DATA_REPORT_DIR,
    FEATURES_DIR,
    PROJECT_ROOT,
    REPORT_OUTPUT_DIR,
)


PROJECT = PROJECT_ROOT
DATA = DATA_PREPROCESSING_DIR
BACKUPS = BACKUP_PARENT


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_parts(root):
    return sum(pq.ParquetFile(path).metadata.num_rows for path in root.rglob("*.parquet"))


def number_in(pattern, output):
    match = re.search(pattern, output, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Required count missing from program output: {pattern}")
    return int(match.group(1).replace(",", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    backup = BACKUPS / args.run_id
    manifest = json.loads((backup / "backup_manifest.json").read_text(encoding="utf-8"))
    if not manifest["verified"] or manifest["run_id"] != args.run_id:
        raise ValueError("Verified backup is required before running the pipeline")
    report_dir = PROJECT / "reports" / args.run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    log_path = report_dir / "execution_log.txt"

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        def record(message):
            line = f"{stamp()} {message}"
            print(line, flush=True)
            log.write(line + "\n")

        def execute(name, command, input_count):
            record(f"BEGIN {name}; input_rows={input_count}; command={' '.join(command)}")
            start = datetime.now().astimezone()
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(PROJECT / "src"), environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace", bufsize=1)
            captured = []
            for line in process.stdout:
                captured.append(line)
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
            elapsed = round((datetime.now().astimezone() - start).total_seconds(), 1)
            record(f"END {name}; exit_code={code}; seconds={elapsed}; status={'PASS' if code == 0 else 'FAIL'}")
            if code:
                raise RuntimeError(f"{name} failed with exit code {code}; outputs were preserved")
            return "".join(captured)

        py = [sys.executable, "-B", "-u"]
        record(f"BACKUP VERIFIED {backup}; files={manifest['old_files']}; bytes={manifest['old_bytes']}")
        try:
            output = execute("step_1_reader", py + ["-m", "ton_iot.data.reader"], "23 source CSV; actual count pending")
            source_rows = number_in(r"Total rows:\s*([\d,]+)", output)
            if number_in(r"Total files read successfully:\s*([\d,]+)", output) != 23:
                raise ValueError("Reader did not read 23 source CSVs")
            record(f"COUNTS step_1_reader; source_rows={source_rows}; data_output_rows=0")

            execute("step_2_analyzer", py + ["-m", "ton_iot.data.analyzer"], source_rows)
            analysis = (DATA / "reports" / "analysis" / "analysis_summary.txt").read_text(encoding="utf-8")
            analyzed = number_in(r"Total rows:\s*([\d,]+)", analysis)
            if analyzed != source_rows:
                raise ValueError("Analysis row count differs from reader")
            record(f"COUNTS step_2_analyzer; analyzed_rows={analyzed}; data_output_rows=0")

            output = execute("step_2_verification", py + ["-m", "ton_iot.data.verifier"], source_rows)
            verified_rows = number_in(r"Rows:\s*([\d,]+)", output)
            required = ("invalid_records.csv", "raw_source_comparison.csv",
                        "duplicate_verification.csv", "verification_summary.md")
            verification = DATA / "reports" / "verification"
            if verified_rows != source_rows or any(not (verification / name).is_file() for name in required):
                raise ValueError("Verification is incomplete or row counts differ")
            suspects = len(csv_rows(verification / "invalid_records.csv"))
            record(f"COUNTS step_2_verification; verified_rows={verified_rows}; suspect_rows={suspects}; data_output_rows=0")

            execute("step_3_cleaner", py + ["-m", "ton_iot.data.cleaner"], source_rows)
            file_counts = csv_rows(DATA / "reports" / "cleaning" / "file_counts.csv")
            source_count = sum(int(row["input_rows"]) for row in file_counts)
            clean_count = sum(int(row["clean_rows"]) for row in file_counts)
            quarantine_count = sum(int(row["quarantine_rows"]) for row in file_counts)
            if source_count != source_rows or clean_count + quarantine_count != source_rows:
                raise ValueError("Cleaning row conservation failed")
            clean_root = DATA / "outputs" / "cleaned"
            if count_parts(clean_root / "network") != clean_count or count_parts(clean_root / "quarantine") != quarantine_count:
                raise ValueError("Cleaning Parquet counts differ from this run's report")
            record(f"COUNTS step_3_cleaner; input_rows={source_count}; clean_rows={clean_count}; quarantine_rows={quarantine_count}")

            execute("step_4_splitter", py + ["-m", "ton_iot.data.splitter"], clean_count)
            split_rows = {name: 0 for name in ("train", "validation", "test")}
            for row in csv_rows(DATA / "reports" / "splitting" / "split_distribution.csv"):
                split_rows[row["split"]] += int(row["count"])
            if sum(split_rows.values()) != clean_count or not all(split_rows.values()):
                raise ValueError("Split row conservation failed")
            current_boundaries = csv_rows(DATA / "reports" / "splitting" / "split_boundaries.csv")
            old_boundaries = csv_rows(backup / "dataPreprocessing" / "reports" / "splitting" / "split_boundaries.csv")
            if [(row["boundary"], row["ts"]) for row in current_boundaries] != [(row["boundary"], row["ts"]) for row in old_boundaries]:
                raise ValueError("Split boundaries changed from the preserved protocol")
            for name, expected in split_rows.items():
                if count_parts(DATA / "outputs" / "split" / name) != expected:
                    raise ValueError(f"Split {name} Parquet count differs from report")
            record(f"COUNTS step_4_splitter; input_rows={clean_count}; output_rows={split_rows}; boundaries_unchanged=True")

            execute("step_5_preprocessor", py + ["-m", "ton_iot.data.preprocessor"], split_rows)
            feature_root = FEATURES_DIR
            rules_path = feature_root / "metadata" / "train_fitted_rules.json"
            old_rules_path = backup / "dataPreprocessing" / "outputs" / "features" / "metadata" / "train_fitted_rules.json"
            new_rules = json.loads(rules_path.read_text(encoding="utf-8"))
            old_rules = json.loads(old_rules_path.read_text(encoding="utf-8"))
            if new_rules != old_rules:
                raise ValueError("Feature rules changed from the previous full-data run")
            feature_rows = {}
            for name, expected in split_rows.items():
                observed = {branch: count_parts(feature_root / name / branch)
                            for branch in ("X", "y", "metadata")}
                if any(count != expected for count in observed.values()):
                    raise ValueError(f"Feature {name} X/y/metadata count differs from split")
                feature_rows[name] = observed
            record(f"COUNTS step_5_preprocessor; input_rows={split_rows}; output_rows={feature_rows}; features={len(new_rules['feature_columns'])}; rules_unchanged=True")

            output = execute("step_6_dry_run", py + ["-m", "ton_iot.training.train", "--model", "all", "--dry-run"], split_rows)
            if "DRY-RUN PASS; no model fitted" not in output:
                raise ValueError("Training dry-run did not report PASS")
            record(f"COUNTS step_6_dry_run; audited_rows={split_rows}; trained_rows=0; models_fitted=0")
            record("PIPELINE PASS; independent final dataset audit is the next gate")
        except Exception as error:
            record(f"PIPELINE FAIL; reason={type(error).__name__}: {error}; all created outputs preserved")
            raise


if __name__ == "__main__":
    main()
