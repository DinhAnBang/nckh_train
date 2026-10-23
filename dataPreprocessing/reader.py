"""Read and validate the ToN-IoT Network CSV files.

This script only reads and summarizes the source files.  It does not modify
the source data, clean records, split data, or train a model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_FILE_COUNT = 23
CHUNK_SIZE = 100_000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREFERRED_DATASET_DIRECTORY = PROJECT_ROOT / "Processed_datasets" / "Processed_Network_dataset"

NORMAL_LABELS = {"0", "normal", "benign", "false", "no"}
ATTACK_LABELS = {"1", "attack", "anomaly", "malicious", "true", "yes"}


def numeric_dataset_number(path: Path) -> int:
    """Return the numeric suffix of Network_dataset_<number>.csv."""

    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def discover_network_files() -> tuple[Path, list[Path], list[str]]:
    """Find the one Network dataset source and return its files.

    If multiple directories contain matching files, the directory with the
    largest complete set is selected.  The preferred processed-data directory
    wins ties.  No files from different directories are combined.
    """

    matches = sorted(
        PROJECT_ROOT.rglob("Network_dataset_*.csv"),
        key=lambda path: (str(path.parent).lower(), numeric_dataset_number(path)),
    )
    by_directory: dict[Path, list[Path]] = defaultdict(list)
    for path in matches:
        by_directory[path.parent].append(path)

    if not by_directory:
        raise FileNotFoundError(
            f"No Network_dataset_*.csv files were found below {PROJECT_ROOT}"
        )

    def directory_score(directory: Path) -> tuple[int, int, int]:
        files = by_directory[directory]
        numbers = {numeric_dataset_number(path) for path in files}
        complete_count = len(numbers.intersection(range(1, EXPECTED_FILE_COUNT + 1)))
        is_preferred = int(directory.resolve() == PREFERRED_DATASET_DIRECTORY.resolve())
        return (complete_count, is_preferred, len(files))

    source_directory = max(by_directory, key=directory_score)
    selected_files = sorted(by_directory[source_directory], key=numeric_dataset_number)
    warnings: list[str] = []

    if len(by_directory) > 1:
        directory_list = ", ".join(str(directory) for directory in sorted(by_directory))
        warnings.append(
            "Multiple directories contain Network_dataset_*.csv. "
            f"Selected '{source_directory}' and ignored the other directory(s): {directory_list}."
        )

    expected_names = {f"Network_dataset_{number}.csv" for number in range(1, EXPECTED_FILE_COUNT + 1)}
    actual_names = {path.name for path in selected_files}
    missing_names = sorted(expected_names - actual_names, key=numeric_dataset_number)
    unexpected_names = sorted(actual_names - expected_names, key=numeric_dataset_number)

    if missing_names:
        warnings.append("Missing expected files: " + ", ".join(missing_names))
    if unexpected_names:
        warnings.append("Unexpected matching files: " + ", ".join(unexpected_names))
    if len(selected_files) != EXPECTED_FILE_COUNT:
        warnings.append(
            f"Expected {EXPECTED_FILE_COUNT} Network files but found {len(selected_files)} "
            f"in the selected directory."
        )

    return source_directory, selected_files, warnings


def normalize_column_name(name: object) -> str:
    """Normalize a column name for label-column detection."""

    return "".join(character for character in str(name).strip().lower() if character.isalnum())


def normalized_value(value: object) -> str | None:
    """Return a comparable label value, or None for a missing value."""

    if pd.isna(value):
        return None
    value_text = str(value).strip().lower()
    if not value_text or value_text in {"nan", "none", "null", "na", "n/a"}:
        return None
    return value_text


def detect_label_columns(sample: pd.DataFrame) -> tuple[str | None, str | None]:
    """Detect the binary normal/attack column and the attack-type column."""

    columns = list(sample.columns)
    normalized_names = {column: normalize_column_name(column) for column in columns}

    label_name_candidates = {
        "label",
        "target",
        "attacklabel",
        "isattack",
        "normalattack",
    }
    type_name_candidates = {
        "type",
        "attacktype",
        "classtype",
        "category",
        "class",
    }

    label_column = next(
        (column for column in columns if normalized_names[column] in label_name_candidates),
        None,
    )
    type_column = next(
        (column for column in columns if normalized_names[column] in type_name_candidates),
        None,
    )

    # Fallbacks use the values in the real sample, in case a future file uses
    # a different column name.
    if label_column is None:
        for column in columns:
            values = {normalized_value(value) for value in sample[column].dropna()}
            values.discard(None)
            if values and values.issubset(NORMAL_LABELS | ATTACK_LABELS) and len(values) <= 2:
                label_column = column
                break

    if type_column is None:
        for column in columns:
            if column == label_column:
                continue
            values = {normalized_value(value) for value in sample[column].dropna()}
            values.discard(None)
            if "normal" in values and len(values) >= 2:
                type_column = column
                break

    return label_column, type_column


def classify_binary_label(value: object) -> str:
    """Classify a binary label as normal, attack, missing, or invalid."""

    value_text = normalized_value(value)
    if value_text is None:
        return "missing"
    if value_text in NORMAL_LABELS:
        return "normal"
    if value_text in ATTACK_LABELS:
        return "attack"
    return "invalid"


def classify_type_label(value: object) -> str:
    """Classify an attack-type value as normal, attack, or missing."""

    value_text = normalized_value(value)
    if value_text is None:
        return "missing"
    if value_text in {"normal", "benign"}:
        return "normal"
    return "attack"


def update_dtype_registry(
    dtype_registry: dict[str, set[str]],
    frame: pd.DataFrame,
) -> None:
    """Record inferred pandas dtypes without retaining the data chunk."""

    for column, dtype in frame.dtypes.items():
        dtype_registry[column].add(str(dtype))


def format_column_dtypes(
    columns: Iterable[object],
    dtype_registry: dict[str, set[str]],
) -> str:
    """Format each column name and its observed inferred dtype."""

    formatted = []
    for column in columns:
        dtypes = "/".join(sorted(dtype_registry.get(str(column), {"unknown"})))
        formatted.append(f"{column}:{dtypes}")
    return ", ".join(formatted)


def empty_file_statistics() -> dict[str, object]:
    """Create counters for one file."""

    return {
        "rows": 0,
        "normal": 0,
        "attack": 0,
        "invalid": 0,
        "missing_label": 0,
        "invalid_label": 0,
        "conflict": 0,
        "attack_types": Counter(),
        "columns": tuple(),
        "dtypes": defaultdict(set),
        "read_successfully": False,
        "error": None,
    }


def read_one_file(
    path: Path,
    label_column: str | None,
    type_column: str | None,
) -> dict[str, object]:
    """Read one CSV in chunks and return only aggregate statistics."""

    statistics = empty_file_statistics()

    try:
        header = pd.read_csv(path, nrows=0, low_memory=False)
        statistics["columns"] = tuple(str(column) for column in header.columns)

        if label_column is not None and label_column not in header.columns:
            statistics["error"] = f"Missing detected label column '{label_column}'."
            return statistics
        if type_column is not None and type_column not in header.columns:
            statistics["error"] = f"Missing detected attack-type column '{type_column}'."
            return statistics

        reader = pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            on_bad_lines="error",
        )

        for chunk in reader:
            statistics["rows"] = int(statistics["rows"]) + len(chunk)
            update_dtype_registry(statistics["dtypes"], chunk)

            if label_column is None or type_column is None:
                statistics["invalid"] = int(statistics["invalid"]) + len(chunk)
                continue

            label_states = chunk[label_column].map(classify_binary_label)
            type_states = chunk[type_column].map(classify_type_label)

            normal_rows = (label_states == "normal") & (type_states == "normal")
            attack_rows = (label_states == "attack") & (type_states == "attack")
            invalid_rows = ~(normal_rows | attack_rows)

            statistics["normal"] = int(statistics["normal"]) + int(normal_rows.sum())
            statistics["attack"] = int(statistics["attack"]) + int(attack_rows.sum())
            statistics["invalid"] = int(statistics["invalid"]) + int(invalid_rows.sum())

            missing_rows = (label_states == "missing") | (type_states == "missing")
            invalid_label_rows = (label_states == "invalid")
            conflict_rows = (
                label_states.isin(["normal", "attack"])
                & type_states.isin(["normal", "attack"])
                & (label_states != type_states)
            )
            statistics["missing_label"] = int(statistics["missing_label"]) + int(missing_rows.sum())
            statistics["invalid_label"] = int(statistics["invalid_label"]) + int(invalid_label_rows.sum())
            statistics["conflict"] = int(statistics["conflict"]) + int(conflict_rows.sum())

            type_values = chunk[type_column].map(normalized_value)
            type_counts = type_values.value_counts(dropna=False)
            for value, count in type_counts.items():
                display_value = "<MISSING>" if value is None else str(value)
                statistics["attack_types"][display_value] += int(count)

        statistics["read_successfully"] = True
    except Exception as error:  # Keep the file-specific error visible in the report.
        statistics["error"] = f"{type(error).__name__}: {error}"

    return statistics


def compare_schemas(
    file_results: list[tuple[Path, dict[str, object]]],
) -> tuple[bool, list[str]]:
    """Compare column names/order and observed dtypes across files."""

    warnings: list[str] = []
    successful_results = [result for _, result in file_results if result["read_successfully"]]
    if not successful_results:
        return False, ["No successfully read file was available for schema comparison."]

    reference_columns = successful_results[0]["columns"]
    reference_dtypes = {
        column: tuple(sorted(successful_results[0]["dtypes"].get(column, set())))
        for column in reference_columns
    }
    schema_consistent = True

    for path, result in file_results:
        if not result["read_successfully"]:
            schema_consistent = False
            continue

        columns = result["columns"]
        if columns != reference_columns:
            schema_consistent = False
            missing = [column for column in reference_columns if column not in columns]
            extra = [column for column in columns if column not in reference_columns]
            order_difference = not missing and not extra and columns != reference_columns
            details = []
            if missing:
                details.append("missing=" + ", ".join(missing))
            if extra:
                details.append("extra=" + ", ".join(extra))
            if order_difference:
                details.append("column order differs")
            warnings.append(
                f"Schema warning: {path.name} differs from the reference file "
                f"({'; '.join(details)})."
            )

        for column, dtypes in result["dtypes"].items():
            observed = tuple(sorted(dtypes))
            reference = reference_dtypes.get(column)
            if reference is not None and observed != reference:
                warnings.append(
                    f"Data-type warning: {path.name}, column '{column}' "
                    f"has {observed}, reference has {reference}."
                )

    return schema_consistent, warnings


def print_report(
    source_directory: Path,
    files: list[Path],
    discovery_warnings: list[str],
    detected_label_column: str | None,
    detected_type_column: str | None,
    results: list[tuple[Path, dict[str, object]]],
    schema_consistent: bool,
    schema_warnings: list[str],
) -> None:
    """Print the requested terminal report."""

    total_rows = sum(int(result["rows"]) for _, result in results)
    total_normal = sum(int(result["normal"]) for _, result in results)
    total_attack = sum(int(result["attack"]) for _, result in results)
    total_invalid = sum(int(result["invalid"]) for _, result in results)
    total_missing = sum(int(result["missing_label"]) for _, result in results)
    total_invalid_values = sum(int(result["invalid_label"]) for _, result in results)
    total_conflicts = sum(int(result["conflict"]) for _, result in results)
    files_read = sum(bool(result["read_successfully"]) for _, result in results)

    attack_types: Counter[str] = Counter()
    for _, result in results:
        attack_types.update(result["attack_types"])

    print("========== TON-IOT NETWORK DATASET ==========")
    print()
    print("SOURCE DIRECTORY")
    print(f"{source_directory}")
    print(f"Detected normal/attack label column: {detected_label_column or '<not found>'}")
    print(f"Detected attack-type column: {detected_type_column or '<not found>'}")
    print()

    print("1. FILE INFORMATION")
    print()
    print(f"Total files found: {len(files)}")
    print(f"Total files read successfully: {files_read}")
    print()
    print("Files:")
    for number, (path, result) in enumerate(results, start=1):
        status = "OK" if result["read_successfully"] else "ERROR"
        print(f"{number:2}. {path.name:<30} | Rows: {int(result['rows']):,} | {status}")
        if result["error"]:
            print(f"    WARNING: {result['error']}")

    print()
    print("Schema and inferred data types by file (inferred while reading chunks):")
    for number, (path, result) in enumerate(results, start=1):
        columns = result["columns"]
        dtype_text = format_column_dtypes(columns, result["dtypes"])
        print(f"{number:2}. {path.name:<30} | Columns: {len(columns)} | {dtype_text}")

    print()
    print("--------------------------------------------")
    print()
    print("2. DATASET SUMMARY")
    print()
    print(f"Total rows: {total_rows:,}")
    print(f"Normal records: {total_normal:,}")
    print(f"Attack records: {total_attack:,}")
    print(f"Missing/invalid labels: {total_invalid:,}")
    print()
    print("Label warnings:")
    print(f"- Rows with a missing label/type value: {total_missing:,}")
    print(f"- Rows with an invalid binary label value: {total_invalid_values:,}")
    print(f"- Rows with conflicting label and type: {total_conflicts:,}")

    print()
    print("--------------------------------------------")
    print()
    print("3. ATTACK TYPES")
    print()
    if attack_types:
        for attack_type, count in sorted(attack_types.items(), key=lambda item: item[0].lower()):
            print(f"{attack_type:<15}: {count:,}")
    else:
        print("<no attack types detected>")

    print()
    print("--------------------------------------------")
    print()
    print("4. VALIDATION")
    print()
    print(f"Files expected: {EXPECTED_FILE_COUNT}")
    print(f"Files read: {files_read}")
    reconciles = total_rows == total_normal + total_attack + total_invalid
    print(
        "Total rows == Normal + Attack + Invalid: "
        f"{total_rows:,} == {total_normal:,} + {total_attack:,} + {total_invalid:,} -> "
        f"{'PASS' if reconciles else 'FAIL'}"
    )
    print(f"Schema consistency: {'PASS' if schema_consistent else 'FAIL'}")
    label_consistent = total_missing == 0 and total_invalid_values == 0 and total_conflicts == 0
    print(f"Label consistency: {'PASS' if label_consistent else 'FAIL'}")

    for warning in discovery_warnings + schema_warnings:
        print(f"WARNING: {warning}")

    final_status = (
        len(files) == EXPECTED_FILE_COUNT
        and files_read == EXPECTED_FILE_COUNT
        and reconciles
        and schema_consistent
        and label_consistent
    )
    print()
    print(f"Final status: {'PASS' if final_status else 'FAIL'}")
    print()
    print("============================================")


def main() -> int:
    """Run the read-only Network dataset report."""

    try:
        source_directory, files, discovery_warnings = discover_network_files()
    except FileNotFoundError as error:
        print("========== TON-IOT NETWORK DATASET ==========")
        print(f"ERROR: {error}")
        print("Final status: FAIL")
        return 1

    if not files:
        print("No files to read. Final status: FAIL")
        return 1

    try:
        sample = pd.read_csv(files[0], nrows=1_000, low_memory=False)
        label_column, type_column = detect_label_columns(sample)
    except Exception as error:
        print("========== TON-IOT NETWORK DATASET ==========")
        print(f"ERROR: Could not inspect the first file: {type(error).__name__}: {error}")
        print("Final status: FAIL")
        return 1

    results = []
    for path in files:
        print(f"Reading {path.name} ...", flush=True)
        result = read_one_file(path, label_column, type_column)
        results.append((path, result))

    schema_consistent, schema_warnings = compare_schemas(results)
    print()
    print_report(
        source_directory=source_directory,
        files=files,
        discovery_warnings=discovery_warnings,
        detected_label_column=label_column,
        detected_type_column=type_column,
        results=results,
        schema_consistent=schema_consistent,
        schema_warnings=schema_warnings,
    )

    files_read = sum(bool(result["read_successfully"]) for _, result in results)
    total_rows = sum(int(result["rows"]) for _, result in results)
    total_normal = sum(int(result["normal"]) for _, result in results)
    total_attack = sum(int(result["attack"]) for _, result in results)
    total_invalid = sum(int(result["invalid"]) for _, result in results)
    label_consistent = all(
        int(result["missing_label"]) == 0
        and int(result["invalid_label"]) == 0
        and int(result["conflict"]) == 0
        for _, result in results
    )
    final_status = (
        len(files) == EXPECTED_FILE_COUNT
        and files_read == EXPECTED_FILE_COUNT
        and total_rows == total_normal + total_attack + total_invalid
        and schema_consistent
        and label_consistent
    )
    return 0 if final_status else 1


if __name__ == "__main__":
    raise SystemExit(main())
