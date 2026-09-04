"""
AGNIPARIKSHA — Stage 1: Data Ingestion.

Loads burn-in test data from CSV or Parquet, validates the schema against
the canonical long-format specification (SPEC.md Section 3.1), and pivots
to wide format (one row per chip) for downstream stages.

Key design principle: NEVER crash on malformed input. On validation failure,
return a ValidationReport listing every problem found.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

from src.constants import (
    TIMEPOINTS,
    COL_CHIP_ID,
    COL_LOT_ID,
    COL_WAFER_ID,
    COL_T_HOURS,
    COL_I_LEAK,
    COL_V_TH,
    COL_T_DELAY,
    COL_TEMP,
    COL_VOLTAGE,
)

# ---------------------------------------------------------------------------
# Required columns in the canonical schema (SPEC.md Section 3.1).
# true_class is intentionally excluded — it is ground-truth metadata that
# the pipeline must NEVER read; only the evaluation module may use it.
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: list[str] = [
    COL_CHIP_ID,
    COL_LOT_ID,
    COL_WAFER_ID,
    COL_T_HOURS,
    COL_I_LEAK,
    COL_V_TH,
    COL_T_DELAY,
    COL_TEMP,
    COL_VOLTAGE,
]

NUMERIC_COLUMNS: list[str] = [
    COL_T_HOURS,
    COL_I_LEAK,
    COL_V_TH,
    COL_T_DELAY,
    COL_TEMP,
    COL_VOLTAGE,
]

# The three measured parameters that get pivoted to wide format
MEASURED_PARAMS: list[str] = [COL_I_LEAK, COL_V_TH, COL_T_DELAY]


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """
    Report of all validation issues found in a dataset.

    Attributes
    ----------
    is_valid : bool
        True if no problems were found.
    errors : list[str]
        Critical issues that prevent processing (e.g., missing columns).
    warnings : list[str]
        Non-critical issues that may affect results but allow processing.
    n_chips : int
        Number of unique chips found.
    n_rows : int
        Total number of rows.
    """
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_chips: int = 0
    n_rows: int = 0

    def add_error(self, msg: str) -> None:
        """Add a critical error and mark the report as invalid."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        """Add a non-critical warning."""
        self.warnings.append(msg)

    def __str__(self) -> str:
        lines = []
        status = "VALID" if self.is_valid else "INVALID"
        lines.append(f"ValidationReport: {status}")
        lines.append(f"  Rows: {self.n_rows}, Chips: {self.n_chips}")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_burn_in_data(
    path_or_buffer: Union[str, Path, io.IOBase, pd.DataFrame],
) -> pd.DataFrame:
    """
    Load burn-in test data from CSV, Parquet, or an existing DataFrame.

    Parameters
    ----------
    path_or_buffer : str, Path, file-like, or pd.DataFrame
        Path to a CSV (.csv) or Parquet (.parquet/.pq) file, a file-like
        object (treated as CSV), or an already-loaded DataFrame.

    Returns
    -------
    pd.DataFrame
        Raw data as loaded. No validation is performed here — call
        validate_schema() separately.

    Raises
    ------
    ValueError
        If the file format is unsupported or the file cannot be read.
    """
    # Already a DataFrame — pass through
    if isinstance(path_or_buffer, pd.DataFrame):
        return path_or_buffer.copy()

    # Convert to Path if string
    if isinstance(path_or_buffer, str):
        path_or_buffer = Path(path_or_buffer)

    # File-like object — treat as CSV
    if isinstance(path_or_buffer, io.IOBase):
        try:
            return pd.read_csv(path_or_buffer)
        except Exception as e:
            raise ValueError(f"Failed to read CSV from buffer: {e}") from e

    # Path-based loading
    if isinstance(path_or_buffer, Path):
        if not path_or_buffer.exists():
            raise ValueError(f"File not found: {path_or_buffer}")

        suffix = path_or_buffer.suffix.lower()
        if suffix == ".csv":
            try:
                return pd.read_csv(path_or_buffer)
            except Exception as e:
                raise ValueError(f"Failed to read CSV '{path_or_buffer}': {e}") from e
        elif suffix in (".parquet", ".pq"):
            try:
                return pd.read_parquet(path_or_buffer)
            except Exception as e:
                raise ValueError(f"Failed to read Parquet '{path_or_buffer}': {e}") from e
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                f"Expected .csv, .parquet, or .pq"
            )

    raise ValueError(
        f"Unsupported input type '{type(path_or_buffer).__name__}'. "
        f"Expected str, Path, file-like, or DataFrame."
    )


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> ValidationReport:
    """
    Validate a DataFrame against the canonical burn-in schema.

    Checks performed (SPEC.md Section 4, Stage 1):
      1. Required columns present
      2. Numeric columns are actually numeric
      3. Exactly 4 timepoints per chip
      4. No duplicate (chip_id, t_hours) pairs
      5. Timepoints are from the expected set {0, 24, 96, 168}

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate.

    Returns
    -------
    ValidationReport
        Report listing all problems found. Never raises on bad input.
    """
    report = ValidationReport()
    report.n_rows = len(df)

    # --- Check 1: Required columns ---
    present_cols = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in present_cols]
    if missing:
        report.add_error(f"Missing required columns: {missing}")
        # Can't do further checks without the key columns
        return report

    report.n_chips = df[COL_CHIP_ID].nunique()

    # --- Check 2: Numeric columns ---
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Try coercing
                coerced = pd.to_numeric(df[col], errors="coerce")
                n_failed = coerced.isna().sum() - df[col].isna().sum()
                if n_failed > 0:
                    report.add_error(
                        f"Column '{col}' has {n_failed} non-numeric values"
                    )
                else:
                    report.add_warning(
                        f"Column '{col}' is not numeric dtype but values are "
                        f"convertible (dtype: {df[col].dtype})"
                    )

    # --- Check 3: Timepoints per chip ---
    rows_per_chip = df.groupby(COL_CHIP_ID).size()
    bad_counts = rows_per_chip[rows_per_chip != len(TIMEPOINTS)]
    if len(bad_counts) > 0:
        # Report up to 10 examples
        examples = bad_counts.head(10).to_dict()
        report.add_error(
            f"{len(bad_counts)} chip(s) do not have exactly "
            f"{len(TIMEPOINTS)} timepoints. Examples: {examples}"
        )

    # --- Check 4: Duplicate (chip_id, t_hours) pairs ---
    dupes = df.duplicated(subset=[COL_CHIP_ID, COL_T_HOURS], keep=False)
    if dupes.any():
        n_dupes = dupes.sum()
        dupe_chips = df.loc[dupes, COL_CHIP_ID].unique()[:10]
        report.add_error(
            f"{n_dupes} duplicate (chip_id, t_hours) rows found. "
            f"Affected chips (up to 10): {list(dupe_chips)}"
        )

    # --- Check 5: Valid timepoints ---
    if pd.api.types.is_numeric_dtype(df[COL_T_HOURS]):
        unique_times = set(df[COL_T_HOURS].unique())
        expected_times = set(TIMEPOINTS)
        unexpected = unique_times - expected_times
        if unexpected:
            report.add_warning(
                f"Unexpected timepoint values found: {sorted(unexpected)}. "
                f"Expected: {sorted(expected_times)}"
            )

    # --- Check 6: NaN in measured parameters ---
    for col in MEASURED_PARAMS:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            report.add_warning(
                f"Column '{col}' has {n_nan} NaN value(s)"
            )

    return report


# ---------------------------------------------------------------------------
# Pivot to wide format
# ---------------------------------------------------------------------------

def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format burn-in data to wide format (one row per chip).

    Creates columns like i_leak_ua_0h, i_leak_ua_24h, i_leak_ua_96h,
    i_leak_ua_168h for each measured parameter.

    Metadata columns (lot_id, wafer_id, temp_c, voltage_v) are carried
    through from the first row of each chip. The true_class column is
    preserved if present (for evaluation use only — the pipeline must
    NEVER read it).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with the canonical schema.

    Returns
    -------
    pd.DataFrame
        Wide-format data with one row per chip.
    """
    # Pivot each measured parameter
    pivoted_parts = []
    for param in MEASURED_PARAMS:
        pivot = df.pivot(
            index=COL_CHIP_ID,
            columns=COL_T_HOURS,
            values=param,
        )
        # Rename columns: e.g., 0 -> i_leak_ua_0h, 168 -> i_leak_ua_168h
        pivot.columns = [f"{param}_{int(t)}h" for t in pivot.columns]
        pivoted_parts.append(pivot)

    # Combine all pivoted parameters
    wide = pd.concat(pivoted_parts, axis=1)

    # Add metadata columns (take from the first row of each chip)
    meta_cols = [COL_LOT_ID, COL_WAFER_ID, COL_TEMP, COL_VOLTAGE]
    # Include true_class if present (for evaluation, never for pipeline)
    if "true_class" in df.columns:
        meta_cols.append("true_class")

    meta = df.groupby(COL_CHIP_ID)[meta_cols].first()
    wide = meta.join(wide)

    # Reset index so chip_id is a regular column
    wide = wide.reset_index()

    return wide
