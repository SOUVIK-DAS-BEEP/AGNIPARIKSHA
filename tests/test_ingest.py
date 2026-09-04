"""
Tests for src/ingest.py — Stage 1: Data Ingestion.

Validates loading (CSV/Parquet), schema validation (with deliberately
broken inputs), and wide-format pivoting.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingest import (
    load_burn_in_data,
    validate_schema,
    to_wide,
    ValidationReport,
    REQUIRED_COLUMNS,
    MEASURED_PARAMS,
)
from src.constants import (
    TIMEPOINTS,
    BURN_IN_HOURS,
    COL_CHIP_ID,
    COL_LOT_ID,
    COL_T_HOURS,
    COL_I_LEAK,
    COL_V_TH,
    COL_T_DELAY,
    STRESS_TEMP_C,
    NOMINAL_VOLTAGE_V,
)
from src.generator import generate_lot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_lot() -> pd.DataFrame:
    """A valid generated lot for testing."""
    return generate_lot(n_chips=50, seed=42)


@pytest.fixture
def valid_lot_no_truth(valid_lot: pd.DataFrame) -> pd.DataFrame:
    """Valid lot with true_class removed (simulates real uploaded data)."""
    return valid_lot.drop(columns=["true_class"])


# ---------------------------------------------------------------------------
# Loading tests
# ---------------------------------------------------------------------------

class TestLoadBurnInData:
    """Test load_burn_in_data for CSV, Parquet, DataFrame, and error cases."""

    def test_load_from_dataframe(self, valid_lot: pd.DataFrame):
        """Pass-through should return an identical copy."""
        loaded = load_burn_in_data(valid_lot)
        pd.testing.assert_frame_equal(loaded, valid_lot)
        # Must be a copy, not the same object
        assert loaded is not valid_lot

    def test_load_csv(self, valid_lot: pd.DataFrame, tmp_path: Path):
        """Write and read back a CSV file."""
        csv_path = tmp_path / "test_data.csv"
        valid_lot.to_csv(csv_path, index=False)
        loaded = load_burn_in_data(csv_path)
        assert len(loaded) == len(valid_lot)
        assert set(loaded.columns) == set(valid_lot.columns)

    def test_load_csv_from_string_path(self, valid_lot: pd.DataFrame, tmp_path: Path):
        """Load CSV using a string path instead of Path object."""
        csv_path = tmp_path / "test_data.csv"
        valid_lot.to_csv(csv_path, index=False)
        loaded = load_burn_in_data(str(csv_path))
        assert len(loaded) == len(valid_lot)

    def test_load_parquet(self, valid_lot: pd.DataFrame, tmp_path: Path):
        """Write and read back a Parquet file."""
        pq_path = tmp_path / "test_data.parquet"
        valid_lot.to_parquet(pq_path, index=False)
        loaded = load_burn_in_data(pq_path)
        assert len(loaded) == len(valid_lot)
        assert set(loaded.columns) == set(valid_lot.columns)

    def test_load_csv_from_buffer(self, valid_lot: pd.DataFrame):
        """Load CSV from a StringIO buffer."""
        buf = io.StringIO(valid_lot.to_csv(index=False))
        loaded = load_burn_in_data(buf)
        assert len(loaded) == len(valid_lot)

    def test_file_not_found(self):
        """Non-existent file should raise ValueError, not crash."""
        with pytest.raises(ValueError, match="not found"):
            load_burn_in_data(Path("/nonexistent/file.csv"))

    def test_unsupported_format(self, tmp_path: Path):
        """Unknown file extension should raise ValueError."""
        bad_path = tmp_path / "data.xlsx"
        bad_path.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_burn_in_data(bad_path)


# ---------------------------------------------------------------------------
# Validation tests — deliberately broken inputs
# ---------------------------------------------------------------------------

class TestValidateSchema:
    """Test validate_schema with good and intentionally bad data."""

    def test_valid_data_passes(self, valid_lot: pd.DataFrame):
        report = validate_schema(valid_lot)
        assert report.is_valid, f"Valid data should pass:\n{report}"
        assert len(report.errors) == 0

    def test_reports_chip_count(self, valid_lot: pd.DataFrame):
        report = validate_schema(valid_lot)
        expected_chips = valid_lot[COL_CHIP_ID].nunique()
        assert report.n_chips == expected_chips

    def test_reports_row_count(self, valid_lot: pd.DataFrame):
        report = validate_schema(valid_lot)
        assert report.n_rows == len(valid_lot)

    def test_missing_columns(self):
        """Missing required columns should produce errors."""
        df = pd.DataFrame({"chip_id": ["A"], "foo": [1]})
        report = validate_schema(df)
        assert not report.is_valid
        assert any("Missing required columns" in e for e in report.errors)

    def test_non_numeric_column(self, valid_lot: pd.DataFrame):
        """Non-numeric data in a numeric column should be flagged."""
        bad = valid_lot.copy()
        # Replace the entire column with object dtype containing a bad value
        col_vals = bad[COL_I_LEAK].astype(object).copy()
        col_vals.iloc[0] = "not_a_number"
        bad[COL_I_LEAK] = col_vals
        report = validate_schema(bad)
        assert not report.is_valid or len(report.warnings) > 0

    def test_wrong_timepoint_count(self, valid_lot: pd.DataFrame):
        """Chips with != 4 rows should be flagged."""
        # Remove one row
        bad = valid_lot.drop(index=0).reset_index(drop=True)
        report = validate_schema(bad)
        assert not report.is_valid
        assert any("timepoints" in e.lower() for e in report.errors)

    def test_duplicate_chip_timepoint(self, valid_lot: pd.DataFrame):
        """Duplicate (chip_id, t_hours) pairs should be flagged."""
        # Duplicate the first row
        bad = pd.concat([valid_lot, valid_lot.iloc[:1]], ignore_index=True)
        report = validate_schema(bad)
        assert not report.is_valid
        assert any("duplicate" in e.lower() for e in report.errors)

    def test_unexpected_timepoints(self):
        """Unexpected timepoint values should produce warnings."""
        df = pd.DataFrame({
            COL_CHIP_ID: ["A"] * 4,
            COL_LOT_ID: ["L1"] * 4,
            "wafer_id": ["W1"] * 4,
            COL_T_HOURS: [0, 24, 96, 200],  # 200 is unexpected
            COL_I_LEAK: [10.0] * 4,
            COL_V_TH: [0.45] * 4,
            COL_T_DELAY: [1.8] * 4,
            "temp_c": [125.0] * 4,
            "voltage_v": [1.2] * 4,
        })
        report = validate_schema(df)
        assert any("unexpected" in w.lower() for w in report.warnings)

    def test_nan_values_warned(self, valid_lot: pd.DataFrame):
        """NaN in measured parameters should be warned about."""
        bad = valid_lot.copy()
        bad.loc[0, COL_I_LEAK] = np.nan
        report = validate_schema(bad)
        assert any("nan" in w.lower() for w in report.warnings)

    def test_empty_dataframe(self):
        """Empty DataFrame with correct columns should pass but report 0 chips."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        report = validate_schema(df)
        assert report.n_chips == 0
        assert report.n_rows == 0

    def test_validation_never_crashes(self):
        """Even a totally wrong DataFrame should produce a report, not crash."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        report = validate_schema(df)
        assert isinstance(report, ValidationReport)
        assert not report.is_valid

    def test_str_representation(self, valid_lot: pd.DataFrame):
        """The report should have a readable string form."""
        report = validate_schema(valid_lot)
        s = str(report)
        assert "VALID" in s


# ---------------------------------------------------------------------------
# Wide-format pivot tests
# ---------------------------------------------------------------------------

class TestToWide:
    """Test the long-to-wide pivot."""

    def test_one_row_per_chip(self, valid_lot: pd.DataFrame):
        wide = to_wide(valid_lot)
        n_chips = valid_lot[COL_CHIP_ID].nunique()
        assert len(wide) == n_chips

    def test_expected_columns(self, valid_lot: pd.DataFrame):
        wide = to_wide(valid_lot)
        # Check that each measured param has columns for each timepoint
        for param in MEASURED_PARAMS:
            for tp in TIMEPOINTS:
                col_name = f"{param}_{tp}h"
                assert col_name in wide.columns, (
                    f"Expected column '{col_name}' not found in wide format"
                )

    def test_metadata_preserved(self, valid_lot: pd.DataFrame):
        wide = to_wide(valid_lot)
        assert COL_CHIP_ID in wide.columns
        assert COL_LOT_ID in wide.columns
        assert "wafer_id" in wide.columns

    def test_true_class_preserved_if_present(self, valid_lot: pd.DataFrame):
        """true_class should be carried through for evaluation."""
        wide = to_wide(valid_lot)
        assert "true_class" in wide.columns

    def test_true_class_absent_if_not_present(self, valid_lot_no_truth: pd.DataFrame):
        """Without true_class in input, it should not appear in output."""
        wide = to_wide(valid_lot_no_truth)
        assert "true_class" not in wide.columns

    def test_values_match_long_format(self, valid_lot: pd.DataFrame):
        """Spot-check that pivoted values match the original long format."""
        wide = to_wide(valid_lot)
        # Check a specific chip
        chip_id = valid_lot[COL_CHIP_ID].iloc[0]
        chip_long = valid_lot[valid_lot[COL_CHIP_ID] == chip_id]
        chip_wide = wide[wide[COL_CHIP_ID] == chip_id].iloc[0]

        for tp in TIMEPOINTS:
            long_val = chip_long[chip_long[COL_T_HOURS] == tp][COL_I_LEAK].values[0]
            wide_val = chip_wide[f"{COL_I_LEAK}_{tp}h"]
            assert long_val == pytest.approx(wide_val), (
                f"Mismatch for {chip_id} at t={tp}h: "
                f"long={long_val}, wide={wide_val}"
            )

    def test_column_count(self, valid_lot: pd.DataFrame):
        """Wide format should have the right number of columns."""
        wide = to_wide(valid_lot)
        # 3 params * 4 timepoints = 12 measurement cols
        # + chip_id, lot_id, wafer_id, temp_c, voltage_v, true_class = 6 meta
        # Total = 18
        n_measure = len(MEASURED_PARAMS) * len(TIMEPOINTS)
        n_meta = 6  # chip_id + lot_id + wafer_id + temp_c + voltage_v + true_class
        assert len(wide.columns) == n_measure + n_meta


# ---------------------------------------------------------------------------
# Round-trip: generate -> validate -> pivot
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """End-to-end test: generate, validate, pivot, verify."""

    def test_generated_data_validates(self):
        df = generate_lot(n_chips=100, seed=42)
        report = validate_schema(df)
        assert report.is_valid, f"Generated data should be valid:\n{report}"

    def test_generate_validate_pivot(self):
        """Full pipeline: generate -> validate -> to_wide -> verify."""
        df = generate_lot(n_chips=100, seed=42)
        report = validate_schema(df)
        assert report.is_valid

        wide = to_wide(df)
        # Every chip should have a row
        assert len(wide) == df[COL_CHIP_ID].nunique()
        # No NaN in the measured columns
        for param in MEASURED_PARAMS:
            for tp in TIMEPOINTS:
                col = f"{param}_{tp}h"
                assert wide[col].notna().all(), f"NaN found in {col}"
