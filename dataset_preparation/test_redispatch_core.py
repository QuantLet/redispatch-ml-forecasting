"""
Validation Tests for Redispatch Core Module.

This script validates that the extracted logic in redispatch_core.py
works correctly by running basic tests against sample data.

Usage:
    python -m models.test_redispatch_core
    
Or run as a script:
    python models/test_redispatch_core.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_preparation.tso_config import (
    GERMAN_TSOS,
    ALL_NEIGHBOR_COUNTRIES,
    TSO_NEIGHBORS,
    normalize_tso_name,
    get_neighbors,
    get_tso_for_bundesland,
    get_split_method,
    is_german_tso,
)
from dataset_preparation.redispatch_core import (
    TARGET_RELEVANT_MEASUREMENT_REASONS,
    bucket_overlap_split,
    split_block_allocation,
    apply_hourly_split_method,
    add_data_driven_features,
    add_runlength_features,
    add_missing_intervals,
)


def test_tso_config():
    """Test TSO configuration module functions."""
    print("=" * 60)
    print("Testing TSO Configuration Module")
    print("=" * 60)
    
    # Test 1: All German TSOs present
    print("\n[Test 1] All German TSOs present...")
    expected_tsos = {"50Hertz", "TenneT DE", "Amprion", "TransnetBW"}
    actual_tsos = set(GERMAN_TSOS)
    assert actual_tsos == expected_tsos, f"Expected {expected_tsos}, got {actual_tsos}"
    print("  ✓ All 4 German TSOs correctly defined")
    
    # Test 2: TSO name normalization
    print("\n[Test 2] TSO name normalization...")
    test_cases = [
        ("TenneT", "TenneT DE"),
        ("50HzT", "50Hertz"),
        ("DE_Amprion", "Amprion"),
        ("TransnetBW_DE", "TransnetBW"),
        ("Unknown TSO", "Unknown TSO"),  # Should return unchanged
    ]
    for raw_name, expected in test_cases:
        result = normalize_tso_name(raw_name)
        assert result == expected, f"normalize_tso_name('{raw_name}') = '{result}', expected '{expected}'"
        print(f"  ✓ '{raw_name}' → '{result}'")
    
    # Test 3: TSO neighbors correctly mapped
    print("\n[Test 3] TSO neighbors correctly mapped...")
    assert "DK_2" in get_neighbors("50Hertz"), "50Hertz should have DK_2 as neighbor"
    assert "DK_1" in get_neighbors("TenneT DE"), "TenneT DE should have DK_1 as neighbor"
    assert "FR" in get_neighbors("Amprion"), "Amprion should have FR as neighbor"
    assert "CH" in get_neighbors("TransnetBW"), "TransnetBW should have CH as neighbor"
    print("  ✓ 50Hertz neighbors:", get_neighbors("50Hertz"))
    print("  ✓ TenneT DE neighbors:", get_neighbors("TenneT DE"))
    print("  ✓ Amprion neighbors:", get_neighbors("Amprion"))
    print("  ✓ TransnetBW neighbors:", get_neighbors("TransnetBW"))
    
    # Test 4: Bundesland to TSO mapping
    print("\n[Test 4] Bundesland to TSO mapping...")
    assert get_tso_for_bundesland("Bayern") == "TenneT DE"
    assert get_tso_for_bundesland("Berlin") == "50Hertz"
    assert get_tso_for_bundesland("Nordrhein-Westfalen") == "Amprion"
    assert get_tso_for_bundesland("Baden-Württemberg") == "TransnetBW"
    print("  ✓ Bundesland mappings correct")
    
    # Test 5: Split method defaults
    print("\n[Test 5] Split method defaults...")
    assert get_split_method("50Hertz") == "equal"
    assert get_split_method("Amprion") == "split"
    assert get_split_method("TenneT DE") == "equal"
    assert get_split_method("TransnetBW") == "equal"
    print("  ✓ Split methods correctly assigned")
    
    # Test 6: is_german_tso
    print("\n[Test 6] is_german_tso function...")
    assert is_german_tso("50Hertz") == True
    assert is_german_tso("TenneT") == True  # Tests normalization
    assert is_german_tso("APG") == False  # Austrian TSO
    print("  ✓ is_german_tso working correctly")
    
    print("\n✓ All TSO configuration tests passed!")


def test_hourly_split_methods():
    """Test hourly split allocation methods."""
    print("\n" + "=" * 60)
    print("Testing Hourly Split Methods")
    print("=" * 60)
    
    # Test 1: bucket_overlap_split with single hour intervention
    print("\n[Test 1] bucket_overlap_split - single hour...")
    row = pd.Series({
        "begin_date": pd.Timestamp("2023-01-01 10:00:00"),
        "end_date": pd.Timestamp("2023-01-01 11:00:00"),
        "total_load": 100.0,
        "mean_load": 100.0
    })
    result = bucket_overlap_split(row)
    assert len(result) == 1, f"Expected 1 bucket, got {len(result)}"
    assert np.isclose(result["total_load"].sum(), 100.0), "Energy not conserved"
    print(f"  ✓ Single hour: {len(result)} bucket, total_load = {result['total_load'].sum():.2f}")
    
    # Test 2: bucket_overlap_split with multi-hour intervention
    print("\n[Test 2] bucket_overlap_split - 2.5 hours...")
    row = pd.Series({
        "begin_date": pd.Timestamp("2023-01-01 10:30:00"),
        "end_date": pd.Timestamp("2023-01-01 13:00:00"),
        "total_load": 250.0,
        "mean_load": 100.0
    })
    result = bucket_overlap_split(row)
    assert len(result) == 3, f"Expected 3 buckets, got {len(result)}"
    assert np.isclose(result["total_load"].sum(), 250.0, rtol=1e-5), "Energy not conserved"
    print(f"  ✓ Multi-hour: {len(result)} buckets, total_load = {result['total_load'].sum():.2f}")
    
    # Test 3: split_block_allocation
    print("\n[Test 3] split_block_allocation - 4 hour intervention with 2h active...")
    row = pd.Series({
        "begin_date": pd.Timestamp("2023-01-01 10:00:00"),
        "end_date": pd.Timestamp("2023-01-01 14:00:00"),
        "total_load": 200.0,
        "mean_load": 100.0  # active_duration = 200/100 = 2 hours
    })
    result = split_block_allocation(row)
    assert np.isclose(result["total_load"].sum(), 200.0, rtol=1e-5), "Energy not conserved"
    print(f"  ✓ Split allocation: {len(result)} buckets, total_load = {result['total_load'].sum():.2f}")
    
    # Test 4: Energy conservation in apply_hourly_split_method
    print("\n[Test 4] apply_hourly_split_method - energy conservation...")
    test_data = pd.DataFrame({
        "begin_date": [
            pd.Timestamp("2023-01-01 10:00:00"),
            pd.Timestamp("2023-01-01 14:00:00"),
        ],
        "end_date": [
            pd.Timestamp("2023-01-01 12:00:00"),
            pd.Timestamp("2023-01-01 16:30:00"),
        ],
        "total_load": [100.0, 150.0],
        "mean_load": [50.0, 60.0],
        "operator": ["TestTSO", "TestTSO"],
        "direction": ["up", "down"],
    })
    
    result_equal = apply_hourly_split_method(test_data.copy(), method="equal")
    result_split = apply_hourly_split_method(test_data.copy(), method="split")
    
    original_total = test_data["total_load"].sum()
    equal_total = result_equal["total_load"].sum()
    split_total = result_split["total_load"].sum()
    
    assert np.isclose(equal_total, original_total, rtol=1e-5), f"Equal method: {equal_total} != {original_total}"
    assert np.isclose(split_total, original_total, rtol=1e-5), f"Split method: {split_total} != {original_total}"
    print(f"  ✓ Original total: {original_total:.2f}")
    print(f"  ✓ Equal method total: {equal_total:.2f}")
    print(f"  ✓ Split method total: {split_total:.2f}")
    
    print("\n✓ All hourly split method tests passed!")


def test_data_driven_features():
    """Test data-driven feature computation."""
    print("\n" + "=" * 60)
    print("Testing Data-Driven Features")
    print("=" * 60)
    
    # Create sample data
    test_data = pd.DataFrame({
        "begin_date": pd.to_datetime([
            "2023-01-01 10:00", "2023-01-01 10:00", "2023-01-01 11:00"
        ]),
        "end_date": pd.to_datetime([
            "2023-01-01 11:00", "2023-01-01 11:30", "2023-01-01 12:00"
        ]),
        "total_load": [100.0, 50.0, 200.0],
        "mean_load": [100.0, 100.0, 200.0],
        "max_load": [100.0, 60.0, 200.0],
        "direction": ["up", "up", "up"],
    })
    
    print("\n[Test 1] Computing data-driven features...")
    features = add_data_driven_features(test_data, add_active_duration_features=True)
    
    expected_cols = [
        "begin_date", "direction",
        "avg_active_duration_ratio", "total_active_duration", "max_active_duration",
        "n_interventions", "max_load_max", "max_load_q90"
    ]
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"
    print(f"  ✓ All expected columns present: {list(features.columns)}")
    
    # Test 2: Check n_interventions
    print("\n[Test 2] Checking n_interventions...")
    row_10h = features[features["begin_date"] == pd.Timestamp("2023-01-01 10:00")]
    assert row_10h["n_interventions"].values[0] == 2, "Hour 10 should have 2 interventions"
    print(f"  ✓ Hour 10:00 has {int(row_10h['n_interventions'].values[0])} interventions")
    
    # Test 3: max_load features
    print("\n[Test 3] Checking max_load features...")
    assert row_10h["max_load_max"].values[0] == 100.0, "max_load_max should be 100"
    print(f"  ✓ max_load_max = {row_10h['max_load_max'].values[0]}")
    
    print("\n✓ All data-driven feature tests passed!")


def test_runlength_features():
    """Test run-length feature computation."""
    print("\n" + "=" * 60)
    print("Testing Run-Length Features")
    print("=" * 60)
    
    # Create sample data with alternating zero/non-zero pattern
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    test_data = pd.DataFrame({
        "begin_date": dates,
        "total_load": [0, 100, 100, 0, 0, 0, 50, 50, 50, 0],
        "direction": ["up"] * 10,
    })
    
    print("\n[Test 1] Computing run-length features...")
    features = add_runlength_features(
        test_data.copy(),
        y_col="total_load",
        id_col="direction",
        time_col="begin_date",
        window_size=5
    )
    
    expected_cols = [
        "indicator", "run_len_pos", "run_len_zero",
        "run_switches", "time_since_last_positive",
        "pos_count_window", "pos_mean_window", "switch_rate_normalized"
    ]
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"
    print(f"  ✓ All expected columns present")
    
    # Test 2: Check indicator
    print("\n[Test 2] Checking indicator values...")
    expected_indicator = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    actual_indicator = features["indicator"].tolist()
    assert actual_indicator == expected_indicator, f"Indicator mismatch: {actual_indicator}"
    print(f"  ✓ Indicator correct: {actual_indicator}")
    
    # Test 3: Check run_len_pos at position 8 (3 consecutive positives)
    print("\n[Test 3] Checking run_len_pos...")
    assert features["run_len_pos"].iloc[8] == 3, "run_len_pos at idx 8 should be 3"
    print(f"  ✓ run_len_pos at idx 8 = {int(features['run_len_pos'].iloc[8])}")
    
    print("\n✓ All run-length feature tests passed!")


def test_missing_intervals():
    """Test missing interval handling."""
    print("\n" + "=" * 60)
    print("Testing Missing Interval Handling")
    print("=" * 60)
    
    # Create data with gaps
    test_data = pd.DataFrame({
        "begin_date": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 02:00"]),
        "total_load": [100.0, 200.0],
    })
    
    start_date = datetime(2023, 1, 1, 0, 0)
    end_date = datetime(2023, 1, 1, 3, 0)
    
    print("\n[Test 1] Filling missing intervals...")
    result = add_missing_intervals(
        test_data, direction="up",
        start_date=start_date, end_date=end_date, freq="h"
    )
    
    # Should have 4 hourly entries (00:00, 01:00, 02:00, 03:00)
    assert len(result) == 4, f"Expected 4 intervals, got {len(result)}"
    print(f"  ✓ Result has {len(result)} intervals")
    
    # Check that gap at 01:00 is filled with 0
    row_01h = result[result["begin_date"] == pd.Timestamp("2023-01-01 01:00")]
    assert len(row_01h) == 1, "Should have one row for 01:00"
    assert row_01h["total_load"].values[0] == 0.0, "Gap should be filled with 0"
    print(f"  ✓ Gap at 01:00 filled with total_load = {row_01h['total_load'].values[0]}")
    
    print("\n✓ All missing interval tests passed!")


def test_basic_validation():
    """Basic validation tests (requires actual data file)."""
    print("\n" + "=" * 60)
    print("Testing Basic Validation (with actual data if available)")
    print("=" * 60)
    
    data_path = Path(__file__).parent.parent / "data" / "redispatch_data_9_dec_2025.csv"
    translations_path = Path(__file__).parent.parent / "data_processing" / "translations.json"
    
    if not data_path.exists():
        print(f"\n⚠ Skipping: Data file not found at {data_path}")
        return
    
    if not translations_path.exists():
        print(f"\n⚠ Skipping: Translations file not found at {translations_path}")
        return
    
    from dataset_preparation.redispatch_core import read_redispatch_data
    
    print("\n[Test 1] Loading redispatch data...")
    try:
        redispatch = read_redispatch_data(
            str(data_path),
            str(translations_path),
            timezone="UTC",
            start_date=datetime(2021, 10, 1)
        )
        print(f"  ✓ Loaded {len(redispatch)} rows")
        
        # Test 2: Check columns exist
        print("\n[Test 2] Checking required columns...")
        required_cols = [
            "begin_date", "end_date", "total_load", "mean_load",
            "directing_operator", "direction", "affected_unit"
        ]
        for col in required_cols:
            assert col in redispatch.columns, f"Missing column: {col}"
        print(f"  ✓ All required columns present")
        
        # Test 3: Check operators
        print("\n[Test 3] Checking operators...")
        unique_operators = redispatch["directing_operator"].unique()
        german_tsos_in_data = [op for op in unique_operators if op in GERMAN_TSOS]
        print(f"  ✓ Found German TSOs: {german_tsos_in_data}")
        
        # Test 4: Check directions
        print("\n[Test 4] Checking directions...")
        unique_directions = redispatch["direction"].unique()
        assert "up" in unique_directions, "Missing 'up' direction"
        assert "down" in unique_directions, "Missing 'down' direction"
        print(f"  ✓ Directions: {list(unique_directions)}")
        
        # Test 5: Check total_load is reasonable
        print("\n[Test 5] Checking total_load values...")
        assert redispatch["total_load"].min() > 0, "Should have no zero loads (filtered earlier)"
        max_load = redispatch["total_load"].max()
        assert max_load < 50000, f"Suspicious max load: {max_load} MWh"
        print(f"  ✓ total_load range: {redispatch['total_load'].min():.2f} - {max_load:.2f} MWh")
        
        print("\n✓ All basic validation tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        raise


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("REDISPATCH CORE MODULE VALIDATION")
    print("=" * 60)
    
    try:
        test_tso_config()
        test_hourly_split_methods()
        test_data_driven_features()
        test_runlength_features()
        test_missing_intervals()
        test_basic_validation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    sys.exit(run_all_tests())
