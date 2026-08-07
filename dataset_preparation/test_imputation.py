"""
Unit Tests for Imputation Module.

This script tests the imputation functions in the feature_loaders/imputation.py module.
Tests cover gap detection, seasonal median computation, and feature-specific imputation.

Usage:
    python -m dataset_preparation.test_imputation
    
Or run as a script:
    python dataset_preparation/test_imputation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_preparation.feature_loaders.imputation import (
    detect_gap_lengths,
    compute_seasonal_median,
    impute_with_gap_detection,
    impute_forward_fill_seasonal,
    impute_forward_fill_only,
    impute_fill_zero,
    impute_day_ahead_prices,
    impute_production,
    impute_consumption,
    impute_wind_pv,
    impute_cross_border_flows,
    impute_bloomberg,
    impute_redispatch_core,
    get_imputation_stats,
    validate_no_missing,
    impute_all_features,
)


def create_sample_timeseries(
    n_rows: int = 100,
    freq: str = "h",
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Create a sample time series DataFrame for testing."""
    dates = pd.date_range(start=start_date, periods=n_rows, freq=freq)
    return pd.DataFrame({"begin_date": dates})


def test_detect_gap_lengths():
    """Test gap length detection."""
    print("=" * 60)
    print("Testing detect_gap_lengths")
    print("=" * 60)
    
    # Test 1: Simple consecutive gaps
    print("\n[Test 1] Simple consecutive gaps...")
    s = pd.Series([1, np.nan, np.nan, 2, np.nan])
    result = detect_gap_lengths(s)
    expected = pd.Series([0, 1, 2, 0, 1])
    assert (result == expected).all(), f"Expected {expected.tolist()}, got {result.tolist()}"
    print("  ✓ Correctly detected gap lengths for simple series")
    
    # Test 2: No gaps
    print("\n[Test 2] No gaps...")
    s = pd.Series([1, 2, 3, 4, 5])
    result = detect_gap_lengths(s)
    assert (result == 0).all(), "Expected all zeros for series with no gaps"
    print("  ✓ Correctly handled series with no gaps")
    
    # Test 3: All gaps
    print("\n[Test 3] All gaps...")
    s = pd.Series([np.nan, np.nan, np.nan])
    result = detect_gap_lengths(s)
    expected = pd.Series([1, 2, 3])
    assert (result == expected).all(), f"Expected {expected.tolist()}, got {result.tolist()}"
    print("  ✓ Correctly detected all-gap series")
    
    # Test 4: Multiple separate gaps
    print("\n[Test 4] Multiple separate gaps...")
    s = pd.Series([1, np.nan, 2, np.nan, np.nan, np.nan, 3])
    result = detect_gap_lengths(s)
    expected = pd.Series([0, 1, 0, 1, 2, 3, 0])
    assert (result == expected).all(), f"Expected {expected.tolist()}, got {result.tolist()}"
    print("  ✓ Correctly detected multiple separate gaps")
    
    print("\n✓ All detect_gap_lengths tests passed!")


def test_compute_seasonal_median():
    """Test seasonal median computation."""
    print("\n" + "=" * 60)
    print("Testing compute_seasonal_median")
    print("=" * 60)
    
    # Create sample data with hour-of-day pattern
    df = create_sample_timeseries(n_rows=48 * 7)  # One week of hourly data
    # Create values with hour-of-day pattern
    df["value"] = df["begin_date"].dt.hour + np.random.normal(0, 0.5, len(df))
    # Add some NaN values
    df.loc[df.index[10:15], "value"] = np.nan
    
    # Test 1: Seasonal median by hour
    print("\n[Test 1] Seasonal median by hour...")
    seasonal = compute_seasonal_median(df, "value", ["hour"])
    assert seasonal.notna().all(), "Seasonal median should not have NaN"
    # Check that values are roughly in the expected range (0-23 for hours)
    assert seasonal.min() >= -1 and seasonal.max() <= 24, "Seasonal values out of expected range"
    print("  ✓ Seasonal median by hour computed correctly")
    
    # Test 2: Seasonal median by hour × weekday
    print("\n[Test 2] Seasonal median by hour × weekday...")
    seasonal = compute_seasonal_median(df, "value", ["hour", "weekday"])
    assert seasonal.notna().all(), "Seasonal median should not have NaN"
    print("  ✓ Seasonal median by hour × weekday computed correctly")
    
    # Test 3: Missing grouping column
    print("\n[Test 3] Missing grouping column...")
    seasonal = compute_seasonal_median(df, "value", ["nonexistent_column"])
    # Should return NaN when no valid grouping
    # Note: The function should handle this gracefully
    print("  ✓ Missing grouping column handled gracefully")
    
    print("\n✓ All compute_seasonal_median tests passed!")


def test_impute_with_gap_detection():
    """Test imputation with gap detection (short=ffill, long=seasonal)."""
    print("\n" + "=" * 60)
    print("Testing impute_with_gap_detection")
    print("=" * 60)
    
    # Create sample data with hour-of-day pattern
    df = create_sample_timeseries(n_rows=24 * 7)  # One week of hourly data
    # Create values with hour-of-day pattern (easier to verify)
    df["value"] = df["begin_date"].dt.hour * 10.0
    
    # Add short gap (2 hours)
    df.loc[df.index[10:12], "value"] = np.nan
    # Add long gap (24 hours)
    df.loc[df.index[50:74], "value"] = np.nan
    
    # Test 1: Short gap should be forward-filled
    print("\n[Test 1] Short gap forward-fill...")
    result = impute_with_gap_detection(
        df, "value", 
        forward_fill_threshold_hours=10,
        seasonal_groups=["hour"],
    )
    # Short gap values should match the last valid value (value at index 9)
    expected_val = df.loc[df.index[9], "value"]
    assert result.iloc[10] == expected_val, f"Expected {expected_val}, got {result.iloc[10]}"
    assert result.iloc[11] == expected_val, f"Expected {expected_val}, got {result.iloc[11]}"
    print("  ✓ Short gap correctly forward-filled")
    
    # Test 2: Long gap should use seasonal median
    print("\n[Test 2] Long gap seasonal median...")
    result = impute_with_gap_detection(
        df, "value",
        forward_fill_threshold_hours=10,
        seasonal_groups=["hour"],
    )
    # Long gap values should be imputed with seasonal values
    assert result.iloc[50:74].notna().all(), "Long gap should be fully imputed"
    print("  ✓ Long gap correctly imputed with seasonal median")
    
    # Test 3: No missing values
    print("\n[Test 3] No missing values...")
    df_complete = create_sample_timeseries(n_rows=100)
    df_complete["value"] = np.arange(100, dtype=float)
    result = impute_with_gap_detection(
        df_complete, "value",
        forward_fill_threshold_hours=10,
        seasonal_groups=["hour"],
    )
    assert (result == df_complete["value"]).all(), "Complete series should be unchanged"
    print("  ✓ Complete series unchanged")
    
    print("\n✓ All impute_with_gap_detection tests passed!")


def test_impute_forward_fill_seasonal():
    """Test forward-fill + seasonal imputation."""
    print("\n" + "=" * 60)
    print("Testing impute_forward_fill_seasonal")
    print("=" * 60)
    
    df = create_sample_timeseries(n_rows=24 * 14)  # Two weeks
    df["col1"] = np.arange(len(df), dtype=float)
    df["col2"] = df["begin_date"].dt.hour * 5.0
    
    # Add gaps
    df.loc[df.index[5:8], "col1"] = np.nan  # Short gap
    df.loc[df.index[100:150], "col2"] = np.nan  # Long gap
    
    # Test imputation
    print("\n[Test 1] Multiple columns imputation...")
    result = impute_forward_fill_seasonal(
        df, ["col1", "col2"],
        gap_threshold_hours=48,
        seasonal_groups=["hour", "weekday"],
    )
    
    assert result["col1"].notna().all(), "col1 should have no NaN after imputation"
    assert result["col2"].notna().all(), "col2 should have no NaN after imputation"
    print("  ✓ Multiple columns correctly imputed")
    
    print("\n✓ All impute_forward_fill_seasonal tests passed!")


def test_impute_forward_fill_only():
    """Test forward-fill only imputation."""
    print("\n" + "=" * 60)
    print("Testing impute_forward_fill_only")
    print("=" * 60)
    
    df = create_sample_timeseries(n_rows=100)
    df["bloomberg_price"] = np.arange(100, dtype=float)
    df.loc[df.index[10:15], "bloomberg_price"] = np.nan
    
    # Test imputation
    print("\n[Test 1] Forward-fill only...")
    result = impute_forward_fill_only(df, ["bloomberg_price"])
    
    # All values in gap should equal the last valid value (9.0)
    expected = 9.0
    for i in range(10, 15):
        assert result.loc[result.index[i], "bloomberg_price"] == expected, \
            f"Expected {expected} at index {i}, got {result.loc[result.index[i], 'bloomberg_price']}"
    print("  ✓ Forward-fill only correctly applied")
    
    # Test 2: Gap at start (cannot forward-fill)
    print("\n[Test 2] Gap at start...")
    df2 = create_sample_timeseries(n_rows=100)
    df2["value"] = np.arange(100, dtype=float)
    df2.loc[df2.index[0:5], "value"] = np.nan
    result2 = impute_forward_fill_only(df2, ["value"])
    # First 5 values should still be NaN (nothing to forward-fill from)
    assert result2["value"].iloc[0:5].isna().all(), "Gap at start should remain NaN"
    print("  ✓ Gap at start correctly handled (remains NaN)")
    
    print("\n✓ All impute_forward_fill_only tests passed!")


def test_impute_fill_zero():
    """Test zero-fill imputation."""
    print("\n" + "=" * 60)
    print("Testing impute_fill_zero")
    print("=" * 60)
    
    df = create_sample_timeseries(n_rows=100)
    df["redispatch_feature"] = np.random.randn(100)
    df.loc[df.index[20:30], "redispatch_feature"] = np.nan
    
    print("\n[Test 1] Zero-fill...")
    result = impute_fill_zero(df, ["redispatch_feature"])
    
    # Gap should be filled with zeros
    assert (result.loc[result.index[20:30], "redispatch_feature"] == 0).all(), \
        "Gap should be filled with zeros"
    print("  ✓ Zero-fill correctly applied")
    
    # Values outside gap should be unchanged
    assert np.allclose(
        result.loc[result.index[:20], "redispatch_feature"],
        df.loc[df.index[:20], "redispatch_feature"],
        equal_nan=True,
    ), "Values outside gap should be unchanged"
    print("  ✓ Values outside gap unchanged")
    
    print("\n✓ All impute_fill_zero tests passed!")


def test_feature_specific_imputation():
    """Test feature-specific imputation functions."""
    print("\n" + "=" * 60)
    print("Testing Feature-Specific Imputation Functions")
    print("=" * 60)
    
    # Create a comprehensive test DataFrame with multiple feature types
    df = create_sample_timeseries(n_rows=24 * 30)  # One month
    
    # Day-ahead prices
    df["day_ahead_price_de_lu"] = 50 + df["begin_date"].dt.hour * 2 + np.random.randn(len(df))
    df["day_ahead_price_fr"] = 45 + df["begin_date"].dt.hour * 2 + np.random.randn(len(df))
    df.loc[df.index[100:110], "day_ahead_price_de_lu"] = np.nan
    
    # Bloomberg data
    df["bloomberg_gas_price"] = 30 + np.random.randn(len(df))
    df.loc[df.index[200:220], "bloomberg_gas_price"] = np.nan
    
    # Production forecast
    df["production_forecast_dk_1"] = 1000 + np.random.randn(len(df)) * 50
    df.loc[df.index[300:310], "production_forecast_dk_1"] = np.nan
    
    # Consumption forecast
    df["consumption_forecast_load_de_50hzt"] = 500 + df["begin_date"].dt.hour * 20
    df.loc[df.index[400:410], "consumption_forecast_load_de_50hzt"] = np.nan
    
    # Wind/PV forecast
    df["wind_forecast_de_50hzt_wind_onshore"] = 200 + np.sin(df["begin_date"].dt.hour * np.pi / 12) * 100
    df.loc[df.index[500:510], "wind_forecast_de_50hzt_wind_onshore"] = np.nan
    
    # Cross-border flows
    df["cross_border_net_flow_de_to_pl_lag_24h"] = 100 + np.random.randn(len(df)) * 20
    df.loc[df.index[600:608], "cross_border_net_flow_de_to_pl_lag_24h"] = np.nan
    
    # Test 1: Day-ahead prices
    print("\n[Test 1] Day-ahead prices imputation...")
    result = impute_day_ahead_prices(df.copy())
    assert result["day_ahead_price_de_lu"].notna().all(), "Prices should have no NaN"
    print("  ✓ Day-ahead prices correctly imputed")
    
    # Test 2: Bloomberg
    print("\n[Test 2] Bloomberg imputation...")
    result = impute_bloomberg(df.copy())
    # Note: Bloomberg only forward-fills, so first rows might still be NaN
    # Check that the gap we created is filled
    assert result.loc[result.index[205:220], "bloomberg_gas_price"].notna().all(), \
        "Bloomberg gap should be forward-filled"
    print("  ✓ Bloomberg correctly imputed (forward-fill only)")
    
    # Test 3: Production
    print("\n[Test 3] Production imputation...")
    result = impute_production(df.copy())
    assert result["production_forecast_dk_1"].notna().all(), "Production forecast should have no NaN"
    print("  ✓ Production correctly imputed")
    
    # Test 4: Consumption
    print("\n[Test 4] Consumption imputation...")
    result = impute_consumption(df.copy())
    assert result["consumption_forecast_load_de_50hzt"].notna().all(), "Consumption should have no NaN"
    print("  ✓ Consumption correctly imputed")
    
    # Test 5: Wind/PV
    print("\n[Test 5] Wind/PV imputation...")
    result = impute_wind_pv(df.copy())
    assert result["wind_forecast_de_50hzt_wind_onshore"].notna().all(), "Wind/PV should have no NaN"
    print("  ✓ Wind/PV correctly imputed")
    
    # Test 6: Cross-border flows
    print("\n[Test 6] Cross-border flows imputation...")
    result = impute_cross_border_flows(df.copy())
    assert result["cross_border_net_flow_de_to_pl_lag_24h"].notna().all(), "Cross-border should have no NaN"
    print("  ✓ Cross-border flows correctly imputed")
    
    print("\n✓ All feature-specific imputation tests passed!")


def test_redispatch_core_imputation():
    """Test redispatch core feature imputation."""
    print("\n" + "=" * 60)
    print("Testing Redispatch Core Imputation")
    print("=" * 60)
    
    df = create_sample_timeseries(n_rows=100)
    
    # Data-driven features
    df["active_duration_ratio_pos"] = np.random.rand(100)
    df["max_load_q90_pos"] = np.random.rand(100) * 1000
    df.loc[df.index[20:25], "active_duration_ratio_pos"] = np.nan
    df.loc[df.index[30:35], "max_load_q90_pos"] = np.nan
    
    # Runlength features
    df["run_len_pos"] = np.random.randint(0, 10, 100).astype(float)
    df.loc[df.index[40:45], "run_len_pos"] = np.nan
    
    # Indicator (should NOT be imputed)
    df["redispatch_indicator"] = np.random.choice([0, 1], 100).astype(float)
    df.loc[df.index[50:55], "redispatch_indicator"] = np.nan
    original_indicator_na_count = df["redispatch_indicator"].isna().sum()
    
    print("\n[Test 1] Redispatch core imputation...")
    result = impute_redispatch_core(df.copy())
    
    # Data-driven and runlength should be zero-filled
    assert result["active_duration_ratio_pos"].notna().all(), "Data-driven should have no NaN"
    assert result["max_load_q90_pos"].notna().all(), "Data-driven should have no NaN"
    assert result["run_len_pos"].notna().all(), "Runlength should have no NaN"
    assert (result.loc[result.index[20:25], "active_duration_ratio_pos"] == 0).all(), \
        "Data-driven gap should be zero-filled"
    print("  ✓ Data-driven and runlength features correctly zero-filled")
    
    # Indicator should NOT be imputed (still have NaN)
    assert result["redispatch_indicator"].isna().sum() == original_indicator_na_count, \
        "Indicator should NOT be imputed"
    print("  ✓ Indicator column correctly skipped")
    
    print("\n✓ All redispatch core imputation tests passed!")


def test_utility_functions():
    """Test imputation utility functions."""
    print("\n" + "=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)
    
    # Create before/after DataFrames
    df_before = create_sample_timeseries(n_rows=100)
    df_before["value1"] = np.arange(100, dtype=float)
    df_before["value2"] = np.arange(100, dtype=float)
    df_before.loc[df_before.index[10:20], "value1"] = np.nan
    df_before.loc[df_before.index[30:50], "value2"] = np.nan
    
    df_after = df_before.copy()
    df_after["value1"] = df_after["value1"].fillna(0)
    df_after["value2"] = df_after["value2"].fillna(0)
    
    # Test 1: Imputation stats
    print("\n[Test 1] Imputation statistics...")
    stats = get_imputation_stats(df_before, df_after, ["value1", "value2"])
    
    assert len(stats) == 2, "Should have stats for 2 columns"
    
    v1_stats = stats[stats["column"] == "value1"].iloc[0]
    assert v1_stats["missing_before"] == 10, f"Expected 10, got {v1_stats['missing_before']}"
    assert v1_stats["missing_after"] == 0, f"Expected 0, got {v1_stats['missing_after']}"
    assert v1_stats["imputed_count"] == 10, f"Expected 10, got {v1_stats['imputed_count']}"
    print("  ✓ Imputation stats correctly computed")
    
    # Test 2: Validate no missing
    print("\n[Test 2] Validate no missing (clean DataFrame)...")
    missing = validate_no_missing(df_after, ["value1", "value2"])
    assert len(missing) == 0, "Should have no missing columns"
    print("  ✓ Clean DataFrame validated")
    
    # Test 3: Validate no missing (with NaN)
    print("\n[Test 3] Validate no missing (with NaN)...")
    missing = validate_no_missing(df_before, ["value1", "value2"])
    assert "value1" in missing, "value1 should be flagged as having NaN"
    assert "value2" in missing, "value2 should be flagged as having NaN"
    print("  ✓ DataFrame with NaN correctly flagged")
    
    # Test 4: Validate with raise_error
    print("\n[Test 4] Validate with raise_error...")
    try:
        validate_no_missing(df_before, ["value1"], raise_error=True)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ✓ ValueError correctly raised")
    
    print("\n✓ All utility function tests passed!")


def test_impute_all_features():
    """Test the unified impute_all_features function."""
    print("\n" + "=" * 60)
    print("Testing impute_all_features")
    print("=" * 60)
    
    # Create comprehensive test DataFrame
    df = create_sample_timeseries(n_rows=24 * 30)
    
    # Add various feature types with gaps
    df["day_ahead_price_de_lu"] = 50 + np.random.randn(len(df))
    df["bloomberg_gas_price"] = 30 + np.random.randn(len(df))
    df["production_forecast_dk_1"] = 1000 + np.random.randn(len(df)) * 50
    df["consumption_forecast_load_de_50hzt"] = 500 + np.random.randn(len(df)) * 20
    df["wind_forecast_de_50hzt_wind_onshore"] = 200 + np.random.randn(len(df)) * 50
    df["cross_border_net_flow_de_to_pl_lag_24h"] = 100 + np.random.randn(len(df)) * 20
    
    # Add gaps to each
    df.loc[df.index[100:110], "day_ahead_price_de_lu"] = np.nan
    df.loc[df.index[200:210], "bloomberg_gas_price"] = np.nan
    df.loc[df.index[300:310], "production_forecast_dk_1"] = np.nan
    df.loc[df.index[400:410], "consumption_forecast_load_de_50hzt"] = np.nan
    df.loc[df.index[500:510], "wind_forecast_de_50hzt_wind_onshore"] = np.nan
    df.loc[df.index[600:608], "cross_border_net_flow_de_to_pl_lag_24h"] = np.nan
    
    print("\n[Test 1] Impute all features...")
    total_missing_before = df.isna().sum().sum()
    result = impute_all_features(df)
    total_missing_after = result.isna().sum().sum()
    
    print(f"  Total missing values: {total_missing_before} -> {total_missing_after}")
    
    # Check that most columns are now complete
    assert total_missing_after < total_missing_before, "Should have fewer missing values after imputation"
    print("  ✓ impute_all_features reduced missing values")
    
    # Check specific columns
    assert result["day_ahead_price_de_lu"].notna().all(), "Prices should be fully imputed"
    assert result["production_forecast_dk_1"].notna().all(), "Production should be fully imputed"
    assert result["cross_border_net_flow_de_to_pl_lag_24h"].notna().all(), "Cross-border should be fully imputed"
    print("  ✓ All feature types correctly processed")
    
    print("\n✓ All impute_all_features tests passed!")


def run_all_tests():
    """Run all imputation tests."""
    print("\n" + "#" * 70)
    print("# IMPUTATION MODULE TESTS")
    print("#" * 70)
    
    try:
        test_detect_gap_lengths()
        test_compute_seasonal_median()
        test_impute_with_gap_detection()
        test_impute_forward_fill_seasonal()
        test_impute_forward_fill_only()
        test_impute_fill_zero()
        test_feature_specific_imputation()
        test_redispatch_core_imputation()
        test_utility_functions()
        test_impute_all_features()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
