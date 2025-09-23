from __future__ import annotations
import pandas as pd

def prepare_cancellation_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the cancellation target variable for hotel booking prediction.

    Args:
        df: Hotel booking DataFrame with 'is_canceled' column

    Returns:
        DataFrame with properly formatted target variable
    """
    result = df.copy()

    # Ensure target column exists
    if 'is_canceled' not in result.columns:
        raise ValueError("Target column 'is_canceled' not found in DataFrame")

    # Ensure target is binary (0/1)
    result['is_canceled'] = result['is_canceled'].astype(int)

    # Validate target values
    valid_values = result['is_canceled'].isin([0, 1])
    if not valid_values.all():
        invalid_count = (~valid_values).sum()
        print(f"Warning: Found {invalid_count} invalid target values, removing these rows")
        result = result[valid_values]

    return result

def make_day_ahead_label(events: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy function for backwards compatibility.
    For hotel cancellation, this simply returns the input DataFrame.
    """
    return events