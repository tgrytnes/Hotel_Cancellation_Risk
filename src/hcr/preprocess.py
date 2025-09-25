from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def load_config(path: str):
    """Simple YAML config loader."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
from .utils import ensure_dir

def clean_hotel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean hotel booking data based on preprocessing steps from main.ipynb.

    Args:
        df: Raw hotel booking DataFrame

    Returns:
        Cleaned DataFrame ready for modeling
    """
    if df.empty:
        return pd.DataFrame()

    df_cleaned = df.copy()

    # Handle missing values
    df_cleaned['children'] = df_cleaned['children'].fillna(0)
    df_cleaned['agent'] = df_cleaned['agent'].fillna(0)
    df_cleaned['company'] = df_cleaned['company'].fillna(0)
    df_cleaned['country'] = df_cleaned['country'].fillna('Unknown')

    # Remove data leakage features
    leakage_cols = [
        'reservation_status',
        'reservation_status_date',
        'deposit_type',
        'booking_changes',
        'assigned_room_type',
    ]
    existing_leakage_cols = [col for col in leakage_cols if col in df_cleaned.columns]
    if existing_leakage_cols:
        df_cleaned = df_cleaned.drop(columns=existing_leakage_cols)

    # Remove undefined entries (early data collection errors)
    df_cleaned = df_cleaned[
        ~df_cleaned['market_segment'].isin(['Undefined']) &
        ~df_cleaned['distribution_channel'].isin(['Undefined'])
    ]

    # Clean ADR outliers
    df_cleaned = df_cleaned[df_cleaned['adr'] >= 0]  # Remove negative ADR
    cap_value = df['adr'].quantile(0.9999)
    df_cleaned['adr'] = df_cleaned['adr'].clip(upper=cap_value)

    # Drop temporal components (insufficient data for reliable patterns)
    temporal_cols = ['arrival_date_year', 'arrival_date_month',
                    'arrival_date_day_of_month', 'arrival_date_week_number']
    existing_temporal_cols = [col for col in temporal_cols if col in df_cleaned.columns]
    if existing_temporal_cols:
        df_cleaned = df_cleaned.drop(columns=existing_temporal_cols)

    return df_cleaned

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)

    # Load raw hotel booking data
    raw_data_path = Path(cfg.paths['raw_data']) / "hotel_bookings.csv"
    print(f"Loading hotel booking data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"Loaded raw data: {len(df):,} rows, {len(df.columns)} columns")

    # Clean the data
    df_cleaned = clean_hotel_data(df)
    print(f"After cleaning: {len(df_cleaned):,} rows, {len(df_cleaned.columns)} columns")

    # Save cleaned data
    out = Path(cfg.paths['artifacts']) / "cleaned_hotel_data.csv"
    ensure_dir(out.parent)
    df_cleaned.to_csv(out, index=False)
    print(f"Saved cleaned hotel data to {out} (rows={len(df_cleaned):,})")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")
