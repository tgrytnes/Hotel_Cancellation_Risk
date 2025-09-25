from __future__ import annotations
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from pathlib import Path
import yaml

def load_config(path: str):
    """Simple YAML config loader."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
from .utils import ensure_dir

LEAD_TIME_BINS = [-1, 7, 30, 90, np.inf]
LEAD_TIME_LABELS = ["Last_Minute", "Short", "Medium", "Long"]
MIN_SEGMENT_CHANNEL_FREQ = 200


def build_hotel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build hotel booking features based on EDA insights from main.ipynb.

    The notebook prioritises a compact feature set: preserve the strongest
    raw signals, add lead-time buckets, total stay nights, and the
    market-segment/ distribution interaction. We intentionally avoid the
    larger collection of heuristic flags that were never discussed so the
    code now mirrors the written methodology.

    Args:
        df: Cleaned hotel booking DataFrame

    Returns:
        DataFrame with engineered, numeric features
    """
    if df.empty:
        return df.copy()

    features = df.copy()

    # Total stay length (weekend + week nights) as highlighted in the report.
    features['total_stay_nights'] = (
        features['stays_in_weekend_nights'] + features['stays_in_week_nights']
    )

    # Lead time buckets that capture the threshold effects described in the EDA.
    lead_time_bucket = pd.cut(
        features['lead_time'],
        bins=LEAD_TIME_BINS,
        labels=LEAD_TIME_LABELS,
        include_lowest=True,
        right=True,
    )
    features['lead_time_bucket'] = (
        lead_time_bucket.cat.add_categories(['Missing']).fillna('Missing')
    )

    # Market segment × distribution channel interaction feature.
    features['segment_channel'] = (
        features['market_segment'].fillna('Unknown').astype(str)
        + '__'
        + features['distribution_channel'].fillna('Unknown').astype(str)
    )

    if 'segment_channel' in features:
        counts = features['segment_channel'].value_counts(dropna=False)
        keep = counts[counts >= MIN_SEGMENT_CHANNEL_FREQ].index
        features.loc[~features['segment_channel'].isin(keep), 'segment_channel'] = 'Other'

    numeric_cols = [
        'lead_time',
        'adults',
        'children',
        'babies',
        'is_repeated_guest',
        'previous_cancellations',
        'previous_bookings_not_canceled',
        'booking_changes',
        'agent',
        'company',
        'days_in_waiting_list',
        'adr',
        'required_car_parking_spaces',
        'total_of_special_requests',
        'stays_in_weekend_nights',
        'stays_in_week_nights',
        'total_stay_nights',
    ]
    numeric_cols = [col for col in numeric_cols if col in features.columns]

    categorical_cols = [
        'hotel',
        'customer_type',
        'market_segment',
        'distribution_channel',
        'lead_time_bucket',
        'segment_channel',
    ]

    cat_feature_frames = []
    for col in categorical_cols:
        if col not in features.columns:
            continue
        series = features[col]
        if isinstance(series.dtype, CategoricalDtype):
            series = series.astype(str)
        else:
            series = series.fillna('Unknown').astype(str)
        cat_feature_frames.append(
            pd.get_dummies(series, prefix=col, prefix_sep='__', dtype=float)
        )

    categorical_features = (
        pd.concat(cat_feature_frames, axis=1) if cat_feature_frames else pd.DataFrame(index=features.index)
    )

    target_col = 'is_canceled'
    output_parts = []
    if target_col in features.columns:
        output_parts.append(features[[target_col]].astype(int))
    if numeric_cols:
        output_parts.append(features[numeric_cols].astype(float))
    if not categorical_features.empty:
        output_parts.append(categorical_features)

    if not output_parts:
        return pd.DataFrame(index=features.index)

    return pd.concat(output_parts, axis=1)

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)

    # Load cleaned hotel data
    inp = Path(cfg.paths['artifacts']) / "cleaned_hotel_data.csv"
    print(f"Loading cleaned data from {inp}")
    df = pd.read_csv(inp)

    # Build features
    features = build_hotel_features(df)
    print(f"After feature engineering: {features.shape}")

    # Features are ready
    print(f"Final dataset: {features.shape}")

    # Save features
    out = Path(cfg.paths['artifacts']) / "features.csv"
    ensure_dir(out.parent)
    features.to_csv(out, index=False)

    print(f"Saved features to {out}")
    if 'is_canceled' in features.columns:
        print(f"Feature columns: {len([col for col in features.columns if col != 'is_canceled'])}")
        print(f"Cancellation rate: {features['is_canceled'].mean():.3f}")
    else:
        print(f"Feature columns: {len(features.columns)}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")
