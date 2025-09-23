from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from .config import load_config
from .utils import ensure_dir
from .labels import prepare_cancellation_target

def build_hotel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build hotel booking features based on EDA insights from main.ipynb.

    Args:
        df: Cleaned hotel booking DataFrame

    Returns:
        DataFrame with engineered features
    """
    features = df.copy()

    # Lead time categories (strong predictor from EDA)
    features['lead_time_category'] = pd.cut(
        features['lead_time'],
        bins=[0, 7, 30, 90, 400],
        labels=['Last_Minute', 'Short', 'Medium', 'Long'],
        include_lowest=True
    )

    # Total nights stayed
    features['total_nights'] = features['stays_in_weekend_nights'] + features['stays_in_week_nights']

    # Create stay pattern features
    features['weekend_ratio'] = features['stays_in_weekend_nights'] / (features['total_nights'] + 1e-6)
    features['is_weekend_only'] = (features['stays_in_weekend_nights'] > 0) & (features['stays_in_week_nights'] == 0)
    features['is_long_stay'] = features['total_nights'] >= 7

    # Guest composition features
    features['total_guests'] = features['adults'] + features['children'] + features['babies']
    features['has_children'] = (features['children'] > 0).astype(int)
    features['has_babies'] = (features['babies'] > 0).astype(int)
    features['is_solo_traveler'] = (features['total_guests'] == 1).astype(int)
    features['is_family'] = ((features['children'] > 0) | (features['babies'] > 0)).astype(int)

    # ADR (Average Daily Rate) features
    features['adr_per_guest'] = features['adr'] / features['total_guests']
    features['total_cost'] = features['adr'] * features['total_nights']
    features['is_complimentary'] = (features['adr'] == 0).astype(int)

    # Booking behavior features
    features['has_special_requests'] = (features['total_of_special_requests'] > 0).astype(int)
    features['high_maintenance'] = (features['total_of_special_requests'] >= 3).astype(int)
    features['needs_parking'] = (features['required_car_parking_spaces'] > 0).astype(int)

    # Guest history features
    features['total_previous_bookings'] = features['previous_cancellations'] + features['previous_bookings_not_canceled']
    features['cancellation_history_ratio'] = features['previous_cancellations'] / (features['total_previous_bookings'] + 1e-6)
    features['has_cancellation_history'] = (features['previous_cancellations'] > 0).astype(int)

    # Channel/Market features (combine related categories)
    features['is_direct_booking'] = (features['distribution_channel'] == 'Direct').astype(int)
    features['is_online_ta'] = (features['market_segment'] == 'Online TA').astype(int)
    features['is_corporate'] = (features['market_segment'] == 'Corporate').astype(int)
    features['is_group_booking'] = (features['market_segment'] == 'Groups').astype(int)

    # Advance booking patterns
    features['is_last_minute'] = (features['lead_time'] <= 7).astype(int)
    features['is_far_advance'] = (features['lead_time'] >= 90).astype(int)

    # Hotel type
    features['is_city_hotel'] = (features['hotel'] == 'City Hotel').astype(int)

    # Risk factors combination
    features['low_commitment'] = (
        (features['lead_time'] > 90) &
        (features['total_of_special_requests'] == 0) &
        (features['required_car_parking_spaces'] == 0)
    ).astype(int)

    features['high_commitment'] = (
        (features['is_repeated_guest'] == 1) |
        (features['total_of_special_requests'] >= 2) |
        (features['required_car_parking_spaces'] > 0)
    ).astype(int)

    return features

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)

    # Load cleaned hotel data
    inp = Path(cfg.paths['artifacts']) / "cleaned_hotel_data.csv"
    print(f"Loading cleaned data from {inp}")
    df = pd.read_csv(inp)

    # Build features
    features = build_hotel_features(df)
    print(f"After feature engineering: {features.shape}")

    # Prepare final dataset with target
    final_data = prepare_cancellation_target(features)
    print(f"Final dataset with target: {final_data.shape}")

    # Save features
    out = Path(cfg.paths['artifacts']) / "features.csv"
    ensure_dir(out.parent)
    final_data.to_csv(out, index=False)

    print(f"Saved features to {out}")
    print(f"Feature columns: {len([col for col in final_data.columns if col != 'is_canceled'])}")
    print(f"Cancellation rate: {final_data['is_canceled'].mean():.3f}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")