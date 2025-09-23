from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from .config import load_config
from .utils import ensure_dir
from .preprocess import clean_hotel_data
from .features import build_hotel_features
from .labels import prepare_cancellation_target

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)

    # Create synthetic hotel booking data for smoke testing
    np.random.seed(42)
    n_bookings = 100

    # Generate realistic hotel booking features
    synthetic_data = pd.DataFrame({
        'hotel': np.random.choice(['City Hotel', 'Resort Hotel'], n_bookings),
        'is_canceled': np.random.choice([0, 1], n_bookings, p=[0.63, 0.37]),  # 37% cancellation rate
        'lead_time': np.random.randint(0, 365, n_bookings),
        'adults': np.random.choice([1, 2, 3, 4], n_bookings, p=[0.3, 0.5, 0.15, 0.05]),
        'children': np.random.choice([0, 1, 2], n_bookings, p=[0.8, 0.15, 0.05]),
        'babies': np.random.choice([0, 1], n_bookings, p=[0.95, 0.05]),
        'meal': np.random.choice(['BB', 'HB', 'FB', 'SC'], n_bookings),
        'country': np.random.choice(['PRT', 'GBR', 'USA', 'ESP', 'DEU', 'Unknown'], n_bookings, p=[0.3, 0.15, 0.1, 0.1, 0.1, 0.25]),
        'market_segment': np.random.choice(['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Groups'], n_bookings),
        'distribution_channel': np.random.choice(['Direct', 'Corporate', 'TA/TO'], n_bookings),
        'is_repeated_guest': np.random.choice([0, 1], n_bookings, p=[0.97, 0.03]),
        'previous_cancellations': np.random.poisson(0.1, n_bookings),
        'previous_bookings_not_canceled': np.random.poisson(0.5, n_bookings),
        'booking_changes': np.random.poisson(0.2, n_bookings),
        'agent': np.random.choice([0] + list(range(1, 20)), n_bookings, p=[0.183] + [0.043] * 19),
        'company': np.random.choice([0] + list(range(1, 5)), n_bookings, p=[0.94] + [0.015] * 4),
        'days_in_waiting_list': np.random.choice([0, 1, 2, 3], n_bookings, p=[0.95, 0.03, 0.015, 0.005]),
        'customer_type': np.random.choice(['Transient', 'Contract', 'Transient-Party', 'Group'], n_bookings),
        'adr': np.random.lognormal(mean=4.5, sigma=0.5, size=n_bookings),  # Average daily rate
        'required_car_parking_spaces': np.random.choice([0, 1], n_bookings, p=[0.92, 0.08]),
        'total_of_special_requests': np.random.poisson(0.6, n_bookings),
        'stays_in_weekend_nights': np.random.randint(0, 4, n_bookings),
        'stays_in_week_nights': np.random.randint(1, 8, n_bookings),
    })

    print(f"Generated synthetic hotel data: {synthetic_data.shape}")
    print(f"Cancellation rate: {synthetic_data['is_canceled'].mean():.3f}")

    # Test preprocessing
    cleaned_data = clean_hotel_data(synthetic_data)
    print(f"After cleaning: {cleaned_data.shape}")

    # Test feature engineering
    features = build_hotel_features(cleaned_data)
    print(f"After feature engineering: {features.shape}")

    # Test target preparation
    final_data = prepare_cancellation_target(features)
    print(f"Final dataset: {final_data.shape}")

    # Basic validation checks
    assert 'is_canceled' in final_data.columns, "Target column missing"
    assert final_data['is_canceled'].isin([0, 1]).all(), "Target should be binary"
    assert not final_data.empty, "Dataset should not be empty"

    # Save smoke test output
    out = Path(cfg.paths['artifacts']) / "hotel_smoke_test.csv"
    ensure_dir(out.parent)
    final_data.to_csv(out, index=False)

    print(f"Smoke test PASSED ✅")
    print(f"Saved smoke test data to {out}")
    print(f"Features: {[col for col in final_data.columns if col != 'is_canceled']}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")